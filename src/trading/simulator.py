"""Paper-trading simulator for WNBA game markets.

Same behaviors as the World Cup / tennis simulators, on two-way
per-team binary markets (each position is YES on one KXWNBAGAME
market):

  1. settle positions whose market finalized (result yes/no)
  1b. orphan sweep — position open but market gone from the open set:
      query Kalshi directly and settle from its status/result
  2. mark open positions to market from the current watchlist
  2b. profit-lock exit when our side trades >= 95¢
  3. open new positions from buy_eligible rows, best edge first,
     one position per game (event), capped total, pre-match only

State file (data/outputs/sim_state.json) uses the exact schema the
dashboard's sport adapter reads: open_positions / closed_positions /
stats / last_settled_at_by_match_id.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("wnba.simulator")

_REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = _REPO_ROOT / "data" / "outputs" / "sim_state.json"

DEFAULTS = {
    "stake": 1.0,
    "slippage": 0.02,
    # A WNBA slate is ~6 games/night — allow more concurrent
    # positions than the WC bot's 6, still one per game.
    "max_open_positions": 25,  # raised 10 → 25 for $1-stake testing (2026-07-10)
    "max_positions_per_match": 1,
    "profit_lock_price": 0.95,
    "high_edge_taper": 0.20,   # edges above this get a reduced stake —
    "high_edge_stake": 0.75,   # a huge Pinnacle-vs-Kalshi gap usually
                               # means a stale quote (lineup news, rain
                               # delay), not free money
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            log.exception("sim_state unreadable — starting fresh")
    return {
        "started_at": _now_iso(),
        "last_tick_at": None,
        "open_positions": [],
        "closed_positions": [],
        "stats": {},
        "last_settled_at_by_match_id": {},
    }


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


def _recompute_stats(state: Dict[str, Any]) -> None:
    closed = state["closed_positions"]
    wins = sum(1 for p in closed if p.get("won"))
    losses = sum(1 for p in closed if p.get("won") is False)
    realized = sum(p.get("realized_pnl") or 0.0 for p in closed)
    unrealized = sum(p.get("unrealized_pnl") or 0.0
                     for p in state["open_positions"])
    staked = sum(p.get("stake") or 0.0
                 for p in closed + state["open_positions"])
    state["stats"] = {
        "total_opened": len(closed) + len(state["open_positions"]),
        "total_closed": len(closed),
        "open_count": len(state["open_positions"]),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / (wins + losses)) if (wins + losses) else 0.0,
        "total_realized_pnl": round(realized, 4),
        "total_unrealized_pnl": round(unrealized, 4),
        "total_staked": round(staked, 4),
        "roi": round(realized / staked, 4) if staked else 0.0,
    }


def _settle(pos: Dict[str, Any], result: str, reason: str) -> Dict[str, Any]:
    # Positions are always YES buys on their own team's market.
    won = result == "yes"
    stake, entry, slip = pos["stake"], pos["entry_market_prob"], pos["slippage"]
    pnl = stake * (1 - entry - slip) if won else -stake * (entry + slip)
    pos.update({
        "closed_at": _now_iso(),
        "winner_side": pos["side"] if won else (
            "PLAYER_B" if pos["side"] == "PLAYER_A" else "PLAYER_A"),
        "won": won,
        "settle_market_prob": 1.0 if won else 0.0,
        "realized_pnl": round(pnl, 4),
        "close_reason": reason,
        "result": "SETTLED",
    })
    return pos


def _close_at_market(pos: Dict[str, Any], exit_price: float) -> Dict[str, Any]:
    stake, entry, slip = pos["stake"], pos["entry_market_prob"], pos["slippage"]
    pnl = stake * (exit_price - entry - slip)
    pos.update({
        "closed_at": _now_iso(),
        "winner_side": None,
        "won": pnl > 0,
        "settle_market_prob": exit_price,
        "realized_pnl": round(pnl, 4),
        "close_reason": "profit_lock",
        "result": "PROFIT_LOCK",
    })
    return pos


def tick(watchlist_rows: List[Dict[str, Any]],
         records: List[Dict[str, Any]],
         cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(cfg or {})}
    state = _load_state()
    # Map every side's market ticker -> (row, 'a'|'b') so positions on
    # either side of a game find their row for mark-to-market.
    rows_by_ticker: Dict[str, tuple] = {}
    for r in watchlist_rows:
        if r.get("ticker_a"):
            rows_by_ticker[r["ticker_a"]] = (r, "a")
        if r.get("ticker_b"):
            rows_by_ticker[r["ticker_b"]] = (r, "b")
        rows_by_ticker.setdefault(r["match_id"], (r, "a"))

    # Flatten the per-event records into a per-market map for status.
    mkt_by_ticker: Dict[str, Dict[str, Any]] = {}
    event_by_ticker: Dict[str, str] = {}
    for rec in records:
        for mkt in (rec.get("markets") or {}).values():
            t = mkt.get("ticker")
            if t:
                mkt_by_ticker[t] = mkt
                event_by_ticker[t] = rec["event_ticker"]

    still_open: List[Dict[str, Any]] = []
    for pos in state["open_positions"]:
        ticker = pos.get("ticker") or pos["match_id"]
        mkt = mkt_by_ticker.get(ticker)

        # 1. settle finalized markets present in the live snapshot
        if mkt and (mkt.get("status") or "") in ("finalized", "settled") \
                and (mkt.get("result") or "") in ("yes", "no"):
            state["closed_positions"].append(
                _settle(pos, mkt["result"], "settlement"))
            state["last_settled_at_by_match_id"][ticker] = _now_iso()
            continue

        # 1b. orphan sweep — market vanished from the open set
        if mkt is None:
            from ..data import kalshi_markets as km
            info = km.get_market_status(ticker)
            if info and (info.get("status") or "") in ("finalized", "settled") \
                    and (info.get("result") or "") in ("yes", "no"):
                state["closed_positions"].append(
                    _settle(pos, info["result"],
                            "auto-settle from Kalshi"))
                state["last_settled_at_by_match_id"][ticker] = _now_iso()
                continue
            still_open.append(pos)  # unknown — hold and retry next tick
            continue

        # 2. mark-to-market
        cur = mkt.get("yes_ask")
        if cur is not None:
            pos["current_market_prob"] = cur
            pos["unrealized_pnl"] = round(
                pos["stake"] * (cur - pos["entry_market_prob"]), 4)
        row_entry = rows_by_ticker.get(ticker)
        if row_entry:
            row, row_side = row_entry
            p = (row.get("live_prob_a") if row_side == "a"
                 else row.get("live_prob_b"))
            if p is not None:
                pos["current_model_prob"] = p

        # 2b. profit-lock exit — our side effectively decided
        if cur is not None and cur >= cfg["profit_lock_price"]:
            state["closed_positions"].append(_close_at_market(pos, cur))
            state["last_settled_at_by_match_id"][ticker] = _now_iso()
            continue

        still_open.append(pos)
    state["open_positions"] = still_open

    # 3. open new positions
    open_events = {event_by_ticker.get(p.get("ticker") or p["match_id"],
                                       p.get("event_ticker"))
                   for p in state["open_positions"]}
    # Never re-enter a game we already traded (either side).
    traded_events = set(open_events)
    for t in state["last_settled_at_by_match_id"]:
        traded_events.add(event_by_ticker.get(t) or t.rsplit("-", 1)[0])

    candidates = sorted(
        (r for r in watchlist_rows if r.get("buy_eligible")),
        key=lambda r: r.get("buy_score") or 0.0, reverse=True)
    for row in candidates:
        if len(state["open_positions"]) >= cfg["max_open_positions"]:
            break
        event = row.get("event_ticker") or ""
        if event in traded_events:
            continue
        side = "PLAYER_A" if row.get("buy_side") == "A" else "PLAYER_B"
        entry = (row.get("market_prob_a") if side == "PLAYER_A"
                 else row.get("market_prob_b"))
        model_p = (row.get("live_prob_a") if side == "PLAYER_A"
                   else row.get("live_prob_b"))
        edge = row.get("buy_side_edge") or 0.0
        if entry is None or model_p is None:
            continue
        # Each side is a YES buy on its own team's market.
        pos_ticker = (row.get("ticker_a") if side == "PLAYER_A"
                      else row.get("ticker_b")) or row["match_id"]
        stake = (cfg["high_edge_stake"] if edge >= cfg["high_edge_taper"]
                 else cfg["stake"])
        side_label = (row["player_a"] if side == "PLAYER_A"
                      else row["player_b"])
        pos = {
            "position_id": f"{pos_ticker}-{side}-{int(time.time())}",
            "match_id": row["match_id"],
            "ticker": pos_ticker,
            "market_side": "YES",
            "event_ticker": event,
            "tournament": row.get("tournament") or "WNBA",
            "surface": row.get("surface") or "Basketball",
            "player_a": row["player_a"],
            "player_b": row.get("player_b") or "Field",
            "side": side,
            "side_player": side_label,
            "entry_market_prob": entry,
            "entry_model_prob": model_p,
            "label_at_open": row.get("recommended_action") or "",
            "stake": stake,
            "slippage": cfg["slippage"],
            "opened_at": _now_iso(),
            "current_market_prob": entry,
            "current_model_prob": model_p,
            "unrealized_pnl": 0.0,
            "reason_at_open": (
                f"Pinnacle {edge * 100:+.1f}pp vs market on "
                f"{side_label} winning ({row.get('event_title')})"),
        }
        state["open_positions"].append(pos)
        traded_events.add(event)
        log.info("opened %s @ %.2f (%s)", pos["position_id"], entry,
                 pos["reason_at_open"])

    state["last_tick_at"] = _now_iso()
    _recompute_stats(state)
    _save_state(state)
    return state
