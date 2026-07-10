"""Build and export the WNBA watchlist (benchmark-era, 2026-07).

One row per game, exactly like a tennis match row: team_a (away) vs
team_b (home), each side backed by its own binary Kalshi market.

The probability source IS the sharp-book benchmark: the devigged
Pinnacle NBA moneyline — Pinnacle's own guest feed first (fresher and
key-free), The Odds API cascade (Pinnacle → Betfair Exchange) filling
any gaps — via kalshi_sdk.pinnacle. The legacy Elo model (nba_bot
package) no longer drives trading; edge is "Pinnacle's fair prob minus
Kalshi's ask", pure cross-venue price discrepancy. Rows with no
benchmark line are WATCH-only and never tradeable.

Buy gates mirror the MLB exporter: benchmark edge over the chosen
side's ask, EV after slippage, entry-price band, open interest,
spread cap — plus a pre-match gate (Pinnacle pre-match lines go stale
at tip-off, so no opens once the game is inside the buffer).
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("wnba.watchlist")

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = _REPO_ROOT / "data" / "outputs"
WATCHLIST_JSON = OUT_DIR / "watchlist.json"
WATCHLIST_CSV = OUT_DIR / "watchlist.csv"

# Trading gates — keep in lockstep with the MLB exporter's DEFAULTS.
DEFAULTS = {
    "slippage": 0.02,
    "min_edge": 0.05,            # SMALL_EDGE floor (benchmark prob − ask)
    "strong_edge": 0.10,
    "min_entry_price": 0.15,     # per-side ask band, dollars
    "max_entry_price": 0.80,
    "max_spread_cents": 6,
    "min_open_interest": 1,
    "prematch_buffer_minutes": 10,
}

TRADEABLE_LABELS = {"STRONG_EDGE", "SMALL_EDGE"}

# WNBA is a single Odds-API key; discovery under the prefix keeps
# any future variants covered. The guest feed uses the friendly
# "basketball" sport (id 4 — includes the WNBA league) and
# pair-matching by full team names keeps other leagues in that feed
# from colliding.
BENCHMARK_SPORT_KEY_PREFIX = "basketball_wnba"
BENCHMARK_FALLBACK_KEYS = ["basketball_wnba"]
GUEST_SPORT = "basketball"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _benchmark_lookup() -> Dict[frozenset, Dict[str, float]]:
    """Devigged Pinnacle (guest feed → Odds-API cascade) win
    probabilities keyed by team pair. Graceful empty dict when both
    sources are unavailable — every row just renders WATCH."""
    try:
        from kalshi_sdk.pinnacle import (
            benchmark_probs_by_pair_with_guest, discover_sport_keys)
    except ImportError:
        return {}
    try:
        keys = (discover_sport_keys(BENCHMARK_SPORT_KEY_PREFIX)
                or BENCHMARK_FALLBACK_KEYS)
        return benchmark_probs_by_pair_with_guest(
            keys, guest_sport=GUEST_SPORT) or {}
    except Exception:  # noqa: BLE001
        log.exception("benchmark lookup failed")
        return {}


def _benchmark_for(lookup: Dict[frozenset, Dict[str, float]],
                   team_a: str, team_b: str):
    """(prob_a, prob_b, start_iso) from the lookup, tolerant of
    name-casing and alias drift between Kalshi codes and the books.
    ``start_iso`` is the book's scheduled tip-off — the only reliable
    start time for NBA (Kalshi's occurrence_datetime is the game
    end), so the prematch gate keys off it."""
    probs = lookup.get(frozenset({team_a, team_b}))
    if probs is None:
        want_a, want_b = team_a.lower(), team_b.lower()
        for names, p in lookup.items():
            lowered = {n.lower() for n in names if not n.startswith("_")}
            if all(any(w in n or n in w for n in lowered)
                   for w in (want_a, want_b)):
                probs = p
                break
    if not probs:
        return None, None, None

    def _match(team: str):
        t = team.lower()
        for k, v in probs.items():
            if k.startswith("_"):
                continue
            kl = k.lower()
            if kl == t or t in kl or kl in t:
                return v
        return None

    return _match(team_a), _match(team_b), probs.get("_start")


def build_watchlist_records(records: List[Dict[str, Any]],
                            cfg: Dict[str, Any] | None = None,
                            ) -> List[Dict[str, Any]]:
    cfg = {**DEFAULTS, **(cfg or {})}
    now = _now()
    rows: List[Dict[str, Any]] = []
    benchmark = _benchmark_lookup()

    for rec in records:
        mkts = rec.get("markets") or {}
        mkt_a, mkt_b = mkts.get("team_a"), mkts.get("team_b")
        if not (mkt_a and mkt_b and mkt_a.get("ticker")
                and mkt_b.get("ticker")):
            continue

        p_a, p_b, bench_start = _benchmark_for(
            benchmark, rec["team_a"], rec["team_b"])
        if p_a is None and p_b is not None:
            p_a = 1.0 - p_b
        if p_b is None and p_a is not None:
            p_b = 1.0 - p_a

        # Tip-off comes from the benchmark feed (Kalshi has no usable
        # start time — see collapse_to_matches). No start time → the
        # prematch gate below fails CLOSED, so a benchmarked game with
        # a missing/unparseable start can never be bought mid-game.
        kickoff_iso = bench_start or rec.get("kickoff")
        kickoff = _parse_ts(kickoff_iso)
        minutes_to_kickoff = ((kickoff - now).total_seconds() / 60.0
                              if kickoff else None)
        prematch = (minutes_to_kickoff is not None
                    and minutes_to_kickoff > cfg["prematch_buffer_minutes"])

        ask_a, ask_b = mkt_a.get("yes_ask"), mkt_b.get("yes_ask")
        edge_a = (p_a - ask_a) if (p_a is not None and ask_a) else None
        edge_b = (p_b - ask_b) if (p_b is not None and ask_b) else None
        ev_a = (edge_a - cfg["slippage"]) if edge_a is not None else None
        ev_b = (edge_b - cfg["slippage"]) if edge_b is not None else None

        side, side_edge, side_ev = "A", edge_a, ev_a
        side_price, side_mkt = ask_a, mkt_a
        if (edge_b is not None) and (edge_a is None or edge_b > edge_a):
            side, side_edge, side_ev = "B", edge_b, ev_b
            side_price, side_mkt = ask_b, mkt_b

        gates: Dict[str, bool] = {}
        if p_a is None:
            label, reason = "WATCH", "no Pinnacle line for this game yet"
            eligible = False
        else:
            gates = {
                "edge": (side_edge or 0) >= cfg["min_edge"],
                "ev": (side_ev or 0) > 0,
                "price_band": (side_price is not None
                               and cfg["min_entry_price"] <= side_price
                               <= cfg["max_entry_price"]),
                "open_interest": (side_mkt.get("open_interest") or 0)
                >= cfg["min_open_interest"],
                "spread": (side_mkt.get("spread_cents") is not None
                           and side_mkt["spread_cents"]
                           <= cfg["max_spread_cents"]),
                "prematch": prematch,
            }
            eligible = all(gates.values())
            if (side_edge or 0) >= cfg["strong_edge"]:
                label = "STRONG_EDGE"
            elif (side_edge or 0) >= cfg["min_edge"]:
                label = "SMALL_EDGE"
            else:
                label = "WATCH"
            eligible = eligible and label in TRADEABLE_LABELS
            blocked = [k for k, ok in gates.items() if not ok]
            side_team = rec["team_a"] if side == "A" else rec["team_b"]
            if eligible:
                reason = (f"Pinnacle {side_edge * 100:+.1f}pp vs market on "
                          f"{side_team} winning")
            elif label in TRADEABLE_LABELS and blocked:
                reason = "edge present but blocked: " + ", ".join(blocked)
            else:
                reason = (f"Pinnacle within {cfg['min_edge'] * 100:.0f}pp "
                          f"of market")

        rows.append({
            "match_id": rec["event_ticker"],
            "event_ticker": rec["event_ticker"],
            "ticker_a": mkt_a["ticker"],
            "ticker_b": mkt_b["ticker"],
            "tournament": rec.get("tournament") or "WNBA",
            "surface": "Basketball",
            "round_label": "REG",
            "player_a": rec["team_a"],
            "player_b": rec["team_b"],
            "player_a_canonical": rec["team_a"].lower(),
            "player_b_canonical": rec["team_b"].lower(),
            "current_score": "",
            "outcome": "win",
            # Benchmark-sourced tip-off — the executor's own prematch
            # check reads this off the row.
            "kickoff": kickoff_iso,
            "match_date": rec.get("date"),

            # The benchmark IS the model for this bot.
            "pre_match_prob_a": p_a,
            "pre_match_prob_b": p_b,
            "live_prob_a": p_a,
            "live_prob_b": p_b,
            "market_prob_a": ask_a,
            "market_prob_b": ask_b,
            "pinnacle_prob_a": p_a,
            "pinnacle_prob_b": p_b,
            "_skip_oi_filter": True,

            "edge_a": edge_a,
            "edge_b": edge_b,
            "ev_a": ev_a,
            "ev_b": ev_b,
            "confidence_score": (max(p_a, p_b)
                                 if p_a is not None else None),
            "volatility_score": 0.0,
            "injury_news_flag": False,

            "recommended_action": label,
            "recommendation": "Buy" if eligible else (
                "Watch" if label == "WATCH" else "Hold"),
            "reason_for_signal": reason,
            "last_updated": now.isoformat(timespec="seconds"),

            "open_interest": ((mkt_a.get("open_interest") or 0)
                              + (mkt_b.get("open_interest") or 0)),
            "volume": ((mkt_a.get("volume") or 0)
                       + (mkt_b.get("volume") or 0)),
            "spread_cents": mkt_a.get("spread_cents"),
            "spread_cents_a": mkt_a.get("spread_cents"),
            "spread_cents_b": mkt_b.get("spread_cents"),
            "yes_ask_cents_a": (round(ask_a * 100)
                                if ask_a is not None else None),
            "yes_ask_cents_b": (round(ask_b * 100)
                                if ask_b is not None else None),

            "title_a": f"Will {rec['team_a']} win? — "
                       f"{rec['event_title']}",
            "title_b": f"Will {rec['team_b']} win? — "
                       f"{rec['event_title']}",
            "title": None,
            "event_title": rec["event_title"],
            "rules_primary": ((mkt_a if (p_a or 0) >= 0.5 else mkt_b)
                              .get("rules_primary")
                              or mkt_a.get("rules_primary") or ""),

            "buy_eligible": eligible,
            "buy_score": float(side_edge or 0.0),
            "buy_side": side,
            "buy_side_edge": side_edge,
            "buy_side_ev": side_ev,
            "buy_gates": gates,
            "buy_blockers": [k for k, ok in gates.items() if not ok],
        })

    rows.sort(key=lambda r: r["buy_score"], reverse=True)
    return rows


def export(records: List[Dict[str, Any]]) -> None:
    """Write watchlist.json (+ CSV) the dashboard reads."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": _now().isoformat(timespec="seconds"),
               "rows": records}
    tmp = WATCHLIST_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(WATCHLIST_JSON)
    if records:
        cols = list(records[0].keys())
        with WATCHLIST_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list))
                                else v) for k, v in r.items()})
    log.info("exported %d watchlist rows", len(records))
