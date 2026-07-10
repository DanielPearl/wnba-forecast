"""Kalshi market discovery for WNBA game markets (KXWNBAGAME).

Benchmark-era data layer (2026-07 rearchitecture): the dashboard's
in-process bot drives these modules the same way it drives the MLB /
NBA bots — the legacy wnba_bot package (Elo model + sim.db) is
retired from the trading path and kept only for its historical
artifacts.

Each KXNBAGAME event is one game holding two binary markets — one per
team, each settling YES if that team wins. Two complementary sides per
event means the watchlist gets one tennis-shape X-vs-Y row per game.

Event ticker anatomy (no start time, unlike MLB)::

    KXWNBAGAME-26JUN09PHXGS-GS
               └┬┘└┬┘└┬┘└─┬─┘ └┬┘
               yy mon day away+home  market team code
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("wnba.kalshi_markets")

# (series ticker, tournament label the watchlist rows carry)
SERIES_TICKERS = (
    ("KXWNBAGAME", "WNBA"),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_STATE_PATH = _REPO_ROOT / "data" / "outputs" / "live_state.json"

_client_singleton = None

# Kalshi ticker team code → full NBA team name (The Odds API /
# Pinnacle name them this way, so benchmark lookups match directly).
# Alternate codes included where conventions disagree.
TEAM_BY_CODE: Dict[str, str] = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun", "CONN": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "GS": "Golden State Valkyries", "GSV": "Golden State Valkyries",
    "IND": "Indiana Fever",
    "LV": "Las Vegas Aces", "LVA": "Las Vegas Aces",
    "LA": "Los Angeles Sparks", "LAS": "Los Angeles Sparks",
    "MIN": "Minnesota Lynx",
    "NY": "New York Liberty", "NYL": "New York Liberty",
    "PHX": "Phoenix Mercury", "PHO": "Phoenix Mercury",
    "POR": "Portland Fire",
    "SEA": "Seattle Storm",
    "TOR": "Toronto Tempo",
    "WAS": "Washington Mystics", "WSH": "Washington Mystics",
}


def _client():
    global _client_singleton
    if _client_singleton is None:
        from kalshi_sdk import KalshiClient
        api_key = os.environ.get("KALSHI_API_KEY_ID", "").strip()
        pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
        _client_singleton = KalshiClient(api_key_id=api_key,
                                         private_key_path=pkey)
    return _client_singleton


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_wnba_markets() -> List[Dict[str, Any]]:
    """All open KXWNBAGAME markets. Settlement of positions whose market left
    the open set is handled by the simulator's orphan sweep (it
    queries each position's market directly).

    Paginates at limit=100: Kalshi's query-exchange 503s on the busy
    sport series at the SDK's hardcoded limit=200 (first observed on
    KXMLBGAME, 2026-07-09; 100 is reliably fine).
    """
    c = _client()
    out: List[Dict[str, Any]] = []
    for series, _label in SERIES_TICKERS:
        n_before = len(out)
        cursor = None
        while True:
            resp = c.get_markets(limit=100, cursor=cursor, status="open",
                                 series_ticker=series)
            out.extend(resp.get("markets", []) or [])
            cursor = resp.get("cursor")
            if not cursor:
                break
        log.info("fetched %d open %s markets", len(out) - n_before, series)
    return out


_EVENT_BODY_RE = re.compile(
    r"^(?P<series>KXWNBAGAME)-"
    r"(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<codes>[A-Z]{2,8})$",
)
_TOURNAMENT_BY_SERIES = {s: label for s, label in SERIES_TICKERS}
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _parse_event_ticker(event_ticker: str) -> Dict[str, Any]:
    """Date + away/home codes from the event ticker. Returns {} when
    the ticker doesn't match the expected anatomy."""
    m = _EVENT_BODY_RE.match(event_ticker or "")
    if not m:
        return {}
    month = _MONTHS.get(m.group("mon"))
    if not month:
        return {}
    year = 2000 + int(m.group("yy"))
    return {
        "date": f"{year}-{month:02d}-{int(m.group('dd')):02d}",
        "codes": m.group("codes"),
        "tournament": _TOURNAMENT_BY_SERIES.get(m.group("series"), "WNBA"),
    }


def _split_codes(codes: str, suffixes: List[str]) -> tuple[str, str] | None:
    """Split the ticker body's away+home concatenation using the two
    market suffix codes. The body always ends with the HOME code."""
    for home in suffixes:
        if codes.endswith(home):
            away = codes[: -len(home)]
            if away in suffixes:
                return away, home
    return None


def collapse_to_matches(markets: List[Dict[str, Any]],
                        prev_markets_by_ticker: Dict[str, Dict] | None = None,
                        ) -> List[Dict[str, Any]]:
    """Group KXWNBAGAME markets by event into one record per game —
    same shape as the MLB collapse (markets.team_a / markets.team_b,
    team_a = away, team_b = home). ``kickoff`` carries the market's
    occurrence_datetime (tip-off) when Kalshi provides it."""
    by_event: Dict[str, List[Dict[str, Any]]] = {}
    for m in markets:
        et = m.get("event_ticker") or ""
        if et:
            by_event.setdefault(et, []).append(m)

    records: List[Dict[str, Any]] = []
    for et, ms in sorted(by_event.items()):
        if len(ms) < 2:
            continue
        parsed = _parse_event_ticker(et)
        suffix_by_ticker = {
            m.get("ticker", ""): m.get("ticker", "").rsplit("-", 1)[-1]
            for m in ms
        }
        suffixes = [s for s in suffix_by_ticker.values() if s]
        split = (_split_codes(parsed["codes"], suffixes)
                 if parsed.get("codes") else None)
        if split is None:
            log.warning("could not split team codes for %s (markets %s)",
                        et, suffixes)
            continue
        away_code, home_code = split

        def _mkt_for(code: str) -> Dict[str, Any] | None:
            for m in ms:
                if suffix_by_ticker.get(m.get("ticker", "")) == code:
                    return m
            return None

        outcome_markets: Dict[str, Dict[str, Any]] = {}
        names: Dict[str, str] = {}
        for outcome, code in (("team_a", away_code), ("team_b", home_code)):
            m = _mkt_for(code)
            if m is None:
                break
            yes_bid = _f(m.get("yes_bid_dollars"))
            yes_ask = _f(m.get("yes_ask_dollars"))
            spread_cents = None
            if yes_bid is not None and yes_ask is not None:
                spread_cents = round((yes_ask - yes_bid) * 100)
            names[outcome] = (TEAM_BY_CODE.get(code)
                              or (m.get("yes_sub_title") or code).strip())
            outcome_markets[outcome] = {
                "ticker": m.get("ticker") or "",
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "status": m.get("status"),
                "result": m.get("result"),
                "open_interest": _f(m.get("open_interest_fp")) or 0.0,
                "volume": _f(m.get("volume_fp")) or 0.0,
                "spread_cents": spread_cents,
                "label": (m.get("yes_sub_title") or "").strip(),
                "rules_primary": (m.get("rules_primary") or "").strip(),
            }
        if len(outcome_markets) != 2:
            continue

        # NO kickoff from Kalshi: KXNBA*GAME tickers carry no start
        # time and occurrence_datetime is the expected game END
        # (== expected_expiration_time) — treating it as tip-off left
        # the prematch gate open for the entire game and the live
        # executor bought two Summer League games 17 and 34 minutes
        # in-play on 2026-07-09. The exporter fills ``kickoff`` from
        # the benchmark feed's scheduled start (Pinnacle guest
        # startTime / Odds-API commence_time); rows without it stay
        # ineligible (the prematch gate fails closed on kickoff=None).
        records.append({
            "match_id": et,
            "event_ticker": et,
            "event_title": f"{names['team_a']} vs {names['team_b']}",
            "team_a": names["team_a"],
            "team_b": names["team_b"],
            "tournament": parsed.get("tournament", "WNBA"),
            "date": parsed.get("date"),
            "kickoff": None,
            "markets": outcome_markets,
        })
    return records


def get_market_status(ticker: str) -> Dict[str, Any] | None:
    """Direct market lookup for the simulator's orphan sweep (position
    open, market no longer in the open set → probably finalized)."""
    try:
        m = _client().get_market(ticker)
        if isinstance(m, dict) and "market" in m:
            m = m["market"]
        return {"status": m.get("status"), "result": m.get("result"),
                "yes_bid": _f(m.get("yes_bid_dollars"))}
    except Exception:  # noqa: BLE001
        log.exception("get_market_status(%s) failed", ticker)
        return None


def write_live_state(records: List[Dict[str, Any]]) -> None:
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    tmp = LIVE_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(LIVE_STATE_PATH)
