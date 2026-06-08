"""Discover and parse Kalshi WNBA game-outcome markets.

Series: ``KXWNBAGAME``

Ticker pattern: ``KXWNBAGAME-{YY}{MMM}{DD}{AWAY}{HOME}-{TEAM}``
  e.g. ``KXWNBAGAME-26JUN09PHXGS-GS`` resolves YES if Golden State beats
  Phoenix in the 2026-06-09 game played at Golden State (home).

Unlike the NBA bot, WNBA team codes are **variable length** (2–4
chars: GS, NY, LV, LA = 2; CONN = 4; the rest = 3), so the body
``PHXGS`` can't be split with a fixed-width regex. We split it against
the known WNBA team set (``WNBA_TEAM_CODES``), disambiguating with the
``-{TEAM}`` suffix on a market ticker (which is always one of the two
sides).

Each event (e.g. ``KXWNBAGAME-26JUN09PHXGS``) has TWO markets — one
asking "does AWAY win?" and one asking "does HOME win?". Their YES
prices should sum to ~1.00 (modulo the spread). The bot's decision
engine evaluates each market independently — buy YES on the side
whose model probability beats the Kalshi ask by enough.

Kept the legacy ``GasMarket`` class name for cross-bot dashboard /
simulator compatibility — semantically it's a generic Kalshi market
dataclass.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .client import KalshiClient, Orderbook
from .data_sources import WNBA_TEAM_CODES, normalize_tricode

log = logging.getLogger(__name__)


@dataclass
class GasMarket:
    """One Kalshi NBA game market.

    Field semantics for NBA (kept compatible with the cross-bot
    ``GasMarket`` shape — `strike_low`/`strike_high` are not used in
    the usual numeric sense, only `team_being_asked` matters):
      ``team_being_asked``  three-letter team code the YES side wins on
      ``home_tricode``       home team in the matchup
      ``away_tricode``       away team in the matchup
      ``game_date``          game date (UTC date the game starts)
    """
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    yes_sub_title: str
    strike_low: Optional[float]      # always None for NBA — kept for compat
    strike_high: Optional[float]     # always None for NBA
    direction: str                   # always "team_wins" for NBA
    close_time: datetime
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    yes_bid_cents: Optional[int]
    volume: int
    open_interest: int = 0
    rules_primary: str = ""
    rules_secondary: str = ""
    event_title: str = ""
    event_sub_title: str = ""
    # NBA-specific:
    team_being_asked: str = ""
    home_tricode: str = ""
    away_tricode: str = ""
    game_date: Optional[datetime] = None
    raw: dict = None  # type: ignore[assignment]


def minutes_to_close(market: GasMarket,
                     now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    delta = (market.close_time - now).total_seconds() / 60.0
    return max(0.0, delta)


def time_to_close_str(mtc_minutes: float | None) -> str:
    if mtc_minutes is None:
        return "—"
    if mtc_minutes < 60:
        return f"{int(round(mtc_minutes))}m"
    if mtc_minutes < 60 * 24:
        return f"{mtc_minutes / 60:.1f}h"
    return f"{mtc_minutes / (60 * 24):.1f}d"


# --------------------------------------------------------------------------- #
# Ticker / event parsing
# --------------------------------------------------------------------------- #

# Event ticker: KXWNBAGAME-YYMMMDD{AWAY}{HOME} (e.g. 26JUN09PHXGS)
# Sub-event suffix: -TEAM (e.g. -GS)
#
# We capture the date prefix and the concatenated team body separately,
# then split the body against the known WNBA team set since codes are
# variable length (GS/NY/LV/LA=2, CONN=4, rest=3).
_EVENT_RE = re.compile(
    r"KXWNBAGAME-(?P<yy>\d{2})(?P<mmm>[A-Z]{3})(?P<dd>\d{2})(?P<body>[A-Z]{4,8})$"
)
_TICKER_RE = re.compile(
    r"KXWNBAGAME-(?P<yy>\d{2})(?P<mmm>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<body>[A-Z]{4,8})-(?P<team>[A-Z]{2,4})$"
)
_MMM_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _split_body(body: str, team: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Split a concatenated ``{AWAY}{HOME}`` body into (away, home).

    ``team`` (when provided, from the market ticker's -TEAM suffix) is
    one of the two sides — we use it to disambiguate. Falls back to a
    known-team-set search when no team hint is available (event ticker).
    Returns codes normalized to Kalshi convention, or None if unparseable.
    """
    body = (body or "").upper()
    if team:
        team = normalize_tricode(team)
        # team is one of the two sides; figure out which end it occupies.
        if body.startswith(team):
            other = body[len(team):]
            if other:  # team is AWAY (listed first), other is HOME
                return team, other
        if body.endswith(team):
            other = body[:len(body) - len(team)]
            if other:  # team is HOME (listed last), other is AWAY
                return other, team
        return None
    # No hint: search for a split point where both halves are known teams.
    for i in range(2, len(body) - 1):
        away, home = body[:i], body[i:]
        if normalize_tricode(away) in WNBA_TEAM_CODES and \
           normalize_tricode(home) in WNBA_TEAM_CODES:
            return normalize_tricode(away), normalize_tricode(home)
    return None


def _parse_event_ticker(event_ticker: str) -> Optional[Tuple[datetime, str, str]]:
    """Decode an event ticker into (game_date_utc, away_tri, home_tri).

    Tickers don't carry a tipoff time — only the date. We anchor at
    midnight UTC of that date and rely on the market's ``close_time``
    field for the actual close cadence.
    """
    m = _EVENT_RE.match(event_ticker or "")
    if not m:
        return None
    try:
        year = 2000 + int(m.group("yy"))
        month = _MMM_TO_NUM[m.group("mmm")]
        day = int(m.group("dd"))
    except (ValueError, KeyError):
        return None
    split = _split_body(m.group("body"))
    if split is None:
        return None
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return dt, split[0], split[1]


def _parse_market_ticker(ticker: str) -> Optional[Tuple[datetime, str, str, str]]:
    """Decode a market ticker into (game_date, away, home, team_being_asked).

    ``team_being_asked`` must equal either ``away`` or ``home``;
    otherwise the ticker is malformed and we return None.
    """
    m = _TICKER_RE.match(ticker or "")
    if not m:
        return None
    try:
        year = 2000 + int(m.group("yy"))
        month = _MMM_TO_NUM[m.group("mmm")]
        day = int(m.group("dd"))
    except (ValueError, KeyError):
        return None
    team = normalize_tricode(m.group("team"))
    split = _split_body(m.group("body"), team=team)
    if split is None:
        return None
    away, home = split
    if team not in (away, home):
        return None
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return dt, away, home, team


# --------------------------------------------------------------------------- #
# Misc parsing helpers
# --------------------------------------------------------------------------- #

def _to_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _volume_from_market(raw: dict) -> int:
    fp = raw.get("volume_fp")
    if fp not in (None, ""):
        try:
            return int(round(float(fp)))
        except (TypeError, ValueError):
            pass
    legacy = raw.get("volume")
    if legacy not in (None, ""):
        try:
            return int(legacy)
        except (TypeError, ValueError):
            pass
    return 0


def _open_interest_from_market(raw: dict) -> int:
    fp = raw.get("open_interest_fp")
    if fp not in (None, ""):
        try:
            return int(round(float(fp)))
        except (TypeError, ValueError):
            pass
    legacy = raw.get("open_interest")
    if legacy not in (None, ""):
        try:
            return int(legacy)
        except (TypeError, ValueError):
            pass
    return 0


def _parse_close_time(raw_close: Optional[str]) -> datetime:
    if not raw_close:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw_close.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _effective_close_time(raw: dict) -> datetime:
    """Return when the market actually expects to close.

    Kalshi NBA game markets have ``can_close_early: true`` — the
    nominal ``close_time`` is set ~2 weeks out as a fallback (in case
    of repeated postponements), but ``expected_expiration_time`` /
    ``occurrence_datetime`` carries the actual tipoff time when the
    market closes early. Without this fix the bot saw 15+ day
    minutes_to_close on games that were 1 day away.
    """
    for field in ("expected_expiration_time", "occurrence_datetime",
                  "close_time"):
        v = raw.get(field)
        if v:
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                continue
    return datetime.now(timezone.utc)


def _cents_from_dollars(value) -> Optional[int]:
    """Kalshi's modern fields are decimal-dollar strings ('0.7600').
    Some legacy fields are still cent ints. Handle both.
    """
    if value is None or value == "":
        return None
    try:
        # If it parses as a float < 100, treat as dollars.
        f = float(value)
        if 0 <= f <= 100 and "." in str(value):
            return int(round(f * 100))
        return int(round(f))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_wnba_markets(
    client: KalshiClient,
    series_prefixes: List[str],
    max_markets: int = 200,
) -> List[GasMarket]:
    """Pull every open Kalshi WNBA game market.

    Each game's event has TWO markets (one per team) — both are
    returned. Validators downstream filter by volume / time-to-close.
    """
    seen: dict[str, dict] = {}
    for pref in series_prefixes:
        cursor: Optional[str] = None
        while True:
            resp = client.get_markets(
                limit=200, cursor=cursor, status="open",
                series_ticker=pref,
            )
            for m in resp.get("markets", []):
                tk = m.get("ticker", "")
                if tk.startswith(pref):
                    seen[tk] = m
            cursor = resp.get("cursor")
            if not cursor or len(seen) >= max_markets:
                break

    event_titles: dict[str, tuple[str, str]] = {}
    for raw in seen.values():
        et = raw.get("event_ticker")
        if et and et not in event_titles:
            try:
                ev = client.get_event(et).get("event", {})
                event_titles[et] = (
                    ev.get("title", "") or "",
                    ev.get("sub_title", "") or "",
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("event title fetch failed for %s: %s", et, exc)
                event_titles[et] = ("", "")

    out: List[GasMarket] = []
    for raw in seen.values():
        ticker = raw.get("ticker", "")
        parsed = _parse_market_ticker(ticker)
        if parsed is None:
            log.debug("could not parse NBA ticker %s — skipping", ticker)
            continue
        game_date, away, home, team_being_asked = parsed
        et = raw.get("event_ticker", "")
        # Kalshi's modern API uses *_dollars fields (decimal strings);
        # the older int-cents fields are kept too. Try the legacy
        # ones first since the bot's other code paths still expect cents.
        ya = (_to_int(raw.get("yes_ask"))
              or _cents_from_dollars(raw.get("yes_ask_dollars")))
        na = (_to_int(raw.get("no_ask"))
              or _cents_from_dollars(raw.get("no_ask_dollars")))
        yb = (_to_int(raw.get("yes_bid"))
              or _cents_from_dollars(raw.get("yes_bid_dollars")))
        out.append(GasMarket(
            ticker=ticker,
            event_ticker=et,
            title=raw.get("title", "") or "",
            subtitle=raw.get("subtitle", "") or "",
            yes_sub_title=raw.get("yes_sub_title", "") or "",
            strike_low=None,
            strike_high=None,
            direction="team_wins",
            close_time=_effective_close_time(raw),
            yes_ask_cents=ya,
            no_ask_cents=na,
            yes_bid_cents=yb,
            volume=_volume_from_market(raw),
            open_interest=_open_interest_from_market(raw),
            rules_primary=raw.get("rules_primary", "") or "",
            rules_secondary=raw.get("rules_secondary", "") or "",
            event_title=event_titles.get(et, ("", ""))[0],
            event_sub_title=event_titles.get(et, ("", ""))[1],
            team_being_asked=team_being_asked,
            home_tricode=home,
            away_tricode=away,
            game_date=game_date,
            raw=raw,
        ))
    return out


# Aliases for cross-bot symmetry.
discover_nba_markets = discover_wnba_markets
discover_gas_markets = discover_wnba_markets
discover_cpi_markets = discover_wnba_markets


def fetch_orderbook(client: KalshiClient, market: GasMarket) -> Optional[Orderbook]:
    try:
        return client.get_orderbook(market.ticker, depth=10)
    except Exception as exc:  # noqa: BLE001
        log.debug("orderbook fetch failed for %s: %s", market.ticker, exc)
        return None
