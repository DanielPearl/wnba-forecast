"""WNBA data layer.

The WNBA equivalent of the NBA bot's data layer. Unlike the NBA bot
(which pulls stats.nba.com via the ``nba_api`` package), the WNBA bot
sources **everything from ESPN's public endpoints** — stats.wnba.com is
unreliable / IP-blocked from cloud hosts, whereas ESPN's JSON API is
stable, unauthenticated, and carries full box scores back to ~2010.

  *** No API key is required. ***

Three sources, each with its own role and cache:

  1. **ESPN core "events" + "summary"** — historical box scores for
     training. The core API lists every regular-season game id for a
     season in one call; the summary endpoint returns each game's
     team-level box score (FG / 3P / FT / OREB / DREB / TOV / …) — i.e.
     everything ``features.py`` needs to compute Dean-Oliver Four
     Factors and team ratings. Cached one CSV per season so a re-train
     only re-fetches the active season.

  2. **ESPN scoreboard JSON** — current schedule, upcoming games,
     home/away assignments, tipoff times, live scores. Used at
     inference time to match Kalshi tickers to real matchups and to
     drive the in-game adjustment layer.

  3. **ESPN injuries JSON** — best-effort live availability feed. Used
     at inference time for a roster-strength posterior nudge. Degrades
     gracefully when the feed hiccups.

Everything here is read-only.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

log = logging.getLogger(__name__)

ESPN_HEADERS = {
    "User-Agent": "kalshi-wnba-bot/1.0 (data fetcher; +contact: ops)",
    "Accept": "application/json",
}

# Site API (scoreboard / summary / teams) and core API (season schedule).
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba"
ESPN_SCOREBOARD_URL = f"{ESPN_SITE}/scoreboard"
ESPN_SUMMARY_URL = f"{ESPN_SITE}/summary"
ESPN_TEAMS_URL = f"{ESPN_SITE}/teams"
ESPN_INJURIES_URL = "https://www.espn.com/wnba/injuries"

# Canonical team codes = the codes Kalshi uses in KXWNBAGAME tickers.
# Kalshi codes are variable length (GS, NY, LV, LA = 2 chars; CONN = 4).
# ESPN's abbreviations match EXCEPT Connecticut, which ESPN calls "CON"
# and Kalshi calls "CONN". We normalize everything to the Kalshi form so
# the model's per-team ELO / rolling stats line up with the market's
# ``team_being_asked`` without a second translation step.
TEAM_TRICODE_ALIASES: Dict[str, str] = {
    "CON": "CONN",   # ESPN Connecticut Sun -> Kalshi CONN
    "CONN": "CONN",
    # Defensive aliases for legacy / alternate spellings seen in older
    # ESPN payloads. Harmless when they never appear.
    "LVA": "LV",     # Las Vegas Aces
    "WAS": "WSH",    # Washington Mystics (ESPN uses WSH)
    "GSV": "GS",     # Golden State Valkyries
    # Kalshi codes the Portland Fire as PDX in KXWNBAGAME tickers while
    # ESPN (our canonical set) uses POR. Without this alias the ticker
    # parser rejects every LV@PDX-style market and the game silently
    # never appears on the watchlist (found 2026-07-09).
    "PDX": "POR",    # Portland Fire (Kalshi ticker code)
}

# Current Kalshi WNBA team universe (used to split concatenated event
# tickers like "PHXGS" -> away=PHX, home=GS unambiguously).
WNBA_TEAM_CODES = {
    "ATL", "CHI", "CONN", "DAL", "GS", "IND", "LV", "LA",
    "MIN", "NY", "PHX", "POR", "SEA", "TOR", "WSH",
}


def normalize_tricode(code: str) -> str:
    code = (code or "").strip().upper()
    return TEAM_TRICODE_ALIASES.get(code, code)


# Repo root (…/WNBA Forecast), two levels up from src/wnba_bot. Cache
# dirs are anchored here so the cache is found regardless of the
# process's cwd — critical when the bot runs IN-PROCESS inside the
# dashboard (cwd=/root/trading-dashboard), not from its own repo dir.
# Without this the in-dashboard bot would re-download every ESPN box
# score on each panel refresh instead of reusing the trained cache.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_cache(cache_dir: str) -> str:
    p = Path(cache_dir)
    return str(p if p.is_absolute() else _REPO_ROOT / p)


# WNBA regulation is 4 × 10 min = 40 min → 5 players × 40 = 200 team-min.
# Each overtime adds 5 min × 5 players = 25 team-min. Used only to scale
# PACE, which the model consumes as a relative feature, so the exact
# value matters little — but getting OT right keeps pace honest.
_REGULATION_TEAM_MINUTES = 200.0
_OT_TEAM_MINUTES = 25.0


# --------------------------------------------------------------------------- #
# Low-level ESPN fetch helper
# --------------------------------------------------------------------------- #

def _get_json(url: str, timeout: int = 25, max_attempts: int = 3) -> Optional[dict]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=ESPN_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.4 * (attempt + 1))
    log.debug("ESPN fetch failed after %d attempts: %s (%s)",
              max_attempts, url, last_exc)
    return None


# --------------------------------------------------------------------------- #
# Box-score parsing
# --------------------------------------------------------------------------- #

def _split_made_attempted(value: Optional[str]) -> Tuple[float, float]:
    """Parse an ESPN 'made-attempted' string ('28-69') into (28.0, 69.0)."""
    if not value or "-" not in str(value):
        return (float("nan"), float("nan"))
    try:
        made, att = str(value).split("-", 1)
        return float(made), float(att)
    except (TypeError, ValueError):
        return (float("nan"), float("nan"))


def _stat_map(team_box: dict) -> Dict[str, str]:
    return {s.get("name"): s.get("displayValue")
            for s in (team_box.get("statistics") or [])}


def _parse_summary_to_rows(summary: dict) -> List[dict]:
    """Turn one ESPN game summary into two team-game rows (home + away).

    Returns [] if the game isn't a completed two-team game we can parse.
    The row schema matches what ``features.build_training_table`` expects
    from the NBA panel: GAME_ID, GAME_DATE, TEAM_ABBREVIATION,
    OPP_TRICODE, HOME, WIN, and the raw box columns (FGM, FGA, FG3M,
    FG3A, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, MIN).
    """
    header = summary.get("header") or {}
    comps = header.get("competitions") or []
    if not comps:
        return []
    comp = comps[0]
    competitors = comp.get("competitors") or []
    if len(competitors) != 2:
        return []
    # Only train on completed games.
    status_state = (((comp.get("status") or {}).get("type") or {})
                    .get("state", "")).lower()
    if status_state and status_state != "post":
        return []

    game_id = str(header.get("id") or summary.get("id") or "")
    date_iso = comp.get("date") or header.get("date") or ""
    try:
        game_date = (datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                     .astimezone(timezone.utc).replace(tzinfo=None))
    except (ValueError, AttributeError):
        return []

    # period count → overtime detection for team minutes.
    period = int(((comp.get("status") or {}).get("period")) or 4)
    team_minutes = _REGULATION_TEAM_MINUTES + max(0, period - 4) * _OT_TEAM_MINUTES

    # Map abbreviation -> (homeAway, winner, score) from the header.
    meta_by_abbr: Dict[str, dict] = {}
    for c in competitors:
        abbr = normalize_tricode((c.get("team") or {}).get("abbreviation", ""))
        try:
            score = float(c.get("score"))
        except (TypeError, ValueError):
            score = float("nan")
        meta_by_abbr[abbr] = {
            "home": (c.get("homeAway") == "home"),
            "winner": bool(c.get("winner")),
            "score": score,
        }
    if len(meta_by_abbr) != 2:
        return []

    box_teams = (summary.get("boxscore") or {}).get("teams") or []
    if len(box_teams) != 2:
        return []

    abbrs = list(meta_by_abbr.keys())
    rows: List[dict] = []
    for tb in box_teams:
        abbr = normalize_tricode((tb.get("team") or {}).get("abbreviation", ""))
        if abbr not in meta_by_abbr:
            continue
        opp = next((a for a in abbrs if a != abbr), "")
        sm = _stat_map(tb)
        fgm, fga = _split_made_attempted(sm.get("fieldGoalsMade-fieldGoalsAttempted"))
        fg3m, fg3a = _split_made_attempted(
            sm.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
        ftm, fta = _split_made_attempted(sm.get("freeThrowsMade-freeThrowsAttempted"))

        def _num(key: str) -> float:
            v = sm.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        meta = meta_by_abbr[abbr]
        rows.append({
            "GAME_ID": game_id,
            "GAME_DATE": game_date,
            "TEAM_ABBREVIATION": abbr,
            "OPP_TRICODE": opp,
            "HOME": 1 if meta["home"] else 0,
            "WL": "W" if meta["winner"] else "L",
            "WIN": 1 if meta["winner"] else 0,
            "PTS": meta["score"],
            "FGM": fgm, "FGA": fga,
            "FG3M": fg3m, "FG3A": fg3a,
            "FTM": ftm, "FTA": fta,
            "OREB": _num("offensiveRebounds"),
            "DREB": _num("defensiveRebounds"),
            "REB": _num("totalRebounds"),
            "AST": _num("assists"),
            "STL": _num("steals"),
            "BLK": _num("blocks"),
            "TOV": _num("turnovers"),
            "PF": _num("fouls"),
            "MIN": team_minutes,
            "MATCHUP": (f"{abbr} vs. {opp}" if meta["home"]
                        else f"{abbr} @ {opp}"),
        })
    if len(rows) != 2:
        return []
    return rows


# --------------------------------------------------------------------------- #
# Season schedule (core API)
# --------------------------------------------------------------------------- #

def _season_event_ids(season: str, season_type: int = 2) -> List[str]:
    """Return every game id for a WNBA season (type 2 = regular season).

    One core-API call lists all events; we extract the numeric ids from
    each item's ``$ref``.
    """
    url = (f"{ESPN_CORE}/seasons/{season}/types/{season_type}/events"
           f"?limit=1000")
    data = _get_json(url)
    if not data:
        return []
    ids: List[str] = []
    for item in data.get("items", []):
        ref = item.get("$ref", "")
        # .../events/401620178?lang=en
        try:
            tail = ref.split("/events/")[1]
            gid = tail.split("?")[0].strip()
            if gid:
                ids.append(gid)
        except (IndexError, AttributeError):
            continue
    return ids


def fetch_season_game_logs(
    season: str,
    cache_dir: str = "data/wnba_api_cache",
    season_type: int = 2,
    refetch_if_stale_hours: float = 12.0,
) -> pd.DataFrame:
    """Return one row per team-game (two rows per game) for one season.

    Built entirely from ESPN. Cached to ``gamelog_<season>.csv``. The
    in-progress (current) season's cache is considered stale after
    ``refetch_if_stale_hours`` so a re-train picks up new games; past
    seasons are cached forever (they never change).
    """
    cache_dir = _resolve_cache(cache_dir)
    cache_path = Path(cache_dir) / f"gamelog_{season}.csv"
    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
        current_season = str(_current_wnba_season())
        is_current = (str(season) == current_season)
        if (not is_current) or age_h < refetch_if_stale_hours:
            try:
                df = pd.read_csv(cache_path, parse_dates=["GAME_DATE"])
                log.info("loaded %d game-log rows for %s season from cache",
                         len(df), season)
                return df
            except Exception as exc:  # noqa: BLE001
                log.debug("cache read failed for %s: %s; refetching", season, exc)

    event_ids = _season_event_ids(season, season_type=season_type)
    if not event_ids:
        log.warning("no events listed for WNBA season %s", season)
        if cache_path.exists():
            try:
                return pd.read_csv(cache_path, parse_dates=["GAME_DATE"])
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame()

    log.info("fetching %d game summaries for WNBA season %s", len(event_ids), season)
    rows: List[dict] = []
    for i, gid in enumerate(event_ids):
        summary = _get_json(f"{ESPN_SUMMARY_URL}?event={gid}")
        if not summary:
            continue
        rows.extend(_parse_summary_to_rows(summary))
        if (i + 1) % 50 == 0:
            log.info("  ... %d/%d games fetched for %s", i + 1, len(event_ids), season)
        time.sleep(0.05)  # be polite to ESPN
    if not rows:
        log.warning("season %s produced 0 parsable game rows", season)
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        log.info("cached %d game-log rows for WNBA season %s", len(df), season)
    except Exception as exc:  # noqa: BLE001
        log.debug("cache write failed for %s: %s", season, exc)
    return df


def fetch_multi_season_panel(
    seasons: List[str],
    cache_dir: str = "data/wnba_api_cache",
) -> pd.DataFrame:
    """Concatenate ESPN game logs across many seasons into one panel.

    The per-season builder already supplies ``HOME``, ``OPP_TRICODE`` and
    ``WIN`` (derived from the ESPN header), so unlike the NBA version we
    don't need to re-parse a MATCHUP string here.
    """
    frames: List[pd.DataFrame] = []
    for s in seasons:
        try:
            df = fetch_season_game_logs(str(s), cache_dir=cache_dir)
            if df.empty:
                continue
            df = df.copy()
            df["SEASON"] = str(s)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping WNBA season %s due to fetch error: %s", s, exc)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["TEAM_ABBREVIATION"] = panel["TEAM_ABBREVIATION"].apply(normalize_tricode)
    panel["OPP_TRICODE"] = panel["OPP_TRICODE"].apply(normalize_tricode)
    # ESPN's "regular season" (type 2) event list includes the All-Star
    # Game, whose competitors are captain squads ("Team USA", "Team
    # Clark", "WNBA Stars", …) — phantom teams that would pollute the
    # ELO chain and rolling stats. Drop any game where either side isn't
    # a real franchise.
    before = len(panel)
    panel = panel[panel["TEAM_ABBREVIATION"].isin(WNBA_TEAM_CODES)
                  & panel["OPP_TRICODE"].isin(WNBA_TEAM_CODES)].copy()
    if len(panel) != before:
        log.info("dropped %d non-franchise (All-Star) team-game rows",
                 before - len(panel))
    panel["HOME"] = panel["HOME"].astype(int)
    panel["WIN"] = panel["WIN"].astype(int)
    panel = panel.sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)
    return panel


def _current_wnba_season(today: Optional[datetime] = None) -> str:
    """Active WNBA season string (a calendar year, e.g. '2026').

    The WNBA plays May–October. Before May we still point at the prior
    completed season so inference has data; once games tip in the spring
    the current year takes over.
    """
    today = today or datetime.now(timezone.utc)
    if today.month >= 5:
        return str(today.year)
    return str(today.year - 1)


# --------------------------------------------------------------------------- #
# ESPN scoreboard (live schedule)
# --------------------------------------------------------------------------- #

def fetch_espn_scoreboard(
    date_yyyymmdd: Optional[str] = None,
    cache_ttl_seconds: int = 60,
    cache_dir: str = "data/espn_cache",
) -> List[dict]:
    """Return today's (or a given date's) WNBA games as a list of dicts.

    Same shape the NBA bot's live layer expects: ``home_tricode``,
    ``away_tricode``, scores, linescores, period, status, tipoff, plus
    best-effort consensus moneyline. Cached to disk for
    ``cache_ttl_seconds``.
    """
    cache_key = date_yyyymmdd or "today"
    cache_dir = _resolve_cache(cache_dir)
    cache_path = Path(cache_dir) / f"scoreboard_{cache_key}.json"
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < cache_ttl_seconds:
                with open(cache_path) as f:
                    return json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.debug("scoreboard cache read failed: %s", exc)

    url = ESPN_SCOREBOARD_URL
    if date_yyyymmdd:
        url += "?" + urllib.parse.urlencode({"dates": date_yyyymmdd})
    raw = _get_json(url, timeout=15)
    if raw is None:
        log.warning("ESPN WNBA scoreboard fetch failed")
        return []
    out: List[dict] = []
    for ev in raw.get("events", []):
        comps = ev.get("competitions") or [{}]
        c = comps[0]
        competitors = c.get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        odds_list = c.get("odds") or []
        odds = odds_list[0] if odds_list else {}

        def _linescores(competitor: dict) -> List[int]:
            try:
                return [int((ls or {}).get("value") or 0)
                        for ls in (competitor.get("linescores") or [])]
            except (TypeError, ValueError):
                return []

        status_block = (ev.get("status") or {})
        period = status_block.get("period") or 0
        clock_str = status_block.get("displayClock") or ""
        seconds_remaining = None
        try:
            if ":" in clock_str:
                m, s = clock_str.split(":", 1)
                seconds_remaining = int(m) * 60 + int(s)
        except (TypeError, ValueError):
            seconds_remaining = None
        out.append({
            "espn_game_id": ev.get("id"),
            "date_iso": ev.get("date"),
            "home_tricode": normalize_tricode((home.get("team") or {}).get("abbreviation", "")),
            "away_tricode": normalize_tricode((away.get("team") or {}).get("abbreviation", "")),
            "home_score": int((home.get("score") or 0)),
            "away_score": int((away.get("score") or 0)),
            "home_linescores": _linescores(home),
            "away_linescores": _linescores(away),
            "period": int(period or 0),
            "seconds_remaining": seconds_remaining,
            "status": ((ev.get("status") or {}).get("type") or {}).get("name", "").lower(),
            "tipoff_iso": ev.get("date"),
            "moneyline_home": odds.get("homeTeamOdds", {}).get("moneyLine"),
            "moneyline_away": odds.get("awayTeamOdds", {}).get("moneyLine"),
            "spread": odds.get("spread"),
            "over_under": odds.get("overUnder"),
        })
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)
    except Exception as exc:  # noqa: BLE001
        log.debug("scoreboard cache write failed: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# ESPN injuries (live availability feed)
# --------------------------------------------------------------------------- #

def fetch_espn_injuries(
    cache_ttl_seconds: int = 600,
    cache_dir: str = "data/espn_cache",
) -> Dict[str, List[dict]]:
    """Return dict[team_tricode -> list[{player, position, status, comment}]].

    Best-effort scrape of ESPN's public WNBA injuries page. Mirrors the
    NBA bot's failure-backoff behaviour so a broken page doesn't hammer
    ESPN every 10 minutes.
    """
    cache_dir = _resolve_cache(cache_dir)
    cache_path = Path(cache_dir) / "injuries.json"
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < cache_ttl_seconds:
                with open(cache_path) as f:
                    return json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.debug("injuries cache read failed: %s", exc)

    try:
        r = requests.get(ESPN_INJURIES_URL, headers={
            "User-Agent": "Mozilla/5.0 (compatible; kalshi-wnba-bot/1.0)",
            "Accept": "text/html",
        }, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
    except Exception as exc:  # noqa: BLE001
        FAILURE_BACKOFF_SECONDS = 3600
        in_backoff = (cache_path.exists()
                      and cache_path.stat().st_mtime > time.time())
        if in_backoff:
            log.debug("ESPN WNBA injuries still failing inside backoff: %s", exc)
        else:
            log.warning("ESPN WNBA injuries fetch failed: %s — using empty "
                        "cache for %ds", exc, FAILURE_BACKOFF_SECONDS)
        try:
            cached = (json.load(open(cache_path))
                      if cache_path.exists() else {})
        except Exception:  # noqa: BLE001
            cached = {}
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cached, f)
            future = time.time() + FAILURE_BACKOFF_SECONDS
            os.utime(cache_path, (future, future))
        except Exception:  # noqa: BLE001
            pass
        return cached

    import re as _re
    headings = _re.findall(
        r'<a[^>]+class="[^"]*injuries__teamName[^"]*"[^>]*>([^<]+)</a>', r.text)
    if not headings:
        headings = _re.findall(r'<h2[^>]*>([A-Z][a-zA-Z\. ]{3,30})</h2>', r.text)
    out: Dict[str, List[dict]] = {}
    name_to_tri = _team_name_to_tricode_map()
    for i, table in enumerate(tables):
        if "NAME" not in [str(c).upper() for c in table.columns]:
            continue
        team_name = headings[i] if i < len(headings) else ""
        tri = name_to_tri.get(team_name.strip().lower())
        if not tri:
            continue
        rows: List[dict] = []
        for _, r2 in table.iterrows():
            rows.append({
                "player": str(r2.get("NAME", "")).strip(),
                "position": str(r2.get("POS", "")).strip(),
                "date": str(r2.get("EST. RETURN DATE", "") or r2.get("DATE", "")).strip(),
                "status": str(r2.get("STATUS", "")).strip(),
                "comment": str(r2.get("COMMENT", "") or r2.get("INJURY", "")).strip(),
            })
        out[tri] = rows
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)
    except Exception as exc:  # noqa: BLE001
        log.debug("injuries cache write failed: %s", exc)
    return out


def _team_name_to_tricode_map() -> Dict[str, str]:
    """Build {lowercase team name} -> Kalshi tricode from ESPN's teams
    endpoint, with a hardcoded fallback (WNBA names rarely change)."""
    fallback = {
        "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CONN",
        "dallas wings": "DAL", "golden state valkyries": "GS",
        "indiana fever": "IND", "las vegas aces": "LV",
        "los angeles sparks": "LA", "la sparks": "LA",
        "minnesota lynx": "MIN", "new york liberty": "NY",
        "phoenix mercury": "PHX", "portland fire": "POR",
        "seattle storm": "SEA", "toronto tempo": "TOR",
        "washington mystics": "WSH",
    }
    try:
        data = _get_json(ESPN_TEAMS_URL, timeout=10)
        teams = ((((data or {}).get("sports") or [{}])[0]
                  .get("leagues") or [{}])[0].get("teams")) or []
        live = {}
        for t in teams:
            team = t.get("team") or {}
            name = (team.get("displayName") or "").strip().lower()
            tri = normalize_tricode(team.get("abbreviation") or "")
            if name and tri:
                live[name] = tri
        if live:
            fallback.update(live)
    except Exception as exc:  # noqa: BLE001
        log.debug("ESPN WNBA teams endpoint failed: %s — using fallback", exc)
    return fallback


# --------------------------------------------------------------------------- #
# Vegas line (optional, requires THE_ODDS_API_KEY)
# --------------------------------------------------------------------------- #

def fetch_odds_api_moneylines(
    cache_ttl_seconds: int = 600,
    cache_dir: str = "data/odds_cache",
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Best-effort pre-game WNBA moneylines from The Odds API.

    Returns dict keyed by (away_tricode, home_tricode). Empty if no
    ``THE_ODDS_API_KEY`` is set or the call fails — Vegas line is an
    OPTIONAL inference signal, never a training requirement.
    """
    api_key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        return {}
    cache_dir = _resolve_cache(cache_dir)
    cache_path = Path(cache_dir) / "moneylines.json"
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < cache_ttl_seconds:
                with open(cache_path) as f:
                    return {tuple(k.split("|")): v for k, v in json.load(f).items()}
        except Exception as exc:  # noqa: BLE001
            log.debug("odds cache read failed: %s", exc)
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_wnba/odds",
            params={"apiKey": api_key, "regions": "us",
                    "markets": "h2h", "oddsFormat": "decimal"},
            timeout=15,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("WNBA Odds API fetch failed: %s", exc)
        return {}
    name_to_tri = _team_name_to_tricode_map()
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ev in events:
        home_tri = name_to_tri.get((ev.get("home_team") or "").lower())
        away_tri = name_to_tri.get((ev.get("away_team") or "").lower())
        if not (home_tri and away_tri):
            continue
        home_prices: List[float] = []
        away_prices: List[float] = []
        for book in ev.get("bookmakers") or []:
            for mkt in book.get("markets") or []:
                if mkt.get("key") != "h2h":
                    continue
                for outcome in mkt.get("outcomes") or []:
                    name = (outcome.get("name") or "").lower()
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if name == (ev.get("home_team") or "").lower():
                        home_prices.append(float(price))
                    elif name == (ev.get("away_team") or "").lower():
                        away_prices.append(float(price))
        if home_prices and away_prices:
            out[(away_tri, home_tri)] = {
                "home_price_decimal": sum(home_prices) / len(home_prices),
                "away_price_decimal": sum(away_prices) / len(away_prices),
            }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"|".join(k): v for k, v in out.items()}, f)
    except Exception as exc:  # noqa: BLE001
        log.debug("odds cache write failed: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# Pinnacle (sharp) devigged probabilities — used as the tennis-style
# reference line for edge calculation vs Kalshi.
# --------------------------------------------------------------------------- #

def fetch_pinnacle_probs_by_pair(
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return Pinnacle-devigged pre-game WNBA win probabilities keyed by
    ``(away_tricode, home_tricode)``.

    Mirrors ``Tennis Forecast``'s Pinnacle path so both bots feed the
    same shared BUY gate the same reference. Backed by
    ``kalshi_sdk.pinnacle.pinnacle_probs_by_pair`` — the network call,
    devig, and cache all live there. This wrapper just maps the
    returned ESPN-style team names onto WNBA tricodes so downstream
    code can join on the same key as ``fetch_espn_scoreboard`` etc.

    Silent no-op when ``THE_ODDS_API_KEY`` isn't set — returns ``{}``.
    Each entry contains ``{"home_prob": float, "away_prob": float}``
    with the two summing to 1.0.
    """
    from kalshi_sdk.pinnacle import pinnacle_probs_by_pair

    raw = pinnacle_probs_by_pair(["basketball_wnba"])
    if not raw:
        return {}
    name_to_tri = _team_name_to_tricode_map()
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for pair_set, name_to_prob in raw.items():
        # frozenset preserves the two names but not their order — we
        # rebuild "away @ home" by joining against ESPN's team map.
        names = list(pair_set)
        if len(names) != 2:
            continue
        tri_by_name = {n: name_to_tri.get(n.lower()) for n in names}
        if not all(tri_by_name.values()):
            continue
        # The devig helper stamped {home_name: p_home, away_name: p_away}
        # in the SDK — but here we don't know which name is home/away.
        # Both callers key by ``(away_tri, home_tri)`` so we emit both
        # orderings pointing at the same probability dict.
        (n1, n2) = names
        p1, p2 = float(name_to_prob[n1]), float(name_to_prob[n2])
        t1, t2 = tri_by_name[n1], tri_by_name[n2]
        record = {n1: p1, n2: p2, t1: p1, t2: p2}
        out[(t1, t2)] = record
        out[(t2, t1)] = record
    return out
