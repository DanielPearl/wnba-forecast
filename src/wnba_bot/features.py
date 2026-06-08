"""Feature engineering for NBA single-game outcome prediction.

Each row in the feature table corresponds to ONE upcoming/past game
from the home team's perspective. The target is binary
``home_wins`` (1 if home won, 0 if away won). Every feature must be
computable from information available BEFORE tipoff — no in-game stats,
no post-game adjustments. The training panel is built game-by-game so
each row sees only the prior games of both teams.

Five families of features:

  1. **ELO rating** — 538-style. One scalar per team that updates after
     every game with a margin-of-victory bump. Rating differential
     (home_elo − away_elo + home-court advantage) is by itself a
     ~65% accuracy predictor, comparable to a Vegas-line baseline.

  2. **Rolling Four Factors** — for each team over its last N games:
        eFG%      effective field-goal percentage
        TOV%      turnover rate
        OREB%     offensive rebound rate
        FT/FGA    free-throw rate
     Differentials (home − away) at multiple windows (5, 10, 20).

  3. **Net rating, pace, scoring** — rolling offensive rating,
     defensive rating, net rating, pace, point differential. The
     single strongest team-level feature is ``net_rating_diff_10``.

  4. **Schedule context** — home/away, days of rest (each team), B2B
     indicator, days into season, road-trip length, recent travel.

  5. **Recent form / matchup** — win streak, win % last 5/10, head-to-
     head record this season, opponent strength of schedule (avg net
     rating of recent opponents).

The injury-impact feature is built separately at INFERENCE time
(``compute_injury_impact``) because historical injury reports aren't
in the nba_api game-log feed. The model trains without it; live
inference adds it as a posterior adjustment when available.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Rolling windows for stat aggregations (in games).
ROLLING_WINDOWS = [5, 10, 20]

# ELO knobs — 538-style with mild MoV adjustment.
ELO_DEFAULT = 1500.0
ELO_K_BASE = 20.0          # base sensitivity per game
ELO_HOME_COURT_ADV = 65.0  # ~3.0 point HCA in NBA terms
# Reset toward mean between seasons so a team's prior-season rating
# doesn't dominate a year of roster turnover.
ELO_SEASON_REGRESSION = 0.25  # 25% pulled back to 1500 at season start


# --------------------------------------------------------------------------- #
# Per-team-game stat enrichment
# --------------------------------------------------------------------------- #

def _enrich_team_game_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Add Four-Factors and rating columns to each team-game row.

    Operates on rows of nba_api LeagueGameLog output. Each row already
    has FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB, AST, STL,
    BLK, TOV, PF, PTS, MIN, PLUS_MINUS. We compute team-level rates
    that don't depend on opponent (defensive rating needs both teams
    of the same game; we'll compute that in a second pass).
    """
    df = panel.copy()
    # Possession estimate (Hollinger): FGA - OREB + TOV + 0.44*FTA
    df["POSS"] = (df["FGA"].astype(float)
                  - df["OREB"].astype(float)
                  + df["TOV"].astype(float)
                  + 0.44 * df["FTA"].astype(float))
    # Effective FG% — weights threes by 1.5
    df["eFG_PCT"] = ((df["FGM"].astype(float)
                      + 0.5 * df["FG3M"].astype(float))
                     / df["FGA"].astype(float).replace(0, np.nan))
    df["TOV_PCT"] = (df["TOV"].astype(float)
                      / df["POSS"].replace(0, np.nan))
    # OREB% needs the opponent's DREB on the same game, so we'll
    # compute a per-team OREB-rate proxy (OREB / (OREB+DREB)) as an
    # imperfect but team-only stand-in. The opponent-aware version
    # gets stitched in later.
    df["OREB_RATE"] = (df["OREB"].astype(float)
                        / (df["OREB"].astype(float)
                           + df["DREB"].astype(float)).replace(0, np.nan))
    df["FT_PER_FGA"] = (df["FTA"].astype(float)
                        / df["FGA"].astype(float).replace(0, np.nan))
    df["OFF_RATING"] = 100.0 * df["PTS"].astype(float) / df["POSS"].replace(0, np.nan)
    df["PACE"] = 48.0 * 5.0 * df["POSS"] / df["MIN"].astype(float).replace(0, np.nan)
    return df


def _join_opponent_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """For each team-game row, attach the opponent's same-game stats.

    Required for defensive rating (= opponent's offensive rating) and
    a true OREB% (= OREB / (own OREB + opponent DREB)). Joins on
    GAME_ID so each game's two team-rows see each other.
    """
    df = panel.copy()
    # Build a lookup: GAME_ID + TEAM -> row index, then derive opp.
    paired: List[pd.Series] = []
    by_game = df.groupby("GAME_ID")
    for game_id, group in by_game:
        if len(group) != 2:
            # Some old or playoff rows might be missing one side; skip.
            continue
        a, b = group.iloc[0].copy(), group.iloc[1].copy()
        # a's opponent stats:
        a["OPP_PTS"] = b["PTS"]
        a["OPP_POSS"] = b["POSS"]
        a["OPP_DREB"] = b["DREB"]
        a["OPP_OFF_RATING"] = b["OFF_RATING"]
        b["OPP_PTS"] = a["PTS"]
        b["OPP_POSS"] = a["POSS"]
        b["OPP_DREB"] = a["DREB"]
        b["OPP_OFF_RATING"] = a["OFF_RATING"]
        paired.append(a)
        paired.append(b)
    if not paired:
        return df
    out = pd.DataFrame(paired)
    out["DEF_RATING"] = out["OPP_OFF_RATING"]
    out["NET_RATING"] = out["OFF_RATING"] - out["DEF_RATING"]
    # True OREB%: own OREB / (own OREB + opp DREB).
    out["OREB_PCT"] = (out["OREB"].astype(float)
                       / (out["OREB"].astype(float)
                          + out["OPP_DREB"].astype(float)).replace(0, np.nan))
    out["MARGIN"] = out["PTS"].astype(float) - out["OPP_PTS"].astype(float)
    return out.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# ELO rating system
# --------------------------------------------------------------------------- #

@dataclass
class _EloState:
    rating: Dict[str, float] = field(default_factory=dict)
    last_season: Dict[str, str] = field(default_factory=dict)

    def get(self, team: str, season: str) -> float:
        # Inter-season regression toward the mean.
        last = self.last_season.get(team)
        cur = self.rating.get(team, ELO_DEFAULT)
        if last is not None and last != season:
            cur = ELO_DEFAULT + (cur - ELO_DEFAULT) * (1.0 - ELO_SEASON_REGRESSION)
            self.rating[team] = cur
        self.last_season[team] = season
        return cur

    def update(self, home: str, away: str, home_won: int, margin: float,
                season: str) -> Tuple[float, float]:
        """Return (home_elo_pre, away_elo_pre) before the update fires."""
        h = self.get(home, season)
        a = self.get(away, season)
        # Home team carries HCA in their effective rating for the calc.
        expected_home = 1.0 / (1.0 + 10.0 ** (-((h + ELO_HOME_COURT_ADV) - a) / 400.0))
        actual_home = float(home_won)
        # MoV multiplier — 538-style: log(margin+1) × elo-diff factor.
        # Caps the swing on blowouts so 30-pt and 5-pt wins look similar.
        elo_diff = abs((h + ELO_HOME_COURT_ADV) - a)
        mov_mult = (np.log(max(1.0, abs(margin)) + 1.0)
                    * (2.2 / (0.001 * (elo_diff if home_won else -elo_diff) + 2.2)))
        delta = ELO_K_BASE * mov_mult * (actual_home - expected_home)
        self.rating[home] = h + delta
        self.rating[away] = a - delta
        return h, a


def _attach_elo_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute the chronological ELO series for every team and attach
    each game's PRE-game home/away ELO ratings as columns. The bot
    reads these as features (the post-game rating leaks).
    """
    state = _EloState()
    home_elos: List[float] = []
    away_elos: List[float] = []
    # Iterate one row per GAME (not per team-game). Group by GAME_ID
    # and take the home row to know who's home/away.
    games = (panel.groupby("GAME_ID", sort=False)
             .agg({"GAME_DATE": "first", "SEASON": "first"})
             .reset_index())
    games = games.merge(
        panel[panel["HOME"] == 1][["GAME_ID", "TEAM_ABBREVIATION", "OPP_TRICODE",
                                    "WIN", "MARGIN"]],
        on="GAME_ID", how="left",
    )
    games = games.rename(columns={
        "TEAM_ABBREVIATION": "HOME_TRI",
        "OPP_TRICODE": "AWAY_TRI",
        "WIN": "HOME_WIN",
        "MARGIN": "HOME_MARGIN",
    })
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    home_elos: List[float] = []
    away_elos: List[float] = []
    for _, r in games.iterrows():
        if pd.isna(r["HOME_TRI"]):
            home_elos.append(np.nan)
            away_elos.append(np.nan)
            continue
        h_pre, a_pre = state.update(
            home=r["HOME_TRI"], away=r["AWAY_TRI"],
            home_won=int(r["HOME_WIN"]),
            margin=float(r["HOME_MARGIN"]),
            season=r["SEASON"],
        )
        home_elos.append(h_pre)
        away_elos.append(a_pre)
    games["HOME_ELO_PRE"] = home_elos
    games["AWAY_ELO_PRE"] = away_elos
    return games


# --------------------------------------------------------------------------- #
# Rolling team stats
# --------------------------------------------------------------------------- #

ROLLING_STAT_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "eFG_PCT", "TOV_PCT", "OREB_PCT", "FT_PER_FGA",
    "MARGIN", "WIN",
    "FG3M", "FG3A",
]


def _rolling_team_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """For each team-game row, compute rolling means over its last N
    games (excluding the current one — leakage-safe). Adds columns like
    ``TEAM_NET_RATING_R10``, ``TEAM_eFG_R5``, etc.

    Implementation: group by team, shift by 1 (so the current row's
    own stats can't leak), then rolling mean over the window. Uses
    transform so the result aligns with the input index.
    """
    df = panel.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]).copy()
    for col in ROLLING_STAT_COLS:
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            df[f"TEAM_{col}_R{w}"] = (
                df.groupby("TEAM_ABBREVIATION")[col]
                .transform(lambda s, w=w: s.shift(1).rolling(
                    w, min_periods=max(2, w // 2)).mean())
            )
    return df


def _add_schedule_context(panel: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: days of rest, b2b indicator, games-into-season.

    Pruned: ``LONG_REST`` (4+ days off categorical flag) used to live
    here too, but the walk-forward selector rejected it every fold —
    the continuous DAYS_REST already encodes the same information.
    Dropping the flag cuts a candidate feature without affecting model
    quality.
    """
    df = panel.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["DAYS_REST"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days
    df["B2B"] = (df["DAYS_REST"] == 1).astype(int)
    df["GAMES_INTO_SEASON"] = df.groupby(
        ["TEAM_ABBREVIATION", "SEASON"]).cumcount()
    return df


# --------------------------------------------------------------------------- #
# Build the per-game feature table from the home team's perspective
# --------------------------------------------------------------------------- #

def _pivot_to_per_game(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per GAME (instead of per team-game), with HOME_* and AWAY_*
    columns prefixed appropriately. The model trains on these rows."""
    home = panel[panel["HOME"] == 1].copy()
    away = panel[panel["HOME"] == 0].copy()
    keep = [c for c in panel.columns
            if c.startswith("TEAM_") or c in (
                "GAME_ID", "GAME_DATE", "SEASON", "TEAM_ABBREVIATION",
                "OPP_TRICODE", "DAYS_REST", "B2B",
                "GAMES_INTO_SEASON", "WIN", "MARGIN")]
    home = home[keep].rename(columns=lambda c: (
        c if c in ("GAME_ID", "GAME_DATE", "SEASON")
        else "HOME_" + c
    ))
    away = away[keep].rename(columns=lambda c: (
        c if c in ("GAME_ID", "GAME_DATE", "SEASON")
        else "AWAY_" + c
    ))
    g = home.merge(away, on=["GAME_ID", "GAME_DATE", "SEASON"], how="inner")
    g["HOME_WIN"] = g["HOME_WIN"].astype(int)
    return g


def _add_diff_features(g: pd.DataFrame) -> pd.DataFrame:
    """Most predictive features in the literature are differentials
    (home minus away). Compute them once and let the selector decide
    which lags / windows survive.
    """
    out = g.copy()
    for col in ROLLING_STAT_COLS:
        for w in ROLLING_WINDOWS:
            h = f"HOME_TEAM_{col}_R{w}"
            a = f"AWAY_TEAM_{col}_R{w}"
            if h in out.columns and a in out.columns:
                out[f"DIFF_{col}_R{w}"] = out[h] - out[a]
    # Rest differential — popular literature feature.
    out["REST_DIFF"] = out["HOME_DAYS_REST"] - out["AWAY_DAYS_REST"]
    out["B2B_DIFF"] = out["HOME_B2B"] - out["AWAY_B2B"]
    return out


def _attach_h2h(panel: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head record between the two teams over their prior games
    THIS season. Computed by a per-pair rolling mean of HOME_WIN.
    """
    home_only = panel[panel["HOME"] == 1].sort_values(["GAME_DATE"]).copy()
    home_only["pair_key"] = home_only.apply(
        lambda r: tuple(sorted([r["TEAM_ABBREVIATION"], r["OPP_TRICODE"]])), axis=1)
    home_only["TEAM_WIN_FOR_KEY"] = home_only["WIN"]  # home-perspective
    home_only["H2H_WINS_BEFORE"] = (
        home_only.groupby(["pair_key", "SEASON"])["TEAM_WIN_FOR_KEY"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    )
    g = g.merge(
        home_only[["GAME_ID", "H2H_WINS_BEFORE"]],
        on="GAME_ID", how="left",
    )
    return g


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #

def build_training_table(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build the model's per-game training table.

    Input ``panel`` is the multi-season output of
    ``data_sources.fetch_multi_season_panel`` — one row per team-game.

    Returns ``(df, feature_columns)`` where ``df`` has one row per game
    (sorted by date) and ``feature_columns`` is the list of columns the
    model should consume. The target is ``HOME_WIN``.
    """
    if panel.empty:
        return pd.DataFrame(), []
    enriched = _enrich_team_game_stats(panel)
    enriched = _join_opponent_stats(enriched)
    enriched = _rolling_team_stats(enriched)
    enriched = _add_schedule_context(enriched)
    # ELO needs MARGIN, which is computed in _join_opponent_stats — pass
    # the enriched (post-join) panel, not the raw input.
    elo_per_game = _attach_elo_history(enriched)
    g = _pivot_to_per_game(enriched)
    g = g.merge(
        elo_per_game[["GAME_ID", "HOME_ELO_PRE", "AWAY_ELO_PRE"]],
        on="GAME_ID", how="left",
    )
    g["ELO_DIFF"] = g["HOME_ELO_PRE"] - g["AWAY_ELO_PRE"] + ELO_HOME_COURT_ADV
    # ELO_WIN_PROB_HOME (sigmoid of ELO_DIFF) used to live here as a
    # second feature, but it carries no extra information once the GBT
    # has ELO_DIFF — the walk-forward selector rejected it every fold.
    # Pruned to drop a redundant candidate. The Elo-only logistic
    # baseline in model.py uses ELO_DIFF directly.
    g = _add_diff_features(g)
    g = _attach_h2h(panel, g)
    # Drop anything still NaN in the target.
    g = g.dropna(subset=["HOME_WIN"]).reset_index(drop=True)
    # Numeric features only. The HOME_TEAM_/AWAY_TEAM_ prefix is shared
    # with non-numeric columns from nba_api (TEAM_NAME, TEAM_ID,
    # TEAM_ABBREVIATION); restrict to rolling-window suffixes ``_R{N}``.
    schedule_features = {
        "HOME_ELO_PRE", "AWAY_ELO_PRE", "ELO_DIFF", "ELO_WIN_PROB_HOME",
        "REST_DIFF", "B2B_DIFF",
        "HOME_DAYS_REST", "AWAY_DAYS_REST",
        "HOME_B2B", "AWAY_B2B", "HOME_LONG_REST", "AWAY_LONG_REST",
        "HOME_GAMES_INTO_SEASON", "AWAY_GAMES_INTO_SEASON",
        "H2H_WINS_BEFORE",
    }
    rolling_suffixes = tuple(f"_R{w}" for w in ROLLING_WINDOWS)
    feature_cols = sorted([
        c for c in g.columns
        if c.startswith("DIFF_")
        or (c.startswith("HOME_TEAM_") and c.endswith(rolling_suffixes))
        or (c.startswith("AWAY_TEAM_") and c.endswith(rolling_suffixes))
        or c in schedule_features
    ])
    # Replace any ±inf (from rate stats with a tiny/degenerate denominator
    # in a team's first games) with NaN so the median imputer handles them.
    # The GBT tolerates inf, but the logistic meta / Elo baseline /
    # bake-off linear models do not — leaving them in produced matmul
    # overflow warnings and degraded the linear learners' calibration.
    if feature_cols:
        g[feature_cols] = g[feature_cols].replace([np.inf, -np.inf], np.nan)
    return g, feature_cols


# --------------------------------------------------------------------------- #
# Inference: build a single feature row for an upcoming game
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Fast per-tick inference (caches the heavy rolling/enrichment work)
# --------------------------------------------------------------------------- #

@dataclass
class TeamStateCache:
    """Per-tick cache of the heavy panel transforms.

    ``team_latest`` is keyed by team tricode → a Series with the team's
    most-recent enriched + rolled feature values. Used by
    ``build_inference_row_fast`` for O(1) per-game scoring.

    ``elo_state`` carries the post-history ELO state so we can compute
    home/away ELO for upcoming games without re-walking the panel.
    """
    team_latest: Dict[str, pd.Series]
    elo_state: _EloState


def precompute_team_states(panel: pd.DataFrame) -> TeamStateCache:
    """Run the heavy transforms (enrich, join opp, rolling stats,
    schedule context, ELO) ONCE on the panel and cache per-team latest
    snapshots. ~10s on a multi-season panel; much faster than calling
    ``build_inference_row`` per market.
    """
    enriched = _enrich_team_game_stats(panel)
    enriched = _join_opponent_stats(enriched)
    enriched = _rolling_team_stats(enriched)
    enriched = _add_schedule_context(enriched)
    # Latest row per team (the rolling features there reflect the
    # team's prior-N-games stats, which is exactly what we want for
    # an upcoming-game prediction).
    latest = (enriched.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
              .groupby("TEAM_ABBREVIATION").tail(1))
    team_latest = {row["TEAM_ABBREVIATION"]: row
                    for _, row in latest.iterrows()}
    # Walk the panel chronologically once to populate ELO state.
    state = _EloState()
    games = (enriched.groupby("GAME_ID", sort=False)
             .agg({"GAME_DATE": "first", "SEASON": "first"})
             .reset_index())
    games = games.merge(
        enriched[enriched["HOME"] == 1][["GAME_ID", "TEAM_ABBREVIATION",
                                          "OPP_TRICODE", "WIN", "MARGIN"]],
        on="GAME_ID", how="left",
    ).rename(columns={
        "TEAM_ABBREVIATION": "HOME_TRI", "OPP_TRICODE": "AWAY_TRI",
        "WIN": "HOME_WIN", "MARGIN": "HOME_MARGIN",
    }).sort_values(["GAME_DATE", "GAME_ID"])
    for _, r in games.iterrows():
        if pd.isna(r["HOME_TRI"]):
            continue
        state.update(home=r["HOME_TRI"], away=r["AWAY_TRI"],
                     home_won=int(r["HOME_WIN"]),
                     margin=float(r["HOME_MARGIN"]),
                     season=r["SEASON"])
    return TeamStateCache(team_latest=team_latest, elo_state=state)


def build_inference_row_fast(
    cache: TeamStateCache,
    home_tri: str, away_tri: str,
    game_date: pd.Timestamp, season: str,
) -> Optional[pd.DataFrame]:
    """O(1) per-market inference row builder using a pre-tick cache.

    Returns a single-row DataFrame matching the training-time feature
    columns. Trade-off vs the slow path: H2H is stubbed to 0.5
    (neutral) — the rolling-Four-Factors and ELO features carry the
    great majority of the signal anyway, and walking H2H per market
    costs more than it adds.
    """
    if home_tri not in cache.team_latest or away_tri not in cache.team_latest:
        return None
    home = cache.team_latest[home_tri]
    away = cache.team_latest[away_tri]
    # Match the panel's tz-naive timestamps (nba_api parses GAME_DATE
    # without timezone). market.game_date is typically tz-aware (UTC).
    target_ts = pd.Timestamp(game_date)
    if target_ts.tz is not None:
        target_ts = (target_ts.tz_convert(None) if hasattr(target_ts, "tz_convert")
                     else target_ts.tz_localize(None))
    target_ts = target_ts.normalize()
    row: Dict[str, float] = {}
    # Mirror everything the slow path produces (HOME_TEAM_*_RN /
    # AWAY_TEAM_*_RN + DIFFs).
    for col in ROLLING_STAT_COLS:
        for w in ROLLING_WINDOWS:
            stat_col = f"TEAM_{col}_R{w}"
            if stat_col in home.index and stat_col in away.index:
                row[f"HOME_{stat_col}"] = float(home[stat_col]) if pd.notna(home[stat_col]) else np.nan
                row[f"AWAY_{stat_col}"] = float(away[stat_col]) if pd.notna(away[stat_col]) else np.nan
                if pd.notna(home[stat_col]) and pd.notna(away[stat_col]):
                    row[f"DIFF_{col}_R{w}"] = float(home[stat_col]) - float(away[stat_col])
                else:
                    row[f"DIFF_{col}_R{w}"] = np.nan
    # Schedule context: compute fresh based on the upcoming game date.
    home_last = home.get("GAME_DATE")
    away_last = away.get("GAME_DATE")
    home_days_rest = (
        (target_ts - pd.Timestamp(home_last)).days
        if pd.notna(home_last) else np.nan
    )
    away_days_rest = (
        (target_ts - pd.Timestamp(away_last)).days
        if pd.notna(away_last) else np.nan
    )
    row["HOME_DAYS_REST"] = float(home_days_rest) if pd.notna(home_days_rest) else np.nan
    row["AWAY_DAYS_REST"] = float(away_days_rest) if pd.notna(away_days_rest) else np.nan
    row["HOME_B2B"] = int(home_days_rest == 1) if pd.notna(home_days_rest) else 0
    row["AWAY_B2B"] = int(away_days_rest == 1) if pd.notna(away_days_rest) else 0
    row["HOME_LONG_REST"] = int(home_days_rest >= 4) if pd.notna(home_days_rest) else 0
    row["AWAY_LONG_REST"] = int(away_days_rest >= 4) if pd.notna(away_days_rest) else 0
    row["HOME_GAMES_INTO_SEASON"] = float(home.get("GAMES_INTO_SEASON", 0) or 0)
    row["AWAY_GAMES_INTO_SEASON"] = float(away.get("GAMES_INTO_SEASON", 0) or 0)
    row["REST_DIFF"] = (row["HOME_DAYS_REST"] - row["AWAY_DAYS_REST"]
                        if pd.notna(row["HOME_DAYS_REST"])
                        and pd.notna(row["AWAY_DAYS_REST"]) else np.nan)
    row["B2B_DIFF"] = row["HOME_B2B"] - row["AWAY_B2B"]
    # ELO from the cached state (current pre-game rating).
    home_elo = cache.elo_state.get(home_tri, season)
    away_elo = cache.elo_state.get(away_tri, season)
    row["HOME_ELO_PRE"] = float(home_elo)
    row["AWAY_ELO_PRE"] = float(away_elo)
    row["ELO_DIFF"] = float(home_elo - away_elo + ELO_HOME_COURT_ADV)
    row["ELO_WIN_PROB_HOME"] = float(
        1.0 / (1.0 + 10.0 ** (-row["ELO_DIFF"] / 400.0))
    )
    # Stubbed neutral H2H (the slow path computes it; for inference
    # the rolling stats already capture most matchup-level signal).
    row["H2H_WINS_BEFORE"] = 0.5
    return pd.DataFrame([row]).replace([np.inf, -np.inf], np.nan)


def build_inference_row(
    panel: pd.DataFrame,
    home_tri: str,
    away_tri: str,
    game_date: pd.Timestamp,
    season: str,
) -> Optional[pd.DataFrame]:
    """Produce a single-row DataFrame matching the training-time
    feature columns for an upcoming home/away matchup.

    Walks the same enrichment pipeline as ``build_training_table`` but
    treats the current row as having NULL stats (it hasn't happened
    yet). All rolling features are computed over the team's prior
    history; ELO uses the current state.
    """
    if panel.empty:
        return None
    # Append a synthetic row for the upcoming game so the rolling
    # transforms have a target index. Stats are NaN; only the schedule
    # fields matter. nba_api parses GAME_DATE as tz-naive, so strip
    # any tz info on the inference timestamp to avoid mixed-tz sort
    # failures inside pandas.
    target_ts = pd.Timestamp(game_date)
    if target_ts.tz is not None:
        target_ts = target_ts.tz_convert(None) if hasattr(target_ts, "tz_convert") else target_ts.tz_localize(None)
    target_ts = target_ts.normalize()
    rows: List[dict] = []
    for tri, opp, home_flag in (
        (home_tri, away_tri, 1),
        (away_tri, home_tri, 0),
    ):
        rows.append({
            "GAME_ID": "INFER",
            "GAME_DATE": target_ts,
            "SEASON": season,
            "TEAM_ABBREVIATION": tri,
            "OPP_TRICODE": opp,
            "HOME": home_flag,
            "WL": "U",
            "WIN": 0,
            "MIN": np.nan,
            "FGM": np.nan, "FGA": np.nan, "FG3M": np.nan, "FG3A": np.nan,
            "FTM": np.nan, "FTA": np.nan, "OREB": np.nan, "DREB": np.nan,
            "REB": np.nan, "AST": np.nan, "STL": np.nan, "BLK": np.nan,
            "TOV": np.nan, "PF": np.nan, "PTS": np.nan, "PLUS_MINUS": np.nan,
            "MATCHUP": f"{home_tri} {'vs.' if home_flag else '@'} {opp}",
        })
    inf_panel = pd.concat(
        [panel, pd.DataFrame(rows)], ignore_index=True,
    ).sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)
    g, _ = build_training_table(inf_panel)
    pick = g[(g["GAME_ID"] == "INFER")]
    if pick.empty:
        return None
    return pick.iloc[[-1]]


# --------------------------------------------------------------------------- #
# Live injury impact (best-effort posterior adjustment)
# --------------------------------------------------------------------------- #

# ESPN injury-report status weights — fraction of "lost minutes" we
# attribute to each status. Day-to-day implies the player MAY play.
INJURY_STATUS_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.85,
    "questionable": 0.4,
    "probable": 0.1,
    "day-to-day": 0.3,
}


def compute_injury_impact(
    home_tri: str,
    away_tri: str,
    injuries_by_team: Dict[str, List[dict]],
) -> Tuple[float, float]:
    """Return (home_impact_score, away_impact_score) — each a
    non-negative float estimating "how much availability damage the
    team is taking into this game". Higher = more players out.

    We don't have per-player BPM/RAPTOR readily; the score is a
    simple "weighted count of out/doubtful/questionable players"
    that proxies roster-strength loss. The bot's decision engine
    blends this as an EV nudge against the team with higher impact.
    """
    def _score(team: str) -> float:
        rows = injuries_by_team.get(team, []) or []
        s = 0.0
        for r in rows:
            status = (r.get("status") or "").lower().strip()
            s += INJURY_STATUS_WEIGHTS.get(status, 0.0)
        return s
    return _score(home_tri), _score(away_tri)
