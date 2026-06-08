"""Live in-game adjustment layer for NBA.

Mirror of the tennis bot's ``live_adjustment_model.py``: a transparent,
rules-based layer that nudges the pre-game model probability using
in-progress game state — score differential, period, momentum
(last-quarter delta), late-game volatility (close game in Q4 / OT),
and injury / foul-out events surfaced by the live feed.

Why rules-based instead of an ML adjustment model: with only a few
seasons of point-by-point data and no historical paper-trade feedback,
a small ML adjustment overfits to the wrong things. The rules here
each express a calibrated point estimate from the literature
(e.g. mid-game lead-to-win lookups from inpredictable / fivethirtyeight)
plus a conservative cap so a single noisy quarter can't dominate.

Outputs

  ``LiveAdjustment``:
    pre_game_prob_home   — what the pre-tip model said
    live_prob_home       — adjusted P(home wins) given current state
    volatility_score     — 0..1; high = avoid trading
    market_overreaction  — bool, True when the market moved further
                            than the model's adjustment justifies
    rules_fired          — list of human-readable reasons (drives the
                            "Why?" column on the dashboard)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class LiveGameState:
    """Live game snapshot in a shape the rules engine can consume.

    All fields tolerate ``None`` so a half-populated ESPN payload
    doesn't crash the layer — every rule short-circuits when its
    input is missing instead of guessing.
    """
    home_tricode: str
    away_tricode: str
    status: str                    # "scheduled" | "in" | "final" | ...
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    period: Optional[int] = None       # 1..4 quarters; 5+ for OT
    seconds_remaining: Optional[int] = None  # in current period
    home_linescores: List[int] = field(default_factory=list)
    away_linescores: List[int] = field(default_factory=list)
    # Posterior signals — set when the live feed reports them.
    home_star_out: bool = False
    away_star_out: bool = False
    home_star_foul_trouble: bool = False  # 4+ fouls on a starter
    away_star_foul_trouble: bool = False
    # Prior tick's market price for player_a (= home), used to detect
    # market overreactions. None on first tick of the game.
    market_prob_home_prev: Optional[float] = None
    market_prob_home_curr: Optional[float] = None


@dataclass
class LiveAdjustment:
    pre_game_prob_home: float
    live_prob_home: float
    volatility_score: float
    market_overreaction: bool
    rules_fired: List[str]
    # Raw trained-model output (None when the artifact is missing or
    # the gate is off) and what the rules-only path produced.
    model_prob_home: Optional[float] = None
    rules_prob_home: Optional[float] = None


# --- Constants ----------------------------------------------------------- #
# Each completed possession of margin is worth ~0.6pp of in-game win
# probability under typical NBA pace (literature: live win-prob curves
# from BigBallData / inpredictable). We cap the overall in-game shift
# at ±35pp — even a 25-point lead in Q4 doesn't make a comeback
# impossible (the model still keeps a calibration tail).
_PT_TO_PROB = 0.006
_MAX_LIVE_SHIFT = 0.35

# Period weighting: a 5-point lead at half is way less predictive than
# a 5-point lead with 3 minutes left. We scale the score-state nudge
# by these factors.
_PERIOD_WEIGHT = {
    1: 0.30,   # Q1
    2: 0.55,   # Q2
    3: 0.75,   # Q3
    4: 1.00,   # Q4
    5: 1.20,   # OT
    6: 1.30,   # 2OT+
}

# Volatility:
_VOL_BASE = 0.05
_VOL_LATE_CLOSE = 0.40        # within 5 in Q4 with <3 min
_VOL_OT = 0.50                # any OT
_VOL_INJURY = 0.30            # star out announced mid-game
_VOL_BLOWOUT_REVERSE = 0.25   # 15+ swing across one quarter
_MAX_VOLATILITY = 1.0

# Market-overreaction band — fires when |market move| >> |model adj|.
_OVERREACTION_MARKET_MOVE = 0.07
_OVERREACTION_MODEL_MOVE = 0.025


def _clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def _safe_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def adjust(pre_game_prob_home: float, state: LiveGameState) -> LiveAdjustment:
    """Apply rules to ``pre_game_prob_home`` based on the live ``state``.

    Returns the adjusted live probability, volatility score, and a
    list of fired rules so the dashboard can show "why" the bot's view
    moved between ticks.
    """
    fired: List[str] = []
    delta = 0.0
    volatility = _VOL_BASE

    # If the game hasn't started, no adjustment.
    if state.status != "in":
        return LiveAdjustment(
            pre_game_prob_home=pre_game_prob_home,
            live_prob_home=_clamp(pre_game_prob_home),
            volatility_score=volatility,
            market_overreaction=False,
            rules_fired=[],
        )

    # ---- 1) Score-state nudge ------------------------------------------
    # Lead in points × period weight × _PT_TO_PROB. Capped at ±_MAX_LIVE_SHIFT.
    h = _safe_int(state.home_score)
    a = _safe_int(state.away_score)
    if h is not None and a is not None:
        margin = h - a  # positive = home leading
        period = state.period or 1
        weight = _PERIOD_WEIGHT.get(period, 1.0)
        score_delta = max(
            -_MAX_LIVE_SHIFT,
            min(_MAX_LIVE_SHIFT, margin * _PT_TO_PROB * weight),
        )
        if abs(score_delta) > 0.005:
            delta += score_delta
            fired.append(
                f"score {h}-{a} (margin {margin:+d}, period {period}, "
                f"weight {weight:.2f}) → {score_delta*100:+.1f}pp on home"
            )

    # ---- 2) Momentum from the last completed quarter -------------------
    # Compare the trailing team's last-quarter scoring to the leader's
    # — if the trailing team JUST scored 35 vs the leader's 18, the
    # comeback narrative deserves a small bump.
    if state.home_linescores and state.away_linescores:
        # Use the most recently *completed* quarter, not the in-progress
        # one (which has zero structural meaning until it ends).
        last_q_home = (state.home_linescores[-2]
                        if len(state.home_linescores) >= 2 else 0)
        last_q_away = (state.away_linescores[-2]
                        if len(state.away_linescores) >= 2 else 0)
        # Treat a 12-pt cross-quarter swing as the threshold for "real"
        # momentum. Bump by ~3pp toward the surging team.
        swing = last_q_home - last_q_away
        if abs(swing) >= 12:
            momentum_delta = 0.03 if swing > 0 else -0.03
            delta += momentum_delta
            fired.append(
                f"last-quarter swing {swing:+d} → {momentum_delta*100:+.1f}pp"
            )
            # Big within-quarter swing also bumps volatility — the game
            # is volatile enough that the next quarter is hard to predict.
            volatility = min(_MAX_VOLATILITY,
                             volatility + _VOL_BLOWOUT_REVERSE)

    # ---- 3) Late-game close-game volatility ----------------------------
    period = state.period or 0
    sec_left = state.seconds_remaining or 999
    abs_margin = (abs((h or 0) - (a or 0))
                  if (h is not None and a is not None) else 999)
    if period >= 4 and sec_left <= 180 and abs_margin <= 5:
        volatility = min(_MAX_VOLATILITY, volatility + _VOL_LATE_CLOSE)
        fired.append(
            f"late-game close: {abs_margin}pt game in Q{period} "
            f"with {sec_left}s left → high volatility"
        )
    if period >= 5:
        volatility = min(_MAX_VOLATILITY, volatility + _VOL_OT)
        fired.append(f"overtime (period {period}) → high volatility")

    # ---- 4) Injury / availability flags --------------------------------
    if state.home_star_out:
        delta -= 0.05
        volatility = min(_MAX_VOLATILITY, volatility + _VOL_INJURY)
        fired.append("home star ruled out mid-game → -5pp + volatility")
    if state.away_star_out:
        delta += 0.05
        volatility = min(_MAX_VOLATILITY, volatility + _VOL_INJURY)
        fired.append("away star ruled out mid-game → +5pp + volatility")
    if state.home_star_foul_trouble:
        delta -= 0.02
        fired.append("home starter in foul trouble → -2pp")
    if state.away_star_foul_trouble:
        delta += 0.02
        fired.append("away starter in foul trouble → +2pp")

    rules_prob = _clamp(pre_game_prob_home + delta)

    # ---- 4b) Trained in-game model (replaces rules nudge if enabled) ----
    # The trained model was fit on real ESPN PBP snapshots from the
    # 2023-24 NBA season; on a held-out 88-game test set it beats the
    # rules baseline by 26.8% Brier. We keep the rules path producing
    # ``rules_fired`` + volatility + injury flags so the dashboard's
    # "why?" column still works when the model is on.
    from . import predict_inmatch  # local import — keeps the rules-only
    # path importable in environments without joblib/sklearn.

    model_prob: Optional[float] = None
    use_model = bool(getattr(state, "use_trained_inmatch_model", True))
    if use_model:
        try:
            model_prob = predict_inmatch.predict(state)
        except Exception as exc:  # noqa: BLE001 — never fail closed
            log.warning("inmatch predict failed: %s", exc)
            model_prob = None
    if model_prob is not None:
        # Progress-weighted blend with the pre-game prior. Late-game
        # snapshots get nearly all weight on the model; early-game the
        # pre-game prior dominates.
        period = state.period or 1
        sec_left = state.seconds_remaining or 0
        if period <= 4:
            elapsed = (period - 1) * 12 * 60 + (12 * 60 - sec_left)
        else:
            elapsed = 4 * 12 * 60 + (period - 5) * 5 * 60 + (5 * 60 - sec_left)
        prog = min(1.0, elapsed / (4 * 12 * 60))
        w_model = max(0.0, min(1.0, 0.25 + 1.0 * prog))
        live_prob = _clamp(w_model * model_prob +
                            (1.0 - w_model) * pre_game_prob_home)
        fired.append(
            f"in-game model {model_prob*100:.1f}% blended w={w_model:.2f} "
            f"with pre-game {pre_game_prob_home*100:.1f}%"
        )
        effective_delta = live_prob - pre_game_prob_home
    else:
        live_prob = rules_prob
        effective_delta = delta

    # ---- 5) Market overreaction detection -------------------------------
    overreaction = False
    if (state.market_prob_home_curr is not None
            and state.market_prob_home_prev is not None):
        market_move = state.market_prob_home_curr - state.market_prob_home_prev
        if (abs(market_move) >= _OVERREACTION_MARKET_MOVE
                and abs(effective_delta) < _OVERREACTION_MODEL_MOVE):
            overreaction = True
            fired.append(
                f"market moved {market_move*100:+.1f}pp but model only "
                f"{effective_delta*100:+.1f}pp — possible overreaction"
            )

    return LiveAdjustment(
        pre_game_prob_home=pre_game_prob_home,
        live_prob_home=live_prob,
        volatility_score=volatility,
        market_overreaction=overreaction,
        rules_fired=fired,
        model_prob_home=model_prob,
        rules_prob_home=rules_prob,
    )


def state_from_espn_event(event: dict,
                           market_prob_home_curr: Optional[float] = None,
                           market_prob_home_prev: Optional[float] = None,
                           ) -> LiveGameState:
    """Build a ``LiveGameState`` from an ``fetch_espn_scoreboard`` row.

    The scoreboard payload has the running totals + status. Per-quarter
    line scores live under the competitor object; we pull them when
    ``data_sources`` plumbs them through. Status mapping:
      ESPN "STATUS_IN_PROGRESS" → "in"
      ESPN "STATUS_FINAL"       → "final"
      anything else             → "scheduled"
    """
    raw_status = (event.get("status") or "").lower()
    if "in_progress" in raw_status or raw_status == "in":
        status = "in"
    elif "final" in raw_status:
        status = "final"
    else:
        status = "scheduled"
    return LiveGameState(
        home_tricode=event.get("home_tricode", ""),
        away_tricode=event.get("away_tricode", ""),
        status=status,
        home_score=_safe_int(event.get("home_score")),
        away_score=_safe_int(event.get("away_score")),
        period=_safe_int(event.get("period")),
        seconds_remaining=_safe_int(event.get("seconds_remaining")),
        home_linescores=list(event.get("home_linescores") or []),
        away_linescores=list(event.get("away_linescores") or []),
        market_prob_home_prev=market_prob_home_prev,
        market_prob_home_curr=market_prob_home_curr,
    )
