"""Decision engine.

Given a Kalshi market (with strike + question direction), the live order
book, and a model distribution for the underlying gas price, this module:

1. Computes the model's probability that the YES side resolves true.
2. Compares against the market's implied probability (the YES ask, in
   dollars). The "edge" is `model_prob - market_ask`.
3. Recommends a side (YES / NO / NONE) based on configured edge floors.

The output is a `Decision` record written to the audit log every poll —
including the no-bet snapshots — so we can post-hoc evaluate calibration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

from kalshi_sdk.validators import (
    ev_for_side as _shared_ev_for_side,
    select_side_by_ev,
)

from .client import Orderbook
from .config import EdgeCfg
from .market_scanner import GasMarket
from .model import GasDistribution

log = logging.getLogger(__name__)


@dataclass
class Decision:
    timestamp: str           # ISO8601 UTC
    ticker: str
    title: str
    direction: str           # market direction: above/below/between
    strike_low: Optional[float]
    strike_high: Optional[float]
    current_gas_price: float
    median_predicted_price: float
    model_prob_yes: float
    yes_ask_dollars: Optional[float]
    no_ask_dollars: Optional[float]
    edge_yes: Optional[float]      # legacy: prob-edge (deprecated, kept for back-compat)
    edge_no: Optional[float]
    side: str                # "YES" | "NO" | "NONE"
    reason: str
    # ── EV-first decision audit (v2) ─────────────────────────────────
    # All probabilities here are the BLENDED posterior the bot acts on.
    # Stored on the position at open so post-trade diagnostics can
    # reproduce the trade rationale without recomputing.
    model_yes_prob_at_entry: Optional[float] = None
    model_no_prob_at_entry: Optional[float] = None
    kalshi_yes_prob_at_entry: Optional[float] = None
    kalshi_no_prob_at_entry: Optional[float] = None
    break_even_probability: Optional[float] = None     # = entry_price_dollars on the side we'd buy
    ev_yes_per_contract: Optional[float] = None        # net of half-spread
    ev_no_per_contract: Optional[float] = None
    selected_side_ev: Optional[float] = None           # EV on the chosen side, or None
    gates_passed: list = field(default_factory=list)
    gates_failed: list = field(default_factory=list)


def _market_implied_yes_prob(yes_ask_cents: Optional[int]) -> Optional[float]:
    """Treat the YES ask as the implied probability (cents -> dollars)."""
    if yes_ask_cents is None:
        return None
    return yes_ask_cents / 100.0


def model_prob_yes(market: GasMarket, dist: GasDistribution,
                   horizon_weeks: float = 1.0) -> Optional[float]:
    """Compute pure-model P(YES resolves true) for a parsed NBA market.

    NBA markets are binary team-wins: the YES side resolves true iff
    ``market.team_being_asked`` wins the game. The model produces a
    P(home_wins) which we map to P(YES) based on which side YES is.

    ``horizon_weeks`` is accepted for cross-bot API symmetry but is a
    no-op here — NBA games are discrete events, not continuous diffusions.

    NOTE: in the live bot this is the *prior*. The decision engine
    actually trades on `blended_prob_yes(...)` which combines this with
    Kalshi's market-implied probability.
    """
    p_home = float(getattr(dist, "home_win_prob", dist.prob_up))
    p_home = max(0.01, min(0.99, p_home))
    if market.direction != "team_wins" or not getattr(market, "team_being_asked", ""):
        # Fallback to the cross-bot above/below path so generic callers
        # don't break.
        if market.direction == "above" and market.strike_low is not None:
            return dist.prob_above(market.strike_low, horizon_weeks=horizon_weeks)
        if market.direction == "below" and market.strike_low is not None:
            return 1.0 - dist.prob_above(market.strike_low, horizon_weeks=horizon_weeks)
        if (market.direction == "between"
                and market.strike_low is not None
                and market.strike_high is not None):
            return dist.prob_between(market.strike_low, market.strike_high,
                                      horizon_weeks=horizon_weeks)
        return None
    if market.team_being_asked == market.home_tricode:
        return p_home
    if market.team_being_asked == market.away_tricode:
        return 1.0 - p_home
    return None


def blended_prob_yes(
    market: GasMarket,
    dist: GasDistribution,
    market_implied_yes: Optional[float],
    horizon_weeks: float,
    model_accuracy: float = 1.0,
) -> Optional[float]:
    """Posterior P(YES) — Bayesian blend of model prior with market data.

    NBA-specific note: the other macro bots (CPI, retail-gas, claims)
    use sqrt(TTC) horizon decay because they're 1-week-ahead time-series
    forecasters whose calibration degrades intraday. The NBA model is a
    *pre-game binary classifier* — it was trained AND evaluated at
    pre-game time, so there is no shorter horizon where it's "less
    reliable". Applying sqrt(TTC) here silenced the model to ~4% weight
    on the day of a game and left zero edges. We drop the horizon decay
    and weight purely by demonstrated skill.

        skill    = max(0, 2 · model_accuracy − 1)
        w_model  = skill
        p_blend  = w_model · p_model  +  (1 − w_model) · p_market

    Examples:
      • acc=0.60 → skill=0.20 → 20% model, 80% market
      • acc=0.67 → skill=0.34 → 34% model, 66% market   (current NBA)
      • acc=0.90 → skill=0.80 → 80% model, 20% market

    Effect: a marginally-accurate model acts as a meaningful tilt on top
    of the market without overriding it. As the model's track record
    improves (visible on the dashboard's "Actual win %" card), the blend
    automatically shifts more weight to the model. The edge floors in
    config.yaml are calibrated to this weighting so trades only fire
    when the raw model materially disagrees with the market.

    Returns None if the model couldn't evaluate; falls back to pure
    model when the market side has no implied probability.
    """
    p_model = model_prob_yes(market, dist, horizon_weeks=horizon_weeks)
    if p_model is None:
        return None
    if market_implied_yes is None:
        return p_model
    skill = max(0.0, 2.0 * float(model_accuracy) - 1.0)
    w_model = skill
    p_blend = w_model * p_model + (1.0 - w_model) * float(market_implied_yes)
    # Same clamp as prob_above: never claim near-certainty.
    return float(min(0.99, max(0.01, p_blend)))


# Back-compat alias: shared implementation lives in kalshi_sdk.validators
# so every bot computes EV the same way.
ev_for_side = _shared_ev_for_side


def make_decision(
    market: GasMarket,
    orderbook: Orderbook,
    dist: GasDistribution,
    cfg: EdgeCfg,
    horizon_weeks: float = 1.0,
    model_accuracy: float = 1.0,
) -> Decision:
    yes_ask = orderbook.yes_best_ask()
    no_ask = orderbook.no_best_ask()
    yes_ask_dollars = _market_implied_yes_prob(yes_ask)
    no_ask_dollars = _market_implied_yes_prob(no_ask)
    # Use the BLENDED posterior: at long TTC the model dominates, at
    # short TTC we lean on the market. This is the probability the bot
    # actually trades against — model alone is the prior, market is the
    # update, weight is sqrt(time-to-close in weeks).
    market_yes_for_blend = (yes_ask_dollars
                            if yes_ask_dollars is not None
                            else (1.0 - no_ask_dollars
                                  if no_ask_dollars is not None else None))
    p_yes = blended_prob_yes(market, dist,
                             market_implied_yes=market_yes_for_blend,
                             horizon_weeks=horizon_weeks,
                             model_accuracy=model_accuracy)
    p_no = (1.0 - p_yes) if p_yes is not None else None
    # Raw (un-blended) model prob — used for the raw-edge gate. The
    # blend can mask large or zero raw gaps depending on the skill
    # multiplier, so we need the raw view to gate honestly.
    raw_p_yes = model_prob_yes(market, dist, horizon_weeks=horizon_weeks)
    raw_p_no = (1.0 - raw_p_yes) if raw_p_yes is not None else None

    # Round-trip half-spread cost: every fill costs half the spread, then
    # the same again on exit. Charge it once up front in EV calc.
    spread_cents = orderbook.spread_cents() or 0
    half_spread_dollars = (spread_cents / 2.0) / 100.0

    # ── EV per side (the primary decision signal) ────────────────────
    ev_yes = (ev_for_side(p_yes, yes_ask_dollars, half_spread_dollars)
              if p_yes is not None and yes_ask_dollars is not None else None)
    ev_no = (ev_for_side(p_no, no_ask_dollars, half_spread_dollars)
             if p_no is not None and no_ask_dollars is not None else None)
    # Legacy prob-edge fields kept for back-compat with old audit logs.
    # (Same numbers — EV without the half-spread baked in is just p - ask.)
    edge_yes = ev_yes
    edge_no = ev_no

    # ── EV / edge / max-entry side selection (shared with every bot) ─
    #
    # NBA-specific: the SDK gates confidence on BLENDED p_yes, but the
    # blended posterior hugs the market price by construction (see
    # blended_prob_yes above), so the SDK would reject every market priced
    # 45-55¢ regardless of what the model thinks. main._record_view already
    # gates confidence on RAW conviction — mirror that here. If raw passes,
    # zero out the SDK's confidence floor so it still runs the rest of the
    # EV / raw-edge / max-entry checks and populates selected_side_ev (the
    # dashboard's Entry EV column reads from this).
    raw_conviction = raw_p_yes if raw_p_yes is not None else p_yes
    conf_low = float(cfg.min_model_confidence)
    raw_in_band = (raw_conviction is not None
                   and conf_low <= raw_conviction <= 1.0 - conf_low)
    if raw_in_band:
        # Raw model is undecided — short-circuit with SKIP. Don't delegate
        # to the SDK because its blended-prob gate could PASS when blended
        # drifts outside the band (e.g. extreme market price), which would
        # let an opinion-less model trade.
        from kalshi_sdk.validators import SideDecision
        decision = SideDecision(
            side="NONE",
            reason=f"low_model_confidence ({raw_conviction:.2f})",
            selected_ev=None,
            break_even_prob=yes_ask_dollars,
            gates_passed=[],
            gates_failed=[f"raw_confidence_band: raw={raw_conviction:.2f}"],
        )
    else:
        # SDK gate is `conf_low <= p_yes <= 1-conf_low → reject`. Setting
        # conf_low > 0.5 inverts the band (lo > hi) so no p_yes matches →
        # gate effectively disabled. We already validated raw conviction
        # above, so the SDK's blended-prob check is redundant here.
        ev_cfg = SimpleNamespace(
            min_model_confidence=0.51,
            min_ev_per_contract=cfg.min_ev_per_contract,
            min_prob_edge_over_breakeven=cfg.min_prob_edge_over_breakeven,
            min_raw_model_edge=getattr(cfg, "min_raw_model_edge", 0.0),
            max_entry_price_cents=cfg.max_entry_price_cents,
        )
        decision = select_side_by_ev(
            p_yes=p_yes, p_no=p_no,
            yes_ask_dollars=yes_ask_dollars, no_ask_dollars=no_ask_dollars,
            raw_p_yes=raw_p_yes, raw_p_no=raw_p_no,
            half_spread_dollars=half_spread_dollars,
            cfg=ev_cfg,
        )
    side = decision.side
    # Match _record_view's convention: a buy verdict reports "edge_met"
    # (mirrors market_views.rejection_reason in sim.db). The SDK / prior
    # gate's reason only applies when we're skipping.
    reason = "edge_met" if side in ("YES", "NO") else decision.reason
    selected_ev = decision.selected_ev
    break_even_prob = decision.break_even_prob
    gates_passed = decision.gates_passed
    gates_failed = decision.gates_failed

    return Decision(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ticker=market.ticker,
        title=market.title or market.yes_sub_title,
        direction=market.direction,
        strike_low=market.strike_low,
        strike_high=market.strike_high,
        current_gas_price=dist.current_gas_price,
        median_predicted_price=dist.median_price,
        model_prob_yes=p_yes if p_yes is not None else float("nan"),
        yes_ask_dollars=yes_ask_dollars,
        no_ask_dollars=no_ask_dollars,
        edge_yes=edge_yes,
        edge_no=edge_no,
        model_yes_prob_at_entry=p_yes,
        model_no_prob_at_entry=p_no,
        kalshi_yes_prob_at_entry=yes_ask_dollars,
        kalshi_no_prob_at_entry=no_ask_dollars,
        break_even_probability=break_even_prob,
        ev_yes_per_contract=ev_yes,
        ev_no_per_contract=ev_no,
        selected_side_ev=selected_ev,
        gates_passed=gates_passed,
        gates_failed=gates_failed,
        side=side,
        reason=reason,
    )


def decision_to_jsonl(d: Decision) -> str:
    import json
    return json.dumps(asdict(d), default=str)
