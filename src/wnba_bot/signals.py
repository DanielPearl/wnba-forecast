"""Signal taxonomy for NBA — mirrors ``baseline-break/src/trading/signals.py``.

Replaces the binary YES/NO/NONE side with a 7-label taxonomy that
encodes both "is there an edge" and "is the edge tradeable right now":

  INJURY_RISK         → injury / availability flag set; skip
  AVOID_VOLATILE      → volatility above tradeable cap (OT, last-3-min,
                         late-game close); skip even if edge looks good
  MARKET_OVERREACTION → market moved further than model adjustment;
                         tradeable as a fade
  STRONG_EDGE         → |edge| ≥ strong_edge_min
  SMALL_EDGE          → |edge| ≥ small_edge_min
  WATCH               → interesting matchup, no actionable edge
  NO_TRADE            → default

These don't replace the existing decision-engine YES/NO output — the
production bot still writes its existing ``Decision`` for the trade
flow. The signal label is metadata for the dashboard and post-hoc
analytics so we can measure how often each label class actually wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# Default thresholds. Tuned to match the tennis bot — conservative
# floors that print fewer signals than the strict edge gate so the
# label always represents a real opportunity, not a noise edge.
DEFAULT_SMALL_EDGE = 0.04
DEFAULT_STRONG_EDGE = 0.08
DEFAULT_MIN_MARKET_PROB = 0.10
DEFAULT_MAX_MARKET_PROB = 0.90
DEFAULT_MAX_VOLATILITY = 0.55


@dataclass
class SignalResult:
    label: str
    reason: str
    confidence_score: float


def _confidence(model_prob: float, volatility: float) -> float:
    """0-1 confidence. Drops with volatility and at extreme tails of
    the calibration band."""
    base = 0.85
    base -= 0.55 * float(volatility)
    if model_prob > 0.92 or model_prob < 0.08:
        base -= 0.20
    return max(0.0, min(1.0, base))


def label_match(
    model_prob_yes: float,
    market_prob_yes: Optional[float],
    *,
    volatility: float = 0.0,
    injury_flag: bool = False,
    market_overreaction: bool = False,
    rules_fired: Optional[List[str]] = None,
    small_edge_min: float = DEFAULT_SMALL_EDGE,
    strong_edge_min: float = DEFAULT_STRONG_EDGE,
    min_market_prob: float = DEFAULT_MIN_MARKET_PROB,
    max_market_prob: float = DEFAULT_MAX_MARKET_PROB,
    max_volatility: float = DEFAULT_MAX_VOLATILITY,
) -> SignalResult:
    """Return a ``SignalResult`` for the given (model, market, state).

    ``model_prob_yes`` and ``market_prob_yes`` are both YES probabilities
    for the SAME side; the label is symmetric — a strong negative edge
    (model says YES is overpriced) earns a STRONG_EDGE on the NO side
    just like a strong positive edge earns it on YES. The dashboard
    surfaces direction via the sign of ``edge``.
    """
    rules_fired = rules_fired or []

    # 1) Injury risk dominates everything.
    if injury_flag:
        return SignalResult(
            label="INJURY_RISK",
            reason="injury / availability flag set — skip until resolved",
            confidence_score=_confidence(model_prob_yes, volatility) * 0.5,
        )

    edge = (model_prob_yes - market_prob_yes
            if market_prob_yes is not None else 0.0)
    edge_abs = abs(edge)

    # 2) Market overreaction — fades a stretched market move. Note this
    #    BEFORE the volatility cap because an overreaction is itself a
    #    tradeable thesis (the market went too far on news the model
    #    already absorbed).
    if (market_overreaction and market_prob_yes is not None
            and edge_abs >= small_edge_min
            and min_market_prob <= market_prob_yes <= max_market_prob):
        reason = "; ".join(rules_fired) or "market move outpaces model adjustment"
        return SignalResult(
            label="MARKET_OVERREACTION",
            reason=reason,
            confidence_score=_confidence(model_prob_yes, volatility),
        )

    # 3) Volatility cap.
    if volatility >= max_volatility:
        return SignalResult(
            label="AVOID_VOLATILE",
            reason="volatility above tradeable cap — wait for the game to settle",
            confidence_score=_confidence(model_prob_yes, volatility),
        )

    # 4) Edge size, gated on a sane market price band.
    if market_prob_yes is None:
        return SignalResult(
            label="WATCH",
            reason="no market price observed — model-only forecast",
            confidence_score=_confidence(model_prob_yes, volatility) * 0.7,
        )
    if not (min_market_prob <= market_prob_yes <= max_market_prob):
        return SignalResult(
            label="NO_TRADE",
            reason=(f"market price {market_prob_yes:.0%} outside tradeable "
                    f"band ({min_market_prob:.0%}-{max_market_prob:.0%})"),
            confidence_score=_confidence(model_prob_yes, volatility),
        )
    if edge_abs >= strong_edge_min:
        return SignalResult(
            label="STRONG_EDGE",
            reason=f"model {edge*100:+.1f}pp vs market",
            confidence_score=_confidence(model_prob_yes, volatility),
        )
    if edge_abs >= small_edge_min:
        return SignalResult(
            label="SMALL_EDGE",
            reason=f"model {edge*100:+.1f}pp vs market",
            confidence_score=_confidence(model_prob_yes, volatility),
        )

    # 5) Default WATCH for matches where the model has a non-trivial view.
    return SignalResult(
        label="WATCH",
        reason="model view aligned with market — no edge",
        confidence_score=_confidence(model_prob_yes, volatility),
    )
