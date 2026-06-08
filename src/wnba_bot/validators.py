"""Pre-trade validators (thin wrapper over shared kalshi_sdk module).

The structural gates (liquidity, spread, time-to-close, probability
bounds, basis-risk) live in ``kalshi_sdk.validators.validate_market`` so
all bots stay in sync. This file adapts the NBA market scanner output
and the team-wins exception (binary outcomes with no numeric strike).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

from kalshi_sdk.validators import validate_market as _shared_validate_market

from .client import Orderbook
from .config import ValidatorCfg
from .market_scanner import GasMarket, minutes_to_close


def validate_market(
    market: GasMarket,
    orderbook: Orderbook | None,
    side_yes_ask_cents: int | None,
    cfg: ValidatorCfg,
    now: datetime | None = None,
    model_median_price: Optional[float] = None,
    side: Optional[str] = None,
) -> Tuple[bool, str]:
    return _shared_validate_market(
        market,
        orderbook,
        cfg,
        side_yes_ask_cents=side_yes_ask_cents,
        now=now,
        model_median_price=model_median_price,
        side=side,
        directions_without_strike=("team_wins",),
        basis_risk_unit_label="pp",
        minutes_to_close_fn=lambda m, n: minutes_to_close(m, now=n),
    )
