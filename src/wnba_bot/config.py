"""Config loading for the NBA bot.

Reads YAML from config/config.yaml and environment variables from .env.
Mirrors the CPI/claims structure: every module takes a typed Config;
no module reaches into os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml
from dotenv import load_dotenv


@dataclass
class EnvCfg:
    api_key_id: str
    private_key_path: str
    log_path: str

    @property
    def base_url(self) -> str:
        return "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class RunCfg:
    market_series_prefixes: List[str]
    poll_interval_seconds_active: int
    poll_interval_seconds_idle: int
    market_hours_start: str   # ET
    market_hours_end: str     # ET
    max_markets_per_poll: int
    # Seasons used for training game logs (e.g. ["2018-19", ..., "2024-25"]).
    training_seasons: List[str] = field(default_factory=list)


@dataclass
class ModelCfg:
    """NBA model knobs. The horizon is single-event (one game), so
    no monthly/weekly cadence applies — just retrain frequency."""
    test_size_games: int             # ~1230 = 1 season
    walk_forward_splits: int
    retrain_day_of_week: int         # 0=Mon
    retrain_hour_et: int
    artifact_path: str
    ensemble_seeds: int = 5
    calibration_holdout_games: int = 400
    max_features: int = 30
    feature_min_positive_folds: int = 3
    feature_correlation_max: float = 0.92


@dataclass
class EdgeCfg:
    min_edge_yes: float
    min_edge_no: float
    min_model_confidence: float
    min_confidence: float = 0.10
    min_model_accuracy: float = 0.55
    min_ev_per_contract: float = 0.03
    min_prob_edge_over_breakeven: float = 0.04
    max_entry_price_cents: int = 80
    min_raw_model_edge: float = 0.04


@dataclass
class HedgeCfg:
    enabled: bool
    profit_lock_cents: int
    stop_loss_cents: int
    hedge_size_fraction: float


@dataclass
class ValidatorCfg:
    min_book_depth_contracts: int
    max_spread_cents: int
    min_minutes_to_close: int
    max_minutes_to_close: int
    prob_bounds_cents: Tuple[int, int]
    min_volume: int = 0
    min_open_interest: int = 0
    min_depth_at_best_ask: int = 0
    # NBA equivalent of basis-risk: skip last-minute trading inside X
    # minutes of tipoff when the book often goes wide. Field name kept
    # for cross-bot config-loader symmetry.
    basis_risk_strike_window_dollars: float = 0
    basis_risk_max_hours_to_close: float = 0


@dataclass
class RiskCfg:
    bet_size_cents: int
    max_open_positions: int
    max_total_exposure_cents: int
    max_bets_per_day: int
    cooldown_seconds_same_market: int


@dataclass
class ExecutionCfg:
    dry_run: bool
    sim_db_path: str
    decisions_log_path: str


@dataclass
class Config:
    env: EnvCfg
    run: RunCfg
    model: ModelCfg
    edge: EdgeCfg
    hedge: HedgeCfg
    validators: ValidatorCfg
    risk: RiskCfg
    execution: ExecutionCfg
    raw: dict = field(default_factory=dict)


def _must(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing env var {name}. Copy .env.example to .env and fill it in."
        )
    return val


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    load_dotenv()
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    env = EnvCfg(
        api_key_id=_must("KALSHI_API_KEY_ID"),
        private_key_path=_must("KALSHI_PRIVATE_KEY_PATH"),
        log_path=os.getenv("BOT_LOG_PATH", "./data/bot.log"),
    )

    validators_raw = dict(raw["validators"])
    validators_raw["prob_bounds_cents"] = tuple(validators_raw["prob_bounds_cents"])

    return Config(
        env=env,
        run=RunCfg(**raw["run"]),
        model=ModelCfg(**raw["model"]),
        edge=EdgeCfg(**raw["edge"]),
        hedge=HedgeCfg(**raw["hedge"]),
        validators=ValidatorCfg(**validators_raw),
        risk=RiskCfg(**raw["risk"]),
        execution=ExecutionCfg(**raw["execution"]),
        raw=raw,
    )
