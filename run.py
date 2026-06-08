"""NBA bot entry point.

Usage:
    python run.py            # main run loop
    python run.py --train    # train the model and exit
    python run.py --once     # run a single tick and exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wnba_bot.client import KalshiClient, KalshiError  # noqa: E402
from wnba_bot.config import Config, load_config  # noqa: E402
from wnba_bot.logging_setup import setup_logging  # noqa: E402
from wnba_bot.main import Bot, _current_wnba_season  # noqa: E402

log = logging.getLogger("run")


def _fail(msg: str):
    print(f"\n[startup] ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def _check_key_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        _fail(f"Private key not found at {path}.")
    body = p.read_text()
    if "PASTE_YOUR_KALSHI_PRIVATE_KEY_HERE" in body or "PLACEHOLDER" in body:
        _fail(f"{path} still contains placeholder text.")
    if "BEGIN" not in body or "PRIVATE KEY" not in body:
        _fail(f"{path} does not look like a PEM private key.")


def _check_api_key_id(key_id: str) -> None:
    if not key_id or key_id.startswith("REPLACE_"):
        _fail("KALSHI_API_KEY_ID in .env is still the placeholder.")


def _banner(cfg: Config, balance_cents: int | None) -> None:
    bal = f"${balance_cents/100:.2f}" if balance_cents is not None else "n/a"
    lines = [
        "=" * 72,
        f"  Kalshi WNBA forecast bot  |  PROD Kalshi  |  mode=SIMULATION",
        "=" * 72,
        f"  Account balance       : {bal}",
        f"  Markets watched       : {cfg.run.market_series_prefixes}",
        f"  Bet size              : ${cfg.risk.bet_size_cents/100:.2f}/opportunity",
        f"  Max open positions    : {cfg.risk.max_open_positions}",
        f"  Max total exposure    : ${cfg.risk.max_total_exposure_cents/100:.2f}",
        f"  Max bets / day        : {cfg.risk.max_bets_per_day}",
        f"  EV floor              : >= ${cfg.edge.min_ev_per_contract:.2f}/contract",
        f"  Active poll / idle    : {cfg.run.poll_interval_seconds_active}s / "
        f"{cfg.run.poll_interval_seconds_idle}s",
        f"  Sim DB / decisions    : {cfg.execution.sim_db_path} / "
        f"{cfg.execution.decisions_log_path}",
        "=" * 72,
    ]
    for ln in lines:
        print(ln)


def preflight(cfg: Config) -> KalshiClient:
    _check_api_key_id(cfg.env.api_key_id)
    _check_key_file(cfg.env.private_key_path)
    log.info("connecting to %s ...", cfg.env.base_url)
    client = KalshiClient(
        base_url=cfg.env.base_url,
        api_key_id=cfg.env.api_key_id,
        private_key_path=cfg.env.private_key_path,
    )
    try:
        status = client.get_exchange_status()
        log.info("exchange status: %s", status)
    except KalshiError as e:
        _fail(f"Could not reach Kalshi exchange status:\n  {e}")
    except Exception as e:  # noqa: BLE001
        _fail(f"Network error reaching {cfg.env.base_url}: {e}")
    balance_cents: int | None = None
    try:
        bal = client.get_balance()
        balance_cents = int(bal.get("balance", 0))
    except Exception as e:  # noqa: BLE001
        log.warning("balance fetch failed: %s", e)
    _banner(cfg, balance_cents)
    return client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="train several model families and report a "
                             "held-out bake-off, then exit")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.env.log_path)

    def _seasons():
        return cfg.run.training_seasons or [
            "2021", "2022", "2023", "2024", "2025", _current_wnba_season(),
        ]

    if args.compare:
        from wnba_bot.data_sources import fetch_multi_season_panel
        from wnba_bot.features import build_training_table
        from wnba_bot.compare_models import run_bakeoff
        from pathlib import Path as _P

        seasons = _seasons()
        log.info("model bake-off on WNBA seasons: %s", seasons)
        panel = fetch_multi_season_panel(seasons)
        if panel.empty:
            _fail("game-log panel is empty — check ESPN connectivity")
        df, feature_cols = build_training_table(panel)
        out_dir = _P(cfg.model.artifact_path).parent
        run_bakeoff(
            df, feature_cols,
            test_size_games=cfg.model.test_size_games,
            out_dir=str(out_dir),
        )
        return 0

    if args.train:
        from wnba_bot.data_sources import fetch_multi_season_panel
        from wnba_bot.features import build_training_table
        from wnba_bot.model import save_model, train_model
        from pathlib import Path as _P

        seasons = _seasons()
        log.info("training WNBA model on seasons: %s", seasons)
        panel = fetch_multi_season_panel(seasons)
        if panel.empty:
            _fail("game-log panel is empty — check ESPN connectivity")
        df, feature_cols = build_training_table(panel)
        log.info("training table: %d games × %d candidate features",
                 len(df), len(feature_cols))
        importance_csv = _P(cfg.model.artifact_path).parent / "feature_importance.csv"
        model = train_model(
            df, feature_cols,
            test_size_games=cfg.model.test_size_games,
            walk_forward_splits=cfg.model.walk_forward_splits,
            ensemble_seeds=cfg.model.ensemble_seeds,
            calibration_holdout_games=cfg.model.calibration_holdout_games,
            importance_csv_path=str(importance_csv),
            max_features=cfg.model.max_features,
            feature_min_positive_folds=cfg.model.feature_min_positive_folds,
            feature_correlation_max=cfg.model.feature_correlation_max,
        )
        save_model(model, cfg.model.artifact_path)
        print("\nMetrics:")
        for k, v in model.metrics.items():
            print(f"  {k:35s}: {v}")
        return 0

    preflight(cfg)
    bot = Bot(cfg)
    try:
        if args.once:
            bot.tick()
            return 0
        bot.run()
    except KeyboardInterrupt:
        log.info("interrupted by user")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
