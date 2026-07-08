"""NBA bot run loop.

Each tick:
  1. Refresh the historical game-log panel (cached per season — so
     a fresh tick is fast unless we've crossed into a new season).
  2. For every open Kalshi NBA game market, derive a model-row for
     that matchup, compute P(team_being_asked wins), apply injury
     impact as a posterior nudge, and produce a decision.
  3. Validators + risk gates decide whether the bot trades.
  4. Hedge / close existing positions on every tick.

Cadence is intra-day during NBA game hours (typically 17:00-23:00 ET
in the regular season; 11:00-23:00 on weekends and playoffs). Outside
those hours: hourly polls.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, time as dtime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from .client import KalshiClient
from .config import Config
from .data_sources import (
    fetch_espn_injuries, fetch_espn_scoreboard, fetch_multi_season_panel,
    fetch_pinnacle_probs_by_pair,
)
from .decision import make_decision, model_prob_yes
from .features import (
    build_inference_row, build_inference_row_fast,
    compute_injury_impact, precompute_team_states,
)
from .live_adjustment import LiveGameState, adjust as live_adjust, state_from_espn_event
from .market_scanner import discover_wnba_markets, fetch_orderbook, minutes_to_close
from .model import GasModel, GasDistribution, load_model, train_model, save_model
from .reporter import TickReporter, startup_report
from .signals import label_match
from .simulator import Simulator, append_decision
from .validators import validate_market

log = logging.getLogger("wnba_bot")
ET = ZoneInfo("America/New_York")

# WNBA seasons are calendar years ('2024', '2025', ...). The season
# helper lives in data_sources; alias it under the old name so any
# remaining references (and run.py's import) keep working.
from .data_sources import _current_wnba_season  # noqa: E402

_current_nba_season = _current_wnba_season


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = KalshiClient(
            base_url=cfg.env.base_url,
            api_key_id=cfg.env.api_key_id,
            private_key_path=cfg.env.private_key_path,
        )
        self.simulator = Simulator(
            db_path=cfg.execution.sim_db_path,
            risk=cfg.risk,
            hedge=cfg.hedge,
        )
        self._model: Optional[GasModel] = None
        self._panel_cache: Optional[pd.DataFrame] = None
        self._panel_cache_at: Optional[datetime] = None
        # Per-tick cache of the heavy panel transforms — invalidated
        # whenever the panel itself is refreshed.
        self._states_cache = None
        self._states_cache_at: Optional[datetime] = None
        self.reporter = TickReporter(log)

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> None:
        path = Path(self.cfg.model.artifact_path)
        if path.exists() and self._model is None:
            log.info("loading model from %s", path)
            self._model = load_model(path)
            startup_report(log, model_metrics=self._model.metrics,
                           train_end_date=self._model.train_end_date,
                           n_features=len(self._model.feature_columns))
            return
        if self._model is None:
            log.info("no model artifact at %s — training fresh", path)
            self._train_and_save()
            if self._model is not None:
                startup_report(log, model_metrics=self._model.metrics,
                               train_end_date=self._model.train_end_date,
                               n_features=len(self._model.feature_columns))

    def _train_and_save(self) -> None:
        from .features import build_training_table
        seasons = self.cfg.run.training_seasons or [
            "2021", "2022", "2023", "2024", "2025", _current_wnba_season(),
        ]
        log.info("loading game logs for %d seasons", len(seasons))
        panel = fetch_multi_season_panel(seasons)
        if panel.empty:
            raise RuntimeError("game-log panel is empty — cannot train")
        df, feature_cols = build_training_table(panel)
        log.info("built training table: %d games × %d candidate features",
                 len(df), len(feature_cols))
        importance_csv = Path(self.cfg.model.artifact_path).parent / "feature_importance.csv"
        extra_pairs = self._load_training_pairs(feature_cols)
        if extra_pairs is not None and not extra_pairs.empty:
            log.info("loaded %d closed-bet training pairs from sim DB",
                     len(extra_pairs))
        model = train_model(
            df, feature_cols,
            test_size_games=self.cfg.model.test_size_games,
            walk_forward_splits=self.cfg.model.walk_forward_splits,
            ensemble_seeds=self.cfg.model.ensemble_seeds,
            calibration_holdout_games=self.cfg.model.calibration_holdout_games,
            importance_csv_path=str(importance_csv),
            extra_pairs_df=extra_pairs,
            max_features=self.cfg.model.max_features,
            feature_min_positive_folds=self.cfg.model.feature_min_positive_folds,
            feature_correlation_max=self.cfg.model.feature_correlation_max,
        )
        save_model(model, self.cfg.model.artifact_path)
        self._model = model
        self._panel_cache = panel
        self._panel_cache_at = datetime.now(timezone.utc)

    def _load_training_pairs(self, feature_cols):
        import sqlite3
        from contextlib import closing
        try:
            with closing(sqlite3.connect(self.simulator.db_path)) as c:
                c.row_factory = sqlite3.Row
                rows = c.execute(
                    "SELECT features_json, gas_change, horizon_weeks "
                    "FROM training_pairs "
                    "WHERE features_json IS NOT NULL "
                    "  AND gas_change IS NOT NULL "
                    "  AND horizon_weeks IS NOT NULL"
                ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        records = []
        for r in rows:
            try:
                feats = json.loads(r["features_json"])
            except Exception:  # noqa: BLE001
                continue
            feats["gas_change"] = float(r["gas_change"])
            feats["horizon_weeks"] = float(r["horizon_weeks"])
            records.append(feats)
        if not records:
            return None
        return pd.DataFrame.from_records(records)

    def _maybe_retrain(self) -> None:
        """Weekly retrain on configured day-of-week + hour (ET).

        NBA seasons are long (Oct-Apr regular + 2-mo playoffs). A weekly
        retrain keeps recent form features fresh; daily would be wasteful.
        """
        now_et = datetime.now(ET)
        if (now_et.weekday() == self.cfg.model.retrain_day_of_week
                and now_et.hour == self.cfg.model.retrain_hour_et
                and self._model is not None
                and self._model.train_end_date is not None
                and self._model.train_end_date.date() < (now_et.date() - timedelta(days=5))):
            log.info("scheduled retrain triggered at %s ET", now_et.isoformat())
            self._train_and_save()

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #

    def _ensure_panel(self) -> Optional[pd.DataFrame]:
        """Return a recent-enough cached panel, refetching once an hour."""
        now = datetime.now(timezone.utc)
        if (self._panel_cache is not None and self._panel_cache_at is not None
                and (now - self._panel_cache_at).total_seconds() < 3600):
            return self._panel_cache
        try:
            seasons = self.cfg.run.training_seasons or [
                "2024", "2025", _current_wnba_season(),
            ]
            # For inference we only need the most recent season(s) so
            # rolling features have data; not the full training corpus.
            recent = seasons[-2:] if len(seasons) >= 2 else seasons
            panel = fetch_multi_season_panel(recent)
            self._panel_cache = panel
            self._panel_cache_at = now
            return panel
        except Exception as exc:  # noqa: BLE001
            log.error("panel refresh failed: %s", exc)
            return self._panel_cache

    def _build_distribution(self, market, injuries_by_team: Optional[dict] = None
                             ) -> Optional[Tuple[GasDistribution, float]]:
        """Compute the model's P(team_being_asked wins) for one market.

        ``injuries_by_team`` should be passed in from the tick (fetched
        once and reused across all markets). If None, fall back to a
        per-call fetch (used by hedge re-evaluation, where it doesn't
        matter that we miss a per-tick optimization).

        Returns (GasDistribution, model_prob_yes) or None if the matchup
        couldn't be scored.
        """
        if self._model is None:
            return None
        panel = self._ensure_panel()
        if panel is None or panel.empty:
            return None
        # Build / reuse the per-tick state cache. This pre-computes the
        # heavy enrich+roll+ELO once and lets each per-market scoring
        # call do an O(1) lookup. Without this, ticking with N markets
        # is O(N × panel_size) and hits multi-minute tick latency.
        now = datetime.now(timezone.utc)
        if (self._states_cache is None or self._states_cache_at is None
                or (now - self._states_cache_at).total_seconds() > 1800
                or self._states_cache_at < (self._panel_cache_at or now)):
            self._states_cache = precompute_team_states(panel)
            self._states_cache_at = now
            log.info("precomputed team states for %d teams",
                     len(self._states_cache.team_latest))
        season = _current_nba_season()
        row = build_inference_row_fast(
            self._states_cache,
            home_tri=market.home_tricode,
            away_tri=market.away_tricode,
            game_date=pd.Timestamp(market.game_date or datetime.now(timezone.utc)),
            season=season,
        )
        if row is None:
            log.debug("could not build inference row for %s vs %s",
                      market.home_tricode, market.away_tricode)
            return None
        try:
            current_anchor = float(row["HOME_ELO_PRE"].iloc[0])
        except Exception:  # noqa: BLE001
            current_anchor = 1500.0
        if current_anchor != current_anchor:
            current_anchor = 1500.0
        dist = self._model.predict_distribution(current_anchor, row)
        # Posterior injury nudge.
        try:
            inj = (injuries_by_team if injuries_by_team is not None
                    else fetch_espn_injuries())
            home_imp, away_imp = compute_injury_impact(
                market.home_tricode, market.away_tricode, inj)
            # Each unit of impact differential nudges the home win
            # probability by ~3pp (calibrated against literature: a
            # single All-NBA out is ≈ 4-6 point line move ≈ 8-10pp).
            # Conservative weight to avoid double-counting season-level
            # roster changes already in the rolling stats.
            pp_per_unit = 0.03
            adj = (away_imp - home_imp) * pp_per_unit
            dist.home_win_prob = float(min(0.99, max(0.01, dist.home_win_prob + adj)))
            dist.prob_up = dist.home_win_prob
        except Exception as exc:  # noqa: BLE001
            log.debug("injury adjustment failed for %s/%s: %s",
                      market.home_tricode, market.away_tricode, exc)
        # Map to YES probability for THIS market's team.
        if market.team_being_asked == market.home_tricode:
            p_yes = dist.home_win_prob
        elif market.team_being_asked == market.away_tricode:
            p_yes = 1.0 - dist.home_win_prob
        else:
            return None
        return dist, float(p_yes)

    # ------------------------------------------------------------------ #
    # One tick
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        self.reporter.start_tick(market_hours=self._is_market_hours())
        self._ensure_model()
        self._maybe_retrain()
        assert self._model is not None

        markets = discover_wnba_markets(
            self.client,
            series_prefixes=self.cfg.run.market_series_prefixes,
            max_markets=self.cfg.run.max_markets_per_poll,
        )

        # Snapshot the latest model state once per tick (one row per
        # tick is enough — the underlying anchor is per-game so we use
        # a fixed sentinel of 1500 here; per-market dists are recorded
        # in market_views below).
        self.simulator.record_model_snapshot(
            current_gas_price=1500.0,
            median_change=0.0,
            median_price=1500.0,
            prob_up=0.5,
            quantile_05=0.0, quantile_50=0.0, quantile_95=0.0,
            residual_std=0.0,
            feature_count=len(self._model.feature_columns),
            classifier_accuracy=self._model.metrics.get(
                "calibrated_classifier_accuracy", 0.0),
            training_precision=self._model.metrics.get("training_precision"),
            training_recall=self._model.metrics.get("training_recall"),
            training_f1=self._model.metrics.get("training_f1"),
            training_roc_auc=self._model.metrics.get("training_roc_auc"),
            training_brier=self._model.metrics.get("training_brier"),
            rows_train=self._model.metrics.get("n_train"),
            rows_test=self._model.metrics.get("n_test"),
        )

        # Fetch injuries ONCE per tick — caching at the data-source
        # layer is TTL-based, but we still don't want N file reads per
        # tick when we can do one. compute_injury_impact does a dict
        # lookup, so this is the only call needed per tick.
        try:
            injuries_by_team = fetch_espn_injuries()
        except Exception as exc:  # noqa: BLE001
            log.debug("injury fetch failed at tick start: %s", exc)
            injuries_by_team = {}

        skip_counts: Counter = Counter()
        verdicts: list[tuple] = []
        # Live scoreboard for the in-game adjustment layer. One ESPN call
        # per tick, cached at the data-source layer; keyed by (home, away)
        # for fast lookup inside the per-market loop. Best-effort —
        # signals just degrade to "pre-game only" when ESPN is down.
        try:
            scoreboard_events = fetch_espn_scoreboard()
        except Exception as exc:  # noqa: BLE001
            log.debug("scoreboard fetch failed at tick start: %s", exc)
            scoreboard_events = []
        scoreboard_by_pair = {
            (ev["home_tricode"], ev["away_tricode"]): ev
            for ev in scoreboard_events
        }
        # Pinnacle (sharp) devigged probs — same one-shot pull as tennis's
        # export_watchlist. Cached for 5 min in kalshi_sdk.pinnacle so the
        # cost stays under quota. Empty dict when THE_ODDS_API_KEY isn't
        # set — every downstream caller tolerates the missing signal.
        try:
            pinnacle_by_pair = fetch_pinnacle_probs_by_pair()
        except Exception as exc:  # noqa: BLE001
            log.debug("pinnacle fetch failed at tick start: %s", exc)
            pinnacle_by_pair = {}
        # Track previous-tick market YES prices per ticker so the live
        # adjustment can flag market overreactions (big move on price
        # without a corresponding move in the model's adjustment).
        if not hasattr(self, "_prev_market_yes_by_ticker"):
            self._prev_market_yes_by_ticker: dict[str, float] = {}
        for m in markets:
            built = self._build_distribution(m, injuries_by_team=injuries_by_team)
            if built is None:
                skip_counts["could_not_score"] += 1
                continue
            dist, p_yes = built
            ob = fetch_orderbook(self.client, m)
            ok = False
            why = "no_orderbook"
            if ob is not None:
                ok, why = validate_market(m, ob, ob.yes_best_ask(),
                                           self.cfg.validators,
                                           model_median_price=None)
            if not ok:
                skip_counts[why.split(" ")[0]] += 1
            # Pinnacle YES-side prob for this market. WNBA tickers ask
            # about a specific team (team_being_asked); we look up the
            # devigged Pinnacle prob for THAT team and stamp it on the
            # market_view so the sport adapter can surface it and the
            # BUY gate can use it as the tennis-style reference.
            pinnacle_yes_prob: float | None = None
            pinn_entry = pinnacle_by_pair.get(
                (m.away_tricode, m.home_tricode)
            ) or pinnacle_by_pair.get(
                (m.home_tricode, m.away_tricode)
            )
            if pinn_entry and getattr(m, "team_being_asked", ""):
                v = pinn_entry.get(m.team_being_asked)
                if v is not None:
                    try:
                        pinnacle_yes_prob = float(v)
                    except (TypeError, ValueError):
                        pinnacle_yes_prob = None
            verdict, reason, edge = self._record_view(
                m, ob, dist, p_yes,
                validator_passed=ok, validator_reason=why,
                pinnacle_yes_prob=pinnacle_yes_prob,
            )
            verdicts.append((m, ob, dist, verdict, reason, edge))

            # ── Tennis-style signal log (additive, doesn't change trading) ──
            # Compute the live-adjusted probability + signal label for
            # this market and append to data/signals.jsonl. The dashboard
            # reads the latest line per ticker. If anything fails here
            # we just skip the row — production trading already happened
            # above off the existing decision flow.
            try:
                ev = scoreboard_by_pair.get(
                    (m.home_tricode, m.away_tricode))
                pre_game_p_home = float(getattr(dist, "home_win_prob", 0.5))
                yes_ask = ob.yes_best_ask() if ob is not None else None
                market_yes_p = (yes_ask / 100.0) if yes_ask is not None else None
                # Map YES prob back to home: p_yes(home if YES asked-on-home)
                # = market_yes_p; else 1 - market_yes_p.
                if (market_yes_p is not None and m.team_being_asked
                        and m.team_being_asked == m.away_tricode):
                    market_p_home = 1.0 - market_yes_p
                else:
                    market_p_home = market_yes_p
                prev_home = self._prev_market_yes_by_ticker.get(m.ticker)
                state = (state_from_espn_event(
                            ev,
                            market_prob_home_curr=market_p_home,
                            market_prob_home_prev=prev_home)
                          if ev is not None else
                          LiveGameState(home_tricode=m.home_tricode,
                                         away_tricode=m.away_tricode,
                                         status="scheduled"))
                la = live_adjust(pre_game_p_home, state)
                # Compute the YES side's live prob from the home prob.
                if m.team_being_asked == m.home_tricode:
                    live_yes_p = la.live_prob_home
                else:
                    live_yes_p = 1.0 - la.live_prob_home
                # Injury flag — we already pulled injuries above; use a
                # conservative threshold of "any starter listed Out" on
                # either team.
                injury_flag = bool(la.rules_fired and any(
                    "ruled out" in r for r in la.rules_fired
                ))
                sig = label_match(
                    live_yes_p, market_yes_p,
                    volatility=la.volatility_score,
                    injury_flag=injury_flag,
                    market_overreaction=la.market_overreaction,
                    rules_fired=la.rules_fired,
                )
                # Persist current market price so next tick can detect moves.
                if market_p_home is not None:
                    self._prev_market_yes_by_ticker[m.ticker] = market_p_home
                self._append_signal({
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "ticker": m.ticker,
                    "home_tricode": m.home_tricode,
                    "away_tricode": m.away_tricode,
                    "team_being_asked": m.team_being_asked,
                    "status": state.status,
                    "period": state.period,
                    "home_score": state.home_score,
                    "away_score": state.away_score,
                    "pre_game_prob_yes": (pre_game_p_home if m.team_being_asked
                                           == m.home_tricode else
                                           1 - pre_game_p_home),
                    "live_prob_yes": live_yes_p,
                    "market_prob_yes": market_yes_p,
                    "edge_yes": (live_yes_p - market_yes_p
                                  if market_yes_p is not None else None),
                    "volatility_score": la.volatility_score,
                    "confidence_score": sig.confidence_score,
                    "market_overreaction": la.market_overreaction,
                    "injury_news_flag": injury_flag,
                    "signal_label": sig.label,
                    "reason": sig.reason,
                    "rules_fired": la.rules_fired,
                })
            except Exception as exc:  # noqa: BLE001
                log.debug("signal write failed for %s: %s", m.ticker, exc)

        # Show one line per tick describing what we found.
        candidates = [t for t in verdicts
                      if t[3] in ("BUY_YES", "BUY_NO") and t[1] is not None]
        candidates.sort(key=lambda t: t[5], reverse=True)
        best = candidates[:1]

        self.reporter.market_scan(total=len(markets),
                                  valid=len(candidates),
                                  skip_counts=skip_counts)
        self.reporter.decisions_header(n_valid=len(best))

        model_acc = self._model.metrics.get(
            "calibrated_classifier_accuracy",
            self._model.metrics.get("directional_accuracy_from_median", 0.5),
        ) if self._model else 0.5

        for m, ob, dist, verdict, reason, edge in best:
            mtc_min2 = minutes_to_close(m) if m else 60
            ttc_weeks2 = max(0.005, min(1.0, mtc_min2 / (60 * 24 * 7)))
            decision = make_decision(m, ob, dist, self.cfg.edge,
                                      horizon_weeks=ttc_weeks2,
                                      model_accuracy=model_acc)
            append_decision(self.cfg.execution.decisions_log_path, decision)
            self.reporter.decision_line(m, decision)
            log.info("  ** picked highest-edge candidate: %s (edge=%+.3f among %d) **",
                     m.ticker, edge, len(candidates))

            features_json = None  # NBA features are panel-derived; persisting
                                  # the full inference row is not currently used
                                  # by the closed-bet feedback path. Future work:
                                  # serialize HOME_ELO_PRE / AWAY_ELO_PRE / a few
                                  # diff features so closed bets can re-train
                                  # with their own feature snapshot.
            elo_at_open = float(dist.current_gas_price)
            min_ask_depth = self.cfg.validators.min_depth_at_best_ask
            if verdict == "BUY_YES" and ob.yes_best_ask() is not None:
                if ob.yes_ask_depth() < min_ask_depth:
                    log.info("skip BUY_YES %s: shallow_at_ask "
                             "(%d < %d)",
                             m.ticker, ob.yes_ask_depth(), min_ask_depth)
                    continue
                self.simulator.open_position(m.ticker, "YES", ob.yes_best_ask(),
                                              decision,
                                              features_json=features_json,
                                              gas_price_at_open=elo_at_open)
            elif verdict == "BUY_NO" and ob.no_best_ask() is not None:
                if ob.no_ask_depth() < min_ask_depth:
                    log.info("skip BUY_NO %s: shallow_at_ask "
                             "(%d < %d)",
                             m.ticker, ob.no_ask_depth(), min_ask_depth)
                    continue
                self.simulator.open_position(m.ticker, "NO", ob.no_best_ask(),
                                              decision,
                                              features_json=features_json,
                                              gas_price_at_open=elo_at_open)

        # Hedge / close existing open positions.
        raw_floor = getattr(self.cfg.edge, "min_raw_model_edge", 0.0)
        for pos in self.simulator.open_positions():
            try:
                ob = self.client.get_orderbook(pos["ticker"], depth=10)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not refresh book for pid=%d: %s", pos["id"], exc)
                continue
            self.simulator.update_mark(pos["id"], ob)
            # Re-evaluate raw model edge for the side we hold.
            if raw_floor > 0:
                pos_market = next((m for m in markets if m.ticker == pos["ticker"]), None)
                if pos_market is not None:
                    built = self._build_distribution(pos_market)
                    if built is not None:
                        dist, p_yes_now = built
                        side = pos["side"]
                        ya, na = ob.yes_best_ask(), ob.no_best_ask()
                        raw_edge = None
                        if side == "YES" and ya is not None:
                            raw_edge = p_yes_now - (ya / 100.0)
                        elif side == "NO" and na is not None:
                            raw_edge = (1.0 - p_yes_now) - (na / 100.0)
                        if raw_edge is not None and raw_edge < raw_floor:
                            exit_c = int(ob.mid_price() or 0)
                            log.info("[SIM] CLOSE pid=%d raw-edge collapsed: "
                                     "%s raw_edge=%+.3f < floor=%.3f",
                                     pos["id"], side, raw_edge, raw_floor)
                            self.simulator.close_position(pos["id"], exit_c)
                            continue
            self.simulator.maybe_hedge(pos, ob)
            try:
                m = self.client.get_market(pos["ticker"]).get("market", {})
                status = m.get("status", "")
                if status not in ("open", "active"):
                    settle = m.get("yes_settled_price") or m.get("settlement_value")
                    exit_cents = int(settle) if settle is not None else int(ob.mid_price() or 0)
                    self.simulator.close_position(pos["id"], exit_cents)
            except Exception as exc:  # noqa: BLE001
                log.debug("status-check failed for %s: %s", pos["ticker"], exc)

        self.reporter.positions_summary(
            open_positions=self.simulator.open_positions(),
            exposure_cents=self.simulator.total_open_exposure_cents(),
            bets_today=self.simulator.bets_today(),
            max_open=self.cfg.risk.max_open_positions,
            max_exposure_cents=self.cfg.risk.max_total_exposure_cents,
            max_bets_per_day=self.cfg.risk.max_bets_per_day,
        )

    # ------------------------------------------------------------------ #
    # Per-market view recording
    # ------------------------------------------------------------------ #

    def _record_view(self, market, orderbook, dist, p_yes_raw,
                     validator_passed: bool,
                     validator_reason: str,
                     pinnacle_yes_prob: float | None = None,
                     ) -> tuple[str, str, float]:
        from .decision import blended_prob_yes
        if self._model:
            model_accuracy = self._model.metrics.get(
                "calibrated_classifier_accuracy",
                self._model.metrics.get("directional_accuracy_from_median", 0.0),
            )
        else:
            model_accuracy = 0.0
        mtc_min = minutes_to_close(market) if market else 60
        ttc_weeks = max(0.005, min(1.0, mtc_min / (60 * 24 * 7)))
        yes_ask_c = orderbook.yes_best_ask() if orderbook else None
        no_ask_c = orderbook.no_best_ask() if orderbook else None
        yes_bid_c = orderbook.yes_best_bid() if orderbook else None
        spread = orderbook.spread_cents() if orderbook else None
        depth = orderbook.depth_within(3) if orderbook else None
        market_yes_for_blend = ((yes_ask_c / 100.0) if yes_ask_c is not None
                                 else ((100 - no_ask_c) / 100.0
                                       if no_ask_c is not None else None))
        # blended_prob_yes calls model_prob_yes(market, dist, ...) under
        # the hood; for NBA that's already the team-aware path we wrote.
        p_yes = blended_prob_yes(market, dist,
                                  market_implied_yes=market_yes_for_blend,
                                  horizon_weeks=ttc_weeks,
                                  model_accuracy=model_accuracy)
        raw_p_yes = p_yes_raw

        # When Pinnacle is listed for this game, it becomes the reference
        # probability for edge — mirroring tennis's export_watchlist logic
        # (Pinnacle wins over the model as the "true" probability whenever
        # both are present). When Pinnacle is missing we fall back to the
        # blended posterior. This is what makes the WNBA and tennis BUY
        # gates evaluate edge identically.
        reference_p_yes = (pinnacle_yes_prob if pinnacle_yes_prob is not None
                            else p_yes)

        edge_yes = edge_no = None
        half_spread_d = ((spread or 0) / 2.0) / 100.0
        if reference_p_yes is not None and yes_ask_c is not None:
            edge_yes = reference_p_yes - (yes_ask_c / 100.0) - half_spread_d
        if reference_p_yes is not None and no_ask_c is not None:
            edge_no = (1.0 - reference_p_yes) - (no_ask_c / 100.0) - half_spread_d
        # `confidence` is the secondary conviction metric. Compute from raw
        # for the same reason as low_model_confidence above: the blended
        # posterior hugs the market price by design, so using it here just
        # measures "is the market near a coin flip", not "does the model
        # actually have an opinion." Scaled by model_accuracy so a less
        # accurate model produces lower confidence at the same raw view.
        conviction_basis = raw_p_yes if raw_p_yes is not None else p_yes
        confidence = (abs(conviction_basis - 0.5) * 2.0 * max(0.5, min(1.0, model_accuracy))
                       if conviction_basis is not None else 0.0)

        verdict = "SKIP"
        reason = "no_signal"
        if p_yes is None:
            reason = "model_could_not_evaluate"
        elif model_accuracy < self.cfg.edge.min_model_accuracy:
            reason = (f"model_accuracy_too_low "
                      f"({model_accuracy:.2f}<{self.cfg.edge.min_model_accuracy:.2f})")
        else:
            confidence_low = self.cfg.edge.min_model_confidence
            confidence_high = 1.0 - confidence_low
            # Confidence gate tests the RAW model conviction. The blended
            # posterior is by construction near the market price (skill ×
            # raw + (1-skill) × market), so testing the blended would
            # essentially reject any market priced 45-55¢ regardless of
            # whether the raw model has a real opinion. Raw answers the
            # only question that matters: does the model itself disagree
            # enough with a coin flip to be worth listening to.
            conviction_p = raw_p_yes if raw_p_yes is not None else p_yes
            if confidence_low <= conviction_p <= confidence_high:
                reason = f"low_model_confidence ({conviction_p:.2f})"
            elif confidence < self.cfg.edge.min_confidence:
                reason = (f"confidence_too_low "
                          f"({confidence:.2f}<{self.cfg.edge.min_confidence:.2f})")
            elif not validator_passed:
                verdict = "WATCH"
                reason = validator_reason
            else:
                ev_floor = self.cfg.edge.min_ev_per_contract
                be_floor = self.cfg.edge.min_prob_edge_over_breakeven
                raw_floor = getattr(self.cfg.edge, "min_raw_model_edge", 0.0)
                raw_yes_edge = (raw_p_yes - (yes_ask_c / 100.0)
                                 if (raw_p_yes is not None and yes_ask_c is not None)
                                 else None)
                raw_no_edge = ((1.0 - raw_p_yes) - (no_ask_c / 100.0)
                                if (raw_p_yes is not None and no_ask_c is not None)
                                else None)
                # Breakeven cushion is measured against the same
                # reference probability as `edge_yes/edge_no` — Pinnacle
                # when present, blended posterior otherwise. This keeps
                # tennis / WNBA in lockstep.
                yes_qual = (edge_yes is not None and edge_yes >= ev_floor
                             and yes_ask_c is not None
                             and reference_p_yes is not None
                             and (reference_p_yes - yes_ask_c / 100.0) >= be_floor
                             and raw_yes_edge is not None
                             and raw_yes_edge >= raw_floor)
                no_qual = (edge_no is not None and edge_no >= ev_floor
                             and no_ask_c is not None
                             and reference_p_yes is not None
                             and ((1.0 - reference_p_yes) - no_ask_c / 100.0) >= be_floor
                             and raw_no_edge is not None
                             and raw_no_edge >= raw_floor)
                # Hard cap on entry price.
                max_entry = self.cfg.edge.max_entry_price_cents / 100.0
                if yes_qual and yes_ask_c is not None and yes_ask_c / 100.0 > max_entry:
                    yes_qual = False
                if no_qual and no_ask_c is not None and no_ask_c / 100.0 > max_entry:
                    no_qual = False
                if yes_qual or no_qual:
                    verdict = ("BUY_YES" if (yes_qual and
                                (not no_qual or (edge_yes or 0) >= (edge_no or 0)))
                                else "BUY_NO")
                    reason = "edge_met"
                else:
                    reason = "insufficient_edge"

        title_for_view = (
            f"{market.team_being_asked} beats "
            f"{market.away_tricode if market.team_being_asked == market.home_tricode else market.home_tricode}"
        ) if getattr(market, "team_being_asked", "") else (market.title or "")
        self.simulator.record_market_view(
            ticker=market.ticker,
            title=title_for_view,
            direction=market.direction,
            strike_low=market.strike_low,
            strike_high=market.strike_high,
            minutes_to_close=minutes_to_close(market) if market else None,
            model_prob_yes=p_yes,
            yes_ask_cents=yes_ask_c, no_ask_cents=no_ask_c,
            yes_bid_cents=yes_bid_c, spread_cents=spread, book_depth=depth,
            edge_yes=edge_yes, edge_no=edge_no,
            bot_verdict=verdict, rejection_reason=reason,
            rules_primary=getattr(market, "rules_primary", "") or "",
            rules_secondary=getattr(market, "rules_secondary", "") or "",
            event_title=getattr(market, "event_title", "") or "",
            event_sub_title=getattr(market, "event_sub_title", "") or "",
            volume=int(getattr(market, "volume", 0) or 0),
            open_interest=int(getattr(market, "open_interest", 0) or 0),
            yes_ask_depth=(orderbook.yes_ask_depth() if orderbook else None),
            no_ask_depth=(orderbook.no_ask_depth() if orderbook else None),
            raw_model_prob_yes=raw_p_yes,
            pinnacle_yes_prob=pinnacle_yes_prob,
        )
        if verdict == "BUY_YES":
            chosen_edge = edge_yes or 0.0
        elif verdict == "BUY_NO":
            chosen_edge = edge_no or 0.0
        else:
            chosen_edge = 0.0
        return verdict, reason, float(chosen_edge)

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #

    def _append_signal(self, row: dict) -> None:
        """Append one ``data/signals.jsonl`` row.

        The dashboard reads the latest line per ticker to render the
        tennis-style signal label / live prob / edge alongside the
        existing trading verdict. Appending instead of rewriting keeps
        a complete audit trail — useful for backtesting which signal
        labels actually print money.
        """
        try:
            path = Path("data/signals.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.debug("could not append signal row: %s", exc)

    def _is_market_hours(self) -> bool:
        now_et = datetime.now(ET).time()
        start = dtime.fromisoformat(self.cfg.run.market_hours_start)
        end = dtime.fromisoformat(self.cfg.run.market_hours_end)
        return start <= now_et <= end

    def run(self) -> None:
        while True:
            tick_started = datetime.now(timezone.utc)
            try:
                self.tick()
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                log.exception("tick failed")
            mh = self._is_market_hours()
            sleep_for = (self.cfg.run.poll_interval_seconds_active
                          if mh else self.cfg.run.poll_interval_seconds_idle)
            elapsed = (datetime.now(timezone.utc) - tick_started).total_seconds()
            sleep_for = max(5, sleep_for - int(elapsed))
            self.reporter.end_tick(sleep_for=sleep_for, market_hours=mh)
            time.sleep(sleep_for)
