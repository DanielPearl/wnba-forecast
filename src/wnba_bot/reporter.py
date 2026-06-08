"""Tick-level structured logging.

Every tick prints a four-stage report so you can see exactly what the bot
is doing on a remote box (journalctl, tmux, etc). The format is plain
text — no ANSI — so it's readable in journalctl and over an SSH pipe.

Usage:
    reporter = TickReporter(log)
    reporter.start_tick(market_hours=True)
    reporter.model_state(dist, feature_date, n_features)
    reporter.market_scan(total, valid, skip_counts)
    for market, decision in pairs:
        reporter.decision_line(market, decision)
    reporter.positions_summary(open_positions, exposure_c, bets_today,
                               max_open, max_exposure_c, max_bets)
    reporter.end_tick(sleep_for, market_hours)
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, List

from .decision import Decision
from .market_scanner import GasMarket
from .model import GasDistribution

DIVIDER = "=" * 78
THIN = "-" * 78


class TickReporter:
    def __init__(self, log: logging.Logger):
        self.log = log
        self.tick_num = 0
        self.tick_started: datetime | None = None

    # ---------- header / footer ---------- #

    def start_tick(self, market_hours: bool) -> None:
        self.tick_num += 1
        self.tick_started = datetime.now(timezone.utc)
        self.log.info("")
        self.log.info(DIVIDER)
        self.log.info("  TICK #%d @ %s  (market_hours=%s)",
                      self.tick_num,
                      self.tick_started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      market_hours)
        self.log.info(DIVIDER)

    def end_tick(self, sleep_for: int, market_hours: bool) -> None:
        elapsed = "?"
        if self.tick_started is not None:
            elapsed = f"{(datetime.now(timezone.utc) - self.tick_started).total_seconds():.1f}s"
        self.log.info("")
        self.log.info("tick complete in %s -- sleeping %ds (market_hours=%s)",
                      elapsed, sleep_for, market_hours)
        self.log.info(DIVIDER)

    # ---------- stage 1+2: model ---------- #

    def model_state(
        self,
        dist: GasDistribution,
        feature_date,
        n_features: int,
    ) -> None:
        self.log.info("[1/4] FEATURE REFRESH")
        self.log.info("  last realized MoM       : %+.3fpp  (month of %s)",
                      dist.current_gas_price,
                      feature_date.date() if hasattr(feature_date, "date") else feature_date)
        self.log.info("  features computed       : %d", n_features)
        self.log.info("")
        self.log.info("[2/4] MODEL SCORING")
        self.log.info("  predicted next-month MoM: %+.3fpp  (delta vs last: %+.3fpp)",
                      dist.median_price, dist.median_change)
        self.log.info("  prob(MoM up vs last)    : %.2f", dist.prob_up)
        q = dist.change_quantiles
        if 0.05 in q and 0.5 in q and 0.95 in q:
            self.log.info("  quantile 5 / 50 / 95    : %+.3f / %+.3f / %+.3f  (residual std %.4fpp)",
                          q[0.05], q[0.5], q[0.95], dist.residual_std)

    # ---------- stage 3: market scan ---------- #

    def market_scan(self, total: int, valid: int, skip_counts: Counter) -> None:
        self.log.info("")
        self.log.info("[3/4] MARKET SCAN")
        self.log.info("  discovered     : %d", total)
        self.log.info("  passed gates   : %d", valid)
        for reason, count in skip_counts.most_common():
            self.log.info("  skipped: %-28s x%d", reason, count)

    # ---------- stage 4: decisions ---------- #

    def decisions_header(self, n_valid: int) -> None:
        self.log.info("")
        self.log.info("[4/4] DECISIONS  (%d valid markets)", n_valid)
        if n_valid == 0:
            self.log.info("  (nothing scored this tick)")

    def decision_line(self, market: GasMarket, decision: Decision) -> None:
        # NBA: direction == "team_wins" — show "{TEAM} beats {OPP}".
        if getattr(market, "direction", "") == "team_wins":
            asked = getattr(market, "team_being_asked", "")
            home = getattr(market, "home_tricode", "")
            away = getattr(market, "away_tricode", "")
            opp = away if asked == home else home
            qstr = f"{asked} beats {opp}"
        elif market.direction == "between" and market.strike_high is not None:
            qstr = "%.2fpp-%.2fpp" % (market.strike_low or 0, market.strike_high)
        elif market.strike_low is not None:
            qstr = "%s %.2fpp" % (market.direction, market.strike_low)
        else:
            qstr = market.direction

        ticker = (market.ticker[:36] + "..") if len(market.ticker) > 38 else market.ticker

        if decision.side in ("YES", "NO"):
            edge = decision.edge_yes if decision.side == "YES" else decision.edge_no
            ask = decision.yes_ask_dollars if decision.side == "YES" else decision.no_ask_dollars
            self.log.info(
                "  >>> %-38s %-18s model=%.2f ask=%.2f edge=%+.3f  ->  BUY %s @ %dc",
                ticker, qstr,
                decision.model_prob_yes,
                ask if ask is not None else 0.0,
                edge if edge is not None else 0.0,
                decision.side,
                int((ask or 0.0) * 100),
            )
        else:
            # Decision rejected — show why concisely
            ask_str = f"ask={decision.yes_ask_dollars:.2f}" if decision.yes_ask_dollars else "ask=?"
            model_str = (f"model={decision.model_prob_yes:.2f}"
                         if decision.model_prob_yes == decision.model_prob_yes  # not nan
                         else "model=?")
            self.log.info("      %-38s %-18s %s %s  (%s)",
                          ticker, qstr, model_str, ask_str, decision.reason)

    # ---------- positions summary ---------- #

    def positions_summary(
        self,
        open_positions: List,
        exposure_cents: int,
        bets_today: int,
        max_open: int,
        max_exposure_cents: int,
        max_bets_per_day: int,
    ) -> None:
        self.log.info("")
        self.log.info("[POSITIONS]  open=%d/%d  exposure=$%.2f/$%.2f  bets_today=%d/%d",
                      len(open_positions), max_open,
                      exposure_cents / 100, max_exposure_cents / 100,
                      bets_today, max_bets_per_day)
        if not open_positions:
            self.log.info("  (no open positions)")
            return
        for pos in open_positions:
            entry = int(pos["entry_price_cents"])
            contracts = int(pos["contracts"])
            hedge_marker = "  [HEDGED]" if pos["hedge_id"] else ""
            tk = (pos["ticker"][:38] + "..") if len(pos["ticker"]) > 40 else pos["ticker"]
            self.log.info("  pid=%-4d  %-40s  %-3s  entry=%dc x%d%s",
                          pos["id"], tk, pos["side"], entry, contracts, hedge_marker)


# --------------------------------------------------------------------------- #
# Startup banner
# --------------------------------------------------------------------------- #

def startup_report(log: logging.Logger, model_metrics: dict | None,
                   train_end_date, n_features: int) -> None:
    """Print model state at boot so it's clear what's loaded."""
    log.info("")
    log.info(DIVIDER)
    log.info("  MODEL LOADED")
    log.info(DIVIDER)
    if train_end_date is not None:
        log.info("  trained through        : %s",
                 train_end_date.date() if hasattr(train_end_date, "date") else train_end_date)
    log.info("  feature count          : %d", n_features)
    if model_metrics:
        for k, v in model_metrics.items():
            if isinstance(v, float):
                log.info("  %-22s : %.4f", k, v)
            else:
                log.info("  %-22s : %s", k, v)
    log.info(DIVIDER)
    log.info("")
