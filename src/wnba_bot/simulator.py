"""Paper-trading simulator with hedging.

Persists positions and trades to SQLite (`data/sim.db`). The bot never
sends a real order — it just records what it WOULD have done.

Schema:
  positions(id, ticker, side, entry_price_cents, contracts, opened_at,
            status, hedge_id, exit_price_cents, exited_at, realized_pnl_cents,
            decision_json)
  trades(id, position_id, ticker, side, action, price_cents, contracts,
         created_at, kind)
    kind in ("entry", "hedge", "exit")

Hedging:
  When an open position's current opposite-side price moves
   - up by `profit_lock_cents` from entry (we're winning), buy the OTHER side
     at current ask for `hedge_size_fraction * original_contracts`.
   - down by `stop_loss_cents` (we're losing), same hedge structure.
  A position can only be hedged once. After the hedge fills, the original
  position stays open until market close — the hedge offsets risk.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .client import Orderbook
from .config import HedgeCfg, RiskCfg
from .decision import Decision

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL,
    hedge_id INTEGER,
    exit_price_cents INTEGER,
    exited_at TEXT,
    realized_pnl_cents INTEGER,
    decision_json TEXT,
    -- The model's anchor gas price ($/gal, EIA series) at the moment the
    -- position closed. Lets the dashboard show "this bet resolved when
    -- gas was $4.12" alongside the entry/exit prices. Note: this is
    -- EIA, not AAA (which is what Kalshi actually settles on), so it's
    -- accurate to within the EIA-AAA basis (~1-3¢).
    gas_price_at_close REAL,
    -- Snapshotted at OPEN time: the feature row the model used to make
    -- this bet, plus the model's anchor gas price at that moment. These
    -- feed back into training once the contract resolves so the model
    -- learns from real Kalshi outcomes, not just FRED retrospective.
    features_json TEXT,
    gas_price_at_open REAL
);

-- Closed-bet training feedback. One row per closed position; the model
-- training step pulls these in as additional (features, gas_change) rows
-- with sample_weight scaled by horizon_weeks (so a 2-hour bet contributes
-- ~2/168 of a weekly FRED row's weight — never dominates).
CREATE TABLE IF NOT EXISTS training_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER UNIQUE,
    captured_at TEXT NOT NULL,        -- = position.opened_at, for ordering
    features_json TEXT NOT NULL,
    gas_price_at_open REAL,
    gas_price_at_close REAL,
    gas_change REAL,                  -- = at_close - at_open
    horizon_hours REAL,               -- elapsed wall-clock between open/close
    horizon_weeks REAL,               -- = horizon_hours / 168
    direction TEXT,                   -- above/below/between
    strike_low REAL,
    strike_high REAL,
    side TEXT,                        -- YES/NO
    won INTEGER                       -- 1 if pnl > 0, 0 if pnl < 0, NULL if flat
);
CREATE INDEX IF NOT EXISTS idx_training_pairs_at ON training_pairs(captured_at);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id);

-- Live marks for open positions, refreshed each tick. Lets the dashboard
-- compute unrealized P&L without having to call Kalshi itself.
CREATE TABLE IF NOT EXISTS position_marks (
    position_id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    yes_ask_cents INTEGER,
    no_ask_cents INTEGER,
    yes_bid_cents INTEGER,
    mid_cents REAL,
    spread_cents INTEGER,
    updated_at TEXT NOT NULL
);

-- Per-tick model snapshot, so the dashboard can show what the model is
-- currently saying without re-running it.
CREATE TABLE IF NOT EXISTS model_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    current_gas_price REAL,
    median_change REAL,
    median_price REAL,
    prob_up REAL,
    quantile_05 REAL,
    quantile_50 REAL,
    quantile_95 REAL,
    residual_std REAL,
    feature_count INTEGER,
    -- Model's track record from training. Used to scale per-market confidence
    -- so a model that's right 58% of the time can never display 100% confident.
    classifier_accuracy REAL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_at ON model_snapshots(captured_at DESC);

-- One row per market per tick. Captures EVERY market the bot saw (not just
-- the ones it could trade), so the dashboard can show a watchlist:
--   "model believes 70% YES, market asks 65c, would buy 5c edge if book opens"
-- bot_verdict is one of:
--   BUY_YES, BUY_NO  - the bot actually placed a (sim) order this tick
--   WATCH            - model has a view & edge but something blocks (book empty,
--                      spread, depth, cooldown, max_open, etc.)
--   SKIP             - model has no actionable view (low confidence / no signal)
CREATE TABLE IF NOT EXISTS market_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    title TEXT,
    direction TEXT,
    strike_low REAL,
    strike_high REAL,
    minutes_to_close REAL,
    model_prob_yes REAL,
    yes_ask_cents INTEGER,
    no_ask_cents INTEGER,
    yes_bid_cents INTEGER,
    spread_cents INTEGER,
    book_depth INTEGER,
    edge_yes REAL,
    edge_no REAL,
    bot_verdict TEXT NOT NULL,
    rejection_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_views_captured ON market_views(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_views_ticker_at ON market_views(ticker, captured_at DESC);
"""


class Simulator:
    def __init__(self, db_path: str, risk: RiskCfg, hedge: HedgeCfg):
        self.db_path = db_path
        self.risk = risk
        self.hedge = hedge
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # DB plumbing
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with closing(self._conn()) as c, c:
            c.executescript(SCHEMA)
            # Idempotent ALTER TABLE migrations for columns added after the
            # initial schema was deployed. SQLite has no "ADD COLUMN IF
            # NOT EXISTS" so we introspect first.
            existing_cols = {r["name"] for r in
                             c.execute("PRAGMA table_info(positions)").fetchall()}
            if "gas_price_at_close" not in existing_cols:
                c.execute("ALTER TABLE positions "
                          "ADD COLUMN gas_price_at_close REAL")
                # Backfill: for already-closed positions, infer the gas
                # price from the model snapshot closest to (and not later
                # than) the exit time. Better than NULL, accurate to one
                # tick interval.
                c.execute(
                    "UPDATE positions SET gas_price_at_close = ("
                    "  SELECT current_gas_price FROM model_snapshots "
                    "  WHERE captured_at <= positions.exited_at "
                    "  ORDER BY captured_at DESC LIMIT 1"
                    ") "
                    "WHERE gas_price_at_close IS NULL "
                    "  AND exited_at IS NOT NULL"
                )
            if "features_json" not in existing_cols:
                c.execute("ALTER TABLE positions ADD COLUMN features_json TEXT")
            if "gas_price_at_open" not in existing_cols:
                c.execute("ALTER TABLE positions ADD COLUMN gas_price_at_open REAL")
                # Backfill open-anchor: same lookup pattern as close, but
                # using opened_at instead. Past bets get an approximation;
                # bets opened after this migration get the live value.
                c.execute(
                    "UPDATE positions SET gas_price_at_open = ("
                    "  SELECT current_gas_price FROM model_snapshots "
                    "  WHERE captured_at <= positions.opened_at "
                    "  ORDER BY captured_at DESC LIMIT 1"
                    ") "
                    "WHERE gas_price_at_open IS NULL"
                )
            # EV-first audit columns. Decision dataclass already records
            # all of these; we mirror the relevant fields onto the
            # position so the dashboard's per-bet diagnostics don't have
            # to parse decision_json on every render.
            for col_def in (
                "model_yes_prob_at_entry REAL",
                "kalshi_yes_prob_at_entry REAL",
                "break_even_probability REAL",
                "expected_ev_at_entry REAL",
                "selected_side_ev REAL",
                "gates_passed_json TEXT",
                "gates_failed_json TEXT",
                # error_type is set on close: GOOD_BET_BAD_OUTCOME, BAD_BET,
                # MODEL_OVERCONFIDENT, EXECUTION_BAD_PRICE, LOW_CONFIDENCE_TRADE,
                # or NULL when not classified (e.g. flat outcome).
                "error_type TEXT",
            ):
                col_name = col_def.split()[0]
                if col_name not in existing_cols:
                    c.execute(f"ALTER TABLE positions ADD COLUMN {col_def}")

    # ------------------------------------------------------------------ #
    # Read helpers
    # ------------------------------------------------------------------ #

    def open_positions(self, ticker: Optional[str] = None) -> List[sqlite3.Row]:
        q = "SELECT * FROM positions WHERE status = 'open'"
        args: tuple = ()
        if ticker:
            q += " AND ticker = ?"
            args = (ticker,)
        with closing(self._conn()) as c:
            return list(c.execute(q, args))

    @staticmethod
    def _event_id(ticker: str) -> str:
        """Strip the team segment from an NBA ticker so both per-team
        markets on the same game map to the same event id. Tickers
        with the standard ``KXNBAGAME-<event>-<team>`` shape collapse
        to ``KXNBAGAME-<event>``; other ticker shapes pass through
        unchanged (returns the full ticker, which means the
        event-level dedup gate is a no-op for them).
        """
        parts = ticker.split("-")
        if len(parts) >= 3 and parts[0].startswith("KXNBAGAME"):
            return "-".join(parts[:2])
        return ticker

    def open_positions_for_event(self, event_id: str) -> List[sqlite3.Row]:
        """Return every open position whose ticker belongs to the given
        event id. Used by the event-level dedup gate so the bot doesn't
        hold concurrent positions on sibling contracts of the same
        game (NO on the home-team market and YES on the away-team
        market are the same directional bet)."""
        like = event_id + "%"
        with closing(self._conn()) as c:
            return list(c.execute(
                "SELECT * FROM positions WHERE status = 'open' "
                "AND ticker LIKE ?",
                (like,),
            ))

    def total_open_exposure_cents(self) -> int:
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT COALESCE(SUM(entry_price_cents * contracts), 0) AS total "
                "FROM positions WHERE status = 'open'"
            ).fetchone()
        return int(row["total"] or 0)

    def bets_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM trades "
                "WHERE kind = 'entry' AND substr(created_at, 1, 10) = ?",
                (today,),
            ).fetchone()
        return int(row["n"] or 0)

    def last_entry_seconds_ago(self, ticker: str) -> Optional[float]:
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT created_at FROM trades "
                "WHERE ticker = ? AND kind IN ('entry', 'hedge') "
                "ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        last = datetime.fromisoformat(row["created_at"])
        return (datetime.now(timezone.utc) - last).total_seconds()

    # ------------------------------------------------------------------ #
    # Risk gates
    # ------------------------------------------------------------------ #

    def can_open_new(self, ticker: str) -> tuple[bool, str]:
        if len(self.open_positions()) >= self.risk.max_open_positions:
            return False, f"max_open_positions ({self.risk.max_open_positions})"
        if self.total_open_exposure_cents() + self.risk.bet_size_cents > self.risk.max_total_exposure_cents:
            return False, "max_total_exposure"
        if self.bets_today() >= self.risk.max_bets_per_day:
            return False, f"max_bets_per_day ({self.risk.max_bets_per_day})"
        if self.open_positions(ticker=ticker):
            return False, "already_have_open_position"
        # Event-level dedup. Each NBA game ships as two Kalshi
        # contracts — "Will <home> win?" and "Will <away> win?" — and
        # holding NO on one is the same directional bet as YES on the
        # other. Without this gate the bot can (and did) double-down
        # on the same game by entering both tickers; we'd then take
        # twice the loss if the model was wrong. Refuse the new entry
        # when a sibling ticker on the same event is already open.
        event_id = self._event_id(ticker)
        if event_id != ticker:
            for sib in self.open_positions_for_event(event_id):
                if sib["ticker"] != ticker:
                    return False, f"already_open_on_event ({sib['ticker']})"
        # One-shot-per-ticker — refuse re-entry once a position on this
        # exact ticker has closed. Stops flap-trades; see Retail Gas
        # simulator for rationale.
        with closing(self._conn()) as c:
            any_closed = c.execute(
                "SELECT 1 FROM positions WHERE ticker = ? AND status = 'closed' LIMIT 1",
                (ticker,),
            ).fetchone()
        if any_closed is not None:
            return False, "already_traded_this_ticker"
        last = self.last_entry_seconds_ago(ticker)
        if last is not None and last < self.risk.cooldown_seconds_same_market:
            return False, f"cooldown_active ({last:.0f}s)"
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def open_position(
        self,
        ticker: str,
        side: str,
        ask_cents: int,
        decision: Decision,
        features_json: Optional[str] = None,
        gas_price_at_open: Optional[float] = None,
    ) -> Optional[int]:
        ok, why = self.can_open_new(ticker)
        if not ok:
            log.info("skip open %s: %s", ticker, why)
            return None
        if ask_cents <= 0 or ask_cents >= 100:
            log.info("skip open %s: invalid ask %s", ticker, ask_cents)
            return None

        # Bet size: $1 -> floor(100c / price_cents) contracts.
        # Fixed 1 contract per simulated bet — keeps every position
        # comparable in size regardless of entry price (a 9c bet and a
        # 82c bet now both put 1 contract at risk; previously the 9c
        # bet would buy 11 contracts to spend ~$1, which made P&L
        # diagnostics non-comparable across price levels).
        contracts = 1
        now = datetime.now(timezone.utc).isoformat()

        # EV-first audit: mirror decision fields onto the position row so
        # the dashboard can render per-bet diagnostics without parsing JSON.
        # Decision stores model_yes_prob for both YES and NO bets, so for a
        # NO bet the relevant "selected side prob" is (1 - model_yes_prob).
        selected_ev = decision.selected_side_ev
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO positions("
                "  ticker, side, entry_price_cents, contracts, opened_at, "
                "  status, decision_json, features_json, gas_price_at_open, "
                "  model_yes_prob_at_entry, kalshi_yes_prob_at_entry, "
                "  break_even_probability, expected_ev_at_entry, "
                "  selected_side_ev, gates_passed_json, gates_failed_json"
                ") VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, side, ask_cents, contracts, now,
                 json.dumps(asdict(decision), default=str),
                 features_json, gas_price_at_open,
                 decision.model_yes_prob_at_entry,
                 decision.kalshi_yes_prob_at_entry,
                 decision.break_even_probability,
                 selected_ev,
                 selected_ev,
                 json.dumps(list(decision.gates_passed or [])),
                 json.dumps(list(decision.gates_failed or []))),
            )
            pid = cur.lastrowid
            c.execute(
                "INSERT INTO trades(position_id, ticker, side, action, price_cents, "
                "contracts, created_at, kind) VALUES (?, ?, ?, 'buy', ?, ?, ?, 'entry')",
                (pid, ticker, side, ask_cents, contracts, now),
            )
        log.info("[SIM] OPEN %s %s @ %dc x %d (pid=%d) — %s",
                 ticker, side, ask_cents, contracts, pid, decision.reason)
        return pid

    def maybe_hedge(self, position: sqlite3.Row, orderbook: Orderbook,
                    gas_price_at_close: Optional[float] = None) -> Optional[int]:
        """EV-aware hedge: re-evaluate the trade against the live mark
        every tick and close when the EV view says it's no longer worth
        holding.

        The previous fixed-cents triggers (+20c profit-lock / -15c
        stop-loss) hedged purely on price movement, which is the wrong
        signal — a +15c move on a sharply-improved-EV bet should NOT
        close, and a -10c move on a now-negative-EV bet SHOULD.

        Triggers (in priority order):
          1. EV inverted: model probability for our side has fallen
             below the live break-even (mark price as a probability) by
             more than the configured cushion. The trade is now a
             negative-EV bet — close it.
          2. Profit lock: live mark is high enough that the remaining
             EV (model_p - mark) is sub-threshold AND the realized
             gain so far is large. Why hold for a few cents of
             remaining edge when we've already won most of it?

        Falls back to the old fixed-cents trigger if entry-time
        diagnostics weren't captured (legacy positions).
        """
        if not self.hedge.enabled:
            return None
        if position["status"] != "open":
            return None

        side = position["side"].upper()
        entry = int(position["entry_price_cents"])
        if side == "YES":
            current = orderbook.yes_best_ask()
        else:
            current = orderbook.no_best_ask()
        if current is None:
            return None

        # Try EV-aware logic first; legacy fall-back if entry-time stats
        # are missing (older positions opened before this code shipped).
        try:
            model_p_yes = position["model_yes_prob_at_entry"]
        except (KeyError, IndexError):
            model_p_yes = None
        triggered: Optional[str] = None
        if model_p_yes is not None:
            p_selected = (float(model_p_yes) if side == "YES"
                          else 1.0 - float(model_p_yes))
            mark_be = current / 100.0      # live break-even prob
            current_ev = p_selected - mark_be
            # 1. EV inverted by more than 5pt cushion → exit.
            if current_ev <= -0.05:
                triggered = "ev_inverted"
            # 2. Most of the EV is already realized — lock it in.
            elif (current - entry) >= 20 and current_ev < 0.02:
                triggered = "ev_realized_lock"

        if triggered is None:
            # Legacy fixed-cents fallback. Kept so older positions still
            # have an exit; new positions will mostly hit the EV branches.
            delta = current - entry
            if delta >= self.hedge.profit_lock_cents:
                triggered = "profit_lock_legacy"
            elif delta <= -self.hedge.stop_loss_cents:
                triggered = "stop_loss_legacy"
        if triggered is None:
            return None

        log.info("[SIM] HEDGE-CLOSE pid=%d trigger=%s entry=%dc current=%dc -> closing",
                 position["id"], triggered, entry, current)
        self.close_position(int(position["id"]), int(current),
                             gas_price_at_close=gas_price_at_close)
        return int(position["id"])

    def update_mark(self, position_id: int, orderbook: Orderbook) -> None:
        """Record the current order-book snapshot for an open position.

        Dashboard reads from this table to show unrealized P&L without
        having to call Kalshi itself.
        """
        ya = orderbook.yes_best_ask()
        na = orderbook.no_best_ask()
        yb = orderbook.yes_best_bid()
        mid = orderbook.mid_price()
        spread = orderbook.spread_cents()
        now = datetime.now(timezone.utc).isoformat()

        with closing(self._conn()) as c, c:
            row = c.execute("SELECT ticker FROM positions WHERE id = ?",
                             (position_id,)).fetchone()
            if not row:
                return
            c.execute(
                "INSERT INTO position_marks(position_id, ticker, yes_ask_cents, "
                "no_ask_cents, yes_bid_cents, mid_cents, spread_cents, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(position_id) DO UPDATE SET "
                "yes_ask_cents=excluded.yes_ask_cents, "
                "no_ask_cents=excluded.no_ask_cents, "
                "yes_bid_cents=excluded.yes_bid_cents, "
                "mid_cents=excluded.mid_cents, "
                "spread_cents=excluded.spread_cents, "
                "updated_at=excluded.updated_at",
                (position_id, row["ticker"], ya, na, yb, mid, spread, now),
            )

    def record_market_view(
        self,
        ticker: str,
        title: str,
        direction: str,
        strike_low: float | None,
        strike_high: float | None,
        minutes_to_close: float | None,
        model_prob_yes: float | None,
        yes_ask_cents: int | None,
        no_ask_cents: int | None,
        yes_bid_cents: int | None,
        spread_cents: int | None,
        book_depth: int | None,
        edge_yes: float | None,
        edge_no: float | None,
        bot_verdict: str,
        rejection_reason: str | None,
        rules_primary: str = "",
        rules_secondary: str = "",
        event_title: str = "",
        event_sub_title: str = "",
        volume: int | None = None,
        open_interest: int | None = None,
        yes_ask_depth: int | None = None,
        no_ask_depth: int | None = None,
        raw_model_prob_yes: float | None = None,
        pinnacle_yes_prob: float | None = None,
    ) -> None:
        with closing(self._conn()) as c, c:
            for col_def in ("rules_primary TEXT", "rules_secondary TEXT",
                            "event_title TEXT", "event_sub_title TEXT",
                            "volume INTEGER", "open_interest INTEGER",
                            "yes_ask_depth INTEGER", "no_ask_depth INTEGER",
                            "raw_model_prob_yes REAL",
                            # Devigged Pinnacle prob for the YES side. Nullable
                            # (Pinnacle not always listed; THE_ODDS_API_KEY may
                            # be absent). Used by _sport_adapter to emit
                            # pinnacle_prob_a/b on the watchlist row and by the
                            # shared BUY gate as the tennis-style reference.
                            "pinnacle_yes_prob REAL"):
                try:
                    c.execute(f"ALTER TABLE market_views ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass
            c.execute(
                "INSERT INTO market_views(captured_at, ticker, title, direction, "
                "strike_low, strike_high, minutes_to_close, model_prob_yes, "
                "yes_ask_cents, no_ask_cents, yes_bid_cents, spread_cents, "
                "book_depth, edge_yes, edge_no, bot_verdict, rejection_reason, "
                "rules_primary, rules_secondary, event_title, event_sub_title, "
                "volume, open_interest, yes_ask_depth, no_ask_depth, "
                "raw_model_prob_yes, pinnacle_yes_prob) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "        ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(),
                 ticker, title, direction, strike_low, strike_high,
                 minutes_to_close, model_prob_yes,
                 yes_ask_cents, no_ask_cents, yes_bid_cents, spread_cents,
                 book_depth, edge_yes, edge_no, bot_verdict, rejection_reason,
                 rules_primary, rules_secondary, event_title, event_sub_title,
                 volume, open_interest, yes_ask_depth, no_ask_depth,
                 raw_model_prob_yes, pinnacle_yes_prob),
            )

    def recently_active_tickers(self, within_seconds: int = 86400) -> List[str]:
        """Tickers we have a recent market_view for. Used by the
        lightweight orderbook tick to know which markets to re-poll
        without doing the full discovery scan again.
        """
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT DISTINCT ticker FROM market_views "
                "WHERE captured_at >= datetime('now', ?)",
                (f'-{int(within_seconds)} seconds',),
            ).fetchall()
        return [r["ticker"] for r in rows]

    def update_market_view_prices(
        self,
        ticker: str,
        yes_ask_cents: int | None,
        no_ask_cents: int | None,
        yes_bid_cents: int | None,
        spread_cents: int | None,
        book_depth: int | None,
        yes_ask_depth: int | None,
        no_ask_depth: int | None,
        volume: int | None,
        open_interest: int | None,
    ) -> None:
        """In-place update of the latest market_views row for `ticker`.

        Used by the lightweight 30s orderbook tick to keep watchlist
        prices/volume fresh between heavy 15-min ticks. Only price-side
        fields are updated; model_prob_yes / strike / direction / etc.
        are intentionally left alone (they don't change between heavy
        ticks, and the lightweight tick can't recompute them anyway).

        Also recomputes edge_yes / edge_no from the existing
        model_prob_yes + the new prices so EV cells re-render
        correctly without waiting for the next heavy tick.
        """
        with closing(self._conn()) as c, c:
            row = c.execute(
                "SELECT id, model_prob_yes FROM market_views "
                "WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if not row:
                return
            view_id = row["id"]
            p_yes = row["model_prob_yes"]
            half_spread_d = ((spread_cents or 0) / 2.0) / 100.0
            edge_yes = (p_yes - (yes_ask_cents / 100.0) - half_spread_d
                        if (p_yes is not None and yes_ask_cents is not None)
                        else None)
            edge_no = ((1.0 - p_yes) - (no_ask_cents / 100.0) - half_spread_d
                       if (p_yes is not None and no_ask_cents is not None)
                       else None)
            c.execute(
                "UPDATE market_views SET "
                "  yes_ask_cents = ?, no_ask_cents = ?, yes_bid_cents = ?, "
                "  spread_cents = ?, book_depth = ?, "
                "  yes_ask_depth = ?, no_ask_depth = ?, "
                "  volume = ?, open_interest = ?, "
                "  edge_yes = ?, edge_no = ?, "
                "  captured_at = ? "
                "WHERE id = ?",
                (yes_ask_cents, no_ask_cents, yes_bid_cents,
                 spread_cents, book_depth,
                 yes_ask_depth, no_ask_depth,
                 volume, open_interest,
                 edge_yes, edge_no,
                 datetime.now(timezone.utc).isoformat(),
                 view_id),
            )

    def record_model_snapshot(
        self,
        current_gas_price: float,
        median_change: float,
        median_price: float,
        prob_up: float,
        quantile_05: float,
        quantile_50: float,
        quantile_95: float,
        residual_std: float,
        feature_count: int,
        classifier_accuracy: float | None,
        training_precision: float | None = None,
        training_recall: float | None = None,
        training_f1: float | None = None,
        training_roc_auc: float | None = None,
        training_brier: float | None = None,
        rows_train: int | None = None,
        rows_test: int | None = None,
    ) -> None:
        with closing(self._conn()) as c, c:
            # Schema migrations for older DBs. ALTER is a no-op if column
            # already exists (caught silently below).
            for col_def in (
                "classifier_accuracy REAL",
                "training_precision REAL",
                "training_recall REAL",
                "training_f1 REAL",
                "training_roc_auc REAL",
                "training_brier REAL",
                "rows_train INTEGER",
                "rows_test INTEGER",
            ):
                try:
                    c.execute(f"ALTER TABLE model_snapshots ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass
            c.execute(
                "INSERT INTO model_snapshots(captured_at, current_gas_price, "
                "median_change, median_price, prob_up, quantile_05, quantile_50, "
                "quantile_95, residual_std, feature_count, classifier_accuracy, "
                "training_precision, training_recall, training_f1, training_roc_auc, "
                "training_brier, rows_train, rows_test) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(),
                 current_gas_price, median_change, median_price, prob_up,
                 quantile_05, quantile_50, quantile_95, residual_std, feature_count,
                 classifier_accuracy, training_precision, training_recall,
                 training_f1, training_roc_auc, training_brier,
                 rows_train, rows_test),
            )

    def close_position(self, position_id: int, exit_price_cents: int,
                       gas_price_at_close: Optional[float] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._conn()) as c, c:
            row = c.execute(
                "SELECT * FROM positions WHERE id = ? AND status = 'open'",
                (position_id,),
            ).fetchone()
            if not row:
                return
            entry = int(row["entry_price_cents"])
            contracts = int(row["contracts"])
            pnl = (exit_price_cents - entry) * contracts
            # Error classification — answers "why did this trade end up
            # where it did?". Read entry-time stats off the row that we
            # mirrored at open. If they're missing (legacy bets) we fall
            # back to UNCLASSIFIED.
            try:
                model_p_yes = (float(row["model_yes_prob_at_entry"])
                               if row["model_yes_prob_at_entry"] is not None
                               else None)
                kalshi_p_yes = (float(row["kalshi_yes_prob_at_entry"])
                                if row["kalshi_yes_prob_at_entry"] is not None
                                else None)
                expected_ev = (float(row["expected_ev_at_entry"])
                               if row["expected_ev_at_entry"] is not None
                               else None)
                break_even = (float(row["break_even_probability"])
                              if row["break_even_probability"] is not None
                              else None)
            except (KeyError, IndexError):
                model_p_yes = kalshi_p_yes = expected_ev = break_even = None
            side = row["side"]
            error_type = _classify_error(side=side, pnl_cents=pnl,
                                          model_p_yes=model_p_yes,
                                          break_even=break_even,
                                          expected_ev=expected_ev)
            c.execute(
                "UPDATE positions SET status = 'closed', exit_price_cents = ?, "
                "exited_at = ?, realized_pnl_cents = ?, "
                "gas_price_at_close = ?, error_type = ? WHERE id = ?",
                (exit_price_cents, now, pnl, gas_price_at_close,
                 error_type, position_id),
            )
            # Feed back into training: only if we captured features at open.
            # Without features we can't form a (X, y) pair. Decode the stored
            # decision to recover strike/direction without re-querying Kalshi.
            features_json = row["features_json"]
            gas_open = row["gas_price_at_open"]
            if features_json and gas_open is not None and gas_price_at_close is not None:
                try:
                    decision = json.loads(row["decision_json"] or "{}")
                except Exception:  # noqa: BLE001
                    decision = {}
                opened_dt = datetime.fromisoformat(row["opened_at"])
                closed_dt = datetime.fromisoformat(now)
                horizon_h = max(0.001, (closed_dt - opened_dt).total_seconds() / 3600.0)
                gas_change = float(gas_price_at_close) - float(gas_open)
                won = 1 if pnl > 0 else (0 if pnl < 0 else None)
                c.execute(
                    "INSERT OR REPLACE INTO training_pairs("
                    "  position_id, captured_at, features_json, "
                    "  gas_price_at_open, gas_price_at_close, gas_change, "
                    "  horizon_hours, horizon_weeks, "
                    "  direction, strike_low, strike_high, side, won"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (position_id, row["opened_at"], features_json,
                     float(gas_open), float(gas_price_at_close), gas_change,
                     horizon_h, horizon_h / 168.0,
                     decision.get("direction"),
                     decision.get("strike_low"), decision.get("strike_high"),
                     row["side"], won),
                )
            c.execute(
                "INSERT INTO trades(position_id, ticker, side, action, price_cents, "
                "contracts, created_at, kind) VALUES (?, ?, ?, 'sell', ?, ?, ?, 'exit')",
                (position_id, row["ticker"], row["side"], exit_price_cents,
                 contracts, now),
            )
        log.info("[SIM] CLOSE pid=%d exit=%dc pnl=%dc", position_id, exit_price_cents, pnl)


# --------------------------------------------------------------------------- #
# Decision audit log
# --------------------------------------------------------------------------- #

def append_decision(path: str, decision: Decision) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(decision), default=str) + "\n")


def _classify_error(
    side: str,
    pnl_cents: int,
    model_p_yes: Optional[float],
    break_even: Optional[float],
    expected_ev: Optional[float],
) -> Optional[str]:
    """Categorize what kind of trade outcome this was.

    Categories (one wins per close, in priority order):

      BAD_BET                — entry-time EV was negative; we shouldn't
                               have taken this trade at all
      EXECUTION_BAD_PRICE    — break-even prob > 0.85, meaning we paid
                               such a high entry that even a strong
                               directional view couldn't overcome it
      LOW_CONFIDENCE_TRADE   — selected-side model prob between 50% and
                               60%; thin signal, results dominated by noise
      MODEL_OVERCONFIDENT    — model said >75% on the side we bought,
                               but we lost. Calibration miss on a strong
                               directional call.
      GOOD_BET_BAD_OUTCOME   — entry-time EV was positive but the trade
                               lost. Expected variance, no process error.
      None                   — flat outcome (pnl = 0) or insufficient
                               data to classify (e.g. legacy bets).
    """
    if pnl_cents == 0:
        return None  # flat — no win/loss, nothing to learn
    won = pnl_cents > 0
    # Map "selected side" prob: for a YES bet it's model_p_yes; for a
    # NO bet it's 1 - model_p_yes.
    if model_p_yes is None:
        return None
    p_selected = model_p_yes if side == "YES" else (1.0 - model_p_yes)

    if expected_ev is not None and expected_ev < 0:
        return "BAD_BET"
    if break_even is not None and break_even > 0.85:
        return "EXECUTION_BAD_PRICE"
    if 0.50 <= p_selected < 0.60:
        return "LOW_CONFIDENCE_TRADE"
    if not won and p_selected >= 0.75:
        return "MODEL_OVERCONFIDENT"
    if not won:
        return "GOOD_BET_BAD_OUTCOME"
    # Won and either had positive EV or we couldn't tell — no error to flag.
    return None
