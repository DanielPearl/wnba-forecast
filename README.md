# Kalshi WNBA forecast bot

Predicts WNBA single-game outcomes and trades Kalshi `KXWNBAGAME`
moneyline markets when the model's view differs from Kalshi by enough
to clear the EV / raw-edge / liquidity gates. Paper-trades by default
(`execution.dry_run: true`).

Built as a sibling of the NBA forecast bot — same EV-first decision
engine, simulator, and dashboard snapshot schema — but with a WNBA data
layer and (importantly) **no API key or vendor SDK**: everything is
sourced from ESPN's public JSON endpoints.

## Do I need an API key?

**No.** The historical training data, the live schedule, and the
injury feed all come from ESPN's unauthenticated endpoints:

| Need | Source | Auth |
|---|---|---|
| Historical box scores (training) | ESPN core `events` + `summary` | none |
| Live schedule / scores | ESPN `scoreboard` | none |
| Injuries | ESPN WNBA injuries page | none |
| (optional) Vegas moneyline | The Odds API | free key, optional |

`stats.wnba.com` (the WNBA mirror of `nba_api`) is unreliable / blocked
from cloud hosts, so we don't use it. The only credential required is
the **Kalshi** API key the rest of the bot suite already uses, for
order/orderbook access.

## Layout

```
WNBA Forecast/
├── README.md                  # this file (also the model card)
├── requirements.txt
├── run.py                     # entry: --train | --compare | --once | (default) loop
├── config/config.yaml
├── data/
│   ├── wnba_api_cache/        # one CSV per season (ESPN box scores), fast re-train
│   ├── espn_cache/            # 60-second scoreboard + 10-min injuries
│   ├── sim.db
│   ├── model.pkl
│   ├── model_comparison.json  # multi-model bake-off (for the dashboard card)
│   └── holdout_predictions.csv
└── src/wnba_bot/
    ├── data_sources.py        # ESPN box scores + scoreboard + injuries (NO key)
    ├── features.py            # ELO + Four Factors + rest/B2B + H2H + injury impact
    ├── model.py               # calibrated GBT ensemble + logistic meta + calibration
    ├── compare_models.py      # NEW: several model families, head-to-head bake-off
    ├── market_scanner.py      # parses KXWNBAGAME-YYMMMDD{AWAY}{HOME}-{TEAM} tickers
    ├── decision.py            # EV-first decision engine (team-aware YES probability)
    ├── simulator.py           # SQLite paper-trading + close-on-hedge
    ├── validators.py          # pre-trade validators
    ├── live_adjustment.py     # in-game probability adjustment
    ├── signals.py             # tennis-style signal labeling
    ├── main.py                # tick loop: discover → score → trade → hedge
    └── reporter.py / client.py / config.py / logging_setup.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cat > .env <<EOF
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/path/to/your/kalshi_api_key.pem
THE_ODDS_API_KEY=optional-the-odds-api-key
EOF

python run.py --compare     # train several models, print the bake-off, exit
python run.py --train        # train + save the production model (~3-8 min cold)
python run.py --once         # one-shot tick (smoke test)
python run.py                # main loop
```

First `--train`/`--compare` pulls ~5-6 seasons of ESPN box scores
(~1,300 games) and caches one CSV per season; subsequent runs only
re-fetch the in-progress season.

# Model card

## Target

**Binary classifier**: P(home team wins). Each Kalshi
`KXWNBAGAME-...-{TEAM}` market is one team's YES — the bot maps
`home_win_prob` to the YES probability for whichever side that market
asks about.

## Several models — the bake-off (`run.py --compare`)

"Which model is best?" is answered empirically. `compare_models.py`
trains these families on the **same chronological split** (most-recent
season held out) and ranks them by **log loss** (a proper score — a
betting edge comes from calibration, not raw accuracy):

| Model | Role |
|---|---|
| `home_baseline` | Always-home sanity floor |
| `elo_logistic` | Logistic regression on `ELO_DIFF` only |
| `logistic_l2` | L2 logistic on the full feature set |
| `random_forest` | Bagged trees |
| `hist_gbt` | Single HistGradientBoosting classifier |
| `calibrated_ensemble` | **The shipped production model** |

The full table is written to `data/model_comparison.{csv,json}`; the
dashboard's model card renders it. The shipped bot uses
`calibrated_ensemble` (below); the bake-off exists to keep us honest
about whether the heavy model actually beats the simple ones.

## Production model architecture

Walk-forward feature selection → calibrated GBT ensemble + logistic-
regression meta-blend.

| Component | Role |
|---|---|
| `HistGradientBoostingClassifier` × 5 seeds, `class_weight="balanced"` | Non-linear interactions (rest × home × ELO, etc.) |
| Holdout `CalibratedClassifierCV(method="sigmoid")` | Well-calibrated probabilities so 60¢ on Kalshi compares fairly to model-60% |
| `LogisticRegressionCV` meta (20% blend) | Linear stabilizer; reduces overfit on the ~1.3K-game panel |
| Walk-forward `permutation_importance` (5 splits) | Selects ~20 stable + uncorrelated features from ~150 raw |

## Features (5 families)

1. **ELO rating** — 538-style with margin-of-victory bump and
   inter-season regression. `ELO_DIFF` = home − away + home-court adv.
2. **Rolling Four Factors** (last 5 / 10 / 20 games) — eFG%, TOV%,
   OREB%, FT/FGA, plus offensive/defensive/net rating, pace, 3PT
   volume. `DIFF_*` (home − away) variants usually survive selection.
3. **Schedule context** — days rest, back-to-back, games into season.
4. **Head-to-head** — rolling home-perspective record this season.
5. **Injury impact** (live, posterior-only) — ESPN injury report
   weighted count, applied as a small inference-time nudge. Training
   metrics are injury-blind.

## Training data

- **ESPN box scores**, regular season, 2021 → current (~240 games/season,
  ~1,300 games total). Cached per season; re-training only re-fetches the
  active season. **No API key.**
- WNBA team codes are normalized to Kalshi's convention (notably ESPN's
  `CON` → Kalshi's `CONN`).

## Out-of-sample evaluation

Most-recent season (~240 games) held out chronologically. The model
card reports Accuracy, Log loss, Brier, F1/Precision/Recall, ROC AUC.

## Decision engine

Same EV-first architecture as the other bots:

1. Score the matchup → P(home wins) → P(YES) for the team asked.
2. Bayesian-blend with Kalshi's market-implied probability
   (weight = skill = 2·accuracy − 1).
3. Compute EV per side after half-spread.
4. Gates: raw model edge ≥ 8pp, EV floor, breakeven cushion, max entry
   price ≤ 70¢, liquidity validators.
5. Trade the single highest-edge qualifying market per tick (paper).

## Risk + simulation

Paper-trading. $1 per bet, `max_open_positions: 5`,
`max_total_exposure_cents: 500` ($5 cap), `max_bets_per_day: 20`,
30-min cooldown per ticker.

## What's deferred to phase 2

- **Player-level features** (star on/off, defense-vs-position) — easy to
  overfit on a ~1.3K-game panel; best added with closed-bet feedback.
- **Live Vegas line as a feature** — cached in
  `data_sources.fetch_odds_api_moneylines` but not yet plumbed into the
  inference row.
- **Play-by-play in-match model** — the NBA bot has one; the WNBA PBP
  feed work is deferred.
