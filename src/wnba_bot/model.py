"""NBA single-game outcome model.

Binary classifier targeting ``HOME_WIN``. The model produces a
calibrated P(home wins) for each upcoming game; the decision engine
then maps that to a Kalshi YES probability for whichever side the
market is asking about.

Architecture (matches the structure the other bots use, simplified
for binary single-target — there's no per-threshold grid for NBA
because each Kalshi game market resolves on a single binary outcome):

  1. **Walk-forward permutation-importance feature selection** —
     prune ~150 raw features to ~30 stable + uncorrelated ones.
  2. **HistGradientBoostingClassifier ensemble** — ~5 seeded members,
     averaged. Handles missing values and high-cardinality team
     interactions natively.
  3. **Holdout-calibrated** with sigmoid (Platt) — gives a probability
     output that maps cleanly onto Kalshi's penny prices.
  4. **ElasticNet meta-voice** on the GBT median's residuals as a
     stabilizer — a small linear blend reduces overfitting on the
     ~12K-game training set without hurting calibration.

The class names ``GasModel`` / ``GasDistribution`` are kept for
cross-bot dashboard compatibility (the simulator + market_views render
identically across bots when the snapshot schema matches).
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# Tennis-style "component" baseline: a logistic regression on JUST the
# Elo features. Bench-mark for the GBT — if the heavy ensemble doesn't
# beat this on Brier/log-loss, we shouldn't ship it.
# We previously included ``ELO_WIN_PROB_HOME`` (a sigmoid transform of
# ELO_DIFF) but it carries no extra information; pruned per the
# walk-forward importance audit. The logistic regression already
# learns the right shape on raw ELO_DIFF.
ELO_BASELINE_FEATURES = ["ELO_DIFF"]


def _full_metrics(y_true, y_prob) -> Dict[str, float]:
    """Standard probability-forecast metrics. Same eight cells the
    tennis bot reports (Accuracy / F1 / Precision / Recall / ROC AUC /
    Brier / log-loss) so the dashboard can render a per-component
    breakdown."""
    y_prob_arr = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y_pred = (y_prob_arr >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob_arr)),
        "brier": float(brier_score_loss(y_true, y_prob_arr)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (float(roc_auc_score(y_true, y_prob_arr))
                    if len(set(y_true)) == 2 else 0.0),
    }


# --------------------------------------------------------------------------- #
# Feature selection — walk-forward permutation importance
# --------------------------------------------------------------------------- #

def _walk_forward_feature_importance(
    X: pd.DataFrame, y: pd.Series,
    n_splits: int = 5, random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    importances: Dict[str, List[float]] = {c: [] for c in X.columns}
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_i, (tr, te) in enumerate(splitter.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05,
                max_depth=4, l2_regularization=0.1,
                random_state=random_state + fold_i,
            )),
        ])
        model.fit(X_tr, y_tr)
        perm = permutation_importance(
            model, X_te, y_te, n_repeats=5,
            random_state=random_state + fold_i,
            scoring="neg_log_loss", n_jobs=1,
        )
        for col, imp in zip(X.columns, perm.importances_mean):
            importances[col].append(float(imp))
    mean_imp = {c: float(np.mean(v)) for c, v in importances.items()}
    pos_folds = {c: int(sum(1 for x in v if x > 0)) for c, v in importances.items()}
    return mean_imp, pos_folds


def _correlation_prune(
    X: pd.DataFrame, keep_order: List[str], correlation_max: float = 0.95,
) -> List[str]:
    kept: List[str] = []
    if not keep_order:
        return kept
    corr = X[keep_order].corr().abs()
    for c in keep_order:
        if any(corr.loc[c, k] > correlation_max for k in kept):
            continue
        kept.append(c)
    return kept


def select_features(
    X: pd.DataFrame, y: pd.Series,
    max_features: int = 30, min_positive_folds: int = 3,
    correlation_max: float = 0.92, n_splits: int = 5,
    random_state: int = 42,
    importance_csv_path: Optional[str] = None,
) -> List[str]:
    mean_imp, pos_folds = _walk_forward_feature_importance(
        X, y, n_splits=n_splits, random_state=random_state)
    n_total = len(X.columns)
    stable = [c for c in X.columns
              if pos_folds[c] >= min_positive_folds and mean_imp[c] > 0]
    log.info("feature selection: %d/%d stable (pos in >= %d/%d folds, mean > 0)",
             len(stable), n_total, min_positive_folds, n_splits)
    stable_sorted = sorted(stable, key=lambda c: mean_imp[c], reverse=True)
    after_corr = _correlation_prune(X, stable_sorted, correlation_max=correlation_max)
    log.info("feature selection: %d -> %d after corr prune (>%.2f)",
             len(stable_sorted), len(after_corr), correlation_max)
    selected = after_corr[:max_features]
    log.info("feature selection: kept top %d (max_features=%d)",
             len(selected), max_features)
    if importance_csv_path:
        rows = pd.DataFrame({
            "feature": list(X.columns),
            "mean_importance": [mean_imp[c] for c in X.columns],
            "positive_folds": [pos_folds[c] for c in X.columns],
            "selected": [c in selected for c in X.columns],
        }).sort_values("mean_importance", ascending=False)
        try:
            from pathlib import Path as _P
            _P(importance_csv_path).parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(importance_csv_path, index=False)
            log.info("feature importance audit written to %s", importance_csv_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write importance audit: %s", exc)
    return selected


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #

@dataclass
class _ClassifierEnsemble:
    members: List[Pipeline]

    def predict_proba(self, X) -> np.ndarray:
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return probs


# --------------------------------------------------------------------------- #
# Model artifact + distribution
# --------------------------------------------------------------------------- #

@dataclass
class GasModel:
    """Trained NBA model. Class name kept for cross-bot dashboard compat.

    Fields:
      • ``classifier`` — calibrated classifier ensemble producing
        P(home_wins). The single source of truth for this bot — there's
        no per-threshold grid like the CPI bot has, since each Kalshi
        NBA game market is one binary outcome.
      • ``meta_classifier`` — optional ElasticNet/LogReg stabilizer.
        Blended at ``meta_weight``.
      • ``feature_columns`` — list of feature names selected by the
        walk-forward selector.
      • ``train_end_date`` — last GAME_DATE the model saw during fit.
      • ``metrics`` — test-set log-loss, Brier, accuracy, etc.
    """
    feature_columns: List[str]
    classifier: _ClassifierEnsemble
    meta_classifier: Optional[Pipeline]
    meta_weight: float
    train_end_date: pd.Timestamp
    metrics: Dict[str, float] = field(default_factory=dict)
    # Kept for cross-bot snapshot compatibility (the dashboard reads these).
    residual_std: float = 0.0
    threshold_classifiers: Dict[float, "_ClassifierEnsemble"] = field(default_factory=dict)
    threshold_grid: List[float] = field(default_factory=list)
    residual_distribution: List[float] = field(default_factory=list)

    def predict_home_win_prob(self, feature_row: pd.DataFrame) -> float:
        """Return calibrated P(home team wins) for a single matchup."""
        row = feature_row[self.feature_columns]
        # meta_weight >= 1.0 means the auto-selector chose a pure linear
        # model (Elo-only or full logistic) over the GBT ensemble — the
        # GBT was trained on a different (wider) feature set, so we must
        # NOT call it here. Return the meta prediction directly.
        if self.meta_classifier is not None and self.meta_weight >= 1.0:
            meta_p = float(self.meta_classifier.predict_proba(row)[0, 1])
            return float(min(0.99, max(0.01, meta_p)))
        gbt_p = float(self.classifier.predict_proba(row)[0])
        if self.meta_classifier is None or self.meta_weight <= 0:
            return float(min(0.99, max(0.01, gbt_p)))
        meta_p = float(self.meta_classifier.predict_proba(row)[0, 1])
        blend = (1.0 - self.meta_weight) * gbt_p + self.meta_weight * meta_p
        return float(min(0.99, max(0.01, blend)))

    def predict_distribution(self, current_value: float,
                             feature_row: pd.DataFrame) -> "GasDistribution":
        """Cross-bot interface: simulator/dashboard call this to get a
        distributional snapshot. For NBA we coerce the binary P(win)
        into the same shape the other bots produce so the snapshot
        schema renders without per-bot conditionals on the dashboard.
        ``current_value`` is the home team's pre-game ELO rating; we
        carry it as the "current" anchor for the dashboard.
        """
        p_home = self.predict_home_win_prob(feature_row)
        return GasDistribution(
            current_gas_price=float(current_value),
            median_change=0.0,
            change_quantiles={0.05: 0.0, 0.25: 0.0, 0.50: 0.0,
                              0.75: 0.0, 0.95: 0.0},
            residual_std=self.residual_std,
            prob_up=p_home,
            captured_at=datetime.now(timezone.utc),
            home_win_prob=p_home,
        )


@dataclass
class GasDistribution:
    """Distributional output for a single matchup.

    For NBA the only meaningful field is ``home_win_prob``. The other
    fields exist so the simulator's ``record_model_snapshot`` schema
    works unchanged.
    """
    current_gas_price: float
    median_change: float
    change_quantiles: Dict[float, float]
    residual_std: float
    prob_up: float
    captured_at: datetime
    threshold_probs: Dict[float, float] = field(default_factory=dict)
    residual_distribution: List[float] = field(default_factory=list)
    home_win_prob: float = 0.5

    @property
    def median_price(self) -> float:
        return self.current_gas_price + self.median_change

    def prob_above(self, strike: float, horizon_weeks: float = 1.0) -> float:
        """Cross-bot stub. The decision engine for NBA doesn't call this
        — it reads ``home_win_prob`` directly via market_scanner — but
        we provide a sane fallback so any generic caller (e.g. a future
        dashboard rendering helper) doesn't crash.

        ``strike`` here is interpreted as "team-side": values >= 0.5
        mean "is HOME going to win" (returns home_win_prob); values <
        0.5 mean "is HOME going to lose" (returns 1 - home_win_prob).
        """
        if strike >= 0.5:
            return float(self.home_win_prob)
        return float(1.0 - self.home_win_prob)

    def prob_between(self, low: float, high: float,
                     horizon_weeks: float = 1.0) -> float:
        return max(0.0, self.prob_above(low, horizon_weeks=horizon_weeks)
                       - self.prob_above(high, horizon_weeks=horizon_weeks))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model(
    df: pd.DataFrame, feature_columns: List[str],
    test_size_games: int = 1230,           # ~one season hold-out
    walk_forward_splits: int = 5,
    ensemble_seeds: int = 5,
    calibration_holdout_games: int = 400,
    random_state: int = 42,
    importance_csv_path: Optional[str] = None,
    meta_model_weight: float = 0.20,
    extra_pairs_df: Optional[pd.DataFrame] = None,
    max_features: int = 30,
    feature_min_positive_folds: int = 3,
    feature_correlation_max: float = 0.92,
) -> GasModel:
    """Train the NBA win classifier end-to-end."""
    target = "HOME_WIN"
    if target not in df.columns:
        raise RuntimeError(f"missing target column {target!r}")
    if len(df) <= test_size_games + 200:
        raise RuntimeError(
            f"not enough rows ({len(df)}) for test_size_games={test_size_games}"
        )

    # Sort by date so the time-ordered split is honest.
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    train = df.iloc[:-test_size_games].copy()
    test = df.iloc[-test_size_games:].copy()

    X_train_raw = train[feature_columns]
    X_test_raw = test[feature_columns]
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    # ---- Walk-forward feature selection ----------------------------- #
    log.info("walk-forward feature selection over %d splits "
             "(start: %d candidates)",
             walk_forward_splits, len(feature_columns))
    selected = select_features(
        X_train_raw, y_train,
        max_features=max_features,
        min_positive_folds=feature_min_positive_folds,
        correlation_max=feature_correlation_max,
        n_splits=walk_forward_splits,
        random_state=random_state,
        importance_csv_path=importance_csv_path,
    )
    if not selected:
        log.warning("feature selector kept 0 features — falling back to full set")
        selected = list(feature_columns)
    X_train = X_train_raw[selected]
    X_test = X_test_raw[selected]
    feature_columns = selected

    # ---- Meta-model: regularized logistic regression ---------------- #
    # Acts as a stabilizer the GBT can blend toward; helps Brier / log
    # loss on the OOS season.
    meta = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=10, cv=5, scoring="neg_log_loss",
            max_iter=2000, random_state=random_state,
        )),
    ])
    meta.fit(X_train, y_train)
    log.info("trained logistic-regression meta (weight=%.2f in blend)",
             meta_model_weight)

    # ---- Closed-bet feedback augmentation --------------------------- #
    X_train_aug = X_train
    y_train_aug = y_train
    weights_aug: Optional[np.ndarray] = None
    if extra_pairs_df is not None and not extra_pairs_df.empty:
        cols_needed = list(feature_columns) + ["gas_change", "horizon_weeks"]
        usable = (extra_pairs_df.dropna(subset=cols_needed)
                  if all(c in extra_pairs_df.columns for c in cols_needed)
                  else pd.DataFrame())
        if not usable.empty:
            # For NBA the closed-bet target is 1 if the home team won,
            # 0 otherwise. We stamp gas_change as +1 (home won) or -1
            # (home lost) at close time; map back here.
            X_extra = usable[list(feature_columns)]
            y_extra = (usable["gas_change"].astype(float) > 0).astype(int)
            w_extra = np.clip(usable["horizon_weeks"].astype(float).values, 0.0, 1.0)
            X_train_aug = pd.concat([X_train, X_extra], axis=0)
            y_train_aug = pd.concat([y_train, y_extra], axis=0)
            weights_aug = np.concatenate([np.ones(len(X_train)), w_extra])
            log.info("augmenting training with %d closed-bet rows", len(usable))

    # ---- GBT ensemble + calibration --------------------------------- #
    fit_end = max(1, len(X_train_aug) - calibration_holdout_games)
    X_fit = X_train_aug.iloc[:fit_end]
    X_cal = X_train_aug.iloc[fit_end:]
    y_fit = y_train_aug.iloc[:fit_end]
    y_cal = y_train_aug.iloc[fit_end:]
    w_fit = weights_aug[:fit_end] if weights_aug is not None else None

    members: List[Pipeline] = []
    for seed_offset in range(ensemble_seeds):
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.04,
                max_depth=5, l2_regularization=0.1,
                class_weight="balanced",
                random_state=random_state + seed_offset * 7919,
            )),
        ])
        if w_fit is not None:
            base.fit(X_fit, y_fit, model__sample_weight=w_fit)
        else:
            base.fit(X_fit, y_fit)
        # Calibrate on the holdout window (only when both classes appear).
        # sklearn 1.6 deprecated ``cv="prefit"`` and 1.8 removed it — the
        # current droplet runs 1.8 and was raising InvalidParameterError on
        # every daily retrain. Wrap the base estimator in FrozenEstimator
        # (added in 1.6) to preserve the "fit once, calibrate on holdout"
        # behaviour against the new API. Fall back to a CV-based calibrator
        # on older sklearn where FrozenEstimator doesn't exist yet so the
        # bot still works on dev machines pinned to <1.6.
        if y_cal.nunique() == 2 and len(y_cal) >= 50:
            try:
                from sklearn.frozen import FrozenEstimator
                cal = CalibratedClassifierCV(
                    FrozenEstimator(base), method="sigmoid",
                )
            except ImportError:
                cal = CalibratedClassifierCV(
                    base, method="sigmoid", cv=5,
                )
            cal.fit(X_cal, y_cal)
            members.append(cal)
        else:
            members.append(base)
    classifier = _ClassifierEnsemble(members=members)
    log.info("trained calibrated GBT ensemble (%d members)", ensemble_seeds)

    # ---- Elo-only logistic baseline (tennis-style component) -------- #
    # Train an interpretable baseline on JUST the Elo features so we
    # can answer "is the heavy GBT actually adding value over Elo?"
    # — a question the production dashboard couldn't previously answer.
    # If both ELO_DIFF and ELO_WIN_PROB_HOME aren't in the panel (e.g.
    # an older training table), skip this and just leave the component
    # metrics empty.
    elo_avail = [c for c in ELO_BASELINE_FEATURES if c in train.columns]
    elo_baseline_pipeline: Optional[Pipeline] = None
    elo_only_metrics: Dict[str, float] = {}
    if len(elo_avail) >= 1:
        elo_baseline_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=400)),
        ])
        elo_baseline_pipeline.fit(train[elo_avail], y_train)
        p_elo_test = elo_baseline_pipeline.predict_proba(test[elo_avail])[:, 1]
        elo_only_metrics = _full_metrics(y_test.values, p_elo_test)
        log.info(
            "Elo-only baseline: acc=%.3f Brier=%.3f log_loss=%.3f AUC=%.3f",
            elo_only_metrics["accuracy"], elo_only_metrics["brier"],
            elo_only_metrics["log_loss"], elo_only_metrics["roc_auc"],
        )

    # ---- Test-set metrics ------------------------------------------ #
    p_home = classifier.predict_proba(X_test)        # GBT ensemble (cal)
    p_meta = meta.predict_proba(X_test)[:, 1]        # logistic blend feed
    p_blend = (1.0 - meta_model_weight) * p_home + meta_model_weight * p_meta

    # Per-component metrics — tennis-style breakdown so the dashboard
    # can show "which sub-model contributes which lift?"
    ensemble_metrics = _full_metrics(y_test.values, p_home)
    full_logistic_metrics = _full_metrics(y_test.values, p_meta)
    blended_metrics = _full_metrics(y_test.values, p_blend)

    # ---- Auto-select the production model by held-out log loss ------- #
    # The WNBA panel (~1K games) is small enough that the GBT ensemble
    # overfits — a `--compare` bake-off consistently ranks the Elo-only
    # and full-logistic linear models ABOVE the calibrated ensemble.
    # Rather than hard-code "always ship the ensemble", we ship whichever
    # candidate wins on the chronological hold-out. As the league (and
    # the panel) grows, the ensemble can win on its own merits and this
    # selector will pick it up automatically — no code change needed.
    #
    # Each candidate carries the (feature_columns, predictor, weight)
    # needed to reproduce it at inference time. weight >= 1.0 routes
    # predict_home_win_prob straight through the linear predictor
    # (skipping the GBT, which was fit on the wider `selected` set).
    candidates = [
        ("blended_ensemble", blended_metrics, list(feature_columns),
         meta, meta_model_weight),
        ("full_logistic", full_logistic_metrics, list(feature_columns),
         meta, 1.0),
    ]
    if elo_baseline_pipeline is not None and elo_only_metrics:
        candidates.append(
            ("elo_logistic", elo_only_metrics, list(elo_avail),
             elo_baseline_pipeline, 1.0))
    prod_name, prod_metrics, prod_features, prod_predictor, prod_weight = min(
        candidates, key=lambda c: c[1]["log_loss"])
    log.info("production model auto-selected: %s "
             "(acc=%.3f log_loss=%.4f Brier=%.4f) over %s",
             prod_name, prod_metrics["accuracy"], prod_metrics["log_loss"],
             prod_metrics["brier"],
             ", ".join(f"{n}={m['log_loss']:.4f}" for n, m, *_ in candidates
                       if n != prod_name))

    metrics = {
        "calibrated_classifier_accuracy": prod_metrics["accuracy"],
        "training_precision": prod_metrics["precision"],
        "training_recall": prod_metrics["recall"],
        "training_f1": prod_metrics["f1"],
        "training_roc_auc": prod_metrics["roc_auc"],
        "training_log_loss": prod_metrics["log_loss"],
        "training_brier": prod_metrics["brier"],
        # Which family actually ships, and how it beat the others.
        "production_model": prod_name,
        # Mirror the per-strike pattern the dashboard renders so the
        # card shows reasonable numbers. There's only one threshold
        # (0.5 = does home win) so we duplicate the headline.
        "directional_accuracy_from_median": prod_metrics["accuracy"],
        "per_strike_avg_accuracy": prod_metrics["accuracy"],
        "per_strike_avg_f1": prod_metrics["f1"],
        "per_strike_avg_roc_auc": prod_metrics["roc_auc"],
        "per_strike_count": 1,
        "n_train": int(len(X_train)),
        "n_test": int(len(y_test)),
        # Component breakdown — keys nested so the dashboard can show
        # them under "Elo-only / GBT / Logistic / Blended". Empty
        # elo_only if the Elo features aren't in the panel.
        "components": {
            "elo_only": elo_only_metrics,
            "ensemble": ensemble_metrics,
            "full_logistic": full_logistic_metrics,
            "blended": blended_metrics,
        },
    }
    log.info("test-set: production=%s acc=%.3f F1=%.3f AUC=%.3f "
             "log_loss=%.3f Brier=%.3f (%d games)",
             prod_name, prod_metrics["accuracy"], prod_metrics["f1"],
             prod_metrics["roc_auc"], prod_metrics["log_loss"],
             prod_metrics["brier"], len(y_test))

    # ---- Dump model_coefficients.json for the dashboard ------------- #
    # Mirrors the tennis bot's ``data/processed/artifacts/model_coefficients.json``.
    # The dashboard reads this when the bot's config plumbs in a
    # ``coefficients_path``. Two coefficient sets:
    #   - elo_only.* — interpretable baseline weights on ELO features
    #   - meta.*     — full logistic regression coefficients on the
    #                  selected feature set (used inside the blend)
    try:
        coef_payload: Dict[str, object] = {
            "blend": {
                "ensemble_weight": float(1.0 - meta_model_weight),
                "logistic_weight": float(meta_model_weight),
            },
            "elo_only": {},
            "meta": {},
        }
        if elo_baseline_pipeline is not None:
            mdl = elo_baseline_pipeline.named_steps["model"]
            coef_payload["elo_only"] = {
                "features": list(elo_avail),
                "coefficients": list(map(float, mdl.coef_.ravel().tolist())),
                "intercept": float(np.array(mdl.intercept_).ravel()[0]),
            }
        try:
            meta_mdl = meta.named_steps["model"]
            coef_payload["meta"] = {
                "features": list(feature_columns),
                "coefficients": list(map(float, meta_mdl.coef_.ravel().tolist())),
                "intercept": float(np.array(meta_mdl.intercept_).ravel()[0]),
            }
        except Exception:
            coef_payload["meta"] = {}
        coef_path = Path("data/model_coefficients.json")
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coef_path, "w") as f:
            json.dump(coef_payload, f, indent=2)
        log.info("wrote model coefficients → %s", coef_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not dump model_coefficients.json: %s", exc)

    # ---- Dump held-out test-set predictions for the dashboard -------- #
    # The dashboard's Models tab reads this file to draw the ROC curve
    # and confusion matrix from training data — i.e. the model's
    # actual evaluation against ground-truth game outcomes — rather
    # than from closed Kalshi bets (which would only kick in once the
    # paper-trade ledger has settles, and double-counts the live
    # market noise on top of model error).
    try:
        # Dump the SHIPPED model's hold-out predictions (not the blend)
        # so the dashboard's ROC curve / confusion matrix reflect what
        # actually trades.
        prod_p = {
            "blended_ensemble": p_blend,
            "full_logistic": p_meta,
            "elo_logistic": (p_elo_test if "p_elo_test" in dir() else p_blend),
        }.get(prod_name, p_blend)
        holdout_path = Path("data/holdout_predictions.csv")
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(holdout_path, "w") as f:
            f.write("predicted_prob,actual_label\n")
            for prob, label in zip(prod_p, y_test.values):
                f.write(f"{float(prob):.6f},{int(label)}\n")
        log.info("wrote holdout predictions (%d rows) → %s",
                  len(y_test), holdout_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not dump holdout_predictions.csv: %s", exc)

    return GasModel(
        # Production fields reflect the auto-selected winner. The GBT
        # `classifier` is retained for the dashboard's component view but
        # is bypassed at inference whenever prod_weight >= 1.0.
        feature_columns=list(prod_features),
        classifier=classifier,
        meta_classifier=prod_predictor,
        meta_weight=prod_weight,
        train_end_date=train["GAME_DATE"].max(),
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_model(model: GasModel, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Optional[GasModel]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
