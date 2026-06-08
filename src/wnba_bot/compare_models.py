"""Multi-model bake-off for WNBA single-game outcome prediction.

The production bot ships ONE model (``model.train_model`` — a calibrated
GBT ensemble blended with a logistic meta-model). But "which model is
best?" is an empirical question, so this module trains several distinct
model families on the *same* chronological train/test split and reports
a head-to-head comparison on the held-out most-recent season.

Models compared
----------------
  • ``home_baseline``   — always predict the home team (sanity floor)
  • ``elo_logistic``    — logistic regression on ELO_DIFF only
  • ``logistic_l2``     — L2 logistic regression on all features
  • ``random_forest``   — bagged trees
  • ``hist_gbt``        — a single HistGradientBoosting classifier
  • ``calibrated_ensemble`` — the SHIPPED production model
    (``model.train_model``'s blended, calibrated output)

Scoring is proper-scoring-first: we rank by **log loss** (then Brier),
because a sports-betting edge comes from calibrated probabilities, not
raw accuracy. The winner and the full table are written to
``data/model_comparison.{csv,json}`` so the dashboard's model card can
render the bake-off, and printed to stdout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import train_model, ELO_BASELINE_FEATURES

log = logging.getLogger(__name__)


def _metrics(y_true, y_prob) -> Dict[str, float]:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (float(roc_auc_score(y_true, y_prob))
                    if len(set(y_true)) == 2 else 0.0),
    }


def _candidate_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "logistic_l2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=10,
                class_weight="balanced", random_state=random_state, n_jobs=-1)),
        ]),
        "hist_gbt": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.04, max_depth=5,
                l2_regularization=0.1, class_weight="balanced",
                random_state=random_state)),
        ]),
    }


def run_bakeoff(
    df: pd.DataFrame,
    feature_columns: List[str],
    test_size_games: int = 240,
    out_dir: str = "data",
    random_state: int = 42,
) -> Dict[str, dict]:
    """Train every candidate on a chronological split and report metrics.

    Returns the results dict (also written to disk). The held-out test
    set is the most-recent ``test_size_games`` games — the same split
    ``model.train_model`` uses, so ``calibrated_ensemble`` here matches
    the shipped model's reported numbers.
    """
    target = "HOME_WIN"
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    if len(df) <= test_size_games + 100:
        # Shrink the hold-out for tiny panels so the bake-off still runs.
        test_size_games = max(40, len(df) // 5)
        log.warning("panel small (%d rows) — using test_size_games=%d",
                    len(df), test_size_games)
    train = df.iloc[:-test_size_games]
    test = df.iloc[-test_size_games:]
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)
    X_train = train[feature_columns]
    X_test = test[feature_columns]

    results: Dict[str, dict] = {}

    # 1) Trivial home-team baseline.
    results["home_baseline"] = _metrics(
        y_test.values, np.full(len(y_test), float(y_train.mean())))

    # 2) Elo-only logistic.
    elo_feats = [c for c in ELO_BASELINE_FEATURES if c in df.columns]
    if elo_feats:
        elo_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        elo_pipe.fit(train[elo_feats], y_train)
        p = elo_pipe.predict_proba(test[elo_feats])[:, 1]
        results["elo_logistic"] = _metrics(y_test.values, p)

    # 3) The generic candidate families.
    for name, pipe in _candidate_pipelines(random_state).items():
        try:
            pipe.fit(X_train, y_train)
            p = pipe.predict_proba(X_test)[:, 1]
            results[name] = _metrics(y_test.values, p)
        except Exception as exc:  # noqa: BLE001
            log.warning("candidate %s failed: %s", name, exc)

    # 4) The shipped production model. NOTE: train_model auto-selects the
    #    best of {elo / full-logistic / calibrated-ensemble} internally,
    #    so this row reports whatever actually ships — it should match the
    #    standalone winner above. Reported under "production_shipped" so
    #    the model card distinguishes "what we deploy" from the raw
    #    candidate families.
    try:
        prod = train_model(
            df, feature_columns,
            test_size_games=test_size_games,
            walk_forward_splits=5,
            ensemble_seeds=5,
            calibration_holdout_games=max(60, test_size_games // 2),
        )
        m = prod.metrics
        results[f"production_shipped[{m.get('production_model', '?')}]"] = {
            "accuracy": m.get("calibrated_classifier_accuracy", 0.0),
            "log_loss": m.get("training_log_loss", 0.0),
            "brier": m.get("training_brier", 0.0),
            "f1": m.get("training_f1", 0.0),
            "roc_auc": m.get("training_roc_auc", 0.0),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("production model bake-off row failed: %s", exc)

    # Rank by log loss (proper score), then Brier.
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1]["log_loss"], kv[1]["brier"]))
    best = ranked[0][0] if ranked else None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # CSV
    rows = [{"model": name, **vals} for name, vals in ranked]
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)
    # JSON (for the dashboard model card)
    payload = {
        "test_games": int(test_size_games),
        "n_train": int(len(train)),
        "ranked_by": "log_loss",
        "best": best,
        "models": rows,
    }
    with open(out / "model_comparison.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Pretty print.
    print("\n" + "=" * 78)
    print(f"WNBA model bake-off  |  held-out games: {test_size_games}  |  "
          f"train: {len(train)}")
    print("=" * 78)
    print(f"{'model':22s} {'acc':>7s} {'logloss':>9s} {'brier':>8s} "
          f"{'roc_auc':>8s} {'f1':>7s}")
    print("-" * 78)
    for name, v in ranked:
        star = "  <-- best" if name == best else ""
        print(f"{name:22s} {v['accuracy']:7.3f} {v['log_loss']:9.4f} "
              f"{v['brier']:8.4f} {v['roc_auc']:8.3f} {v['f1']:7.3f}{star}")
    print("=" * 78)
    print(f"Winner (lowest log loss): {best}")
    print(f"Written: {out/'model_comparison.csv'} , {out/'model_comparison.json'}\n")
    return payload
