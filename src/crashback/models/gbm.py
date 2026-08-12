"""Gradient-boosted (LightGBM) recovery models with validation-only tuning.

Trees capture nonlinear interactions that the logistic baselines (STU-59) miss. LightGBM
handles missing values natively, so no imputation/scaling — we only route ``inf`` (undefined
ratios) to NaN so it takes the native missing path.

**Tuning discipline (CLAUDE.md §20):** the predeclared hyperparameter grid is scored on the
*validation* period only, and early stopping watches the *validation* set to choose the boosting
round count. The test period is never touched here. Because selection and reporting both use
validation, the reported validation metrics are mildly optimistic — the unbiased read is the
held-out test set (STU-62). Class prevalence is preserved (no oversampling / class weighting).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from crashback.evaluation.metrics import binary_metrics

_METRIC_KEY = {"log_loss": "log_loss", "brier": "brier"}


def _matrix(X: pl.DataFrame) -> np.ndarray:
    """Float matrix with inf routed to NaN (LightGBM handles NaN natively as 'missing')."""
    a = X.to_numpy().astype(float)
    a[np.isinf(a)] = np.nan
    return a


def base_params(cfg_lgbm, seed: int) -> dict:
    """Fixed LightGBM params shared across the grid (deterministic, prevalence-preserving)."""
    return {
        "objective": "binary",
        "learning_rate": cfg_lgbm.learning_rate,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
        "verbosity": -1,
    }


def grid(cfg_lgbm) -> list[dict]:
    """Cartesian product of the predeclared search grid → list of param overrides."""
    keys = ("num_leaves", "min_child_samples", "feature_fraction", "lambda_l2")
    combos = itertools.product(
        cfg_lgbm.num_leaves, cfg_lgbm.min_child_samples,
        cfg_lgbm.feature_fraction, cfg_lgbm.lambda_l2,
    )
    return [dict(zip(keys, c, strict=True)) for c in combos]


def fit_one(
    X_tr: pl.DataFrame, y_tr, X_val: pl.DataFrame, y_val, params: dict,
    *, num_boost_round: int, early_stopping_rounds: int, feature_names: list[str],
) -> tuple[lgb.Booster, int]:
    """Train a single booster with early stopping on validation; return (booster, best_iter)."""
    dtr = lgb.Dataset(_matrix(X_tr), label=np.asarray(y_tr, dtype=int),
                      feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(_matrix(X_val), label=np.asarray(y_val, dtype=int),
                       reference=dtr, feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(
        {**params, "metric": "binary_logloss"},
        dtr, num_boost_round=num_boost_round, valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return booster, booster.best_iteration


def predict(booster: lgb.Booster, X: pl.DataFrame) -> np.ndarray:
    return booster.predict(_matrix(X), num_iteration=booster.best_iteration)


@dataclass
class TuneResult:
    booster: lgb.Booster
    best_iteration: int
    best_params: dict
    trials: pl.DataFrame          # every grid point + its validation score


def tune(
    X_tr: pl.DataFrame, y_tr, X_val: pl.DataFrame, y_val, cfg_lgbm, seed: int,
    *, feature_names: list[str],
) -> TuneResult:
    """Grid-search on validation; pick the booster minimizing the predeclared primary metric."""
    fixed = base_params(cfg_lgbm, seed)
    key = _METRIC_KEY[cfg_lgbm.primary_metric]
    rows: list[dict] = []
    best = None
    for override in grid(cfg_lgbm):
        booster, best_iter = fit_one(
            X_tr, y_tr, X_val, y_val, {**fixed, **override},
            num_boost_round=cfg_lgbm.num_boost_round,
            early_stopping_rounds=cfg_lgbm.early_stopping_rounds,
            feature_names=feature_names,
        )
        p_val = booster.predict(_matrix(X_val), num_iteration=best_iter)
        score = binary_metrics(y_val, p_val)[key]
        rows.append({**override, "best_iteration": best_iter, "val_" + key: score})
        if best is None or score < best[0]:
            best = (score, booster, best_iter, override)

    _, booster, best_iter, override = best
    return TuneResult(booster, best_iter, override, pl.DataFrame(rows).sort("val_" + key))


def importance_table(booster: lgb.Booster) -> pl.DataFrame:
    """Feature importance by total gain and by split count, normalized, sorted by gain."""
    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    g_sum = gain.sum() or 1.0
    return (
        pl.DataFrame({"feature": names, "gain": gain, "split": split})
        .with_columns((pl.col("gain") / g_sum).alias("gain_frac"))
        .sort("gain", descending=True)
    )
