"""Gradient-boosted (XGBoost) recovery models — a cross-check on the LightGBM results (STU-60).

Same tuning discipline as ``crashback.models.gbm``: the predeclared grid is scored on the
*validation* period only, early stopping watches validation, and the best per stage is chosen by
the predeclared primary metric. Test is never touched. XGBoost handles missing values natively
(``missing=nan``); ``inf`` (undefined ratios) is routed to NaN so it takes that path. Class
prevalence is preserved (no ``scale_pos_weight``).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.evaluation.metrics import binary_metrics

_METRIC_KEY = {"log_loss": "log_loss", "brier": "brier"}


def _matrix(X: pl.DataFrame) -> np.ndarray:
    a = X.to_numpy().astype(float)
    a[np.isinf(a)] = np.nan
    return a


def base_params(cfg_xgb, seed: int) -> dict:
    """Fixed XGBoost params shared across the grid (deterministic hist, prevalence-preserving)."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": cfg_xgb.learning_rate,
        "subsample": cfg_xgb.subsample,
        "seed": seed,
        "nthread": 4,
    }


def grid(cfg_xgb) -> list[dict]:
    """Cartesian product of the predeclared search grid → list of param overrides."""
    keys = ("max_depth", "min_child_weight", "colsample_bytree", "reg_lambda")
    combos = itertools.product(
        cfg_xgb.max_depth, cfg_xgb.min_child_weight,
        cfg_xgb.colsample_bytree, cfg_xgb.reg_lambda,
    )
    return [dict(zip(keys, c, strict=True)) for c in combos]


def _dmatrix(X: pl.DataFrame, y, feature_names: list[str]) -> xgb.DMatrix:
    return xgb.DMatrix(_matrix(X), label=np.asarray(y, dtype=int),
                       feature_names=feature_names, missing=np.nan)


def fit_one(
    X_tr: pl.DataFrame, y_tr, X_val: pl.DataFrame, y_val, params: dict,
    *, num_boost_round: int, early_stopping_rounds: int, feature_names: list[str],
) -> tuple[xgb.Booster, int]:
    """Train a single booster with early stopping on validation; return (booster, best_iter)."""
    dtr = _dmatrix(X_tr, y_tr, feature_names)
    dval = _dmatrix(X_val, y_val, feature_names)
    booster = xgb.train(
        params, dtr, num_boost_round=num_boost_round,
        evals=[(dval, "val")], early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster, booster.best_iteration


def predict(booster: xgb.Booster, X: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    d = xgb.DMatrix(_matrix(X), feature_names=feature_names, missing=np.nan)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


@dataclass
class TuneResult:
    booster: xgb.Booster
    best_iteration: int
    best_params: dict
    trials: pl.DataFrame


def tune(
    X_tr: pl.DataFrame, y_tr, X_val: pl.DataFrame, y_val, cfg_xgb, seed: int,
    *, feature_names: list[str],
) -> TuneResult:
    """Grid-search on validation; pick the booster minimizing the predeclared primary metric."""
    fixed = base_params(cfg_xgb, seed)
    key = _METRIC_KEY[cfg_xgb.primary_metric]
    rows: list[dict] = []
    best = None
    for override in grid(cfg_xgb):
        booster, best_iter = fit_one(
            X_tr, y_tr, X_val, y_val, {**fixed, **override},
            num_boost_round=cfg_xgb.num_boost_round,
            early_stopping_rounds=cfg_xgb.early_stopping_rounds,
            feature_names=feature_names,
        )
        p_val = predict(booster, X_val, feature_names)
        score = binary_metrics(y_val, p_val)[key]
        rows.append({**override, "best_iteration": best_iter, "val_" + key: score})
        if best is None or score < best[0]:
            best = (score, booster, best_iter, override)

    _, booster, best_iter, override = best
    return TuneResult(booster, best_iter, override, pl.DataFrame(rows).sort("val_" + key))


def importance_table(booster: xgb.Booster, feature_names: list[str]) -> pl.DataFrame:
    """Feature importance by total gain (normalized) and split count, sorted by gain.

    XGBoost only reports features that were actually used in a split, so we reindex against the
    full feature list (unused features get 0).
    """
    gain = booster.get_score(importance_type="total_gain")
    split = booster.get_score(importance_type="weight")
    g = np.array([gain.get(f, 0.0) for f in feature_names], dtype=float)
    s = np.array([split.get(f, 0.0) for f in feature_names], dtype=float)
    g_sum = g.sum() or 1.0
    return (
        pl.DataFrame({"feature": feature_names, "gain": g, "split": s})
        .with_columns((pl.col("gain") / g_sum).alias("gain_frac"))
        .sort("gain", descending=True)
    )
