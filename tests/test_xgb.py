"""XGBoost models: grid expansion, validation-only tuning, importance, inf handling."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from crashback.models.xgb import grid, importance_table, predict, tune

# tiny predeclared grid: 2 * 1 * 1 * 2 = 4 combos
_XG = SimpleNamespace(
    learning_rate=0.1, num_boost_round=100, early_stopping_rounds=10, subsample=1.0,
    primary_metric="log_loss",
    max_depth=[3, 5], min_child_weight=[1.0], colsample_bytree=[1.0], reg_lambda=[0.0, 1.0],
)


def test_grid_is_cartesian_product():
    g = grid(_XG)
    assert len(g) == 2 * 1 * 1 * 2
    assert set(g[0]) == {"max_depth", "min_child_weight", "colsample_bytree", "reg_lambda"}
    assert len({tuple(sorted(d.items())) for d in g}) == len(g)


def _synthetic(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x_signal = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    p = 1 / (1 + np.exp(-2.5 * x_signal))
    y = (rng.uniform(size=n) < p).astype(int)
    return pl.DataFrame({"x_signal": x_signal, "x_noise": x_noise}), y


def test_tune_selects_from_grid_and_predicts_probabilities():
    X, y = _synthetic()
    feats = ["x_signal", "x_noise"]
    res = tune(X[:800], y[:800], X[800:], y[800:], _XG, seed=42, feature_names=feats)
    assert res.best_params in grid(_XG)
    assert res.trials.height == len(grid(_XG))
    assert "val_log_loss" in res.trials.columns
    p = predict(res.booster, X[800:], feats)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_signal_feature_dominates_importance():
    X, y = _synthetic()
    feats = ["x_signal", "x_noise"]
    res = tune(X[:800], y[:800], X[800:], y[800:], _XG, seed=42, feature_names=feats)
    imp = importance_table(res.booster, feats)
    assert imp.row(0, named=True)["feature"] == "x_signal"
    assert abs(imp["gain_frac"].sum() - 1.0) < 1e-6


def test_inf_is_handled_as_missing():
    X, y = _synthetic(n=600)
    xs = X["x_noise"].to_list()
    xs[0], xs[1] = float("inf"), float("-inf")
    X = X.with_columns(pl.Series("x_noise", xs))
    feats = ["x_signal", "x_noise"]
    res = tune(X[:400], y[:400], X[400:], y[400:], _XG, seed=42, feature_names=feats)
    p = predict(res.booster, X[400:], feats)
    assert np.isfinite(p).all()
