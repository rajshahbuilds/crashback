"""LightGBM models: grid expansion, validation-only tuning, importance, inf handling."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from crashback.models.gbm import grid, importance_table, predict, tune

# tiny predeclared grid so tests stay fast: 2 * 1 * 1 * 2 = 4 combos
_LG = SimpleNamespace(
    learning_rate=0.1, num_boost_round=100, early_stopping_rounds=10,
    primary_metric="log_loss",
    num_leaves=[7, 15], min_child_samples=[20], feature_fraction=[1.0], lambda_l2=[0.0, 1.0],
)


def test_grid_is_cartesian_product():
    g = grid(_LG)
    assert len(g) == 2 * 1 * 1 * 2
    assert set(g[0]) == {"num_leaves", "min_child_samples", "feature_fraction", "lambda_l2"}
    # every combination is distinct
    assert len({tuple(sorted(d.items())) for d in g}) == len(g)


def _synthetic(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x_signal = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    p = 1 / (1 + np.exp(-2.5 * x_signal))
    y = (rng.uniform(size=n) < p).astype(int)
    X = pl.DataFrame({"x_signal": x_signal, "x_noise": x_noise})
    return X, y


def test_tune_selects_from_grid_and_predicts_probabilities():
    X, y = _synthetic()
    Xtr, ytr = X[:800], y[:800]
    Xva, yva = X[800:], y[800:]
    feats = ["x_signal", "x_noise"]
    res = tune(Xtr, ytr, Xva, yva, _LG, seed=42, feature_names=feats)

    assert res.best_params in grid(_LG)           # chosen from the declared space
    assert res.trials.height == len(grid(_LG))     # one row per grid point
    assert "val_log_loss" in res.trials.columns
    p = predict(res.booster, Xva)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert res.best_iteration >= 1


def test_signal_feature_dominates_importance():
    X, y = _synthetic()
    feats = ["x_signal", "x_noise"]
    res = tune(X[:800], y[:800], X[800:], y[800:], _LG, seed=42, feature_names=feats)
    imp = importance_table(res.booster)
    assert imp.row(0, named=True)["feature"] == "x_signal"
    assert abs(imp["gain_frac"].sum() - 1.0) < 1e-6


def test_inf_is_handled_as_missing():
    X, y = _synthetic(n=600)
    xs = X["x_noise"].to_list()
    xs[0] = float("inf")
    xs[1] = float("-inf")
    X = X.with_columns(pl.Series("x_noise", xs))
    feats = ["x_signal", "x_noise"]
    res = tune(X[:400], y[:400], X[400:], y[400:], _LG, seed=42, feature_names=feats)
    p = predict(res.booster, X[400:])            # must not raise on inf
    assert np.isfinite(p).all()
