"""Model stages + logistic pipeline: nesting, base rate, learning signal, no target leak."""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.datasets.assemble import FEATURE_COLS, OUTCOME_COLS
from crashback.models.logistic import (
    coefficient_table,
    fit_base_rate,
    fit_logistic,
    predict_logistic,
)
from crashback.models.stages import STAGE_GROUPS, stage_features


def test_stages_are_nested_and_feature_only():
    m1, m2, m3 = (set(stage_features(s)) for s in ("model1", "model2", "model3"))
    assert stage_features("model0") == []
    assert m1 < m2 < m3                                   # strictly nested
    # no stage ever includes an outcome column
    assert set(stage_features("model3")).isdisjoint(OUTCOME_COLS)
    # every stage feature is a real dataset feature column
    assert set(stage_features("model3")) <= set(FEATURE_COLS)
    assert set(STAGE_GROUPS) == {"model0", "model1", "model2", "model3"}


def test_base_rate_predicts_training_prevalence():
    y = [1, 1, 1, 0, 0, 0, 0, 0]         # prevalence 0.375
    model = fit_base_rate(y)
    assert abs(model.rate - 0.375) < 1e-12
    preds = model.predict_proba(5)
    assert preds.shape == (5,)
    assert np.allclose(preds, 0.375)


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x_signal = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    # y depends on x_signal only; probability rises with x_signal
    p = 1 / (1 + np.exp(-2.0 * x_signal))
    y = (rng.uniform(size=n) < p).astype(int)
    return pl.DataFrame({"x_signal": x_signal, "x_noise": x_noise}), y


def test_logistic_learns_signal_direction_and_predicts_range():
    X, y = _synthetic()
    pipe = fit_logistic(X, y, C=1.0, max_iter=1000, seed=42)
    p = predict_logistic(pipe, X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    coef = coefficient_table(pipe, ["x_signal", "x_noise"])
    top = coef.row(0, named=True)
    assert top["feature"] == "x_signal"                  # signal dominates noise
    assert top["coef_std"] > 0                           # positive association
    assert top["direction"] == "increases P(recover)"


def test_logistic_handles_missing_with_indicator():
    X, y = _synthetic()
    # inject missingness into x_noise; imputer should add x_noise__missing
    xn = X["x_noise"].to_list()
    for i in range(0, len(xn), 3):
        xn[i] = None
    X = X.with_columns(pl.Series("x_noise", xn))
    pipe = fit_logistic(X, y, C=1.0, max_iter=1000, seed=42)
    coef = coefficient_table(pipe, ["x_signal", "x_noise"])
    assert "x_noise__missing" in set(coef["feature"].to_list())
    # coefficient vector length == inputs + one indicator
    assert coef.height == 3
