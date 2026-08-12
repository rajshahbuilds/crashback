"""Per-event explanations: additivity (SHAP + bias = margin), logistic terms, ranking."""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.evaluation.explain import (
    logistic_contributions,
    sigmoid,
    top_contributors,
    tree_shap,
)
from crashback.models.logistic import fit_logistic
from crashback.models.xgb import predict as xgb_predict
from crashback.models.xgb import tune as xgb_tune

_XG = __import__("types").SimpleNamespace(
    learning_rate=0.1, num_boost_round=60, early_stopping_rounds=10, subsample=1.0,
    primary_metric="log_loss",
    max_depth=[3], min_child_weight=[1.0], colsample_bytree=[1.0], reg_lambda=[0.0])


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.5 * x1 - 1.0 * x2)))
    y = (rng.uniform(size=n) < p).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2}), y


def test_tree_shap_sums_to_prediction():
    X, y = _data()
    feats = ["x1", "x2"]
    res = xgb_tune(X[:400], y[:400], X[400:], y[400:], _XG, seed=1, feature_names=feats)
    contribs, bias = tree_shap(res.booster, X[400:], feats)
    margin = contribs.sum(axis=1) + bias
    # sigmoid(sum of SHAP + bias) must equal the model's predicted probability
    assert np.allclose(sigmoid(margin), xgb_predict(res.booster, X[400:], feats), atol=1e-5)
    assert contribs.shape == (200, 2)


def test_logistic_contributions_sum_to_logit():
    X, y = _data()
    feats = ["x1", "x2"]
    pipe = fit_logistic(X, y, C=1.0, max_iter=1000, seed=0)
    contribs, intercept, names = logistic_contributions(pipe, X, feats)
    logit = contribs.sum(axis=1) + intercept
    prob = pipe.predict_proba(
        __import__("numpy").nan_to_num(X.to_numpy(), nan=0.0))[:, 1]  # no missing here
    assert np.allclose(sigmoid(logit), prob, atol=1e-6)
    assert names[:2] == ["x1", "x2"]


def test_logistic_sign_matches_coefficient_direction():
    X, y = _data()
    pipe = fit_logistic(X, y, C=1.0, max_iter=1000, seed=0)
    contribs, _, names = logistic_contributions(pipe, X, ["x1", "x2"])
    # x1 has positive coef: its contribution should correlate positively with x1's value
    x1 = X["x1"].to_numpy()
    j = names.index("x1")
    assert np.corrcoef(contribs[:, j], x1)[0, 1] > 0.9


def test_top_contributors_ranks_and_splits_sign():
    row = np.array([0.5, -0.3, 0.1, -0.9, 0.0])
    names = ["a", "b", "c", "d", "e"]
    pos, neg = top_contributors(row, names, k=2)
    assert pos[0] == ("a", 0.5) and pos[1] == ("c", 0.1)     # positives, largest first
    assert neg[0] == ("d", -0.9) and neg[1] == ("b", -0.3)   # negatives, most negative first
