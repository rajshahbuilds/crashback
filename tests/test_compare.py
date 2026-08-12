"""Ablation ladder + paired clustered bootstrap for incremental value (STU-61)."""
from __future__ import annotations

import numpy as np

from crashback.evaluation.compare import paired_cluster_bootstrap
from crashback.models.stages import (
    INCREMENTS,
    LADDER_TO_STAGE,
    ladder_features,
    stage_features,
)


def test_ladder_isolates_recent_crash_and_matches_stages():
    # price+recent must equal price ∪ recent-crash, and the finer rung splits them.
    price = set(ladder_features("price"))
    price_recent = set(ladder_features("price+recent"))
    assert price < price_recent                       # recent-crash strictly adds columns
    assert price_recent - price                       # the added columns are the recent-crash ones
    # ladder rungs that coincide with staged models match them exactly (so reuse is valid)
    assert ladder_features("price+recent") == stage_features("model1")
    assert ladder_features("+market") == stage_features("model2")
    assert ladder_features("+fundamentals") == stage_features("model3")
    assert set(LADDER_TO_STAGE) == {"price+recent", "+market", "+fundamentals"}


def test_increments_form_a_chain():
    # each increment's `to` is the next increment's `from` (a proper ladder)
    tos = [i[1] for i in INCREMENTS]
    froms = [i[0] for i in INCREMENTS]
    assert froms[0] == "base"
    assert froms[1:] == tos[:-1]
    assert tos[-1] == "+fundamentals"


def _clustered_data(n_sec=200, per=5, seed=0):
    rng = np.random.default_rng(seed)
    y, better, worse, clusters = [], [], [], []
    for s in range(n_sec):
        yy = rng.integers(0, 2, per)
        for i in range(per):
            y.append(yy[i])
            # `better` is correlated with y; `worse` is pure noise
            better.append(0.5 + (0.35 if yy[i] else -0.35) + rng.normal(0, 0.05))
            worse.append(rng.uniform(0.4, 0.6))
            clusters.append(s)
    clip = lambda a: np.clip(np.array(a, float), 1e-4, 1 - 1e-4)  # noqa: E731
    return (np.array(y), clip(better), clip(worse), np.array(clusters))


def test_bootstrap_flags_real_improvement_as_material():
    y, better, worse, clusters = _clustered_data()
    preds = {"a": worse, "b": better}
    incs = [("a", "b", "signal")]
    deltas, point = paired_cluster_bootstrap(
        y, preds, clusters, incs, metrics=("log_loss", "roc_auc"), n_boot=200, seed=1)
    ll = deltas.filter(deltas["metric"] == "log_loss").row(0, named=True)
    assert ll["delta"] < 0                              # adding real signal lowers log loss
    assert ll["material"] is True                       # CI excludes 0
    assert ll["ci_hi"] < 0
    assert point["b"]["log_loss"] < point["a"]["log_loss"]


def test_bootstrap_flags_noise_as_immaterial():
    y, _, worse, clusters = _clustered_data()
    rng = np.random.default_rng(3)
    worse2 = np.clip(worse + rng.normal(0, 0.01, worse.shape), 1e-4, 1 - 1e-4)
    preds = {"a": worse, "b": worse2}                   # b adds only noise
    deltas, _ = paired_cluster_bootstrap(
        y, preds, clusters, [("a", "b", "noise")], metrics=("log_loss",), n_boot=200, seed=2)
    assert deltas.row(0, named=True)["material"] is False   # CI spans 0
