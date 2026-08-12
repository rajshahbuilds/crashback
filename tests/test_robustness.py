"""Bootstrap CIs + slice AUC for robustness (STU-63)."""
from __future__ import annotations

import numpy as np

from crashback.evaluation.robustness import auc_by_slice, bootstrap_test


def _clustered(n_sec=300, per=6, seed=0):
    # Each security has a latent recovery propensity shared by all its events → within-cluster
    # correlation (the reason a clustered bootstrap is wider than iid). `good` tracks the latent
    # with noise (moderate, non-saturating AUC); `const` is the baseline.
    rng = np.random.default_rng(seed)
    y, good, const, clusters = [], [], [], []
    for s in range(n_sec):
        latent = rng.uniform(0.1, 0.9)
        for _ in range(per):
            y.append(int(rng.uniform() < latent))
            good.append(latent + rng.normal(0, 0.08))
            const.append(0.5)
            clusters.append(s)
    clip = lambda a: np.clip(np.array(a, float), 1e-4, 1 - 1e-4)  # noqa: E731
    return np.array(y), clip(good), clip(const), np.array(clusters)


def test_ci_brackets_point_and_flags_real_signal():
    y, good, const, clusters = _clustered()
    preds = {"base": const, "good": good}
    mci, dci = bootstrap_test(y, preds, clusters, "base", n_boot=200, seed=1)
    auc = mci.filter((mci["model"] == "good") & (mci["metric"] == "roc_auc")).row(0, named=True)
    assert auc["ci_lo"] <= auc["point"] <= auc["ci_hi"]
    assert auc["ci_lo"] > 0.5                              # solidly above chance
    d = dci.filter(dci["model"] == "good").row(0, named=True)
    assert d["beats_base"] is True                          # beats baseline on log loss
    assert d["ci_hi"] < 0


def test_noise_model_not_flagged():
    y, _, const, clusters = _clustered()
    rng = np.random.default_rng(4)
    noise = np.clip(const + rng.normal(0, 0.01, const.shape), 1e-4, 1 - 1e-4)
    mci, dci = bootstrap_test(y, {"base": const, "noise": noise}, clusters, "base",
                              n_boot=200, seed=2)
    auc = mci.filter((mci["model"] == "noise") & (mci["metric"] == "roc_auc")).row(0, named=True)
    assert auc["ci_lo"] < 0.5 < auc["ci_hi"]               # CI spans chance
    assert dci.filter(dci["model"] == "noise").row(0, named=True)["beats_base"] is False


def test_clustered_ci_wider_than_iid():
    y, good, const, clusters = _clustered()
    preds = {"base": const, "good": good}
    cl, _ = bootstrap_test(y, preds, clusters, "base", n_boot=300, seed=3, by_cluster=True)
    iid, _ = bootstrap_test(y, preds, clusters, "base", n_boot=300, seed=3, by_cluster=False)

    def width(f):
        r = f.filter((f["model"] == "good") & (f["metric"] == "roc_auc")).row(0, named=True)
        return r["ci_hi"] - r["ci_lo"]
    assert width(cl) > width(iid)                          # dependence widens the interval


def test_auc_by_slice_respects_mask():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7])
    mask = np.array([True, True, True, True, False, False, False, False])
    auc, n, base = auc_by_slice(y, p, mask)
    assert n == 4
    assert base == 0.5
    assert auc == 1.0                                       # perfectly ranked within the mask
