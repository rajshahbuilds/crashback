"""Bootstrap confidence bands for test metrics + dependence quantification (STU-63).

Answers "is the out-of-sample signal solidly above chance, or within noise?" by putting a 95%
CI on each model's test metric. Two resampling schemes:

- **cluster** (default): resample *securities* with replacement (block bootstrap) — honest given
  that crash events for the same company cluster in time and are correlated (CLAUDE.md §22).
- **iid**: resample individual events — understates uncertainty; kept only to *quantify* how
  much the dependence widens the interval (the difference is the point).

A model's discrimination is "solidly > chance" iff the ROC-AUC CI lower bound exceeds 0.5; it
"beats baseline" iff the log-loss delta-vs-baseline CI upper bound is below 0.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.evaluation.compare import _cluster_rows, _metric


def _resample_index(rng, cluster_rows, k, n, by_cluster):
    if by_cluster:
        samp = rng.integers(0, k, k)
        return np.concatenate([cluster_rows[c] for c in samp])
    return rng.integers(0, n, n)


def bootstrap_test(
    y, preds: dict[str, np.ndarray], clusters, baseline_key: str,
    *, metrics=("roc_auc", "log_loss"), n_boot: int = 500, seed: int = 42, by_cluster: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-model metric CIs + each model's log-loss delta-vs-baseline CI, one bootstrap pass.

    Returns (metric_ci, delta_ci):
      metric_ci: model, metric, point, ci_lo, ci_hi
      delta_ci : model, delta_logloss_vs_base (point), ci_lo, ci_hi, beats_base
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    rows = _cluster_rows(np.asarray(clusters))
    k, n = len(rows), y.shape[0]
    names = list(preds)

    point = {nm: {m: _metric(y, preds[nm], m) for m in metrics} for nm in names}
    dist = {(nm, m): [] for nm in names for m in metrics}
    ddist = {nm: [] for nm in names}

    for _ in range(n_boot):
        idx = _resample_index(rng, rows, k, n, by_cluster)
        yb = y[idx]
        mv = {(nm, m): _metric(yb, preds[nm][idx], m) for nm in names for m in metrics}
        base_ll = mv[(baseline_key, "log_loss")]
        for nm in names:
            for m in metrics:
                dist[(nm, m)].append(mv[(nm, m)])
            ddist[nm].append(mv[(nm, "log_loss")] - base_ll)

    def ci(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a.size \
            else (float("nan"), float("nan"))

    mrows, drows = [], []
    for nm in names:
        for m in metrics:
            lo, hi = ci(dist[(nm, m)])
            mrows.append({"model": nm, "metric": m, "point": point[nm][m],
                          "ci_lo": lo, "ci_hi": hi})
        lo, hi = ci(ddist[nm])
        delta = point[nm]["log_loss"] - point[baseline_key]["log_loss"]
        drows.append({"model": nm, "delta_logloss_vs_base": delta,
                      "ci_lo": lo, "ci_hi": hi, "beats_base": bool(hi < 0)})
    return pl.DataFrame(mrows), pl.DataFrame(drows)


def auc_by_slice(y, p, mask: np.ndarray) -> tuple[float, int, float]:
    """(ROC-AUC, n, base_rate) on the subset selected by a boolean mask."""
    yy, pp = np.asarray(y, float)[mask], np.asarray(p, float)[mask]
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0, float("nan")
    return _metric(yy, pp, "roc_auc"), n, float(yy.mean())
