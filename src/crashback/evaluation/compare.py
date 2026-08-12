"""Incremental-value comparison with clustered-bootstrap uncertainty (STU-61).

Crash events for the same security cluster in time and are correlated (CLAUDE.md §22), so a
naive bootstrap that resamples individual events would understate uncertainty on metric
*deltas*. We instead resample **securities** (clusters) with replacement — a block bootstrap
that keeps each security's events together — and use the **same** resample across every model
(paired), so the delta distribution reflects the correlation between models on shared events.

For each information-source increment (from-rung → to-rung) we report the point delta, a 95%
percentile CI, and the fraction of resamples in which the metric improved. An increment is
"material" only if its CI excludes 0.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Whether a smaller value is better (proper scores) or larger (discrimination).
LOWER_BETTER: dict[str, bool] = {"log_loss": True, "brier": True, "roc_auc": False}


def _metric(y: np.ndarray, p: np.ndarray, name: str) -> float:
    if len(np.unique(y)) < 2 and name == "roc_auc":
        return float("nan")
    if name == "log_loss":
        return float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
    if name == "brier":
        return float(brier_score_loss(y, p))
    if name == "roc_auc":
        return float(roc_auc_score(y, p))
    raise ValueError(f"unknown metric {name!r}")


def _cluster_rows(clusters: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by cluster id (so a resample can pull a security's rows together)."""
    inv = np.unique(clusters, return_inverse=True)[1]
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(counts))]


def paired_cluster_bootstrap(
    y, preds: dict[str, np.ndarray], clusters, increments,
    *, metrics=("log_loss", "roc_auc"), n_boot: int = 500, seed: int = 42,
) -> tuple[pl.DataFrame, dict]:
    """Bootstrap metric-delta CIs for each increment, resampling by cluster.

    ``preds`` maps rung-name → predicted P(hit) on the evaluation set. ``increments`` is a list
    of (from_rung, to_rung, source_label). Returns (deltas_df, point_metrics) where deltas_df
    has one row per (increment, metric): source, from, to, metric, delta, ci_lo, ci_hi,
    p_improve, material.
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    rows = _cluster_rows(np.asarray(clusters))
    k = len(rows)

    point = {name: {m: _metric(y, preds[name], m) for m in metrics} for name in preds}

    boot: dict[tuple[str, str, str], list[float]] = {
        (frm, to, m): [] for (frm, to, _) in increments for m in metrics
    }
    for _ in range(n_boot):
        samp = rng.integers(0, k, k)
        idx = np.concatenate([rows[c] for c in samp])
        yb = y[idx]
        mv = {(name, m): _metric(yb, preds[name][idx], m) for name in preds for m in metrics}
        for (frm, to, _) in increments:
            for m in metrics:
                boot[(frm, to, m)].append(mv[(to, m)] - mv[(frm, m)])

    out: list[dict] = []
    for (frm, to, label) in increments:
        for m in metrics:
            arr = np.array(boot[(frm, to, m)], dtype=float)
            arr = arr[~np.isnan(arr)]
            lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) \
                if arr.size else (float("nan"), float("nan"))
            delta = point[to][m] - point[frm][m]
            improve = float(np.mean(arr < 0)) if LOWER_BETTER[m] else float(np.mean(arr > 0))
            # "material" = the 95% CI excludes 0 in EITHER direction (a real effect, help or
            # hurt); the sign of `delta` then says which.
            material = bool(lo > 0 or hi < 0)
            out.append({
                "source": label, "from": frm, "to": to, "metric": m,
                "delta": delta, "ci_lo": lo, "ci_hi": hi,
                "p_improve": improve, "material": material,
            })
    return pl.DataFrame(out), point
