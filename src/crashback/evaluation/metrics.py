"""Probability-prediction metrics: discrimination, proper scoring, and calibration.

The project targets *calibrated probabilities* (CLAUDE.md §21), so alongside discrimination
(ROC-AUC, PR-AUC) we report proper scoring rules (log loss, Brier) and an explicit reliability
table with expected calibration error (ECE). All functions take a predicted ``P(hit)`` vector
and a binary outcome vector, both already restricted to *determined* events (non-null label).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def binary_metrics(y_true, p_pred) -> dict:
    """Discrimination + proper-scoring metrics for a probability forecast.

    ROC-AUC and PR-AUC are undefined when the outcome has a single class; ROC-AUC is also
    undefined when every predicted score is identical (Model 0). Those cases return NaN
    rather than a misleading 0.5.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-15, 1 - 1e-15)
    n = int(y.shape[0])
    prevalence = float(y.mean()) if n else float("nan")
    both_classes = 0.0 < prevalence < 1.0
    out = {
        "n": n,
        "prevalence": prevalence,
        "mean_pred": float(p.mean()) if n else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if n else float("nan"),
        "brier": float(brier_score_loss(y, p)) if n else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y, p)) if both_classes and np.unique(p).size > 1 else float("nan")
    )
    return out


def calibration_table(y_true, p_pred, bins: int = 10) -> tuple[pl.DataFrame, float]:
    """Reliability table over equal-width probability bins + expected calibration error.

    Returns (table, ece). The table has one row per bin: (bucket, lo, hi, n, mean_pred,
    actual_rate). ECE = sum over bins of (n_bin / N) * |actual_rate - mean_pred|; empty bins
    contribute nothing. A well-calibrated model has mean_pred ~= actual_rate in every bin.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 0.0, 1.0)
    n = int(y.shape[0])
    edges = np.linspace(0.0, 1.0, bins + 1)
    # digitize on interior edges so p in [lo, hi); the top bin includes p == 1.0.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)

    rows: list[dict] = []
    ece = 0.0
    for b in range(bins):
        m = idx == b
        cnt = int(m.sum())
        mp = float(p[m].mean()) if cnt else None
        ar = float(y[m].mean()) if cnt else None
        if cnt and n:
            ece += cnt / n * abs(ar - mp)
        rows.append({
            "bucket": b + 1, "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": cnt, "mean_pred": mp, "actual_rate": ar,
        })
    return pl.DataFrame(rows), float(ece)
