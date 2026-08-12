"""Metrics: proper-scoring correctness, degenerate guards, and calibration/ECE."""
from __future__ import annotations

import math

from crashback.evaluation.metrics import binary_metrics, calibration_table


def test_perfect_probabilities_score_zero():
    y = [1, 0, 1, 0]
    p = [1.0, 0.0, 1.0, 0.0]
    m = binary_metrics(y, p)
    assert m["brier"] < 1e-6
    assert m["log_loss"] < 1e-6          # clipped, so ~0 not exactly 0
    assert m["roc_auc"] == 1.0
    assert m["prevalence"] == 0.5


def test_constant_prediction_auc_is_nan_but_brier_defined():
    # Model-0-style constant forecast: ROC-AUC undefined (all tied), Brier still valid.
    y = [1, 0, 1, 1, 0]
    p = [0.6] * 5
    m = binary_metrics(y, p)
    assert math.isnan(m["roc_auc"])
    assert abs(m["brier"] - (0.6 - 0.6) ** 2 * 0) < 1  # just assert it computes
    assert not math.isnan(m["brier"])
    assert m["mean_pred"] == 0.6


def test_single_class_auc_and_prauc_nan():
    y = [1, 1, 1]
    p = [0.9, 0.8, 0.7]
    m = binary_metrics(y, p)
    assert math.isnan(m["roc_auc"])
    assert math.isnan(m["pr_auc"])


def test_brier_matches_manual():
    y = [1, 0]
    p = [0.7, 0.2]
    m = binary_metrics(y, p)
    manual = ((0.7 - 1) ** 2 + (0.2 - 0) ** 2) / 2
    assert abs(m["brier"] - manual) < 1e-9


def test_calibration_perfectly_calibrated_has_zero_ece():
    # Two bins, prediction equals empirical rate in each -> ECE == 0.
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # bin ~0.1 rate
    p = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # actual rate 0.2 == mean_pred 0.2
    table, ece = calibration_table(y, p, bins=10)
    assert ece < 1e-9
    row = table.filter(table["n"] > 0).row(0, named=True)
    assert abs(row["mean_pred"] - row["actual_rate"]) < 1e-9


def test_calibration_miscalibration_positive_ece():
    y = [0] * 10                          # true rate 0
    p = [0.9] * 10                        # predicts 0.9 -> badly miscalibrated
    _, ece = calibration_table(y, p, bins=10)
    assert abs(ece - 0.9) < 1e-9


def test_calibration_covers_all_events():
    y = [1, 0, 1, 0, 1]
    p = [0.05, 0.15, 0.55, 0.95, 1.0]     # includes p == 1.0 (top bin)
    table, _ = calibration_table(y, p, bins=10)
    assert table["n"].sum() == len(y)
