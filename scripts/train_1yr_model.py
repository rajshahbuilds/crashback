#!/usr/bin/env python3
"""Train/evaluate the paper's one-year recovery model (P(earn money in a year)).

Thin CLI over crashback.analysis.recovery_model. Target y = 1 if the survivorship-safe one-year
total return after the crash-day close is positive. Base features are the ten screened in the
feature-correlation section; --regime adds five point-in-time market-regime features.

Chronological split with a one-year embargo:
    train <= 2017-12-31 | validation 2019-2020 | test 2022+   (2018, 2021 embargo)

Tuning: predeclared grid scored on VALIDATION only; test touched once. Compared against the
Model 0 base rate (train prevalence as a constant forecast).

Run: PYTHONPATH=src .venv/bin/python scripts/train_1yr_model.py [--regime]
"""
from __future__ import annotations

import argparse

from crashback.analysis.recovery_model import assemble, fit_predict
from crashback.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", action="store_true", help="add point-in-time regime features")
    args = ap.parse_args()

    cfg = load_config()
    print("assembling features ...")
    df = assemble(cfg)
    r = fit_predict(df, cfg, regime=args.regime)

    print(f"\nregime={'on' if args.regime else 'off'}  n_features={len(r.features)}")
    print(r.sizes)
    print(f"\nbest params: {r.best_params}  best_iteration={r.best_iteration}")

    print("\n===== TEST =====")
    print(f"n={r.metrics['n']}  actual P(earn)={r.metrics['prevalence']:.3f}  "
          f"mean_pred={r.metrics['mean_pred']:.3f}")
    print(f"{'metric':10s}{'Model 0':>12s}{'XGBoost':>12s}")
    for k in ("log_loss", "brier", "roc_auc", "pr_auc"):
        print(f"{k:10s}{r.m0[k]:12.4f}{r.metrics[k]:12.4f}")

    print(f"\nECE={r.ece:.4f}")
    print(r.calib.select("bucket", "n", "mean_pred", "actual_rate"))
    print("\nfeature importance (gain):")
    print(r.importance.select("feature", "gain_frac", "split"))


if __name__ == "__main__":
    main()
