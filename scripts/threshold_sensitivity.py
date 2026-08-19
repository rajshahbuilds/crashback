#!/usr/bin/env python3
"""Crash-threshold sensitivity: re-run the pooled walk-forward at -10/-15/-20/-30% crash depths.

A deeper crash is a subset of the -10% events with a more negative crash-day return, so we subset
the assembled events by crash_return and re-run the expanding-window walk-forward at each depth
(same features, same target, same 2005-2024 pooling). Reports how the base recovery rate, the
return distribution, and the model's out-of-sample edge move as crashes get deeper.

Caveat: path features (e.g. prior_crash_count_20d) stay defined at the -10% threshold; only the
event set deepens. Hyperparameters are the fixed set from the main tuning.

Run: PYTHONPATH=src .venv/bin/python scripts/threshold_sensitivity.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.analysis.recovery_model import BASE_FEATURES, REGIME_FEATURES, assemble
from crashback.config import load_config
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models.xgb import base_params

COLS = BASE_FEATURES + REGIME_FEATURES
WON = {"max_depth": 7, "min_child_weight": 10.0, "colsample_bytree": 0.7, "reg_lambda": 0.0}
NUM_ROUNDS = 350
MIN_TRAIN = 3000
THRESHOLDS = [0.10, 0.15, 0.20, 0.30]
START, END = 2005, 2024


def _dmatrix(frame: pl.DataFrame, labelled: bool) -> xgb.DMatrix:
    a = frame.select(COLS).to_numpy().astype(float)
    a[np.isinf(a)] = np.nan
    label = frame["y"].to_numpy() if labelled else None
    return xgb.DMatrix(a, label=label, feature_names=COLS, missing=np.nan)


def walk_forward(sub: pl.DataFrame, params: dict) -> pl.DataFrame:
    preds = []
    for Y in range(START, END + 1):
        test = sub.filter(pl.col("crash_date").dt.year() == Y)
        train = sub.filter(pl.col("crash_date") <= date(Y - 2, 12, 31))
        if test.height == 0 or train.height < MIN_TRAIN:
            continue
        booster = xgb.train(params, _dmatrix(train, True), num_boost_round=NUM_ROUNDS)
        p = booster.predict(_dmatrix(test, False))
        preds.append(test.select("y", "ret").with_columns(p=pl.Series(p)))
    return pl.concat(preds) if preds else pl.DataFrame()


def main():
    cfg = load_config()
    params = {**base_params(cfg.models.xgboost, 42), **WON}
    print("assembling features ...")
    df = assemble(cfg)

    print(f"\n{'thresh':>7} {'n_events':>9} {'base_earn':>10} {'base_med':>9} "
          f"{'AUC':>6} {'ECE':>6} {'bot_earn':>9} {'top_earn':>9} {'top_med':>8}")
    for thr in THRESHOLDS:
        sub = df.filter(pl.col("crash_return") <= -thr)
        pooled = walk_forward(sub, params)
        if pooled.height == 0:
            print(f"{-thr:>7.0%}  insufficient data")
            continue
        y, p, ret = (pooled["y"].to_numpy(), pooled["p"].to_numpy(), pooled["ret"].to_numpy())
        m = binary_metrics(y, p)
        _, ece = calibration_table(y, p, cfg.models.calibration_bins)
        g = np.array_split(np.argsort(p), 10)
        bot, top = ret[g[0]], ret[g[-1]]
        print(f"{-thr:>7.0%} {pooled.height:>9d} {(ret > 0).mean():>10.3f} "
              f"{np.median(ret):>+9.3f} {m['roc_auc']:>6.3f} {ece:>6.3f} "
              f"{(bot > 0).mean():>9.3f} {(top > 0).mean():>9.3f} {np.median(top):>+8.3f}")


if __name__ == "__main__":
    main()
