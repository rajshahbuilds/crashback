#!/usr/bin/env python3
"""How the MODEL performs, by crash depth, in the Section-5 histogram shape.

For each crash definition (-10/-15/-20/-30%) run the pooled walk-forward, take the model's
top-decile picks (highest predicted recovery), and plot THEIR one-year return distribution in the
baseline style. This is the return you would have earned following the model's best calls at each
crash depth -- out-of-sample, pooled 2005-2024.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_threshold_hist.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery_model import BASE_FEATURES, REGIME_FEATURES, assemble
from crashback.config import load_config
from crashback.models.xgb import base_params

COLS = BASE_FEATURES + REGIME_FEATURES
WON = {"max_depth": 7, "min_child_weight": 10.0, "colsample_bytree": 0.7, "reg_lambda": 0.0}
NUM_ROUNDS, MIN_TRAIN, START, END = 350, 3000, 2005, 2024
THRESHOLDS = [(0.10, "thr_10"), (0.15, "thr_15"), (0.20, "thr_20"), (0.30, "thr_30")]


def _dmatrix(frame: pl.DataFrame, labelled: bool) -> xgb.DMatrix:
    a = frame.select(COLS).to_numpy().astype(float)
    a[np.isinf(a)] = np.nan
    label = frame["y"].to_numpy() if labelled else None
    return xgb.DMatrix(a, label=label, feature_names=COLS, missing=np.nan)


def top_decile_returns(sub: pl.DataFrame, params: dict) -> np.ndarray:
    """Pooled walk-forward; return the one-year returns of the model's top-decile picks."""
    preds = []
    for Y in range(START, END + 1):
        test = sub.filter(pl.col("crash_date").dt.year() == Y)
        train = sub.filter(pl.col("crash_date") <= date(Y - 2, 12, 31))
        if test.height == 0 or train.height < MIN_TRAIN:
            continue
        booster = xgb.train(params, _dmatrix(train, True), num_boost_round=NUM_ROUNDS)
        p = booster.predict(_dmatrix(test, False))
        preds.append(test.select("ret").with_columns(p=pl.Series(p)))
    pooled = pl.concat(preds)
    p, ret = pooled["p"].to_numpy(), pooled["ret"].to_numpy()
    return ret[np.argsort(p)[-pooled.height // 10:]]


def main():
    cfg = load_config()
    params = {**base_params(cfg.models.xgboost, 42), **WON}
    print("assembling features ...")
    df = assemble(cfg)
    figdir = Path("paper/figures")

    for thr, key in THRESHOLDS:
        sub = df.filter(pl.col("crash_return") <= -thr)
        top = top_decile_returns(sub, params)
        out = figdir / f"recovery_hist_{key}.pdf"
        title = f"model's top-decile picks, crash $\\leq -{int(thr * 100)}\\%$"
        med, mean, p_earn, n = return_histogram(top, ONE_YEAR, out, title=title)
        print(f"{key}: n={n:5d}  P(earn)={p_earn:.3f}  median={med:+.3f}  mean={mean:+.3f}")


if __name__ == "__main__":
    main()
