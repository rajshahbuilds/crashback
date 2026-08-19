#!/usr/bin/env python3
"""Walk-forward across vintages: is the value edge (and the model's edge) robust, or a 2022+ fluke?

For each 3-year test vintage, train the model only on crashes before it (with a one-year embargo)
and evaluate two strategies against that vintage's OWN base rate:
  - cheap-P/E screen (0 < P/E <= 15), no training
  - XGBoost top decile (15 features, fixed hyperparameters -- the set that won the main tuning;
    not re-grid-searched per window, a standard walk-forward simplification)

Reports per vintage: base recovery rate, then each strategy's earn rate / median / mean return and
the earn-rate lift over base. A robust signal beats base in most/all vintages.

Run: PYTHONPATH=src .venv/bin/python scripts/walk_forward.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.analysis.recovery_model import BASE_FEATURES, REGIME_FEATURES, assemble
from crashback.config import load_config
from crashback.models.xgb import base_params

COLS = BASE_FEATURES + REGIME_FEATURES
# fixed hyperparameters: the override that won the chronological grid search
WON = {"max_depth": 7, "min_child_weight": 10.0, "colsample_bytree": 0.7, "reg_lambda": 0.0}
NUM_ROUNDS = 400
CHEAP_PE = (pl.col("pe") > 0) & (pl.col("pe") <= 15)
# (test-window start year, end year); train ends 2 calendar years before (one-year embargo)
VINTAGES = [(2007, 2009), (2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024)]


def _matrix(frame: pl.DataFrame) -> np.ndarray:
    a = frame.select(COLS).to_numpy().astype(float)
    a[np.isinf(a)] = np.nan
    return a


def summ(ret: np.ndarray, base_earn=None) -> str:
    if len(ret) == 0:
        return "n=0"
    earn = (ret > 0).mean()
    lift = "" if base_earn is None else f"  lift={earn - base_earn:+.3f}"
    return (f"n={len(ret):5d}  earn={earn:.3f}  median={np.median(ret):+.3f}  "
            f"mean={ret.mean():+.3f}{lift}")


def main():
    cfg = load_config()
    df = assemble(cfg)
    params = {**base_params(cfg.models.xgboost, 42), **WON}
    print(f"walk-forward: fixed params {WON}, {NUM_ROUNDS} rounds, target = P(earn money in 1yr)\n")

    for start, end in VINTAGES:
        test = df.filter((pl.col("crash_date") >= date(start, 1, 1))
                         & (pl.col("crash_date") <= date(end, 12, 31)))
        train = df.filter(pl.col("crash_date") <= date(start - 2, 12, 31))
        if test.height == 0 or train.height < 5000:
            print(f"{start}-{end}: insufficient data (train={train.height}, test={test.height})")
            continue

        base = test["ret"].to_numpy()
        base_earn = (base > 0).mean()

        dtr = xgb.DMatrix(_matrix(train), label=train["y"].to_numpy(), missing=np.nan)
        booster = xgb.train(params, dtr, num_boost_round=NUM_ROUNDS)
        p = booster.predict(xgb.DMatrix(_matrix(test), missing=np.nan))
        top = np.argsort(p)[-test.height // 10:]

        print(f"===== vintage {start}-{end}  (train<= {start - 2}, n_train={train.height}) =====")
        print(f"  base (all crashes)   {summ(base)}")
        print(f"  cheap P/E screen     {summ(test.filter(CHEAP_PE)['ret'].to_numpy(), base_earn)}")
        print(f"  model top decile     {summ(test['ret'].to_numpy()[top], base_earn)}\n")


if __name__ == "__main__":
    main()
