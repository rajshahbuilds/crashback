#!/usr/bin/env python3
"""Production training loop: expanding-window walk-forward + a final deployable model.

Two things:
  1. Honest pooled out-of-sample evaluation. For each year Y in [start, end], train on every crash
     with outcome fully before year Y (one-year embargo: crash_date <= Y-2), predict year Y, and
     collect the predictions. Concatenating all years gives ONE out-of-sample set spanning many
     vintages -- a robust performance estimate that no single test period can bias. Metrics:
     AUC / log loss / Brier / ECE, a calibration table, and decile earn/median/mean ROI.
  2. A final production model trained on ALL events with known one-year outcomes (what you would
     deploy to score new crashes). Saved to data/models/, with gain importance.

Fixed hyperparameters (the set that won the chronological grid search); not re-tuned per window --
a standard walk-forward simplification. Target y = P(earn money in a year).

Run: PYTHONPATH=src .venv/bin/python scripts/train_production.py [--start 2005] [--end 2024]
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.analysis.recovery_model import BASE_FEATURES, REGIME_FEATURES, assemble
from crashback.config import load_config
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models.xgb import base_params, importance_table

COLS = BASE_FEATURES + REGIME_FEATURES
WON = {"max_depth": 7, "min_child_weight": 10.0, "colsample_bytree": 0.7, "reg_lambda": 0.0}
NUM_ROUNDS = 350
MIN_TRAIN = 5000


def _matrix(frame: pl.DataFrame) -> np.ndarray:
    a = frame.select(COLS).to_numpy().astype(float)
    a[np.isinf(a)] = np.nan
    return a


def _dmatrix(frame: pl.DataFrame, labelled: bool) -> xgb.DMatrix:
    label = frame["y"].to_numpy() if labelled else None
    return xgb.DMatrix(_matrix(frame), label=label, feature_names=COLS, missing=np.nan)


def walk_forward_oos(df: pl.DataFrame, params: dict, start: int, end: int) -> pl.DataFrame:
    """Expanding-window: train on <= Y-2, predict year Y, for each Y; return pooled predictions."""
    preds = []
    for Y in range(start, end + 1):
        test = df.filter(pl.col("crash_date").dt.year() == Y)
        train = df.filter(pl.col("crash_date") <= date(Y - 2, 12, 31))
        if test.height == 0 or train.height < MIN_TRAIN:
            continue
        booster = xgb.train(params, _dmatrix(train, True), num_boost_round=NUM_ROUNDS)
        p = booster.predict(_dmatrix(test, False))
        preds.append(test.select("event_id", "crash_date", "y", "ret").with_columns(
            p=pl.Series(p), year=pl.lit(Y)))
        top = test["ret"].to_numpy()[np.argsort(p)[-test.height // 10:]]
        print(f"  {Y}: train={train.height:6d}  test={test.height:5d}  "
              f"top-decile earn={_earn(top):.3f}")
    return pl.concat(preds)


def _earn(ret: np.ndarray) -> float:
    return float((ret > 0).mean()) if len(ret) else float("nan")


def decile_table(pooled: pl.DataFrame) -> pl.DataFrame:
    p = pooled["p"].to_numpy()
    ret = pooled["ret"].to_numpy()
    order = np.argsort(p)
    rows = []
    for i, g in enumerate(np.array_split(order, 10), 1):
        rows.append({"decile": i, "n": len(g), "earn": _earn(ret[g]),
                     "median_roi": float(np.median(ret[g])), "mean_roi": float(ret[g].mean())})
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2024)
    args = ap.parse_args()

    cfg = load_config()
    params = {**base_params(cfg.models.xgboost, 42), **WON}
    print("assembling features ...")
    df = assemble(cfg)

    print(f"\n=== expanding-window walk-forward {args.start}-{args.end} ===")
    pooled = walk_forward_oos(df, params, args.start, args.end)

    yy = pooled["y"].to_numpy()
    pp = pooled["p"].to_numpy()
    m = binary_metrics(yy, pp)
    _, ece = calibration_table(yy, pp, cfg.models.calibration_bins)
    print(f"\n=== POOLED out-of-sample ({pooled.height} events, {args.start}-{args.end}) ===")
    print(f"actual P(earn)={m['prevalence']:.3f}  AUC={m['roc_auc']:.4f}  "
          f"log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"ECE={ece:.4f}")
    print("\ndecile earn / median ROI / mean ROI:")
    print(decile_table(pooled))

    proc = cfg.paths.resolve("data_processed")
    pooled.write_parquet(proc / "pred_walkforward_oos.parquet")
    print(f"\nwrote {proc / 'pred_walkforward_oos.parquet'}")

    # final production model: train on ALL events with known outcomes
    print(f"\n=== production model: train on all {df.height} labelled events ===")
    prod = xgb.train(params, _dmatrix(df, True), num_boost_round=NUM_ROUNDS)
    models_dir = cfg.paths.resolve("data_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    prod.save_model(str(models_dir / "recovery_prod.json"))
    print(f"wrote {models_dir / 'recovery_prod.json'}")
    print("\nfeature importance (gain):")
    print(importance_table(prod, COLS).select("feature", "gain_frac"))


if __name__ == "__main__":
    main()
