#!/usr/bin/env python3
"""Model performance on 2022+ crashes only, by crash depth (-10/-15/-20/-30%).

The 2022-2024 predictions in the pooled walk-forward are already out-of-sample (each year was
trained only on data through Y-2), so no refit is needed: filter to those years, then to each
crash depth, and report the model's edge. This is the hard, recent vintage -- no 2008/2020
rebound inflation.

Run: PYTHONPATH=src .venv/bin/python scripts/perf_2022_by_threshold.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.config import load_config
from crashback.evaluation.metrics import binary_metrics, calibration_table

THRESHOLDS = [0.10, 0.15, 0.20, 0.30]


def main():
    cfg = load_config()
    proc = cfg.paths.resolve("data_processed")
    pred = pl.read_parquet(proc / "pred_walkforward_oos.parquet")
    ev = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).select("event_id", "crash_return")
    d = pred.join(ev, on="event_id").filter(pl.col("year") >= 2022)

    print("2022-2024 crashes only, out-of-sample:\n")
    print(f"{'thresh':>7} {'n':>6} {'base_earn':>10} {'base_med':>9} {'AUC':>6} {'ECE':>6} "
          f"{'top_n':>6} {'top_earn':>9} {'top_med':>8} {'top_mean':>9}")
    for thr in THRESHOLDS:
        sub = d.filter(pl.col("crash_return") <= -thr)
        y, p, ret = sub["y"].to_numpy(), sub["p"].to_numpy(), sub["ret"].to_numpy()
        m = binary_metrics(y, p)
        _, ece = calibration_table(y, p, cfg.models.calibration_bins)
        top = ret[np.argsort(p)[-max(1, sub.height // 10):]]
        print(f"{-thr:>7.0%} {sub.height:>6d} {(ret > 0).mean():>10.3f} "
              f"{np.median(ret):>+9.3f} {m['roc_auc']:>6.3f} {ece:>6.3f} "
              f"{len(top):>6d} {(top > 0).mean():>9.3f} {np.median(top):>+8.3f} "
              f"{top.mean():>+9.3f}")


if __name__ == "__main__":
    main()
