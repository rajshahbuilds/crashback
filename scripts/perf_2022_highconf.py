#!/usr/bin/env python3
"""2022+ model performance at high confidence (fixed predicted-probability thresholds).

Filters the out-of-sample walk-forward to 2022-2024 and reports return stats for crashes the model
rated at >= 0.7, 0.8, 0.9. Then breaks the 0.9+ bucket by crash depth. No refit (predictions are
already out-of-sample).

Run: PYTHONPATH=src .venv/bin/python scripts/perf_2022_highconf.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.config import load_config


def row(tag, ret):
    if len(ret) == 0:
        print(f"  {tag:22s} n=0")
        return
    print(f"  {tag:22s} n={len(ret):5d}  earn={(ret > 0).mean():.3f}  "
          f"median={np.median(ret):+.3f}  mean={ret.mean():+.3f}")


def main():
    cfg = load_config()
    proc = cfg.paths.resolve("data_processed")
    pred = pl.read_parquet(proc / "pred_walkforward_oos.parquet")
    ev = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).select("event_id", "crash_return")
    d = pred.join(ev, on="event_id").filter(pl.col("year") >= 2022)

    print("2022-2024 crashes, by confidence threshold (all crash depths, i.e. <= -10%):")
    for conf in (0.7, 0.8, 0.9):
        row(f"p >= {conf}", d.filter(pl.col("p") >= conf)["ret"].to_numpy())

    print("\n2022-2024, p >= 0.9, by crash depth:")
    for thr in (0.10, 0.15, 0.20, 0.30):
        sub = d.filter((pl.col("p") >= 0.9) & (pl.col("crash_return") <= -thr))
        row(f"crash <= -{int(thr * 100)}%", sub["ret"].to_numpy())


if __name__ == "__main__":
    main()
