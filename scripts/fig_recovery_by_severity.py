#!/usr/bin/env python3
"""Figures: one-year return distribution by crash severity (feature-correlation section).

Buckets crash events by the crash-day return into fixed severity bands and renders one
baseline-style one-year return histogram per band. Shows how the recovery distribution shifts as
the initial drop deepens.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_by_severity.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

# (panel title, file, lo exclusive, hi inclusive) on crash_return (negative)
BANDS = [
    ("crash of 10–15%", "recovery_hist_sev_10_15.pdf", -0.15, -0.10),
    ("crash of 15–20%", "recovery_hist_sev_15_20.pdf", -0.20, -0.15),
    ("crash of 20–30%", "recovery_hist_sev_20_30.pdf", -0.30, -0.20),
    ("crash worse than 30%", "recovery_hist_sev_30plus.pdf", -1.01, -0.30),
]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ret = one_year_returns(cfg)
    cr = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "crash_return")
    d = ret.join(cr, on="event_id")

    spec = {**ONE_YEAR, "label": "one-year"}
    for title, file, lo, hi in BANDS:
        band = d.filter((pl.col("crash_return") > lo) & (pl.col("crash_return") <= hi))
        r = band["ret"].to_numpy()
        med, mean, p_earn, n = return_histogram(r, spec, figdir / file, title=title)
        print(f"{title:22s}: n={n:6d}  P(earn)={p_earn:.3f}  median={med:+.3f}  "
              f"mean={mean:+.3f}  -> {file}")


if __name__ == "__main__":
    main()
