#!/usr/bin/env python3
"""Figures: post-crash return distribution by crash severity, at 60 days and one year.

Buckets crash events by the crash-day return into fixed severity bands and renders one baseline-
style return histogram per band, at both the 60-trading-day and one-year horizons. Shows how the
recovery distribution shifts as the initial drop deepens, and how that differs by horizon.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_by_severity.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, SIXTY_DAY, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

# (panel title, key, lo exclusive, hi inclusive) on crash_return (negative)
BANDS = [
    ("crash of 10–15%", "10_15", -0.15, -0.10),
    ("crash of 15–20%", "15_20", -0.20, -0.15),
    ("crash of 20–30%", "20_30", -0.30, -0.20),
    ("crash worse than 30%", "30plus", -1.01, -0.30),
]
# (horizon trading days, display spec, filename suffix)
HORIZONS = [(252, ONE_YEAR, ""), (60, SIXTY_DAY, "_60d")]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    cr = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "crash_return")

    for horizon, spec, suffix in HORIZONS:
        d = one_year_returns(cfg, horizon).join(cr, on="event_id")
        for title, key, lo, hi in BANDS:
            band = d.filter((pl.col("crash_return") > lo) & (pl.col("crash_return") <= hi))
            out = figdir / f"recovery_hist_sev_{key}{suffix}.pdf"
            med, mean, p_earn, n = return_histogram(band["ret"].to_numpy(), spec, out, title=title)
            print(f"h={horizon:3d} {title:22s}: n={n:6d}  P(earn)={p_earn:.3f}  "
                  f"median={med:+.3f}  mean={mean:+.3f}")


if __name__ == "__main__":
    main()
