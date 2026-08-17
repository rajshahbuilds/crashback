#!/usr/bin/env python3
"""Figures: one-year return distribution by market-cap range (feature-correlation section).

Joins the survivorship-safe one-year outcome to each event's point-in-time market cap, splits into
size ranges, and renders one baseline-style return histogram per range (identical structure to the
Model 0 histograms). Shows how the distribution's shape shifts with company size.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_by_marketcap.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

# (label for the figure, file, lo $M inclusive, hi $M exclusive)
RANGES = [
    ("micro-cap (< $100M)", "recovery_hist_mcap_micro.pdf", 0, 100),
    ("small-cap ($100M–$1B)", "recovery_hist_mcap_small.pdf", 100, 1_000),
    ("mid/large-cap ($1B–$10B)", "recovery_hist_mcap_midlarge.pdf", 1_000, 10_000),
    ("mega-cap (> $10B)", "recovery_hist_mcap_mega.pdf", 10_000, 1e12),
]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ret = one_year_returns(cfg)
    mc = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "market_cap")
    d = ret.join(mc, on="event_id").filter(
        pl.col("market_cap").is_not_null() & (pl.col("market_cap") > 0))

    spec = {**ONE_YEAR, "label": "one-year"}
    for label, file, lo, hi in RANGES:
        r = d.filter((pl.col("market_cap") >= lo) & (pl.col("market_cap") < hi))["ret"].to_numpy()
        med, mean, p_earn, n = return_histogram(r, spec, figdir / file, title=label)
        print(f"{label:26s}: n={n:6d}  P(earn)={p_earn:.3f}  median={med:+.3f}  "
              f"mean={mean:+.3f}  -> {file}")


if __name__ == "__main__":
    main()
