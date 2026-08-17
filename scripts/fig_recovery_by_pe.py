#!/usr/bin/env python3
"""Figures: one-year return distribution by valuation (P/E) at the crash.

Splits events by point-in-time P/E---negative earnings (unprofitable) as its own category, then
cheap/fair/expensive among profitable firms---and renders one baseline-style one-year return
histogram per bucket. P/E is the crash-day market cap over trailing-twelve-month net income;
events without fundamentals or with zero earnings (null P/E) are excluded.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_by_pe.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

PE = pl.col("pe")
# (panel title, file, filter expression on P/E)
BUCKETS = [
    ("unprofitable (P/E < 0)", "recovery_hist_pe_unprofitable.pdf", PE < 0),
    ("cheap (P/E 0–15)", "recovery_hist_pe_cheap.pdf", (PE > 0) & (PE <= 15)),
    ("fair (P/E 15–30)", "recovery_hist_pe_fair.pdf", (PE > 15) & (PE <= 30)),
    ("expensive (P/E > 30)", "recovery_hist_pe_expensive.pdf", PE > 30),
]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ret = one_year_returns(cfg)
    pe = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "pe")
    d = ret.join(pe, on="event_id")

    for title, file, expr in BUCKETS:
        band = d.filter(expr)
        med, mean, p_earn, n = return_histogram(band["ret"].to_numpy(), ONE_YEAR,
                                                 figdir / file, title=title)
        print(f"{title:24s}: n={n:6d}  P(earn)={p_earn:.3f}  median={med:+.3f}  mean={mean:+.3f}")


if __name__ == "__main__":
    main()
