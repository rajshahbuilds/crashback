#!/usr/bin/env python3
"""Figures: one-year return distribution by prior-crash count (feature-correlation section).

Splits crash events into cohorts by how many crashes the security had in the preceding 20 trading
days (fresh / 1 prior / 2+ prior) and renders one baseline-style one-year return histogram per
cohort. Today's crash never counts as its own prior.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_by_priorcrash.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

# (panel title, file, lo, hi) on prior_crash_count_20d
COHORTS = [
    ("fresh crash (no prior in 20d)", "recovery_hist_prior_0.pdf", 0, 0),
    ("1 prior crash (within 20d)", "recovery_hist_prior_1.pdf", 1, 1),
    ("2+ prior crashes (within 20d)", "recovery_hist_prior_2plus.pdf", 2, 10_000),
]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ret = one_year_returns(cfg)
    pc = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "prior_crash_count_20d")
    d = ret.join(pc, on="event_id").with_columns(pl.col("prior_crash_count_20d").fill_null(0))

    for title, file, lo, hi in COHORTS:
        c = pl.col("prior_crash_count_20d")
        band = d.filter((c >= lo) & (c <= hi))
        med, mean, p_earn, n = return_histogram(band["ret"].to_numpy(), ONE_YEAR,
                                                 figdir / file, title=title)
        print(f"{title:32s}: n={n:6d}  P(earn)={p_earn:.3f}  median={med:+.3f}  mean={mean:+.3f}")


if __name__ == "__main__":
    main()
