#!/usr/bin/env python3
"""One-year return histograms by crash-depth threshold, in the Section-5 (baseline) style.

For each crash definition (-10/-15/-20/-30% single-day drop) plot the all-history one-year total
return distribution of every qualifying crash, so the shift with depth is visible: same renderer,
break-even / median / mean lines, and earns/loses-money annotation as the feature histograms.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_threshold_hist.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

THRESHOLDS = [(0.10, "thr_10"), (0.15, "thr_15"), (0.20, "thr_20"), (0.30, "thr_30")]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ret = one_year_returns(cfg)
    ev = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).select("event_id", "crash_return")
    d = ret.join(ev, on="event_id")

    for thr, key in THRESHOLDS:
        band = d.filter(pl.col("crash_return") <= -thr)
        out = figdir / f"recovery_hist_{key}.pdf"
        med, mean, p_earn, n = return_histogram(
            band["ret"].to_numpy(), ONE_YEAR, out, title=f"crash $\\leq -{int(thr * 100)}\\%$")
        print(f"{key}: n={n:6d}  P(earn)={p_earn:.3f}  median={med:+.3f}  mean={mean:+.3f}")


if __name__ == "__main__":
    main()
