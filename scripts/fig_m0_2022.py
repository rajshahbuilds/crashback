#!/usr/bin/env python3
"""Model 0 base-rate histograms (Section 3 style) restricted to crashes from 2022 onwards.

Same survivorship-safe forward returns and shared renderer as fig_m0_return_hist.py, but filtered
to crash_date >= 2022-01-01 -- the recent, rebound-free regime. Writes new files only; the
full-history Section-3 figures are untouched. (Late-2024/2025 crashes without a full forward window
are censored out, as everywhere.)

Run: PYTHONPATH=src .venv/bin/python scripts/fig_m0_2022.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from crashback.analysis.plots import ONE_YEAR, SIXTY_DAY, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

SINCE = date(2022, 1, 1)
# horizon (trading days), display spec (label overridden after the spread), output file
SPECS = [
    (60, {**SIXTY_DAY, "label": "60-day (2022 on)"}, "m0_return_hist_60d_2022.pdf"),
    (252, {**ONE_YEAR, "label": "one-year (2022 on)"}, "m0_return_hist_2022.pdf"),
]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    for horizon, spec, fname in SPECS:
        r = one_year_returns(cfg, horizon=horizon).filter(
            pl.col("crash_date") >= SINCE)["ret"].to_numpy()
        med, mean, p_earn, n = return_histogram(r, spec, figdir / fname)
        print(f"{spec['label']:20s} (h={horizon}): n={n:,}  P(earn)={p_earn:.3f}  "
              f"median={med:+.3f}  mean={mean:+.3f}  -> {fname}")


if __name__ == "__main__":
    main()
