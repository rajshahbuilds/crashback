#!/usr/bin/env python3
"""One-year return histograms for the model's top and bottom predicted deciles (chronological
+regime test set), in the baseline/feature-section style. Shows that the probability ranking also
sorts ROI, and that the right-skew persists: even the top decile's median return is negative.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_decile_roi_hist.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery_model import assemble, fit_predict
from crashback.config import load_config


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    df = assemble(cfg)
    r = fit_predict(df, cfg, regime=True)
    t = r.test.join(df.select("event_id", "ret"), on="event_id")
    p = t["p"].to_numpy()
    ret = t["ret"].to_numpy()
    order = np.argsort(p)
    groups = np.array_split(order, 10)

    panels = [
        (groups[-1], "model top decile (highest predicted recovery)", "decile_top"),
        (groups[0], "model bottom decile (lowest predicted recovery)", "decile_bottom"),
    ]
    for g, title, key in panels:
        out = figdir / f"recovery_hist_{key}.pdf"
        med, mean, p_earn, n = return_histogram(ret[g], ONE_YEAR, out, title=title)
        print(f"{key:14s} n={n:5d}  P(earn)={p_earn:.3f}  median={med:+.3f}  mean={mean:+.3f}")


if __name__ == "__main__":
    main()
