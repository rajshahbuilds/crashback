#!/usr/bin/env python3
"""Does ranking by P(earn money) also rank by ROI? Mean/median one-year return per predicted
decile, for the honest chronological +regime model. P(earn) is a hit-rate; ROI needs magnitude.

Run: PYTHONPATH=src .venv/bin/python scripts/roi_by_decile.py
"""
from __future__ import annotations

import numpy as np

from crashback.analysis.recovery_model import assemble, fit_predict
from crashback.config import load_config


def main():
    cfg = load_config()
    df = assemble(cfg)  # has event_id, ret (one-year return), y
    r = fit_predict(df, cfg, regime=True)
    t = r.test.join(df.select("event_id", "ret"), on="event_id")
    p = t["p"].to_numpy()
    ret = t["ret"].to_numpy()
    y = t["y"].to_numpy().astype(float)
    order = np.argsort(p)
    groups = np.array_split(order, 10)
    print(f"\n===== chrono +regime  (n={len(p)}) =====")
    print(f"{'decile':>6} {'n':>6} {'P(earn)':>8} {'mean ret':>9} {'median ret':>11}")
    for i, g in enumerate(groups, 1):
        print(f"{i:>6} {len(g):>6} {y[g].mean():>8.1%} "
              f"{ret[g].mean():>9.1%} {np.median(ret[g]):>11.1%}")
    print(f"{'ALL':>6} {len(ret):>6} {y.mean():>8.1%} "
          f"{ret.mean():>9.1%} {np.median(ret):>11.1%}")


if __name__ == "__main__":
    main()
