#!/usr/bin/env python3
"""Do the big 'good' descriptive slices survive out of sample?

For each large, positive-looking slice from the feature section (cheap P/E, low/high beta,
profitable, moderate growers), compare its all-history stats to its 2022+ test-period stats,
against the contemporaneous base rate and the model's top decile. Absolute levels collapse
with the bad vintage; the honest question is whether a slice still beats the 2022+ base rate.

Run: PYTHONPATH=src .venv/bin/python scripts/simple_rules_oos.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.analysis.recovery_model import assemble, fit_predict, split_chrono
from crashback.config import load_config

C = pl.col
SLICES = [
    ("cheap P/E 0-15",        (C("pe") > 0) & (C("pe") <= 15)),
    ("EBITDA margin 10-25%",  (C("ebitda_margin") > 0.10) & (C("ebitda_margin") <= 0.25)),
    ("EBITDA margin > 25%",    C("ebitda_margin") > 0.25),
    ("beta 0.8-1.2 (market)", (C("beta") >= 0.8) & (C("beta") <= 1.2)),
    ("beta 1.2-1.8 (high)",   (C("beta") > 1.2) & (C("beta") <= 1.8)),
    ("revenue growth 0-10%",  (C("revenue_growth_yoy") > 0) & (C("revenue_growth_yoy") <= 0.10)),
    ("revenue growth 10-30%", (C("revenue_growth_yoy") > 0.10) & (C("revenue_growth_yoy") <= 0.30)),
]


def line(tag, ret, base_earn=None):
    if len(ret) == 0:
        print(f"  {tag:22s} n=0")
        return
    earn = (ret > 0).mean()
    lift = "" if base_earn is None else f"  lift={earn - base_earn:+.3f}"
    print(f"  {tag:22s} n={len(ret):6d}  earn={earn:.3f}  "
          f"median={np.median(ret):+.3f}  mean={ret.mean():+.3f}{lift}")


def main():
    cfg = load_config()
    df = split_chrono(assemble(cfg))
    test = df.filter(C("split") == "test")
    base = test["ret"].to_numpy()
    base_earn = (base > 0).mean()

    print("2022+ TEST base rate (all crashes):")
    line("base", base)

    r = fit_predict(df.drop("split"), cfg, regime=True)
    t = r.test.join(df.select("event_id", "ret"), on="event_id")
    p = t["p"].to_numpy()
    top = np.argsort(p)[-len(p) // 10:]
    line("model top decile", t["ret"].to_numpy()[top], base_earn)

    for name, expr in SLICES:
        print(f"\n{name}:")
        line("all history", df.filter(expr)["ret"].to_numpy())
        line("2022+ test", test.filter(expr)["ret"].to_numpy(), base_earn)


if __name__ == "__main__":
    main()
