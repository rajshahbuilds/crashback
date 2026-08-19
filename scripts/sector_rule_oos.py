#!/usr/bin/env python3
"""Does the 'crashed with or better than sector' rule survive out of sample?

The descriptive histogram (all history) shows rel_sector > -3% recovering 75% / median +34%.
This checks the SAME rule on the 2022+ test period the model is judged on, and compares it to
the model's top decile there --- apples to apples.

Run: PYTHONPATH=src .venv/bin/python scripts/sector_rule_oos.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from crashback.analysis.recovery_model import assemble, fit_predict, split_chrono
from crashback.config import load_config


def stats(ret: np.ndarray) -> str:
    if len(ret) == 0:
        return "n=0"
    return (f"n={len(ret):6d}  P(earn)={(ret > 0).mean():.3f}  "
            f"median={np.median(ret):+.3f}  mean={ret.mean():+.3f}")


def main():
    cfg = load_config()
    df = split_chrono(assemble(cfg))
    rule = pl.col("rel_sector") > -0.03  # "crashed with or better than sector"

    print("=== 'crashed with or better than sector' (rel_sector > -3%) ===")
    print(f"all history      : {stats(df.filter(rule)['ret'].to_numpy())}")
    test = df.filter(pl.col('split') == 'test')
    print(f"2022+ test only  : {stats(test.filter(rule)['ret'].to_numpy())}")
    print(f"2022+ all crashes: {stats(test['ret'].to_numpy())}")

    # model top decile on the same test period
    r = fit_predict(df.drop('split'), cfg, regime=True)
    t = r.test.join(df.select('event_id', 'ret', 'rel_sector'), on='event_id')
    p = t['p'].to_numpy()
    top = np.argsort(p)[-len(p) // 10:]
    print("\n=== model top decile (2022+) ===")
    print(f"top decile       : {stats(t['ret'].to_numpy()[top])}")

    # overlap: how much of the top decile is itself 'crashed with sector'?
    top_relsec = t['rel_sector'].to_numpy()[top]
    frac = np.mean(top_relsec > -0.03)
    print(f"\n{frac:.1%} of the model's top decile already satisfies the sector rule "
          f"(vs {(t['rel_sector'].to_numpy() > -0.03).mean():.1%} of all test crashes)")


if __name__ == "__main__":
    main()
