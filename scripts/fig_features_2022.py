#!/usr/bin/env python3
"""2022+ versions of every Section-4 feature histogram (same buckets, same renderer).

Regenerates each feature-correlation histogram restricted to crashes from 2022 onward, writing new
'_2022' files (the full-history Section-4 figures are untouched). Bucket edges are copied verbatim
from the per-feature scripts. Prints n and stats so small 2022+ buckets are visible.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_features_2022.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from crashback.analysis.extra_features import event_beta, event_ebitda_margin
from crashback.analysis.plots import ONE_YEAR, SIXTY_DAY, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

SINCE = date(2022, 1, 1)
C = pl.col
BIG = 1e12

# one-year buckets: (title, file key, filter expression). Nulls in the feature drop out of filter.
BUCKETS_1YR = [
    ("micro-cap (< $100M)", "mcap_micro", (C("market_cap") > 0) & (C("market_cap") < 100)),
    ("small-cap ($100M-$1B)", "mcap_small", (C("market_cap") >= 100) & (C("market_cap") < 1_000)),
    ("mid/large-cap ($1B-$10B)", "mcap_midlarge",
     (C("market_cap") >= 1_000) & (C("market_cap") < 10_000)),
    ("mega-cap (> $10B)", "mcap_mega", C("market_cap") >= 10_000),
    ("crash of 10-15%", "sev_10_15", (C("crash_return") > -0.15) & (C("crash_return") <= -0.10)),
    ("crash of 15-20%", "sev_15_20", (C("crash_return") > -0.20) & (C("crash_return") <= -0.15)),
    ("crash of 20-30%", "sev_20_30", (C("crash_return") > -0.30) & (C("crash_return") <= -0.20)),
    ("crash worse than 30%", "sev_30plus", C("crash_return") <= -0.30),
    ("fresh crash (no prior in 20d)", "prior_0", C("prior_crash_count_20d") == 0),
    ("1 prior crash (within 20d)", "prior_1", C("prior_crash_count_20d") == 1),
    ("2+ prior crashes (within 20d)", "prior_2plus", C("prior_crash_count_20d") >= 2),
    ("unprofitable (P/E < 0)", "pe_unprofitable", C("pe") < 0),
    ("cheap (P/E 0-15)", "pe_cheap", (C("pe") > 0) & (C("pe") <= 15)),
    ("fair (P/E 15-30)", "pe_fair", (C("pe") > 15) & (C("pe") <= 30)),
    ("expensive (P/E > 30)", "pe_expensive", C("pe") > 30),
    ("pre-crash 1yr return < -50%", "premom_dn", C("return_252d_pre") <= -0.5),
    ("pre-crash 1yr return -50 to 0%", "premom_mild",
     (C("return_252d_pre") > -0.5) & (C("return_252d_pre") <= 0.0)),
    ("pre-crash 1yr return 0 to +50%", "premom_up",
     (C("return_252d_pre") > 0.0) & (C("return_252d_pre") <= 0.5)),
    ("pre-crash 1yr return > +50%", "premom_hot", C("return_252d_pre") > 0.5),
    ("beta < 0.8 (defensive)", "beta_lo", C("beta") < 0.8),
    ("beta 0.8-1.2 (market)", "beta_mkt", (C("beta") >= 0.8) & (C("beta") <= 1.2)),
    ("beta 1.2-1.8 (high)", "beta_hi", (C("beta") > 1.2) & (C("beta") <= 1.8)),
    ("beta > 1.8 (very high)", "beta_vhi", C("beta") > 1.8),
    ("EBITDA margin < 0 (unprofitable)", "ebm_neg", C("ebitda_margin") < 0.0),
    ("EBITDA margin 0-10%", "ebm_lo", (C("ebitda_margin") >= 0.0) & (C("ebitda_margin") <= 0.10)),
    ("EBITDA margin 10-25%", "ebm_mid", (C("ebitda_margin") > 0.10) & (C("ebitda_margin") <= 0.25)),
    ("EBITDA margin > 25%", "ebm_hi", C("ebitda_margin") > 0.25),
    ("revenue declining (YoY < 0)", "revg_dn", C("revenue_growth_yoy") < 0.0),
    ("revenue growth 0-10%", "revg_lo",
     (C("revenue_growth_yoy") >= 0.0) & (C("revenue_growth_yoy") <= 0.10)),
    ("revenue growth 10-30%", "revg_mid",
     (C("revenue_growth_yoy") > 0.10) & (C("revenue_growth_yoy") <= 0.30)),
    ("revenue growth > 30%", "revg_hi", C("revenue_growth_yoy") > 0.30),
    ("market 20d return < -10%", "mkt_crash", C("market_return_20d") < -0.10),
    ("market 20d return -10 to -3%", "mkt_dn",
     (C("market_return_20d") >= -0.10) & (C("market_return_20d") < -0.03)),
    ("market 20d return -3 to 0%", "mkt_flat",
     (C("market_return_20d") >= -0.03) & (C("market_return_20d") < 0.0)),
    ("market 20d return > 0%", "mkt_up", C("market_return_20d") >= 0.0),
    ("crashed far worse than sector (< -10%)", "relsec_worse", C("rel_sector") < -0.10),
    ("crashed moderately worse (-10 to -3%)", "relsec_mid",
     (C("rel_sector") >= -0.10) & (C("rel_sector") < -0.03)),
    ("crashed with or better than sector (> -3%)", "relsec_with", C("rel_sector") >= -0.03),
]
# 60-day severity buckets (crash_return bands)
_cr = C("crash_return")
BUCKETS_60D = [
    ("crash of 10-15%", "sev_10_15_60d", (_cr > -0.15) & (_cr <= -0.10)),
    ("crash of 15-20%", "sev_15_20_60d", (_cr > -0.20) & (_cr <= -0.15)),
    ("crash of 20-30%", "sev_20_30_60d", (_cr > -0.30) & (_cr <= -0.20)),
    ("crash worse than 30%", "sev_30plus_60d", _cr <= -0.30),
]


def render(d, buckets, spec, figdir):
    for title, key, expr in buckets:
        r = d.filter(expr)["ret"].to_numpy()
        out = figdir / f"recovery_hist_{key}_2022.pdf"
        med, mean, p_earn, n = return_histogram(r, spec, out, title=title)
        flag = "  <-- small n" if n < 50 else ""
        print(f"  {key:16s} n={n:5d}  P(earn)={p_earn:.3f}  median={med:+.3f}  "
              f"mean={mean:+.3f}{flag}")


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    proc = cfg.paths.resolve("data_processed")
    ev = pl.read_parquet(proc / "events_v1.parquet").select(
        "event_id", "market_cap", "crash_return", "prior_crash_count_20d", "pe",
        "return_252d_pre", "revenue_growth_yoy", "market_return_20d", "sector_return_1d")
    beta, ebm = event_beta(cfg), event_ebitda_margin(cfg)

    d1 = (one_year_returns(cfg, 252).filter(C("crash_date") >= SINCE)
          .join(ev, on="event_id").join(beta, on="event_id", how="left")
          .join(ebm, on="event_id", how="left")
          .with_columns(rel_sector=C("crash_return") - C("sector_return_1d")))
    d60 = one_year_returns(cfg, 60).filter(C("crash_date") >= SINCE).join(
        ev.select("event_id", "crash_return"), on="event_id")

    print(f"one-year buckets (2022+):  [d1 n={d1.height}]")
    render(d1, BUCKETS_1YR, {**ONE_YEAR, "label": "one-year (2022 on)"}, figdir)
    print(f"\n60-day severity buckets (2022+):  [d60 n={d60.height}]")
    render(d60, BUCKETS_60D, {**SIXTY_DAY, "label": "60-day (2022 on)"}, figdir)


if __name__ == "__main__":
    main()
