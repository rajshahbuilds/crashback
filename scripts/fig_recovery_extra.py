#!/usr/bin/env python3
"""Figures: one-year return distribution by six further features (feature-correlation section).

Pre-crash one-year return, 252-day beta, EBITDA margin, revenue growth, market 20-day return, and
crash-day performance relative to sector. One baseline-style one-year return histogram per fixed
bucket. Beta and EBITDA margin are computed point-in-time (crashback.analysis.extra_features); the
rest come from events_v1.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_extra.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from crashback.analysis.extra_features import event_beta, event_ebitda_margin
from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.analysis.recovery import one_year_returns
from crashback.config import load_config

BIG = 1e9
# feature column -> list of (panel title, file, lo exclusive, hi inclusive)
FEATURES = [
    ("return_252d_pre", [
        ("pre-crash 1yr return < -50%", "premom_dn", -BIG, -0.5),
        ("pre-crash 1yr return -50 to 0%", "premom_mild", -0.5, 0.0),
        ("pre-crash 1yr return 0 to +50%", "premom_up", 0.0, 0.5),
        ("pre-crash 1yr return > +50%", "premom_hot", 0.5, BIG)]),
    ("beta", [
        ("beta < 0.8 (defensive)", "beta_lo", -BIG, 0.8),
        ("beta 0.8–1.2 (market)", "beta_mkt", 0.8, 1.2),
        ("beta 1.2–1.8 (high)", "beta_hi", 1.2, 1.8),
        ("beta > 1.8 (very high)", "beta_vhi", 1.8, BIG)]),
    ("ebitda_margin", [
        ("EBITDA margin < 0 (unprofitable)", "ebm_neg", -BIG, 0.0),
        ("EBITDA margin 0–10%", "ebm_lo", 0.0, 0.10),
        ("EBITDA margin 10–25%", "ebm_mid", 0.10, 0.25),
        ("EBITDA margin > 25%", "ebm_hi", 0.25, BIG)]),
    ("revenue_growth_yoy", [
        ("revenue declining (YoY < 0)", "revg_dn", -BIG, 0.0),
        ("revenue growth 0–10%", "revg_lo", 0.0, 0.10),
        ("revenue growth 10–30%", "revg_mid", 0.10, 0.30),
        ("revenue growth > 30%", "revg_hi", 0.30, BIG)]),
    ("market_return_20d", [
        ("market 20d return < -10%", "mkt_crash", -BIG, -0.10),
        ("market 20d return -10 to -3%", "mkt_dn", -0.10, -0.03),
        ("market 20d return -3 to 0%", "mkt_flat", -0.03, 0.0),
        ("market 20d return > 0%", "mkt_up", 0.0, BIG)]),
    ("rel_sector", [
        ("crashed far worse than sector (< -10%)", "relsec_worse", -BIG, -0.10),
        ("crashed moderately worse (-10 to -3%)", "relsec_mid", -0.10, -0.03),
        ("crashed with or better than sector (> -3%)", "relsec_with", -0.03, BIG)]),
]


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ret = one_year_returns(cfg)
    ev = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "return_252d_pre", "revenue_growth_yoy", "market_return_20d",
        "crash_return", "sector_return_1d")
    print("computing beta ...")
    beta = event_beta(cfg)
    print("computing ebitda margin ...")
    ebm = event_ebitda_margin(cfg)
    d = (ret.join(ev, on="event_id").join(beta, on="event_id", how="left")
         .join(ebm, on="event_id", how="left")
         .with_columns(rel_sector=pl.col("crash_return") - pl.col("sector_return_1d")))

    for col, buckets in FEATURES:
        for title, key, lo, hi in buckets:
            band = d.filter((pl.col(col) > lo) & (pl.col(col) <= hi))
            out = figdir / f"recovery_hist_{key}.pdf"
            med, mean, p_earn, n = return_histogram(band["ret"].to_numpy(), ONE_YEAR, out,
                                                    title=title)
            print(f"  {key:12s} n={n:6d}  P(earn)={p_earn:.3f}  median={med:+.3f}")


if __name__ == "__main__":
    main()
