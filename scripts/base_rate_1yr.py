#!/usr/bin/env python3
"""Model 0 base rate: after a crash, what is P(earn money one year later)?

Survivorship-safe by construction: the one-year outcome is the compounded total daily return
over the next 252 trading days (splits, dividends, and delisting/bankruptcy terminal returns are
all included), measured from the crash-day close. Crashes within one trading year of the data
edge cannot be observed for a full year and are censored, not silently labeled.

Run: PYTHONPATH=src .venv/bin/python scripts/base_rate_1yr.py
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
import polars as pl

from crashback.config import load_config
from crashback.ingestion.prices import scan_daily_prices

HORIZON = 252  # trading days ~ one calendar year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    norm = cfg.paths.resolve("data_normalized")
    events_dir = cfg.paths.resolve("data_events")

    ev = pl.read_parquet(events_dir / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).select("event_id", "security_id", "crash_date")
    sids = ev["security_id"].unique().to_list()

    # per-security cumulative log total return + trading-day index
    px = (scan_daily_prices(norm / "daily_prices")
          .filter(pl.col("security_id").is_in(sids))
          .select("security_id", "date", "daily_return").sort(["security_id", "date"])
          .with_columns(
              td_idx=pl.int_range(pl.len()).over("security_id"),
              cumlog=(pl.col("daily_return").fill_null(0.0) + 1.0).log().cum_sum()
              .over("security_id"))
          .collect())
    last = px.group_by("security_id").agg(last_idx=pl.col("td_idx").max(),
                                          last_date=pl.col("date").max())
    idx = px.select("security_id", "td_idx", "cumlog")
    data_edge = px["date"].max() - timedelta(days=17)  # ~ within-a-year-of-end guard

    e = (ev.join(px.select("security_id", "date", "td_idx", "cumlog"),
                 left_on=["security_id", "crash_date"], right_on=["security_id", "date"],
                 how="inner")
         .rename({"td_idx": "k0", "cumlog": "cl0"})
         .join(last, on="security_id")
         .with_columns(end_idx=pl.min_horizontal(pl.col("k0") + HORIZON, pl.col("last_idx")),
                       fwd=pl.col("last_idx") - pl.col("k0"))
         .join(idx.rename({"td_idx": "end_idx", "cumlog": "cl_end"}),
               on=["security_id", "end_idx"], how="left")
         .with_columns(ret1y=(pl.col("cl_end") - pl.col("cl0")).exp() - 1.0,
                       censored=(pl.col("fwd") < HORIZON) & (pl.col("last_date") >= data_edge)))

    obs = e.filter(~pl.col("censored") & pl.col("ret1y").is_not_null())
    r = obs["ret1y"].to_numpy()
    print(f"CLEAN crash events:            {e.height:,}")
    print(f"censored (< 1yr of data left): {int(e['censored'].sum()):,}")
    print(f"observed one-year outcomes:    {obs.height:,}")
    print(f"P(earn money, 1yr return > 0): {(r > 0).mean():.4f}")
    print(f"P(up >= 10%):                  {(r >= 0.10).mean():.4f}")
    print(f"P(down >= 50%):                {(r <= -0.50).mean():.4f}")
    print(f"mean 1yr return:  {r.mean():+.4f}")
    print(f"median 1yr return: {np.median(r):+.4f}")


if __name__ == "__main__":
    main()
