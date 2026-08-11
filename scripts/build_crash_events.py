#!/usr/bin/env python3
"""STU-49: detect crash events from the daily price history and write versioned Parquet.

Every day with daily_return <= threshold is one event (no cooldown / no suppression).
Reads the STU-48 daily prices + STU-47 security master; writes data/events/crash_events_*.

Run: .venv/bin/python scripts/build_crash_events.py --version v1
"""
from __future__ import annotations

import argparse

import polars as pl

from crashback.config import load_config
from crashback.events.detect import build_crash_events
from crashback.ingestion.prices import scan_daily_prices
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prices = scan_daily_prices(cfg.paths.resolve("data_normalized") / "daily_prices")
    master = pl.read_parquet(
        cfg.paths.resolve("data_normalized") / "security_master" / "security_master_v1.parquet"
    )

    print(f"threshold={cfg.crash.threshold}  min_price={cfg.universe.min_price}")
    events = build_crash_events(
        prices, master, threshold=cfg.crash.threshold, min_price=cfg.universe.min_price
    )

    n = events.height
    in_univ = events["in_universe_at_event"].sum()
    clean = events.filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).height
    print("\n=== summary ===")
    print(f"  crash events (all):      {n:,}")
    print(f"  securities:              {events['security_id'].n_unique():,}")
    dmin, dmax = events["crash_date"].min(), events["crash_date"].max()
    print(f"  date range:              {dmin} .. {dmax}")
    print(f"  in universe at event:    {in_univ:,}")
    print(f"  pass min_price (${cfg.universe.min_price}):     {events['passes_min_price'].sum():,}")
    print(f"  CLEAN (in-univ & >=${cfg.universe.min_price}): {clean:,}")
    print(f"  null crash_close:        {events['crash_close'].null_count():,}")

    # --- sample: consecutive crash days for one security (must be separate events) ---
    with_next = events.sort(["security_id", "crash_date"]).with_columns(
        gap=(pl.col("crash_date") - pl.col("crash_date").shift(1))
        .dt.total_days()
        .over("security_id")
    )
    consec = with_next.filter(pl.col("gap") == 1)
    if consec.height:
        sid = consec.row(0, named=True)["security_id"]
        print("\n=== sample: consecutive crash days (same security, separate events) ===")
        print(
            events.filter(pl.col("security_id") == sid)
            .select(["event_id", "crash_date", "crash_return", "crash_close", "ticker_as_of_event"])
            .head(8).to_pandas().to_string(index=False)
        )

    out_dir = cfg.paths.resolve("data_events")
    path = write_versioned_parquet(
        events, out_dir, "crash_events", args.version,
        meta={"threshold": cfg.crash.threshold, "min_price": cfg.universe.min_price,
              "events": n, "securities": events["security_id"].n_unique()},
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}")
    print("DONE.")


if __name__ == "__main__":
    main()
