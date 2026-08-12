#!/usr/bin/env python3
"""STU-50: generate recovery labels + continuous outcomes for every crash event.

Reads crash events (STU-49), daily prices (STU-48), and the security master (STU-47);
writes data/events/recovery_labels_*.parquet. Processes securities in chunks so the
forward-window fan-out stays within memory.

Run: .venv/bin/python scripts/build_recovery_labels.py --version v1
"""
from __future__ import annotations

import argparse

import polars as pl

from crashback.config import load_config
from crashback.ingestion.prices import scan_daily_prices
from crashback.labels.outcomes import build_labels
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet

N_CHUNKS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    norm = cfg.paths.resolve("data_normalized")
    events_dir = cfg.paths.resolve("data_events")

    horizons = tuple(cfg.labels.horizons_trading_days)
    thresholds = tuple((int(round(f * 100)), f) for f in cfg.labels.recovery_thresholds)
    print(f"horizons={horizons}  thresholds={thresholds}")

    print("loading prices into memory ...")
    prices = scan_daily_prices(norm / "daily_prices").select(
        "security_id", "date", "close", "high"
    ).collect()
    master = pl.read_parquet(norm / "security_master" / "security_master_v1.parquet").select(
        "security_id", "delisting_date", "delisting_return"
    )
    events = pl.read_parquet(events_dir / "crash_events_v1.parquet")
    print(f"prices={prices.height:,}  events={events.height:,}")

    sids = events["security_id"].unique().sort().to_list()
    parts = []
    for k in range(N_CHUNKS):
        group = sids[k::N_CHUNKS]
        ev_c = events.filter(pl.col("security_id").is_in(group))
        if ev_c.height == 0:
            continue
        px_c = prices.filter(pl.col("security_id").is_in(group))
        m_c = master.filter(pl.col("security_id").is_in(group))
        parts.append(build_labels(ev_c, px_c, m_c, horizons=horizons, thresholds=thresholds))
        print(f"  chunk {k + 1}/{N_CHUNKS}: {ev_c.height:,} events")

    labels = pl.concat(parts, how="vertical").sort(["security_id", "crash_date"])

    # --- base rates over the CLEAN pool (in-universe & liquid) ---
    flags = events.select("event_id", "in_universe_at_event", "passes_min_price")
    lab = labels.join(flags, on="event_id", how="left")
    clean = lab.filter(pl.col("in_universe_at_event") & pl.col("passes_min_price"))

    print("\n=== summary ===")
    print(f"  label rows: {labels.height:,}   clean pool: {clean.height:,}")
    for h in horizons:
        cens = labels.filter(pl.col(f"censored_{h}d")).height
        print(f"  censored_{h}d: {cens:,}")
    print("\n=== primary target base rate (hit_10pct_20d, close-based) ===")
    for name, d in (("ALL", lab), ("CLEAN", clean)):
        sub = d.filter(~pl.col("censored_20d") & pl.col("hit_10pct_20d").is_not_null())
        rate = sub["hit_10pct_20d"].mean()
        print(f"  {name}: {rate:.4f} over {sub.height:,} determined events")

    path = write_versioned_parquet(
        labels, events_dir, "recovery_labels", args.version,
        meta={"horizons": list(horizons), "thresholds": [t[1] for t in thresholds],
              "rows": labels.height},
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}\nDONE.")


if __name__ == "__main__":
    main()
