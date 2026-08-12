#!/usr/bin/env python3
"""STU-53: build market/sector context features for every crash event.

Reads crash events (STU-49), daily prices (STU-48), and the security master (STU-47);
writes data/processed/features_market_sector_*.parquet keyed by event_id.

Run: .venv/bin/python scripts/build_market_sector_features.py --version v1
"""
from __future__ import annotations

import argparse

import polars as pl

from crashback.config import load_config
from crashback.features.market_sector import FEATURE_NAMES, build_market_sector_features
from crashback.ingestion.prices import scan_daily_prices
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    norm = cfg.paths.resolve("data_normalized")

    print("loading prices into memory ...")
    prices = scan_daily_prices(norm / "daily_prices").select(
        "security_id", "date", "daily_return"
    ).collect()
    master = pl.read_parquet(norm / "security_master" / "security_master_v1.parquet").select(
        "security_id", "sic_code"
    )
    events = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet")
    print(f"prices={prices.height:,}  events={events.height:,}")

    feats = build_market_sector_features(events, prices, master).sort("event_id")

    print("\n=== summary ===")
    print(f"  feature rows: {feats.height:,}")
    for f in FEATURE_NAMES:
        cov = 1.0 - feats[f].null_count() / feats.height
        print(f"    {f:24s} coverage {cov:6.3f}")

    out_dir = cfg.paths.resolve("data_processed")
    path = write_versioned_parquet(
        feats, out_dir, "features_market_sector", args.version,
        meta={"features": list(FEATURE_NAMES), "rows": feats.height,
              "market_weight": "equal", "sector": "2-digit SIC",
              "focal_excluded": "sector_return_1d"},
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}\nDONE.")


if __name__ == "__main__":
    main()
