#!/usr/bin/env python3
"""STU-52: build recent-crash history features for every crash event.

Reads crash events (STU-49) + daily prices (STU-48); writes
data/processed/features_recent_crash_*.parquet keyed by event_id. Chunked by security so
each security's full crash history is intact within a chunk.

Run: .venv/bin/python scripts/build_recent_crash_features.py --version v1
"""
from __future__ import annotations

import argparse

import polars as pl

from crashback.config import load_config
from crashback.features.recent_crash import FEATURE_NAMES, build_recent_crash_features
from crashback.ingestion.prices import scan_daily_prices
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet

N_CHUNKS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    norm = cfg.paths.resolve("data_normalized")

    print("loading prices into memory ...")
    prices = scan_daily_prices(norm / "daily_prices").select(
        "security_id", "date", "close", "high", "low", "open", "volume", "daily_return"
    ).collect()
    events = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet")
    print(f"prices={prices.height:,}  events={events.height:,}")

    sids = events["security_id"].unique().sort().to_list()
    parts = []
    for k in range(N_CHUNKS):
        group = sids[k::N_CHUNKS]
        ev_c = events.filter(pl.col("security_id").is_in(group))
        if ev_c.height == 0:
            continue
        px_c = prices.filter(pl.col("security_id").is_in(group))
        parts.append(build_recent_crash_features(ev_c, px_c))
        print(f"  chunk {k + 1}/{N_CHUNKS}: {ev_c.height:,} events")

    feats = pl.concat(parts, how="vertical").sort(["security_id", "crash_date"])

    print("\n=== summary ===")
    print(f"  feature rows: {feats.height:,}")
    fresh = feats.filter(pl.col("prior_crash_count_60d") == 0).height
    print(f"  fresh crashes (0 prior in 60d): {fresh:,} ({fresh / feats.height:.1%})")
    print("  distribution of prior_crash_count_20d:")
    print(feats.group_by("prior_crash_count_20d").len().sort("prior_crash_count_20d").head(8)
          .to_pandas().to_string(index=False))

    out_dir = cfg.paths.resolve("data_processed")
    path = write_versioned_parquet(
        feats, out_dir, "features_recent_crash", args.version,
        meta={"features": list(FEATURE_NAMES), "rows": feats.height},
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}\nDONE.")


if __name__ == "__main__":
    main()
