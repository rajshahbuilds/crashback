#!/usr/bin/env python3
"""STU-48: ingest & normalize historical daily prices to partitioned Parquet.

Pulls the common-stock universe's daily bars year by year via the WRDS provider and writes
Hive-partitioned Parquet under data/normalized/daily_prices/. See crashback.ingestion.prices
for the adjustment/delisting policy.

Run: .venv/bin/python scripts/ingest_daily_prices.py --username r43shah \
        --start-year 1962 --end-year 2025
"""
from __future__ import annotations

import argparse
import os

from crashback.config import load_config
from crashback.ingestion.prices import ingest_daily_prices
from crashback.providers.universe import UniverseFilter
from crashback.providers.wrds_provider import WRDSProvider
from crashback.storage.artifacts import current_git_commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    universe = UniverseFilter.from_config(cfg.universe)
    out_dir = cfg.paths.resolve("data_normalized") / "daily_prices"
    print(f"universe: {universe}\nout_dir: {out_dir}\nyears: {args.start_year}..{args.end_year}\n")

    provider = WRDSProvider(username=args.username)
    manifest = ingest_daily_prices(
        provider, out_dir, args.start_year, args.end_year, universe,
        version=args.version, git_commit=current_git_commit(),
    )
    provider.close()

    print(f"\n=== done ===\n  total_rows: {manifest['total_rows']:,}")
    print(f"  years with data: {len(manifest['years'])}")
    print(f"  manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
