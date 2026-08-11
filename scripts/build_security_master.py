#!/usr/bin/env python3
"""STU-47: build the survivorship-safe security master and write versioned Parquet.

Pulls the full CRSP CIZ common-stock universe (active + delisted) via the WRDS provider,
applying the configured universe filter, writes it to versioned Parquet + sidecar under
data/normalized/security_master/, and prints integrity + sample checks (active, renamed,
acquired, delisted).

Run: .venv/bin/python scripts/build_security_master.py --username r43shah --version v1
"""
from __future__ import annotations

import argparse
import os

import polars as pl

from crashback.config import load_config
from crashback.providers.universe import UniverseFilter
from crashback.providers.wrds_provider import WRDSProvider
from crashback.securities.master import build_security_master, summarize_master
from crashback.storage.artifacts import write_versioned_parquet


def _show(df: pl.DataFrame, cols: list[str], n: int = 6) -> str:
    return df.select(cols).head(n).to_pandas().to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    universe = UniverseFilter.from_config(cfg.universe)
    print(f"universe filter: {universe}")

    provider = WRDSProvider(username=args.username)
    df = build_security_master(provider, universe)
    provider.close()

    summary = summarize_master(df)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = cfg.paths.resolve("data_normalized") / "security_master"
    path = write_versioned_parquet(
        df, out_dir, "security_master", args.version,
        meta={"universe_filter": universe.__dict__, "summary": {
            k: (str(v) if hasattr(v, "isoformat") else v) for k, v in summary.items()}},
    )
    print(f"\nwrote {path}")

    # ---- sample checks (acceptance criterion) --------------------------------
    disp = ["security_id", "company_id", "ticker", "ticker_start", "ticker_end",
            "exchange", "delisting_date", "delisting_code"]

    print("\n=== sample: ACTIVE (no delisting) ===")
    active = df.filter(pl.col("delisting_date").is_null())
    print(_show(active, disp))

    print("\n=== sample: RENAMED / TICKER-CHANGED (same security_id, >1 ticker period) ===")
    multi_ids = (
        df.group_by("security_id").agg(pl.col("ticker").n_unique().alias("n_tickers"))
        .filter(pl.col("n_tickers") > 1).sort("n_tickers", descending=True)
    )
    if multi_ids.height:
        eid = multi_ids.row(0, named=True)["security_id"]
        print(f"(security_id={eid} has {multi_ids.row(0, named=True)['n_tickers']} tickers)")
        print(_show(df.filter(pl.col("security_id") == eid), disp, n=10))

    print("\n=== sample: ACQUIRED (delisting_code 200-399 = merger/exchange) ===")
    acquired = df.filter(
        (pl.col("delisting_code") >= 200) & (pl.col("delisting_code") < 400)
    )
    print(_show(acquired, disp))

    print("\n=== sample: DELISTED / DROPPED (delisting_code >= 400) ===")
    dropped = df.filter(pl.col("delisting_code") >= 400)
    print(_show(dropped, disp))

    print("\nDONE.")


if __name__ == "__main__":
    main()
