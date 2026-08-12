#!/usr/bin/env python3
"""STU-54: ingest & normalize point-in-time (unrestated) fundamentals.

Pulls Compustat unrestated quarterly (comp_urq.urqus), normalizes to the canonical
fundamentals schema with a defensible availability date, and writes versioned Parquet to
data/normalized/fundamentals/. See crashback.fundamentals.ingest for the restatement policy.

Run: .venv/bin/python scripts/ingest_fundamentals.py --username r43shah
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import polars as pl

from crashback.config import load_config
from crashback.fundamentals.ingest import (
    FUNDAMENTALS_SCHEMA,
    RAW_FIELDS,
    normalize_urqus,
)
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    import wrds

    db = wrds.Connection(wrds_username=args.username)
    print("pulling comp_urq.urqus (unrestated quarterly) ...")
    pdf = db.raw_sql(f"SELECT {', '.join(RAW_FIELDS)} FROM comp_urq.urqus")
    db.close()
    for c in ("datadate", "rdq"):
        pdf[c] = pd.to_datetime(pdf[c], errors="coerce")
    raw = pl.from_pandas(pdf)
    fund = normalize_urqus(raw)

    print("\n=== summary ===")
    print(f"  rows: {fund.height:,}   companies: {fund['company_id'].n_unique():,}")
    print(f"  period_end: {fund['period_end'].min()} .. {fund['period_end'].max()}")
    ra = fund["rdq_available"].mean()
    print(f"  rdq_available: {ra:.1%}  (rest use period_end + 60d fallback)")
    print("  non-null coverage of key fields:")
    for c in ("revenue", "net_income", "total_assets", "total_debt", "eps",
              "stockholders_equity", "ebitda"):
        print(f"    {c:22s} {1 - fund[c].null_count() / fund.height:6.3f}")

    out_dir = cfg.paths.resolve("data_normalized") / "fundamentals"
    path = write_versioned_parquet(
        fund, out_dir, "fundamentals", args.version,
        meta={"source": "comp_urq.urqus (unrestated)", "restatement_policy": "originally-reported",
              "availability": "rdq else period_end+60d", "fcf": "unavailable (documented gap)",
              "fields": list(FUNDAMENTALS_SCHEMA.keys()), "rows": fund.height},
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}\nDONE.")


if __name__ == "__main__":
    main()
