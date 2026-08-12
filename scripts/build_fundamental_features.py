#!/usr/bin/env python3
"""STU-55: as-of join fundamentals to crash events + derived features (leakage-free).

Pulls the CRSP/Compustat link (crsp.ccmxpf_lnkhist), loads normalized fundamentals (STU-54)
and crash events (STU-49), and writes data/processed/features_fundamentals_*.parquet keyed by
event_id. See crashback.fundamentals.features for the as-of / leakage guarantee.

Run: .venv/bin/python scripts/build_fundamental_features.py --username r43shah --version v1
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import polars as pl

from crashback.config import load_config
from crashback.fundamentals.features import DERIVED_FEATURES, build_fundamental_features
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    norm = cfg.paths.resolve("data_normalized")

    import wrds

    db = wrds.Connection(wrds_username=args.username)
    print("pulling crsp.ccmxpf_lnkhist ...")
    lpdf = db.raw_sql(
        "SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt FROM crsp.ccmxpf_lnkhist"
    )
    db.close()
    for c in ("linkdt", "linkenddt"):
        lpdf[c] = pd.to_datetime(lpdf[c], errors="coerce")
    link = pl.from_pandas(lpdf)

    fundamentals = pl.read_parquet(norm / "fundamentals" / "fundamentals_v1.parquet")
    events = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet").select(
        "event_id", "security_id", "crash_date", "crash_close"
    )
    print(f"link={link.height:,}  fundamentals={fundamentals.height:,}  events={events.height:,}")

    feats = build_fundamental_features(events, fundamentals, link)

    ev = pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet")
    j = feats.join(ev.select("event_id", "in_universe_at_event", "passes_min_price"), on="event_id")
    clean = j.filter(pl.col("in_universe_at_event") & pl.col("passes_min_price"))

    print("\n=== summary ===")
    print(f"  feature rows: {feats.height:,}")
    print(f"  fundamentals_available: {feats['fundamentals_available'].mean():.1%} (all) | "
          f"{clean['fundamentals_available'].mean():.1%} (CLEAN)")
    stale = feats.filter(pl.col("fundamentals_available"))["fundamentals_stale"].mean()
    print(f"  stale (of available): {stale:.1%}")
    print("  CLEAN coverage of key derived features:")
    for f in ("net_margin", "roe", "debt_to_assets", "revenue_growth_yoy", "pe", "ev_to_ebitda"):
        print(f"    {f:22s} {1 - clean[f].null_count() / clean.height:6.3f}")

    out_dir = cfg.paths.resolve("data_processed")
    path = write_versioned_parquet(
        feats, out_dir, "features_fundamentals", args.version,
        meta={"features": list(DERIVED_FEATURES), "rows": feats.height,
              "asof": "public_date <= crash_date", "link": "crsp.ccmxpf_lnkhist"},
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}\nDONE.")


if __name__ == "__main__":
    main()
