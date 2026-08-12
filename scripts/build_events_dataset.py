#!/usr/bin/env python3
"""STU-56: assemble the canonical V1 event dataset (events_v1).

Joins the crash-event spine with recovery labels and the four feature families into one
row per event, validates it, and writes a versioned Parquet + rich provenance sidecar
(column groups, missingness, config snapshot). No new data pulls.

Run: .venv/bin/python scripts/build_events_dataset.py --version v1
"""
from __future__ import annotations

import argparse
import json

import polars as pl

from crashback.config import load_config
from crashback.datasets.assemble import (
    FEATURE_COLS,
    FEATURE_GROUPS,
    FEATURE_META_COLS,
    ID_COLS,
    OUTCOME_COLS,
    OUTCOME_META_COLS,
    PRIMARY_TARGET,
    assemble_events,
    validate_events,
)
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    events_dir = cfg.paths.resolve("data_events")
    proc = cfg.paths.resolve("data_processed")

    def rd(path):
        return pl.read_parquet(path)

    df = assemble_events(
        crash_events=rd(events_dir / "crash_events_v1.parquet"),
        labels=rd(events_dir / "recovery_labels_v1.parquet"),
        price=rd(proc / "features_price_v1.parquet"),
        recent=rd(proc / "features_recent_crash_v1.parquet"),
        market_sector=rd(proc / "features_market_sector_v1.parquet"),
        fundamentals=rd(proc / "features_fundamentals_v1.parquet"),
    )

    report = validate_events(df, crash_threshold=cfg.crash.threshold)
    clean = df.filter(pl.col("in_universe_at_event") & pl.col("passes_min_price"))

    print("=== events_v1 ===")
    print(f"  rows: {df.height:,}  unique events: {report['unique_events']:,}")
    print(f"  features: {report['n_features']}  outcomes: {report['n_outcomes']}")
    print(f"  CLEAN pool: {clean.height:,}")
    det = clean.filter(~pl.col("censored_20d") & pl.col(PRIMARY_TARGET).is_not_null())
    print(f"  primary base rate (CLEAN, determined): {det[PRIMARY_TARGET].mean():.4f}")

    # missingness (fraction null) per feature, over the CLEAN pool
    missing_ness = {c: round(clean[c].null_count() / clean.height, 4) for c in FEATURE_COLS}

    out_dir = proc
    path = write_versioned_parquet(
        df, out_dir, "events", args.version,
        meta={
            "id_cols": list(ID_COLS),
            "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
            "feature_cols": list(FEATURE_COLS),
            "feature_meta_cols": list(FEATURE_META_COLS),
            "outcome_cols": list(OUTCOME_COLS),
            "outcome_meta_cols": list(OUTCOME_META_COLS),
            "primary_target": PRIMARY_TARGET,
            "validation": report,
            "clean_pool_rows": clean.height,
            "clean_feature_missingness": missing_ness,
            "date_range": [str(df["crash_date"].min()), str(df["crash_date"].max())],
            "config": json.loads(cfg.model_dump_json()),
        },
        git_commit=current_git_commit(),
    )
    print(f"\nwrote {path}\nDONE.")


if __name__ == "__main__":
    main()
