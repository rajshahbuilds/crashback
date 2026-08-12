#!/usr/bin/env python3
"""STU-58: assign chronological train/validation/test splits (+ embargo) and summarize.

Loads events_v1 and the real trading calendar (distinct daily-price dates), assigns each
event a split by crash_date with an outcome-window embargo, writes data/processed/splits_v1
(event_id, split), and generates reports/STU-58_splits.md.

Run: .venv/bin/python scripts/build_splits.py --version v1
"""
from __future__ import annotations

import argparse

import polars as pl

from crashback.config import load_config
from crashback.datasets.splits import assign_splits
from crashback.ingestion.prices import scan_daily_prices
from crashback.providers.normalize import sic_division
from crashback.storage.artifacts import current_git_commit, write_versioned_parquet

PRIMARY = "hit_10pct_20d"
ORDER = ["train", "validation", "test", "embargo", "none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    norm = cfg.paths.resolve("data_normalized")

    events = pl.read_parquet(proc / "events_v1.parquet").select(
        "event_id", "security_id", "crash_date", "in_universe_at_event", "passes_min_price",
        "censored_20d", PRIMARY,
    )
    calendar = scan_daily_prices(norm / "daily_prices").select("date").unique().collect()["date"]

    splits = assign_splits(events, cfg.splits, calendar)
    write_versioned_parquet(
        splits, proc, "splits", args.version,
        meta={"train": [str(d) for d in cfg.splits.train],
              "validation": [str(d) for d in cfg.splits.validation],
              "test": [str(d) for d in cfg.splits.test],
              "embargo_trading_days": cfg.splits.embargo_trading_days},
        git_commit=current_git_commit(),
    )

    # sector per security (for the split summary)
    sm = pl.read_parquet(norm / "security_master" / "security_master_v1.parquet").group_by(
        "security_id"
    ).agg(pl.col("sic_code").drop_nulls().first().alias("sic_code"))
    sic_map = {s: sic_division(s) for s in sm["sic_code"].unique().to_list()}
    sm = sm.with_columns(
        pl.col("sic_code").replace_strict(sic_map, default="Unknown").alias("sector")
    )

    e = events.join(splits.select("event_id", "split"), on="event_id").join(
        sm.select("security_id", "sector"), on="security_id", how="left"
    )
    clean = e.filter(pl.col("in_universe_at_event") & pl.col("passes_min_price"))
    determined = ~pl.col("censored_20d") & pl.col(PRIMARY).is_not_null()

    summ = clean.group_by("split").agg(
        pl.len().alias("events"),
        pl.col("crash_date").min().alias("date_min"),
        pl.col("crash_date").max().alias("date_max"),
        pl.col("security_id").n_unique().alias("securities"),
        pl.col("sector").n_unique().alias("sectors"),
        determined.sum().alias("determined"),
        pl.col(PRIMARY).filter(determined).mean().alias("base_rate"),
    )
    order_map = {s: i for i, s in enumerate(ORDER)}
    summ = summ.with_columns(
        pl.col("split").replace_strict(order_map, default=99).alias("_o")
    ).sort("_o").drop("_o")

    rows = ["| split | events | determined | date range | securities | sectors | base rate |",
            "|---|---|---|---|---|---|---|"]
    for r in summ.iter_rows(named=True):
        br = f"{r['base_rate']:.4f}" if r["base_rate"] is not None else "—"
        rows.append(f"| {r['split']} | {r['events']:,} | {r['determined']:,} | "
                    f"{r['date_min']} → {r['date_max']} | {r['securities']:,} | "
                    f"{r['sectors']} | {br} |")
    table = "\n".join(rows)

    report = f"""# STU-58 — Chronological Train / Validation / Test Splits

Assigned by `scripts/build_splits.py` (module `crashback.datasets.splits`) over `events_v1`.
Splits are by `crash_date` only — **never random row sampling** — so training is always on the
past relative to validation/test. Definitions are versioned in `configs/default.yaml`.

## Split definitions (from config)

- **train**: {cfg.splits.train[0]} → {cfg.splits.train[1]}
- **validation**: {cfg.splits.validation[0]} → {cfg.splits.validation[1]}
- **test**: {cfg.splits.test[0]} → {cfg.splits.test[1]}
- **embargo**: {cfg.splits.embargo_trading_days} trading days

`none` = events outside all configured ranges (chiefly pre-{cfg.splits.train[0].year} history,
retained only for robustness work, not primary modeling).

## Embargo (outcome-window boundary treatment)

Outcomes look up to **{cfg.splits.embargo_trading_days} trading days** forward, so an event in
the last {cfg.splits.embargo_trading_days} trading days of train (or validation) has an outcome
window that spills into the next split. Those events are reassigned to **`embargo`** and dropped
from modeling, measured against the **real trading calendar** (not calendar days). This
guarantees no training event's outcome overlaps validation/test — the primary leakage risk of
chronological splitting with forward-looking labels.

## Summary (CLEAN pool)

{table}

- **No event appears in multiple splits** — each `event_id` gets exactly one label.
- **Primary training never sees test-period events or outcomes** — test is strictly later and
  the embargo removes boundary-crossing outcome windows.
- Target prevalence (base rate) is reported per split; drift across splits reflects genuine
  regime differences, and is why we evaluate on the held-out test period rather than in-sample.
"""
    out = cfg.paths.resolve("reports") / "STU-58_splits.md"
    out.write_text(report)
    print(table)
    print(f"\nwrote {out} and splits_{args.version}.parquet")


if __name__ == "__main__":
    main()
