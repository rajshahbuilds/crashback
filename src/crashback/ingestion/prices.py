"""Ingest and normalize the canonical daily price history.

Pulls daily bars for the common-stock universe year by year (bounded queries), validates
the canonical `daily_price` schema, rejects duplicate (security_id, date) rows, and writes
Hive-partitioned Parquet (`year=YYYY/prices.parquet`) for efficient DuckDB/Polars scans.

Adjustment policy (documented + tested):
  * Returns use CRSP ``daily_return`` (dlyret) — total return, already adjusted for splits
    and dividends. Crash detection must use this, never close-to-close deltas, so a split
    never looks like a crash.
  * ``adjusted_close`` = close / cumulative price factor; raw ``close`` is retained too.
  * No forward-filling: gaps are genuine non-trading days or halts; dlyret spans them.
  * Delisting behavior is preserved: the last real trading bar is kept (never dropped);
    the terminal delisting return lives in the security master (`delisting_return`, code
    >= 200), so downstream label logic can apply it without corrupting the price series.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

from crashback.providers.base import MarketDataProvider
from crashback.providers.schemas import DAILY_PRICE_SCHEMA, validate_schema
from crashback.providers.universe import UniverseFilter


def assert_unique_security_date(df: pl.DataFrame) -> None:
    """Reject duplicate (security_id, trading_date) rows — the canonical uniqueness rule."""
    dups = df.group_by(["security_id", "date"]).len().filter(pl.col("len") > 1)
    if dups.height:
        sample = dups.head(5).to_dicts()
        raise ValueError(f"{dups.height} duplicate (security_id, date) rows; sample: {sample}")


def ingest_daily_prices(
    provider: MarketDataProvider,
    out_dir: str | Path,
    start_year: int,
    end_year: int,
    universe: UniverseFilter | None = None,
    *,
    version: str = "v1",
    log_fn: Callable[[str], None] = print,
    git_commit: str | None = None,
) -> dict:
    """Materialize daily prices per year to Hive-partitioned Parquet; return a manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = []
    total_rows = 0
    for year in range(start_year, end_year + 1):
        df = provider.get_daily_prices(
            None, date(year, 1, 1), date(year, 12, 31), universe=universe
        )
        if df.height == 0:
            log_fn(f"{year}: 0 rows (skipped)")
            continue
        df = validate_schema(df, DAILY_PRICE_SCHEMA)
        assert_unique_security_date(df)

        part_dir = out_dir / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / "prices.parquet"
        df.write_parquet(path)

        n_sec = df["security_id"].n_unique()
        years.append({"year": year, "rows": df.height, "securities": n_sec})
        total_rows += df.height
        log_fn(f"{year}: {df.height:>9,} rows, {n_sec:>5} securities -> {path.name}")

    manifest = {
        "name": "daily_prices",
        "version": version,
        "partition": "year",
        "start_year": start_year,
        "end_year": end_year,
        "total_rows": total_rows,
        "years": years,
        "columns": list(DAILY_PRICE_SCHEMA.keys()),
        "universe_filter": universe.__dict__ if universe else None,
        "git_commit": git_commit,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def scan_daily_prices(out_dir: str | Path) -> pl.LazyFrame:
    """Lazy scan over the partitioned daily-price dataset (for DuckDB/Polars downstream)."""
    out_dir = Path(out_dir)
    return pl.scan_parquet(out_dir / "year=*" / "prices.parquet")
