"""Daily-price ingestion: partitioned write, duplicate rejection, round-trip (hermetic)."""
from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from crashback.ingestion.prices import (
    assert_unique_security_date,
    ingest_daily_prices,
    scan_daily_prices,
)
from crashback.providers import SyntheticProvider


def test_assert_unique_security_date_rejects_duplicates():
    dup = pl.DataFrame(
        {"security_id": [1, 1], "date": [date(2020, 1, 2), date(2020, 1, 2)]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        assert_unique_security_date(dup)


def test_ingest_writes_partitioned_parquet_and_manifest(tmp_path):
    manifest = ingest_daily_prices(
        SyntheticProvider.example(), tmp_path, 2020, 2020, log_fn=lambda _m: None
    )
    # example() has 5 price rows, all in 2020
    assert manifest["total_rows"] == 5
    assert manifest["years"] == [{"year": 2020, "rows": 5, "securities": 2}]
    assert (tmp_path / "year=2020" / "prices.parquet").exists()

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["partition"] == "year"

    # round-trip via the scan helper
    back = scan_daily_prices(tmp_path).collect()
    assert back.height == 5
    assert set(back["security_id"].unique().to_list()) == {1001, 2002}


def test_ingest_skips_empty_years(tmp_path):
    # No data in 1900 -> no partition, zero total.
    manifest = ingest_daily_prices(
        SyntheticProvider.example(), tmp_path, 1900, 1900, log_fn=lambda _m: None
    )
    assert manifest["total_rows"] == 0
    assert manifest["years"] == []
    assert not (tmp_path / "year=1900").exists()
