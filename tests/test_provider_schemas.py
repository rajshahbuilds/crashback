"""Canonical schema validation and the synthetic provider contract."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from crashback.providers import (
    SCHEMAS,
    MarketDataProvider,
    SchemaError,
    SyntheticProvider,
    validate_schema,
)
from crashback.providers.schemas import DAILY_PRICE_SCHEMA


def test_validate_schema_coerces_and_orders_columns():
    df = pl.DataFrame({"security_id": [1], "date": ["2020-01-02"]})  # wrong order, str date
    df = df.with_columns(
        pl.col("date").str.to_date(),
        open=pl.lit(1.0), high=pl.lit(1.0), low=pl.lit(1.0), close=pl.lit(1.0),
        adjusted_close=pl.lit(1.0), volume=pl.lit(1.0), daily_return=pl.lit(0.0),
        daily_return_ex_div=pl.lit(0.0), cum_factor_price=pl.lit(1.0),
        cum_factor_shares=pl.lit(1.0), shares_outstanding=pl.lit(1.0),
    )
    out = validate_schema(df, DAILY_PRICE_SCHEMA)
    assert out.columns == list(DAILY_PRICE_SCHEMA.keys())  # canonical order
    assert out.schema["date"] == pl.Date
    assert out.schema["security_id"] == pl.Int64


def test_validate_schema_missing_column_raises():
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_schema(pl.DataFrame({"security_id": [1]}), DAILY_PRICE_SCHEMA)


def test_validate_schema_strict_dtype_mismatch_raises():
    df = SyntheticProvider.example().get_daily_prices(None, date(2000, 1, 1), date(2099, 1, 1))
    bad = df.with_columns(pl.col("security_id").cast(pl.Utf8))
    with pytest.raises(SchemaError, match="dtype mismatch"):
        validate_schema(bad, DAILY_PRICE_SCHEMA, coerce=False)


def test_synthetic_provider_is_a_market_data_provider():
    assert isinstance(SyntheticProvider.example(), MarketDataProvider)


def test_synthetic_provider_outputs_conform_to_schemas():
    p = SyntheticProvider.example()
    frames = {
        "security_master": p.get_security_master(),
        "daily_price": p.get_daily_prices(None, date(2000, 1, 1), date(2099, 1, 1)),
        "fundamentals": p.get_fundamentals(),
        "corporate_action": p.get_corporate_actions(),
        "sector_metadata": p.get_sector_metadata(),
    }
    for name, df in frames.items():
        # Re-validating must be a no-op (already canonical).
        assert df.columns == list(SCHEMAS[name].keys()), name
        assert df.height > 0, name


def test_synthetic_provider_filters_by_id_and_date():
    p = SyntheticProvider.example()
    only = p.get_daily_prices([2002], date(2020, 1, 1), date(2020, 1, 3))
    assert set(only["security_id"].unique().to_list()) == {2002}

    windowed = p.get_daily_prices([1001], date(2020, 1, 3), date(2020, 1, 6))
    assert windowed["date"].min() == date(2020, 1, 3)
    assert windowed["date"].max() == date(2020, 1, 6)


def test_downstream_crash_detection_uses_only_canonical_columns():
    # A stand-in for the crash-event stage: no vendor column names anywhere.
    prices = SyntheticProvider.example().get_daily_prices(None, date(2000, 1, 1), date(2099, 1, 1))
    crashes = prices.filter(pl.col("daily_return") <= -0.10)
    assert crashes.height == 2
    assert set(crashes["security_id"].to_list()) == {1001, 2002}
