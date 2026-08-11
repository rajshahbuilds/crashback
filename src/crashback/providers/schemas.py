"""Canonical, provider-neutral tabular schemas.

Downstream stages (crash detection, labels, features, fundamentals joins) depend ONLY
on these column names and dtypes — never on vendor names like ``dlyret`` or ``permno``.
Adapters normalize into these schemas as early as possible (CLAUDE.md sec 6).

Design note: config uses Pydantic (STU-44); bulk tabular data uses Polars schemas here,
because the price table alone is ~110M rows and Polars/DuckDB are the project's analytics
engines (CLAUDE.md sec 7). `validate_schema()` is the row-agnostic contract check.
"""
from __future__ import annotations

from collections import OrderedDict

import polars as pl

# --- identifiers ------------------------------------------------------------------
# security_id : stable per-security id (CRSP permno)
# company_id  : stable per-company id (CRSP permco; Compustat gvkey for fundamentals)


class SchemaError(ValueError):
    """Raised when a DataFrame does not satisfy a canonical schema."""


# --- canonical schemas (name -> Polars dtype), in canonical column order ----------

DAILY_PRICE_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("date", pl.Date),
        ("security_id", pl.Int64),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),            # raw closing price
        ("adjusted_close", pl.Float64),   # close adjusted by cumulative price factor
        ("volume", pl.Float64),
        ("daily_return", pl.Float64),     # total return (split+dividend adjusted) — crash basis
        ("daily_return_ex_div", pl.Float64),
        ("cum_factor_price", pl.Float64),
        ("cum_factor_shares", pl.Float64),
        ("shares_outstanding", pl.Float64),
    ]
)

SECURITY_MASTER_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        # grain: one row per (security_id, ticker period) — ticker history is preserved.
        ("security_id", pl.Int64),
        ("company_id", pl.Int64),
        ("ticker", pl.Utf8),
        ("ticker_start", pl.Date),
        ("ticker_end", pl.Date),
        ("exchange", pl.Utf8),
        ("security_type", pl.Utf8),
        ("sic_code", pl.Int64),
        ("listing_date", pl.Date),
        ("delisting_date", pl.Date),
        ("delisting_code", pl.Int64),
        ("delisting_return", pl.Float64),
    ]
)

FUNDAMENTALS_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("company_id", pl.Int64),          # gvkey
        ("security_id", pl.Int64),         # linked permno (nullable)
        ("period_end", pl.Date),           # datadate (fiscal period end)
        ("public_date", pl.Date),          # availability date (rdq/pdate) — for as-of joins
        ("freq", pl.Utf8),                 # 'Q' | 'A'
        ("fiscal_year", pl.Int64),
        ("fiscal_quarter", pl.Int64),      # nullable for annual
        ("revenue", pl.Float64),
        ("net_income", pl.Float64),
        ("total_assets", pl.Float64),
        ("total_liabilities", pl.Float64),
        ("cash", pl.Float64),
        ("total_debt", pl.Float64),
        ("shares_outstanding", pl.Float64),
    ]
)

CORPORATE_ACTION_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("security_id", pl.Int64),
        ("effective_date", pl.Date),
        ("action_type", pl.Utf8),          # see ACTION_TYPES
        ("value", pl.Float64),             # delisting_return / split ratio / dividend amount
        ("code", pl.Int64),                # provider code (e.g. CRSP dlstcd), nullable
        ("details", pl.Utf8),
    ]
)

SECTOR_METADATA_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("security_id", pl.Int64),
        ("sic_code", pl.Int64),
        ("sic_division", pl.Utf8),
    ]
)

ACTION_TYPES = ("DELISTING", "SPLIT", "DIVIDEND")

SCHEMAS: dict[str, OrderedDict[str, pl.DataType]] = {
    "daily_price": DAILY_PRICE_SCHEMA,
    "security_master": SECURITY_MASTER_SCHEMA,
    "fundamentals": FUNDAMENTALS_SCHEMA,
    "corporate_action": CORPORATE_ACTION_SCHEMA,
    "sector_metadata": SECTOR_METADATA_SCHEMA,
}


def empty_frame(schema: OrderedDict[str, pl.DataType]) -> pl.DataFrame:
    """An empty DataFrame with exactly the canonical columns/dtypes."""
    return pl.DataFrame(schema={k: v for k, v in schema.items()})


def validate_schema(
    df: pl.DataFrame,
    schema: OrderedDict[str, pl.DataType],
    *,
    coerce: bool = True,
) -> pl.DataFrame:
    """Enforce a canonical schema on ``df``.

    Requires every schema column to be present. With ``coerce=True`` (default), selects
    the columns in canonical order and casts to the declared dtypes (raising on an
    impossible cast); with ``coerce=False``, dtypes must already match exactly. Extra
    columns are dropped. Returns the conforming DataFrame.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    if not coerce:
        mismatched = {
            c: (df.schema[c], dt) for c, dt in schema.items() if df.schema[c] != dt
        }
        if mismatched:
            raise SchemaError(f"dtype mismatch (got, expected): {mismatched}")
        return df.select(list(schema.keys()))

    try:
        return df.select([pl.col(c).cast(dt) for c, dt in schema.items()])
    except Exception as e:  # noqa: BLE001 - surface as a schema error with context
        raise SchemaError(f"could not coerce to schema: {e}") from e
