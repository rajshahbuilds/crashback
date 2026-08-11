"""Pure normalization: CRSP CIZ / Compustat raw frames -> canonical schemas.

These functions contain the only knowledge of vendor column names in the codebase. They
are pure (DataFrame in, DataFrame out) so the mapping is unit-tested without a live WRDS
connection. The WRDS adapter is a thin layer that runs SQL and calls these.
"""
from __future__ import annotations

import polars as pl

from crashback.providers.schemas import (
    CORPORATE_ACTION_SCHEMA,
    DAILY_PRICE_SCHEMA,
    FUNDAMENTALS_SCHEMA,
    SECTOR_METADATA_SCHEMA,
    SECURITY_MASTER_SCHEMA,
    validate_schema,
)

# CIZ single-character primary-exchange codes -> canonical labels (unknowns pass through).
EXCHANGE_LABELS = {"N": "NYSE", "A": "NYSE American", "Q": "Nasdaq"}

# Standard SIC major-division ranges (inclusive) -> label.
_SIC_DIVISIONS = [
    (100, 999, "Agriculture, Forestry & Fishing"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation & Public Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance & Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9729, "Public Administration"),
]


def sic_division(sic: int | None) -> str:
    """Coarse SIC major-division label; 'Unknown' for null/out-of-range codes."""
    if sic is None:
        return "Unknown"
    for lo, hi, label in _SIC_DIVISIONS:
        if lo <= sic <= hi:
            return label
    return "Nonclassifiable"


def _exchange_expr(col: str) -> pl.Expr:
    # Map known CIZ exchange codes to labels; unknown codes pass through unchanged.
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .replace_strict(EXCHANGE_LABELS, default=pl.col(col).cast(pl.Utf8))
        .alias("exchange")
    )


def _sic_division_expr(col: str) -> pl.Expr:
    expr = pl.lit("Unknown")
    # Build nested when/then from the ranges (evaluated in order).
    for lo, hi, label in _SIC_DIVISIONS:
        expr = (
            pl.when((pl.col(col) >= lo) & (pl.col(col) <= hi))
            .then(pl.lit(label))
            .otherwise(expr)
        )
    # Out-of-range but non-null -> Nonclassifiable; null -> Unknown.
    out_of_range = (
        pl.when(expr == pl.lit("Unknown"))
        .then(pl.lit("Nonclassifiable"))
        .otherwise(expr)
    )
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.lit("Unknown"))
        .otherwise(out_of_range)
        .alias("sic_division")
    )


def normalize_ciz_prices(df: pl.DataFrame) -> pl.DataFrame:
    """CRSP CIZ ``dsf_v2`` rows -> canonical daily_price.

    Crash detection must use ``daily_return`` (CRSP ``dlyret`` — already split/dividend
    adjusted), never raw close-to-close deltas, so splits never look like crashes.
    ``adjusted_close`` = close / cumulative price factor (falls back to close if the
    factor is null/zero).
    """
    out = df.select(
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        pl.col("permno").cast(pl.Int64).alias("security_id"),
        pl.col("dlyopen").cast(pl.Float64).alias("open"),
        pl.col("dlyhigh").cast(pl.Float64).alias("high"),
        pl.col("dlylow").cast(pl.Float64).alias("low"),
        pl.col("dlyclose").cast(pl.Float64).alias("close"),
        pl.when(pl.col("dlycumfacpr").cast(pl.Float64).fill_null(0) != 0)
        .then(pl.col("dlyclose").cast(pl.Float64) / pl.col("dlycumfacpr").cast(pl.Float64))
        .otherwise(pl.col("dlyclose").cast(pl.Float64))
        .alias("adjusted_close"),
        pl.col("dlyvol").cast(pl.Float64).alias("volume"),
        pl.col("dlyret").cast(pl.Float64).alias("daily_return"),
        pl.col("dlyretx").cast(pl.Float64).alias("daily_return_ex_div"),
        pl.col("dlycumfacpr").cast(pl.Float64).alias("cum_factor_price"),
        pl.col("dlycumfacshr").cast(pl.Float64).alias("cum_factor_shares"),
        pl.col("shrout").cast(pl.Float64).alias("shares_outstanding"),
    )
    return validate_schema(out, DAILY_PRICE_SCHEMA)


def normalize_ciz_security_master(
    names: pl.DataFrame, delistings: pl.DataFrame | None = None
) -> pl.DataFrame:
    """CRSP CIZ ``stocknames_v2`` (+ optional ``dsedelist``) -> canonical security_master.

    Grain is one row per (security_id, ticker period): ticker history is preserved.
    """
    out = names.select(
        pl.col("permno").cast(pl.Int64).alias("security_id"),
        pl.col("permco").cast(pl.Int64).alias("company_id"),
        pl.col("ticker").cast(pl.Utf8).alias("ticker"),
        pl.col("namedt").cast(pl.Date).alias("ticker_start"),
        pl.col("nameenddt").cast(pl.Date).alias("ticker_end"),
        _exchange_expr("primaryexch"),
        pl.col("securitytype").cast(pl.Utf8).alias("security_type"),
        pl.col("siccd").cast(pl.Int64).alias("sic_code"),
        pl.col("securitybegdt").cast(pl.Date).alias("listing_date"),
        pl.col("securityenddt").cast(pl.Date).alias("delisting_date"),
    )
    if delistings is not None and delistings.height > 0:
        dl = delistings.select(
            pl.col("permno").cast(pl.Int64).alias("security_id"),
            pl.col("dlstcd").cast(pl.Int64).alias("delisting_code"),
            pl.col("dlret").cast(pl.Float64).alias("delisting_return"),
        )
        out = out.join(dl, on="security_id", how="left")
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("delisting_code"),
            pl.lit(None, dtype=pl.Float64).alias("delisting_return"),
        )
    return validate_schema(out, SECURITY_MASTER_SCHEMA)


def normalize_ciz_delistings(dsedelist: pl.DataFrame) -> pl.DataFrame:
    """CRSP ``dsedelist`` rows -> canonical corporate_action (action_type=DELISTING).

    Splits/dividends are captured in the price table's cumulative factors and in
    ``daily_return``; the delisting event is the corporate action we model explicitly.
    """
    out = dsedelist.select(
        pl.col("permno").cast(pl.Int64).alias("security_id"),
        pl.col("dlstdt").cast(pl.Date).alias("effective_date"),
        pl.lit("DELISTING").alias("action_type"),
        pl.col("dlret").cast(pl.Float64).alias("value"),
        pl.col("dlstcd").cast(pl.Int64).alias("code"),
        pl.concat_str([pl.lit("dlstcd="), pl.col("dlstcd").cast(pl.Utf8)]).alias("details"),
    )
    return validate_schema(out, CORPORATE_ACTION_SCHEMA)


def normalize_ciz_sector(names: pl.DataFrame) -> pl.DataFrame:
    """CRSP CIZ ``stocknames_v2`` -> canonical sector_metadata (SIC + coarse division)."""
    out = names.select(
        pl.col("permno").cast(pl.Int64).alias("security_id"),
        pl.col("siccd").cast(pl.Int64).alias("sic_code"),
        _sic_division_expr("siccd"),
    ).unique(subset=["security_id", "sic_code"], keep="first")
    return validate_schema(out, SECTOR_METADATA_SCHEMA)


# Compustat column maps per frequency: canonical_name -> raw column.
_FUND_COLS = {
    "Q": {
        "period_end": "datadate",
        "public_date": "rdq",
        "fiscal_year": "fyearq",
        "fiscal_quarter": "fqtr",
        "revenue": "revtq",
        "net_income": "niq",
        "total_assets": "atq",
        "total_liabilities": "ltq",
        "cash": "cheq",
        "debt_lt": "dlttq",
        "debt_curr": "dlcq",
        "shares_outstanding": "cshoq",
    },
    "A": {
        "period_end": "datadate",
        "public_date": "pdate",
        "fiscal_year": "fyear",
        "fiscal_quarter": None,
        "revenue": "revt",
        "net_income": "ni",
        "total_assets": "at",
        "total_liabilities": "lt",
        "cash": "che",
        "debt_lt": "dltt",
        "debt_curr": "dlc",
        "shares_outstanding": "csho",
    },
}


def normalize_compustat_fundamentals(
    df: pl.DataFrame, freq: str = "Q", links: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Compustat ``fundq`` (freq='Q') or ``funda`` (freq='A') -> canonical fundamentals.

    ``public_date`` (rdq/pdate) is the point-in-time availability date used by leakage-free
    as-of joins (STU-55). ``total_debt`` = long-term + current debt (nulls treated as 0).
    If ``links`` (gvkey->security_id) is given, ``security_id`` is attached, else null.
    """
    if freq not in _FUND_COLS:
        raise ValueError(f"freq must be 'Q' or 'A', got {freq!r}")
    m = _FUND_COLS[freq]

    def num(name: str) -> pl.Expr:
        raw = m[name]
        return pl.col(raw).cast(pl.Float64).alias(name)

    fq_expr = (
        pl.col(m["fiscal_quarter"]).cast(pl.Int64).alias("fiscal_quarter")
        if m["fiscal_quarter"]
        else pl.lit(None, dtype=pl.Int64).alias("fiscal_quarter")
    )

    out = df.select(
        pl.col("gvkey").cast(pl.Int64).alias("company_id"),
        pl.col(m["period_end"]).cast(pl.Date).alias("period_end"),
        pl.col(m["public_date"]).cast(pl.Date).alias("public_date"),
        pl.lit(freq).alias("freq"),
        pl.col(m["fiscal_year"]).cast(pl.Int64).alias("fiscal_year"),
        fq_expr,
        num("revenue"),
        num("net_income"),
        num("total_assets"),
        num("total_liabilities"),
        num("cash"),
        pl.sum_horizontal(
            pl.col(m["debt_lt"]).cast(pl.Float64), pl.col(m["debt_curr"]).cast(pl.Float64)
        ).alias("total_debt"),
        num("shares_outstanding"),
    )

    if links is not None and links.height > 0:
        lk = links.select(
            pl.col("gvkey").cast(pl.Int64).alias("company_id"),
            pl.col("lpermno").cast(pl.Int64).alias("security_id"),
        ).unique(subset=["company_id"], keep="first")
        out = out.join(lk, on="company_id", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("security_id"))

    return validate_schema(out, FUNDAMENTALS_SCHEMA)
