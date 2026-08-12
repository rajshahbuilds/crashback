"""Point-in-time fundamentals ingestion from Compustat **unrestated** quarterly (comp_urq.urqus).

Restatement policy (documented, leakage-safe): we use the UNRESTATED file, whose values are
as *originally reported* — a later restatement of an old quarter is never reflected, so no
future revision leaks into an earlier date. Availability is the earnings report date ``rdq``;
where ``rdq`` is missing (~20%), we fall back to ``period_end + 60 days`` (a conservative
quarterly filing lag) and flag it via ``rdq_available = False`` so the estimate is visible and
the as-of join (STU-55) never assumes availability too early.

Grain: one row per (company_id, period_end) — the source is already one row per firm-quarter.

Known gap: the unrestated file carries no cash-flow-statement items, so free-cash-flow
features (fcf_margin, fcf_yield) are NOT available from this source. Documented; can be
back-filled from restated ``comp.fundq`` later if needed (isolating restatement risk to FCF).
"""
from __future__ import annotations

from collections import OrderedDict

import polars as pl

# Raw comp_urq.urqus columns to pull.
RAW_FIELDS: tuple[str, ...] = (
    "gvkey", "datadate", "rdq", "fqtr", "fyrq",
    "saleq", "cogsq", "oibdpq", "dpq", "niq", "epspxq", "xintq",
    "atq", "ltq", "cheq", "dlttq", "dlcq", "actq", "lctq", "seqq", "ceqq", "cshoq",
)

# Availability fallback when rdq is missing (conservative quarterly filing lag).
RDQ_FALLBACK_DAYS = 60

FUNDAMENTALS_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("company_id", pl.Int64),          # gvkey
        ("period_end", pl.Date),           # datadate (fiscal quarter end)
        ("public_date", pl.Date),          # rdq, or period_end + 60d when rdq missing
        ("rdq_available", pl.Boolean),     # False => public_date is the estimated fallback
        ("freq", pl.Utf8),
        ("fiscal_year", pl.Int64),
        ("fiscal_quarter", pl.Int64),
        # income statement
        ("revenue", pl.Float64),
        ("cogs", pl.Float64),
        ("gross_profit", pl.Float64),
        ("ebitda", pl.Float64),            # oibdpq (operating income before D&A)
        ("depreciation", pl.Float64),
        ("operating_income", pl.Float64),  # oibdpq - dpq (~ oiadpq)
        ("net_income", pl.Float64),
        ("eps", pl.Float64),
        ("interest_expense", pl.Float64),
        # balance sheet
        ("total_assets", pl.Float64),
        ("total_liabilities", pl.Float64),
        ("cash", pl.Float64),
        ("debt_long_term", pl.Float64),
        ("debt_current", pl.Float64),
        ("total_debt", pl.Float64),
        ("current_assets", pl.Float64),
        ("current_liabilities", pl.Float64),
        ("stockholders_equity", pl.Float64),
        ("common_equity", pl.Float64),
        ("shares_outstanding", pl.Float64),
    ]
)


def normalize_urqus(df: pl.DataFrame) -> pl.DataFrame:
    """comp_urq.urqus raw rows -> canonical unrestated point-in-time fundamentals."""
    def f(col: str) -> pl.Expr:
        return pl.col(col).cast(pl.Float64)

    period_end = pl.col("datadate").cast(pl.Date)
    rdq = pl.col("rdq").cast(pl.Date)
    out = df.select(
        pl.col("gvkey").cast(pl.Int64).alias("company_id"),
        period_end.alias("period_end"),
        pl.coalesce([rdq, period_end.dt.offset_by(f"{RDQ_FALLBACK_DAYS}d")]).alias("public_date"),
        rdq.is_not_null().alias("rdq_available"),
        pl.lit("Q").alias("freq"),
        pl.col("fyrq").cast(pl.Int64).alias("fiscal_year"),
        pl.col("fqtr").cast(pl.Int64).alias("fiscal_quarter"),
        f("saleq").alias("revenue"),
        f("cogsq").alias("cogs"),
        (f("saleq") - f("cogsq")).alias("gross_profit"),
        f("oibdpq").alias("ebitda"),
        f("dpq").alias("depreciation"),
        (f("oibdpq") - f("dpq")).alias("operating_income"),
        f("niq").alias("net_income"),
        f("epspxq").alias("eps"),
        f("xintq").alias("interest_expense"),
        f("atq").alias("total_assets"),
        f("ltq").alias("total_liabilities"),
        f("cheq").alias("cash"),
        f("dlttq").alias("debt_long_term"),
        f("dlcq").alias("debt_current"),
        # null only when BOTH debt fields are missing (don't mask missing data as zero debt)
        pl.when(f("dlttq").is_null() & f("dlcq").is_null())
        .then(None)
        .otherwise(pl.sum_horizontal(f("dlttq"), f("dlcq")))
        .alias("total_debt"),
        f("actq").alias("current_assets"),
        f("lctq").alias("current_liabilities"),
        f("seqq").alias("stockholders_equity"),
        f("ceqq").alias("common_equity"),
        f("cshoq").alias("shares_outstanding"),
    )
    return out.select([pl.col(c).cast(dt) for c, dt in FUNDAMENTALS_SCHEMA.items()]).sort(
        ["company_id", "period_end"]
    )
