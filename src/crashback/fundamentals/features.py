"""Leakage-free as-of join of fundamentals to crash events + derived §15 features.

The core guarantee: an event only ever receives the most recent fundamentals whose
**availability date (`public_date`) is <= the crash date** — never a quarter first published
after the crash. That is enforced by a backward `join_asof` on `public_date`, and covered by
dedicated leakage tests that inject future records and confirm they are excluded.

Steps: (1) point-in-time permno->gvkey link via CCM (link valid on the crash date);
(2) enrich the fundamentals series with trailing-twelve-month (TTM) sums + YoY/QoQ growth
(all backward, so point-in-time safe); (3) backward as-of join by `public_date`;
(4) derive profitability / balance-sheet / valuation ratios; (5) flag missing & stale.

Valuation uses the event-date price (`crash_close`) with the latest known shares (`cshoq`).
FCF ratios are omitted (the unrestated fundamentals source has no cash-flow items — STU-54).
"""
from __future__ import annotations

import polars as pl

# A fundamental attached to an event is "stale" if its period ended > this many days before.
STALE_DAYS = 200

DERIVED_FEATURES: tuple[str, ...] = (
    # growth
    "revenue_growth_yoy", "revenue_growth_qoq", "eps_growth_yoy",
    # profitability (TTM)
    "gross_margin", "operating_margin", "net_margin", "roa", "roe",
    # balance sheet
    "net_debt", "current_ratio", "debt_to_assets", "net_debt_to_ebitda", "interest_coverage",
    # valuation (event price x latest fundamentals)
    "market_cap", "pe", "price_to_sales", "ev_to_sales", "ev_to_ebitda",
    # meta / provenance
    "fundamentals_available", "fundamentals_period_end", "fundamentals_public_date",
    "days_since_fundamental", "fundamentals_stale", "fundamentals_rdq_available",
)

_LINK_TYPES = ("LU", "LC")
_LINK_PRIMS = ("P", "C")


def link_permno_gvkey(events: pl.DataFrame, link: pl.DataFrame) -> pl.DataFrame:
    """Attach the gvkey whose CCM link was valid on the crash date (point-in-time).

    `link` columns: gvkey, lpermno, linktype, linkprim, linkdt, linkenddt (null = still active).
    """
    lk = link.filter(
        pl.col("linktype").is_in(_LINK_TYPES) & pl.col("linkprim").is_in(_LINK_PRIMS)
    ).select(
        pl.col("lpermno").cast(pl.Int64).alias("security_id"),
        pl.col("gvkey").cast(pl.Int64).alias("company_id"),
        pl.col("linkprim"),
        pl.col("linkdt").cast(pl.Date).alias("linkdt"),
        pl.col("linkenddt").cast(pl.Date).alias("linkenddt"),
    )
    valid = (pl.col("crash_date") >= pl.col("linkdt")) & (
        pl.col("linkenddt").is_null() | (pl.col("crash_date") <= pl.col("linkenddt"))
    )
    # Left join keeps every event; null out gvkey where the link is not valid on the crash
    # date (so an event is never dropped — it just gets no fundamentals).
    joined = events.join(lk, on="security_id", how="left").with_columns(
        valid.fill_null(False).alias("_valid")
    ).with_columns(
        pl.when(pl.col("_valid")).then(pl.col("company_id")).alias("company_id"),
        (pl.col("_valid") & (pl.col("linkprim") == "P")).alias("_prim"),
    )
    # One row per event: prefer a valid link, primary first.
    return (
        joined.sort(
            ["event_id", pl.col("company_id").is_not_null(), "_prim"],
            descending=[False, True, True],
        )
        .unique(subset="event_id", keep="first")
        .select("event_id", "security_id", "crash_date", "crash_close", "company_id")
    )


def enrich_fundamentals(fund: pl.DataFrame) -> pl.DataFrame:
    """Add TTM sums and YoY/QoQ growth to the normalized fundamentals (per company, backward)."""
    g = "company_id"
    f = fund.sort([g, "period_end"])

    def ttm(col: str) -> pl.Expr:
        return pl.col(col).rolling_sum(4, min_samples=4).over(g)

    return f.with_columns(
        ttm("revenue").alias("revenue_ttm"),
        ttm("net_income").alias("ni_ttm"),
        ttm("gross_profit").alias("gross_profit_ttm"),
        ttm("operating_income").alias("operating_income_ttm"),
        ttm("ebitda").alias("ebitda_ttm"),
        ttm("interest_expense").alias("interest_ttm"),
        (pl.col("revenue") / pl.col("revenue").shift(4).over(g) - 1.0).alias("revenue_growth_yoy"),
        (pl.col("revenue") / pl.col("revenue").shift(1).over(g) - 1.0).alias("revenue_growth_qoq"),
        (pl.col("eps") / pl.col("eps").shift(4).over(g) - 1.0).alias("eps_growth_yoy"),
    )


def _finite(expr: pl.Expr) -> pl.Expr:
    """Replace inf/-inf (division by ~0) with null so ratios stay clean."""
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


def build_fundamental_features(
    events: pl.DataFrame, fundamentals: pl.DataFrame, link: pl.DataFrame
) -> pl.DataFrame:
    """Per-event fundamental features via a leakage-free as-of join (keyed by event_id)."""
    ev = link_permno_gvkey(events, link)
    ef = enrich_fundamentals(fundamentals).sort(["company_id", "public_date"])

    # Backward as-of: the latest fundamentals with public_date <= crash_date, per company.
    matched = ev.filter(pl.col("company_id").is_not_null()).sort("crash_date")
    unmatched = ev.filter(pl.col("company_id").is_null())
    joined = matched.join_asof(
        ef, left_on="crash_date", right_on="public_date", by="company_id", strategy="backward"
    )
    joined = pl.concat([joined, unmatched], how="diagonal")

    mc = pl.col("crash_close") * pl.col("shares_outstanding")
    net_debt = pl.col("total_debt") - pl.col("cash")
    ev_val = mc + net_debt
    days_since = (pl.col("crash_date") - pl.col("period_end")).dt.total_days()

    out = joined.with_columns(
        pl.col("period_end").is_not_null().alias("fundamentals_available"),
        pl.col("period_end").alias("fundamentals_period_end"),
        pl.col("public_date").alias("fundamentals_public_date"),
        days_since.alias("days_since_fundamental"),
        (days_since > STALE_DAYS).alias("fundamentals_stale"),
        pl.col("rdq_available").fill_null(False).alias("fundamentals_rdq_available"),
        _finite(net_debt).alias("net_debt"),
        _finite(mc).alias("market_cap"),
        _finite(pl.col("gross_profit_ttm") / pl.col("revenue_ttm")).alias("gross_margin"),
        _finite(pl.col("operating_income_ttm") / pl.col("revenue_ttm")).alias("operating_margin"),
        _finite(pl.col("ni_ttm") / pl.col("revenue_ttm")).alias("net_margin"),
        _finite(pl.col("ni_ttm") / pl.col("total_assets")).alias("roa"),
        _finite(pl.col("ni_ttm") / pl.col("stockholders_equity")).alias("roe"),
        _finite(pl.col("current_assets") / pl.col("current_liabilities")).alias("current_ratio"),
        _finite(pl.col("total_debt") / pl.col("total_assets")).alias("debt_to_assets"),
        _finite(net_debt / pl.col("ebitda_ttm")).alias("net_debt_to_ebitda"),
        _finite(pl.col("operating_income_ttm") / pl.col("interest_ttm")).alias("interest_coverage"),
        _finite(mc / pl.col("ni_ttm")).alias("pe"),
        _finite(mc / pl.col("revenue_ttm")).alias("price_to_sales"),
        _finite(ev_val / pl.col("revenue_ttm")).alias("ev_to_sales"),
        _finite(ev_val / pl.col("ebitda_ttm")).alias("ev_to_ebitda"),
    )
    return out.select("event_id", *DERIVED_FEATURES).sort("event_id")
