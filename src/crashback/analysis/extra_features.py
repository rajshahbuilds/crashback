"""Two point-in-time features computed on the fly for the descriptive analysis: 252-day market
beta and trailing EBITDA margin. Kept separate from the modeling feature pipeline so events_v1
and the model feature set are untouched.
"""
from __future__ import annotations

import polars as pl

from crashback.fundamentals.features import (
    _finite,
    enrich_fundamentals,
    link_permno_gvkey,
)
from crashback.ingestion.prices import scan_daily_prices


def _clean_events(cfg) -> pl.DataFrame:
    return pl.read_parquet(cfg.paths.resolve("data_events") / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price"))


def event_beta(cfg, window: int = 252, min_samples: int = 120) -> pl.DataFrame:
    """Per-event market beta over the ``window`` trading days ending the day before the crash.

    Market = equal-weighted mean daily return across all securities. beta = cov(stock, mkt) /
    var(mkt) via rolling moments; shifted one day so the crash-day shock is excluded. Returns
    (event_id, beta).
    """
    norm = cfg.paths.resolve("data_normalized")
    mkt = (scan_daily_prices(norm / "daily_prices").select("date", "daily_return")
           .group_by("date").agg(mret=pl.col("daily_return").mean()).sort("date").collect())
    ev = _clean_events(cfg).select("event_id", "security_id", "crash_date")
    sids = ev["security_id"].unique().to_list()
    px = (scan_daily_prices(norm / "daily_prices").filter(pl.col("security_id").is_in(sids))
          .select("security_id", "date", "daily_return").sort(["security_id", "date"]).collect()
          .join(mkt, on="date", how="left")
          .with_columns(x=pl.col("daily_return").fill_null(0.0), y=pl.col("mret").fill_null(0.0)))
    r = {"window_size": window, "min_samples": min_samples}
    px = px.with_columns(
        mx=pl.col("x").rolling_mean(**r).over("security_id"),
        my=pl.col("y").rolling_mean(**r).over("security_id"),
        mxy=(pl.col("x") * pl.col("y")).rolling_mean(**r).over("security_id"),
        myy=(pl.col("y") * pl.col("y")).rolling_mean(**r).over("security_id"))
    px = px.with_columns(
        beta=_finite((pl.col("mxy") - pl.col("mx") * pl.col("my"))
                     / (pl.col("myy") - pl.col("my") ** 2))
    ).with_columns(beta=pl.col("beta").shift(1).over("security_id"))
    return (ev.join(px.select("security_id", "date", "beta"),
                    left_on=["security_id", "crash_date"], right_on=["security_id", "date"],
                    how="left").select("event_id", "beta"))


def event_ebitda_margin(cfg) -> pl.DataFrame:
    """Per-event trailing-twelve-month EBITDA margin (EBITDA_ttm / revenue_ttm), point-in-time.

    Uses the saved CRSP/Compustat link (``data/normalized/ccm_link.parquet``) and a backward
    as-of join on public_date. Returns (event_id, ebitda_margin)."""
    norm = cfg.paths.resolve("data_normalized")
    link = pl.read_parquet(norm / "ccm_link.parquet")
    ev = link_permno_gvkey(
        _clean_events(cfg).select("event_id", "security_id", "crash_date", "crash_close"), link)
    ef = enrich_fundamentals(
        pl.read_parquet(norm / "fundamentals" / "fundamentals_v1.parquet")
    ).sort(["company_id", "public_date"])
    matched = ev.filter(pl.col("company_id").is_not_null()).sort("crash_date")
    joined = matched.join_asof(ef, left_on="crash_date", right_on="public_date",
                               by="company_id", strategy="backward")
    return joined.with_columns(
        ebitda_margin=_finite(pl.col("ebitda_ttm") / pl.col("revenue_ttm"))
    ).select("event_id", "ebitda_margin")
