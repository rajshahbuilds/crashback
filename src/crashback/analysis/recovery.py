"""Shared one-year post-crash outcome, keyed by event (for the descriptive feature analysis).

Single source of truth for the survivorship-safe one-year return used across the writeup: from
each crash-day close, compound total daily returns over the next ``horizon`` trading days (so
splits, dividends, and delisting/bankruptcy terminal returns are all included). Crashes without a
full forward window before the data edge are censored. Returns one row per observed CLEAN event
so any feature can be joined and bucketed against recovery.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from crashback.ingestion.prices import scan_daily_prices

HORIZON = 252


def one_year_returns(cfg, horizon: int = HORIZON) -> pl.DataFrame:
    """DataFrame(event_id, security_id, crash_date, ret) for observed CLEAN crash events."""
    norm = cfg.paths.resolve("data_normalized")
    events_dir = cfg.paths.resolve("data_events")
    ev = pl.read_parquet(events_dir / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).select("event_id", "security_id", "crash_date")
    sids = ev["security_id"].unique().to_list()

    px = (scan_daily_prices(norm / "daily_prices")
          .filter(pl.col("security_id").is_in(sids))
          .select("security_id", "date", "daily_return").sort(["security_id", "date"])
          .with_columns(
              td_idx=pl.int_range(pl.len()).over("security_id"),
              cumlog=(pl.col("daily_return").fill_null(0.0) + 1.0).log().cum_sum()
              .over("security_id"))
          .collect())
    last = px.group_by("security_id").agg(last_idx=pl.col("td_idx").max(),
                                          last_date=pl.col("date").max())
    idx = px.select("security_id", "td_idx", "cumlog")
    edge = px["date"].max() - timedelta(days=17)

    e = (ev.join(px.select("security_id", "date", "td_idx", "cumlog"),
                 left_on=["security_id", "crash_date"], right_on=["security_id", "date"],
                 how="inner")
         .rename({"td_idx": "k0", "cumlog": "cl0"})
         .join(last, on="security_id")
         .with_columns(end_idx=pl.min_horizontal(pl.col("k0") + horizon, pl.col("last_idx")),
                       fwd=pl.col("last_idx") - pl.col("k0"))
         .join(idx.rename({"td_idx": "end_idx", "cumlog": "cl_end"}),
               on=["security_id", "end_idx"], how="left")
         .with_columns(ret=(pl.col("cl_end") - pl.col("cl0")).exp() - 1.0,
                       censored=(pl.col("fwd") < horizon) & (pl.col("last_date") >= edge)))
    return (e.filter(~pl.col("censored") & pl.col("ret").is_not_null())
            .select("event_id", "security_id", "crash_date", "ret"))
