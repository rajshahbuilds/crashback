"""Crash-event detection.

Every day with ``daily_return <= threshold`` is its own crash event — no cooldown, no
suppression based on later recovery or nearby crashes (CLAUDE.md sec 1/11). Because the
crash test uses CRSP ``daily_return`` (split/dividend adjusted), corporate actions never
create false events. Identifiers (company_id, ticker) are attached by a point-in-time join
to the security master's name-period covering the crash date; no crash is ever dropped
(unmatched identifiers are left null, never silently discarded).
"""
from __future__ import annotations

from collections import OrderedDict

import polars as pl

CRASH_EVENT_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("event_id", pl.Utf8),
        ("security_id", pl.Int64),
        ("company_id", pl.Int64),            # nullable if no covering name-period
        ("ticker_as_of_event", pl.Utf8),     # nullable
        ("crash_date", pl.Date),
        ("crash_return", pl.Float64),
        ("crash_close", pl.Float64),         # P0 — the label anchor (null on CRSP no-trade days)
        ("crash_volume", pl.Float64),
        ("in_universe_at_event", pl.Boolean),  # a universe name-period covers the crash date
        ("passes_min_price", pl.Boolean),    # point-in-time liquidity flag (crash_close >= min)
    ]
)


def detect_crashes(prices: pl.DataFrame | pl.LazyFrame, threshold: float) -> pl.DataFrame:
    """One row per (security_id, date) with daily_return <= threshold.

    A null daily_return is not a crash (null <= threshold is null -> excluded).
    """
    lf = prices.lazy()
    return (
        lf.filter(pl.col("daily_return") <= threshold)
        .select(
            pl.col("security_id").cast(pl.Int64),
            pl.col("date").cast(pl.Date).alias("crash_date"),
            pl.col("daily_return").cast(pl.Float64).alias("crash_return"),
            pl.col("close").cast(pl.Float64).alias("crash_close"),
            pl.col("volume").cast(pl.Float64).alias("crash_volume"),
        )
        .collect()
    )


def attach_identifiers(crashes: pl.DataFrame, master: pl.DataFrame) -> pl.DataFrame:
    """Attach company_id + ticker_as_of_event via a point-in-time name-period join.

    Keeps exactly one row per (security_id, crash_date). If a name-period covers the crash
    date, its identifiers are used; otherwise identifiers are null (the event is retained).
    """
    m = master.select(
        "security_id", "company_id", "ticker", "ticker_start", "ticker_end"
    )
    covers = (pl.col("crash_date") >= pl.col("ticker_start")) & (
        pl.col("crash_date") <= pl.col("ticker_end")
    )
    joined = crashes.join(m, on="security_id", how="left").with_columns(
        covers.fill_null(False).alias("_covers")
    )
    # Prefer a covering name-period; null out identifiers when none covers. The event is
    # always retained; `in_universe_at_event` records whether it was in a clean universe
    # name-period on the crash date (point-in-time membership).
    key = ["security_id", "crash_date"]
    return (
        joined.sort([*key, "_covers"], descending=[False, False, True])
        .with_columns(
            pl.when(pl.col("_covers")).then(pl.col("company_id")).alias("company_id"),
            pl.when(pl.col("_covers")).then(pl.col("ticker")).alias("ticker"),
        )
        .unique(subset=key, keep="first")
        .drop("ticker_start", "ticker_end")
        .rename({"ticker": "ticker_as_of_event", "_covers": "in_universe_at_event"})
    )


def build_crash_events(
    prices: pl.DataFrame | pl.LazyFrame,
    master: pl.DataFrame,
    *,
    threshold: float,
    min_price: float | None = None,
) -> pl.DataFrame:
    """Full crash-event table in the canonical CRASH_EVENT_SCHEMA."""
    crashes = detect_crashes(prices, threshold)
    events = attach_identifiers(crashes, master)
    passes = (
        (pl.col("crash_close") >= min_price) if min_price is not None else pl.lit(True)
    )
    events = events.with_columns(
        pl.concat_str(
            [pl.col("security_id").cast(pl.Utf8), pl.col("crash_date").dt.strftime("%Y%m%d")],
            separator="_",
        ).alias("event_id"),
        passes.alias("passes_min_price"),
    )
    return (
        events.select([pl.col(c).cast(dt) for c, dt in CRASH_EVENT_SCHEMA.items()])
        .sort(["security_id", "crash_date"])
    )
