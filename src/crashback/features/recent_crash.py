"""Recent-crash history features (point-in-time, a first-class family — CLAUDE.md sec 12).

For each crash event, these summarize the security's *prior* crashes and its path since the
previous crash. The load-bearing invariant: **today's crash never counts as one of its own
prior crashes** — every window/marker is shifted strictly before the current crash day.

A first-ever crash therefore has zero prior counts and null previous-crash fields.

Note: cumulative_return_* and drawdown_from_*_high (also listed under sec 12) are already
produced point-in-time by STU-51 (`return_{N}d_pre`, `distance_from_{N}d_high`); we do not
duplicate them here.
"""
from __future__ import annotations

import polars as pl

FEATURE_NAMES: tuple[str, ...] = (
    "prior_crash_count_5d",
    "prior_crash_count_20d",
    "prior_crash_count_60d",
    "days_since_previous_crash",
    "previous_crash_return",
    "return_since_previous_crash",
    "max_rebound_since_previous_crash",
)


def build_recent_crash_features(events: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Point-in-time recent-crash features, one row per event (keyed by event_id)."""
    sid = "security_id"
    p = prices.sort([sid, "date"]).with_columns(
        pl.int_range(pl.len()).over(sid).alias("seq")
    )
    # Mark which price rows are crash events.
    p = p.join(
        events.select(sid, "crash_date").with_columns(pl.lit(True).alias("is_crash")),
        left_on=[sid, "date"], right_on=[sid, "crash_date"], how="left",
    ).with_columns(pl.col("is_crash").fill_null(False))

    is_crash_i = pl.col("is_crash").cast(pl.Int64)
    # Prior-crash counts over the N trading days STRICTLY before the current day
    # (rolling_sum ends yesterday via shift(1); a fresh crash gets 0, never null).
    counts = [
        is_crash_i.rolling_sum(n, min_samples=1).shift(1).fill_null(0).over(sid)
        .cast(pl.Int64).alias(f"prior_crash_count_{n}d")
        for n in (5, 20, 60)
    ]
    # Markers of the most recent crash STRICTLY before the current day.
    prev_seq = (
        pl.when(pl.col("is_crash")).then(pl.col("seq"))
        .shift(1).forward_fill().over(sid)
    )
    prev_ret = (
        pl.when(pl.col("is_crash")).then(pl.col("daily_return"))
        .shift(1).forward_fill().over(sid)
    )
    prev_close = (
        pl.when(pl.col("is_crash")).then(pl.col("close"))
        .shift(1).forward_fill().over(sid)
    )
    # Segment that resets the row AFTER each crash, so a running max within it spans
    # (previous crash, current day].
    seg = is_crash_i.cum_sum().shift(1).fill_null(0).over(sid)

    p = p.with_columns(*counts, prev_seq.alias("_pseq"), prev_ret.alias("_pret"),
                       prev_close.alias("_pclose"), seg.alias("_seg"))
    p = p.with_columns(pl.col("close").cum_max().over([sid, "_seg"]).alias("_runmax"))
    p = p.with_columns(
        (pl.col("seq") - pl.col("_pseq")).alias("days_since_previous_crash"),
        pl.col("_pret").alias("previous_crash_return"),
        (pl.col("close") / pl.col("_pclose") - 1.0).alias("return_since_previous_crash"),
        (pl.col("_runmax") / pl.col("_pclose") - 1.0).alias("max_rebound_since_previous_crash"),
    )

    return events.select("event_id", sid, "crash_date").join(
        p.select(sid, "date", *FEATURE_NAMES),
        left_on=[sid, "crash_date"], right_on=[sid, "date"], how="left",
    )
