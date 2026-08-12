"""Chronological train / validation / test splits with an outcome-window embargo.

Splits are assigned purely by ``crash_date`` against the configured date ranges — never by
random row sampling — so the arrangement approximates live deployment (train on the past,
predict the future).

**Embargo (the boundary treatment):** outcomes look up to ``max_horizon`` trading days
forward, so an event near the end of a split has an outcome window that spills into the next
split. Such events are reassigned to ``embargo`` (dropped from modeling) so a training event's
outcome can never overlap the validation/test period. Trading-day distance to the boundary is
measured against the real trading calendar, not calendar days.

Assigned labels: ``train`` | ``validation`` | ``test`` | ``embargo`` | ``none`` (outside all
configured ranges — e.g. pre-train history, kept only for robustness work).
"""
from __future__ import annotations

import polars as pl

SPLIT_NAMES = ("train", "validation", "test", "embargo", "none")


def assign_splits(
    events: pl.DataFrame,
    splits_config,
    calendar: pl.Series,
    *,
    embargo_trading_days: int | None = None,
) -> pl.DataFrame:
    """Return (event_id, crash_date, split) for each event.

    ``calendar`` is the set of real trading dates (e.g. distinct dates in the daily prices).
    ``splits_config`` is a crashback.config.SplitsConfig (train/validation/test as (start,end)
    date tuples + ``embargo_trading_days``).
    """
    emb = (
        splits_config.embargo_trading_days
        if embargo_trading_days is None
        else embargo_trading_days
    )
    cal = pl.DataFrame({"date": calendar.unique().sort()}).with_row_index("cal_idx")

    def end_idx(d) -> int:
        return cal.filter(pl.col("date") <= d)["cal_idx"].max()

    tr_s, tr_e = splits_config.train
    va_s, va_e = splits_config.validation
    te_s, te_e = splits_config.test
    tr_end_i, va_end_i = end_idx(tr_e), end_idx(va_e)

    e = events.select("event_id", "crash_date").join(
        cal, left_on="crash_date", right_on="date", how="left"
    ).rename({"cal_idx": "crash_idx"})

    def between(lo, hi):
        return (pl.col("crash_date") >= lo) & (pl.col("crash_date") <= hi)

    e = e.with_columns(
        pl.when(between(tr_s, tr_e)).then(pl.lit("train"))
        .when(between(va_s, va_e)).then(pl.lit("validation"))
        .when(between(te_s, te_e)).then(pl.lit("test"))
        .otherwise(pl.lit("none")).alias("split")
    )
    # Embargo events whose forward outcome window would cross into the next split.
    e = e.with_columns(
        pl.when((pl.col("split") == "train") & ((tr_end_i - pl.col("crash_idx")) < emb))
        .then(pl.lit("embargo"))
        .when((pl.col("split") == "validation") & ((va_end_i - pl.col("crash_idx")) < emb))
        .then(pl.lit("embargo"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    return e.select("event_id", "crash_date", "split")


def split_summary(
    df: pl.DataFrame, split_col: str = "split", target: str = "hit_10pct_20d",
    censored_col: str = "censored_20d",
) -> pl.DataFrame:
    """Per-split counts, date range, and target prevalence over determined events.

    ``df`` should already be restricted to the modeling pool (e.g. CLEAN) and carry the split,
    target, and censored columns.
    """
    determined = ~pl.col(censored_col) & pl.col(target).is_not_null()
    return (
        df.group_by(split_col)
        .agg(
            pl.len().alias("events"),
            pl.col("crash_date").min().alias("date_min"),
            pl.col("crash_date").max().alias("date_max"),
            determined.sum().alias("determined"),
            pl.col(target).filter(determined).mean().alias("target_prevalence"),
        )
        .sort(split_col)
    )
