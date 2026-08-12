"""Recovery labels and continuous outcomes for crash events.

All outcomes are anchored to the crash-day close ``crash_close`` (P0) and measured over
forward *trading-day* windows. The full binary grid is {+5,+10,+20%} x {5,20,60d}; the
primary target is ``hit_10pct_20d`` (close-based). Intraday-high variants are stored with a
``_hi`` suffix so they can never be confused with the close-based primaries.

Censoring vs delisting (the crucial distinction):
  * An **active** security whose window runs past the data edge is **censored** (label null),
    never counted as a failure.
  * A **delisted** security's outcome is **known**: we augment its price path with a terminal
    node valued at ``last_close * (1 + delisting_return)`` so a bankruptcy that never rebounded
    is a real ``hit=0`` (and its drawdown reflects the loss) — not dropped. Censoring the
    delisted names would re-introduce survivorship bias.

Missing CRSP delisting returns are flagged (`delisting_return_missing`) rather than imputed.
"""
from __future__ import annotations

import polars as pl

HORIZONS: tuple[int, ...] = (5, 20, 60)
# (percent label, fraction)
THRESHOLDS: tuple[tuple[int, float], ...] = ((5, 0.05), (10, 0.10), (20, 0.20))


def hit_col(pct: int, h: int, hi: bool = False) -> str:
    return f"hit_{pct}pct_{h}d" + ("_hi" if hi else "")


def return_col(h: int) -> str:
    return f"return_{h}d"


def rebound_col(h: int) -> str:
    return f"max_rebound_{h}d"


def drawdown_col(h: int) -> str:
    return f"max_drawdown_{h}d"


def censored_col(h: int) -> str:
    return f"censored_{h}d"


def nforward_col(h: int) -> str:
    return f"n_forward_{h}d"


def augment_prices_with_delisting(prices: pl.DataFrame, master: pl.DataFrame) -> pl.DataFrame:
    """Append a terminal node (last_close * (1+delisting_return)) for delisted securities.

    The terminal sits one calendar day after the last real close, so it becomes the next
    trading-day step in the forward window and folds the delisting return into all outcomes.
    """
    base = prices.select("security_id", "date", "close", "high")
    valid = base.filter(pl.col("close").is_not_null()).sort(["security_id", "date"])
    last = valid.group_by("security_id", maintain_order=True).agg(
        last_date=pl.col("date").last(),
        last_close=pl.col("close").last(),
    )
    delist = (
        master.group_by("security_id")
        .agg(
            delisting_date=pl.col("delisting_date").max(),
            delisting_return=pl.col("delisting_return").drop_nulls().first(),
        )
        .filter(pl.col("delisting_date").is_not_null())
    )
    terminal = (
        last.join(delist, on="security_id", how="inner")
        .with_columns(
            (pl.col("last_close") * (1.0 + pl.col("delisting_return").fill_null(0.0)))
            .alias("close"),
            pl.col("last_date").dt.offset_by("1d").alias("date"),
        )
        .select(
            "security_id", "date", "close",
            pl.col("close").alias("high"),
        )
    )
    return pl.concat([base, terminal], how="vertical").sort(["security_id", "date"])


def _security_delisting_flags(master: pl.DataFrame) -> pl.DataFrame:
    return master.group_by("security_id").agg(
        is_delisted=pl.col("delisting_date").max().is_not_null(),
        _dl_ret=pl.col("delisting_return").drop_nulls().first(),
        _any_delist=pl.col("delisting_date").max(),
    ).with_columns(
        delisting_return_missing=(
            pl.col("_any_delist").is_not_null() & pl.col("_dl_ret").is_null()
        )
    ).select("security_id", "is_delisted", "delisting_return_missing")


def build_labels(
    events: pl.DataFrame,
    prices: pl.DataFrame,
    master: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    thresholds: tuple[tuple[int, float], ...] = THRESHOLDS,
) -> pl.DataFrame:
    """Compute the full label grid + continuous outcomes for every crash event."""
    max_h = max(horizons)
    aug = augment_prices_with_delisting(prices, master)
    prices_seq = aug.sort(["security_id", "date"]).with_columns(
        pl.int_range(pl.len()).over("security_id").alias("seq")
    )

    # Crash-day sequence index + P0.
    ev = events.select("event_id", "security_id", "crash_date", "crash_close").join(
        prices_seq.select("security_id", "date", "seq"),
        left_on=["security_id", "crash_date"],
        right_on=["security_id", "date"],
        how="left",
    ).rename({"seq": "i", "crash_close": "p0"})

    # Fan out forward trading-day offsets 1..max_h and pull the forward close/high.
    fwd = (
        ev.with_columns(pl.int_ranges(1, max_h + 1).alias("offset"))
        .explode("offset")
        .with_columns((pl.col("i") + pl.col("offset")).alias("target_seq"))
        .join(
            prices_seq.select("security_id", "seq", "close", "high").with_columns(
                pl.lit(True).alias("_row")
            ),
            left_on=["security_id", "target_seq"],
            right_on=["security_id", "seq"],
            how="left",
        )
        .rename({"close": "f_close", "high": "f_high"})
        .sort(["event_id", "offset"])
    )

    # Per-event, per-horizon aggregates.
    agg_exprs = [
        pl.col("p0").first().alias("p0"),
        pl.col("security_id").first().alias("security_id"),
        pl.col("crash_date").first().alias("crash_date"),
    ]
    for h in horizons:
        within = pl.col("offset") <= h
        agg_exprs += [
            pl.col("f_close").filter(within).max().alias(f"_maxc_{h}"),
            pl.col("f_close").filter(within).min().alias(f"_minc_{h}"),
            pl.col("f_high").filter(within).max().alias(f"_maxh_{h}"),
            pl.col("f_close").filter(within).is_not_null().sum().alias(nforward_col(h)),
            pl.col("_row").filter(within).fill_null(False).sum().alias(f"_nrows_{h}"),
            pl.col("f_close").filter(within).drop_nulls().last().alias(f"_endc_{h}"),
        ]
    grouped = fwd.group_by("event_id").agg(agg_exprs)

    flags = _security_delisting_flags(master)
    grouped = grouped.join(flags, on="security_id", how="left").with_columns(
        pl.col("is_delisted").fill_null(False)
    )

    # Labels + continuous outcomes.
    #   no_close_data : the crash-day close (P0) is null -> nothing is anchorable (CRSP
    #                   no-trade days). All labels null; this is distinct from censoring.
    #   censored_{h}d : valid P0, an ACTIVE security whose window runs past the data edge
    #                   (fewer than h forward trading rows). Genuine horizon censoring.
    #   determined    : valid P0 and (delisted OR >= h forward trading rows).
    has_p0 = pl.col("p0").is_not_null()
    out_cols: list[pl.Expr] = [(~has_p0).alias("no_close_data")]
    for h in horizons:
        enough_rows = pl.col(f"_nrows_{h}") >= h
        determined = has_p0 & (pl.col("is_delisted") | enough_rows)
        censored = has_p0 & (~pl.col("is_delisted")) & ~enough_rows
        out_cols.append(censored.alias(censored_col(h)))
        for pct, frac in thresholds:
            target = pl.col("p0") * (1.0 + frac)
            out_cols.append(
                pl.when(determined)
                .then((pl.col(f"_maxc_{h}") >= target).cast(pl.Int8))
                .alias(hit_col(pct, h))
            )
            out_cols.append(
                pl.when(determined)
                .then((pl.col(f"_maxh_{h}") >= target).cast(pl.Int8))
                .alias(hit_col(pct, h, hi=True))
            )
        p0 = pl.col("p0")
        out_cols += [
            pl.when(determined).then(pl.col(f"_endc_{h}") / p0 - 1.0).alias(return_col(h)),
            pl.when(determined).then(pl.col(f"_maxc_{h}") / p0 - 1.0).alias(rebound_col(h)),
            pl.when(determined).then(pl.col(f"_minc_{h}") / p0 - 1.0).alias(drawdown_col(h)),
        ]

    result = grouped.with_columns(out_cols)

    keep = ["event_id", "security_id", "crash_date", "p0", "no_close_data",
            "delisting_return_missing"]
    for h in horizons:
        keep += [nforward_col(h), censored_col(h), return_col(h), rebound_col(h), drawdown_col(h)]
        for pct, _ in thresholds:
            keep += [hit_col(pct, h), hit_col(pct, h, hi=True)]
    return (
        result.select([c for c in keep if c in result.columns])
        .rename({"p0": "crash_close"})
        .sort(["security_id", "crash_date"])
    )
