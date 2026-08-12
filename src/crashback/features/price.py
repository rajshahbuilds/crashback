"""Crash-day and pre-crash price-action features (point-in-time).

Every feature is computable from data available by the crash-day close and uses only
observations with date <= crash_date (per-security ``.over`` window expressions on a
(security_id, date)-sorted frame). Pre-crash momentum/volatility windows exclude the crash
day itself (via ``shift(1)``) so the crash does not contaminate them; distance-from-high and
drawdown include the crash day (they describe the post-crash-close position).

Missing history is explicit: a rolling window shorter than its size, or a shift before the
start of the series, yields null (never a partial or forward-filled value).
"""
from __future__ import annotations

import polars as pl

# Feature columns produced, in order.
FEATURE_NAMES: tuple[str, ...] = (
    # crash day
    "crash_return", "opening_gap", "intraday_range", "close_vs_low", "close_vs_open",
    "crash_volume", "relative_volume_20d",
    # trailing volatility (prior N days, excludes the crash day)
    "volatility_20d", "volatility_60d",
    # pre-crash returns (window ends the day before the crash)
    "return_5d_pre", "return_20d_pre", "return_60d_pre", "return_252d_pre",
    # distance below trailing high (includes the crash day)
    "distance_from_20d_high", "distance_from_60d_high", "distance_from_52w_high",
    # drawdown from trailing close peak (includes the crash day)
    "drawdown_20d", "drawdown_60d", "drawdown_252d",
)


def _feature_exprs() -> list[pl.Expr]:
    close, high, low, openp = pl.col("close"), pl.col("high"), pl.col("low"), pl.col("open")
    vol, ret = pl.col("volume"), pl.col("daily_return")
    prev_close = close.shift(1)
    raw = {
        "crash_return": ret,
        "opening_gap": openp / prev_close - 1.0,
        "intraday_range": (high - low) / prev_close,
        "close_vs_low": close / low - 1.0,
        "close_vs_open": close / openp - 1.0,
        "crash_volume": vol,
        "relative_volume_20d": vol / vol.rolling_mean(20).shift(1),
        "volatility_20d": ret.rolling_std(20).shift(1),
        "volatility_60d": ret.rolling_std(60).shift(1),
        "return_5d_pre": close.shift(1) / close.shift(6) - 1.0,
        "return_20d_pre": close.shift(1) / close.shift(21) - 1.0,
        "return_60d_pre": close.shift(1) / close.shift(61) - 1.0,
        "return_252d_pre": close.shift(1) / close.shift(253) - 1.0,
        "distance_from_20d_high": close / high.rolling_max(20) - 1.0,
        "distance_from_60d_high": close / high.rolling_max(60) - 1.0,
        "distance_from_52w_high": close / high.rolling_max(252) - 1.0,
        "drawdown_20d": close / close.rolling_max(20) - 1.0,
        "drawdown_60d": close / close.rolling_max(60) - 1.0,
        "drawdown_252d": close / close.rolling_max(252) - 1.0,
    }
    # Every expression is evaluated per security (never across security boundaries).
    return [expr.over("security_id").alias(name) for name, expr in raw.items()]


def build_price_features(events: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Point-in-time crash-day / pre-crash features, one row per event (keyed by event_id)."""
    featured = prices.sort(["security_id", "date"]).with_columns(_feature_exprs())
    return events.select("event_id", "security_id", "crash_date").join(
        featured.select("security_id", "date", *FEATURE_NAMES),
        left_on=["security_id", "crash_date"],
        right_on=["security_id", "date"],
        how="left",
    )
