"""Market- and sector-context features (point-in-time — CLAUDE.md sec 14).

We do NOT filter out broad selloffs; instead we give the model both the individual move and
its environment. Market and sector daily-return series are built **equal-weighted from the
research universe** (breadth of the selloff matters more than mega-cap moves, and it keeps
the series self-contained). Sector = 2-digit SIC major group, assigned per security from the
security master. The focal company is **excluded from its own sector's crash-day return**
(sec 14) so a stock's own crash cannot masquerade as a sector-wide crash.

All values on the crash day use that day's cross-section (known at the close); trailing
windows use only dates <= crash_date. No forward information is used.
"""
from __future__ import annotations

import polars as pl

FEATURE_NAMES: tuple[str, ...] = (
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",       # trailing 1-month market move (regime proxy)
    "market_volatility_20d",
    "sector_return_1d",        # excludes the focal company
    "sector_return_5d",
    "sector_volatility_20d",
    "sector_n_members",
)


def _cum_return(col: str, window: int, over: str | None = None) -> pl.Expr:
    """Trailing `window`-day cumulative return (inclusive) via summed log returns."""
    logsum = (pl.col(col) + 1.0).log().rolling_sum(window)
    logsum = logsum.over(over) if over else logsum
    return logsum.exp() - 1.0


def sector_map(master: pl.DataFrame) -> pl.DataFrame:
    """One (static) 2-digit-SIC sector per security, from the master."""
    return master.group_by("security_id").agg(
        (pl.col("sic_code").drop_nulls().first() // 100).alias("sector")
    )


def build_market_sector_features(
    events: pl.DataFrame, prices: pl.DataFrame, master: pl.DataFrame
) -> pl.DataFrame:
    """Point-in-time market/sector context, one row per event (keyed by event_id)."""
    smap = sector_map(master)

    # --- market: equal-weighted daily return across the whole universe ---
    market = (
        prices.group_by("date")
        .agg(
            pl.col("daily_return").sum().alias("_msum"),
            pl.col("daily_return").count().alias("_mn"),
        )
        .sort("date")
        .with_columns((pl.col("_msum") / pl.col("_mn")).alias("market_return_1d"))
        .with_columns(
            _cum_return("market_return_1d", 5).alias("market_return_5d"),
            _cum_return("market_return_1d", 20).alias("market_return_20d"),
            pl.col("market_return_1d").rolling_std(20).alias("market_volatility_20d"),
        )
    )

    # --- sector: equal-weighted daily return per 2-digit SIC ---
    ps = prices.join(smap, on="security_id", how="left")
    sector = (
        ps.group_by(["sector", "date"])
        .agg(
            pl.col("daily_return").sum().alias("sector_sum_ret"),
            pl.col("daily_return").count().alias("sector_n_members"),
        )
        .sort(["sector", "date"])
        .with_columns((pl.col("sector_sum_ret") / pl.col("sector_n_members")).alias("_sret1d"))
        .with_columns(
            _cum_return("_sret1d", 5, over="sector").alias("sector_return_5d"),
            pl.col("_sret1d").rolling_std(20).over("sector").alias("sector_volatility_20d"),
        )
    )

    ev = events.select("event_id", "security_id", "crash_date", "crash_return").join(
        smap, on="security_id", how="left"
    )
    ev = ev.join(
        market.select("date", "market_return_1d", "market_return_5d",
                      "market_return_20d", "market_volatility_20d"),
        left_on="crash_date", right_on="date", how="left",
    )
    ev = ev.join(
        sector.select("sector", "date", "sector_sum_ret", "sector_n_members",
                      "sector_return_5d", "sector_volatility_20d"),
        left_on=["sector", "crash_date"], right_on=["sector", "date"], how="left",
    )
    ev = ev.with_columns(
        # equal-weighted sector return with the focal company removed
        pl.when(pl.col("sector_n_members") > 1)
        .then((pl.col("sector_sum_ret") - pl.col("crash_return"))
              / (pl.col("sector_n_members") - 1))
        .alias("sector_return_1d")
    )
    return ev.select("event_id", *FEATURE_NAMES)
