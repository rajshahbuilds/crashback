"""Build the survivorship-safe security master.

Pulls the full common-stock universe (active + delisted) from a provider, applying the
configured universe filter, and returns the canonical `security_master` schema. Grain is
one row per (security_id, ticker period): permno is the stable identity, so a ticker/name
change is additional rows for the SAME security, never a new one — no survivorship or
duplicate-company bias.
"""
from __future__ import annotations

import polars as pl

from crashback.providers.base import MarketDataProvider
from crashback.providers.universe import UniverseFilter


def build_security_master(
    provider: MarketDataProvider,
    universe: UniverseFilter | None = None,
) -> pl.DataFrame:
    """Return the canonical security master for the given universe (active + delisted)."""
    df = provider.get_security_master(universe=universe)
    # Deterministic ordering; ticker periods within a security ordered chronologically.
    return df.sort(["security_id", "ticker_start"], nulls_last=True)


def summarize_master(df: pl.DataFrame) -> dict:
    """Quick integrity summary used by the build script / report."""
    n_secs = df["security_id"].n_unique()
    n_delisted = df.filter(pl.col("delisting_date").is_not_null())["security_id"].n_unique()
    return {
        "rows": df.height,
        "securities": n_secs,
        "companies": df["company_id"].n_unique(),
        "with_delisting": n_delisted,
        "active": n_secs - n_delisted,
        "date_min": df["ticker_start"].min(),
        "date_max": df["ticker_end"].max(),
        "exchanges": sorted(x for x in df["exchange"].unique().to_list() if x is not None),
    }
