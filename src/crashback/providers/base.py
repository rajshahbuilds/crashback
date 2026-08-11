"""Provider-neutral market-data interface.

A ``MarketDataProvider`` returns Polars DataFrames in the canonical schemas defined in
``crashback.providers.schemas``. Downstream code depends on this interface and those
schemas only — never on a concrete provider or vendor column names.

Only one concrete provider (WRDS) is implemented for now; the abstraction exists to keep
the normalization boundary explicit and to allow a synthetic in-memory provider for
tests. Multi-vendor support is intentionally out of scope (see STU-45 scope decision).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

import polars as pl

from crashback.providers.universe import UniverseFilter


class MarketDataProvider(ABC):
    """Canonical read interface over a market-data source.

    Every method returns a Polars DataFrame conforming to the matching canonical schema.
    ``security_ids`` filters are optional; ``None`` means "the whole available universe"
    (callers should pass explicit id lists in production to bound the query).
    """

    @abstractmethod
    def get_security_master(
        self,
        security_ids: Sequence[int] | None = None,
        universe: UniverseFilter | None = None,
    ) -> pl.DataFrame:
        """Canonical security master (schema: security_master). Preserves ticker history.

        ``universe`` optionally restricts to the research universe (e.g. US common stock on
        major exchanges). Providers that cannot express the filter may ignore it.
        """

    @abstractmethod
    def get_daily_prices(
        self,
        security_ids: Sequence[int] | None,
        start: date,
        end: date,
        universe: UniverseFilter | None = None,
    ) -> pl.DataFrame:
        """Canonical daily prices in [start, end] inclusive (schema: daily_price).

        ``universe`` optionally restricts to the research universe (same semantics as
        ``get_security_master``); providers that cannot express it may ignore it.
        """

    @abstractmethod
    def get_fundamentals(
        self,
        *,
        company_ids: Sequence[int] | None = None,
        security_ids: Sequence[int] | None = None,
        freq: str = "Q",
    ) -> pl.DataFrame:
        """Canonical point-in-time fundamentals (schema: fundamentals). ``freq`` is 'Q'|'A'."""

    @abstractmethod
    def get_corporate_actions(
        self, security_ids: Sequence[int] | None = None
    ) -> pl.DataFrame:
        """Canonical corporate actions incl. delistings (schema: corporate_action)."""

    @abstractmethod
    def get_sector_metadata(
        self, security_ids: Sequence[int] | None = None
    ) -> pl.DataFrame:
        """Canonical sector metadata (schema: sector_metadata)."""
