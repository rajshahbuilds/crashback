"""In-memory MarketDataProvider for tests and offline development.

Holds canonical DataFrames directly, so downstream logic (crash detection, labels,
features) can be unit-tested on tiny deterministic fixtures without a live WRDS
connection. ``SyntheticProvider.example()`` builds a minimal schema-valid universe
containing a crash day and a delisting.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from crashback.providers.base import MarketDataProvider
from crashback.providers.schemas import (
    CORPORATE_ACTION_SCHEMA,
    DAILY_PRICE_SCHEMA,
    FUNDAMENTALS_SCHEMA,
    SECTOR_METADATA_SCHEMA,
    SECURITY_MASTER_SCHEMA,
    empty_frame,
    validate_schema,
)


class SyntheticProvider(MarketDataProvider):
    def __init__(
        self,
        *,
        prices: pl.DataFrame | None = None,
        security_master: pl.DataFrame | None = None,
        fundamentals: pl.DataFrame | None = None,
        corporate_actions: pl.DataFrame | None = None,
        sector_metadata: pl.DataFrame | None = None,
    ):
        # validate_schema guarantees every stored frame conforms to its canonical schema.
        self._prices = validate_schema(
            prices if prices is not None else empty_frame(DAILY_PRICE_SCHEMA),
            DAILY_PRICE_SCHEMA,
        )
        self._security_master = validate_schema(
            security_master if security_master is not None else empty_frame(SECURITY_MASTER_SCHEMA),
            SECURITY_MASTER_SCHEMA,
        )
        self._fundamentals = validate_schema(
            fundamentals if fundamentals is not None else empty_frame(FUNDAMENTALS_SCHEMA),
            FUNDAMENTALS_SCHEMA,
        )
        self._corporate_actions = validate_schema(
            corporate_actions
            if corporate_actions is not None
            else empty_frame(CORPORATE_ACTION_SCHEMA),
            CORPORATE_ACTION_SCHEMA,
        )
        self._sector_metadata = validate_schema(
            sector_metadata if sector_metadata is not None else empty_frame(SECTOR_METADATA_SCHEMA),
            SECTOR_METADATA_SCHEMA,
        )

    @staticmethod
    def _filter_ids(df: pl.DataFrame, security_ids: Sequence[int] | None) -> pl.DataFrame:
        if security_ids is None:
            return df
        return df.filter(pl.col("security_id").is_in(list(security_ids)))

    def get_security_master(self, security_ids=None) -> pl.DataFrame:
        return self._filter_ids(self._security_master, security_ids)

    def get_daily_prices(self, security_ids, start: date, end: date) -> pl.DataFrame:
        df = self._filter_ids(self._prices, security_ids)
        return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    def get_corporate_actions(self, security_ids=None) -> pl.DataFrame:
        return self._filter_ids(self._corporate_actions, security_ids)

    def get_sector_metadata(self, security_ids=None) -> pl.DataFrame:
        return self._filter_ids(self._sector_metadata, security_ids)

    def get_fundamentals(self, *, company_ids=None, security_ids=None, freq="Q") -> pl.DataFrame:
        df = self._fundamentals.filter(pl.col("freq") == freq)
        if company_ids is not None:
            df = df.filter(pl.col("company_id").is_in(list(company_ids)))
        if security_ids is not None:
            df = df.filter(pl.col("security_id").is_in(list(security_ids)))
        return df

    # --- fixtures ----------------------------------------------------------------

    @classmethod
    def example(cls) -> SyntheticProvider:
        """A minimal two-security universe with a crash day and a delisting."""
        prices = validate_schema(
            pl.DataFrame(
                {
                    "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6),
                             date(2020, 1, 2), date(2020, 1, 3)],
                    "security_id": [1001, 1001, 1001, 2002, 2002],
                    "open": [100.0, 99.0, 84.0, 50.0, 49.0],
                    "high": [101.0, 100.0, 86.0, 51.0, 49.5],
                    "low": [98.0, 83.0, 83.0, 48.0, 20.0],
                    "close": [99.0, 85.0, 88.0, 49.0, 22.0],
                    "adjusted_close": [99.0, 85.0, 88.0, 49.0, 22.0],
                    "volume": [1e6, 3e6, 2e6, 5e5, 4e6],
                    # crashes: 1001 on day 2 (-14%), 2002 on day 2 (-55%)
                    "daily_return": [-0.01, -0.1414, 0.0353, -0.02, -0.5510],
                    "daily_return_ex_div": [-0.01, -0.1414, 0.0353, -0.02, -0.5510],
                    "cum_factor_price": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "cum_factor_shares": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "shares_outstanding": [1e7, 1e7, 1e7, 2e6, 2e6],
                }
            ),
            DAILY_PRICE_SCHEMA,
        )
        security_master = validate_schema(
            pl.DataFrame(
                {
                    "security_id": [1001, 2002],
                    "company_id": [11, 22],
                    "ticker": ["AAA", "BBB"],
                    "ticker_start": [date(2010, 1, 1), date(2012, 1, 1)],
                    "ticker_end": [date(2099, 12, 31), date(2020, 1, 3)],
                    "exchange": ["NYSE", "Nasdaq"],
                    "security_type": ["EQTY", "EQTY"],
                    "sic_code": [3571, 6021],
                    "listing_date": [date(2010, 1, 1), date(2012, 1, 1)],
                    "delisting_date": [None, date(2020, 1, 3)],
                    "delisting_code": [None, 574],
                    "delisting_return": [None, -0.551],
                }
            ),
            SECURITY_MASTER_SCHEMA,
        )
        corporate_actions = validate_schema(
            pl.DataFrame(
                {
                    "security_id": [2002],
                    "effective_date": [date(2020, 1, 3)],
                    "action_type": ["DELISTING"],
                    "value": [-0.551],
                    "code": [574],
                    "details": ["dlstcd=574"],
                }
            ),
            CORPORATE_ACTION_SCHEMA,
        )
        sector_metadata = validate_schema(
            pl.DataFrame(
                {
                    "security_id": [1001, 2002],
                    "sic_code": [3571, 6021],
                    "sic_division": ["Manufacturing", "Finance, Insurance & Real Estate"],
                }
            ),
            SECTOR_METADATA_SCHEMA,
        )
        fundamentals = validate_schema(
            pl.DataFrame(
                {
                    "company_id": [11],
                    "security_id": [1001],
                    "period_end": [date(2019, 12, 31)],
                    "public_date": [date(2020, 1, 30)],
                    "freq": ["Q"],
                    "fiscal_year": [2019],
                    "fiscal_quarter": [4],
                    "revenue": [500.0],
                    "net_income": [80.0],
                    "total_assets": [2000.0],
                    "total_liabilities": [800.0],
                    "cash": [300.0],
                    "total_debt": [400.0],
                    "shares_outstanding": [10.0],
                }
            ),
            FUNDAMENTALS_SCHEMA,
        )
        return cls(
            prices=prices,
            security_master=security_master,
            fundamentals=fundamentals,
            corporate_actions=corporate_actions,
            sector_metadata=sector_metadata,
        )
