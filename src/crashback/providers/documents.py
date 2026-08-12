"""Provider-neutral contemporaneous-document interface (V2 news retrieval, STU-65).

Mirrors ``MarketDataProvider``: a ``DocumentProvider`` returns Polars DataFrames in the canonical
``DOCUMENT_SCHEMA``, and downstream code depends on that schema only — never on EDGAR, a news API,
or vendor field names. The **point-in-time guarantee lives here**, in the base layer, so every
future source (SEC EDGAR now; a timestamped news API later) inherits leakage-safety by
construction: a document is admissible only if its availability timestamp is at or before the
crash **prediction timestamp** (the crash-day close, 16:00 ET).

CLAUDE.md §3 (no look-ahead) and §24 (V2 retrieves only documents available by the prediction
timestamp) are enforced by ``available_as_of`` / ``DocumentQuery.as_of`` — not left to callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, time

import polars as pl

# 16:00 America/New_York — the crash-day close, i.e. the prediction timestamp (§3).
MARKET_CLOSE_ET = time(16, 0)

# Canonical document schema (name -> Polars dtype), in canonical column order.
# available_at is the single point-in-time field every source MUST populate.
DOCUMENT_SCHEMA: OrderedDict[str, pl.DataType] = OrderedDict(
    [
        ("doc_id", pl.Utf8),               # stable per-document id (e.g. EDGAR accession no.)
        ("security_id", pl.Int64),         # our permno (nullable until resolved)
        ("company_id", pl.Int64),          # our gvkey (nullable)
        ("source", pl.Utf8),               # 'sec_edgar' | 'news_api' | ...
        ("source_type", pl.Utf8),          # '8-K' | '10-Q' | 'press_release' | 'news_article' ...
        ("title", pl.Utf8),                # form description or headline
        ("available_at", pl.Datetime("us")),   # publication/availability timestamp (ET, naive)
        ("period_of_report", pl.Date),     # fiscal period the doc concerns (nullable)
        ("url", pl.Utf8),                  # canonical source URL
        ("snippet", pl.Utf8),              # retrieved text/snippet metadata (nullable)
        ("retrieved_at", pl.Datetime("us")),   # retrieval metadata (when we fetched it)
    ]
)


def empty_documents() -> pl.DataFrame:
    """An empty DataFrame with the canonical document schema."""
    return pl.DataFrame(schema=DOCUMENT_SCHEMA)


def prediction_cutoff(crash_date: date, close: time = MARKET_CLOSE_ET) -> datetime:
    """The prediction timestamp for a crash: crash-day close (16:00 ET by default)."""
    return datetime.combine(crash_date, close)


def available_as_of(docs: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Keep only documents available at or before ``as_of`` — the leakage guard (§3).

    Documents with a null availability timestamp are **dropped** (cannot prove they were
    available in time), never admitted.
    """
    return docs.filter(pl.col("available_at").is_not_null() & (pl.col("available_at") <= as_of))


@dataclass(frozen=True)
class DocumentQuery:
    """A point-in-time retrieval request for one company at one crash prediction timestamp.

    ``as_of`` is the hard cutoff (typically ``prediction_cutoff(crash_date)``). Identifier fields
    are provided best-effort; each provider uses whichever it can resolve (EDGAR → cik/ticker, a
    news provider → ticker/company_name). ``lookback_days`` bounds how far back to retrieve.
    """

    as_of: datetime
    ticker: str | None = None
    cik: str | None = None
    company_name: str | None = None
    security_id: int | None = None
    company_id: int | None = None
    lookback_days: int = 90


class DocumentProvider(ABC):
    """Canonical read interface over a contemporaneous-document source.

    Concrete providers implement ``_fetch`` (source-specific retrieval + normalization into
    ``DOCUMENT_SCHEMA``); the base ``get_documents`` applies the point-in-time filter and lookback
    window so leakage-safety cannot be forgotten by a subclass.
    """

    source: str = "unknown"

    @abstractmethod
    def _fetch(self, query: DocumentQuery) -> pl.DataFrame:
        """Retrieve + normalize all candidate documents for the query (pre-filter)."""

    def get_documents(self, query: DocumentQuery) -> pl.DataFrame:
        """Canonical documents available strictly at/before ``query.as_of`` (point-in-time)."""
        docs = self._fetch(query)
        docs = available_as_of(docs, query.as_of)
        if query.lookback_days is not None:
            floor = query.as_of - _timedelta_days(query.lookback_days)
            docs = docs.filter(pl.col("available_at") >= floor)
        return docs.sort("available_at", descending=True)


def _timedelta_days(days: int):
    from datetime import timedelta
    return timedelta(days=days)
