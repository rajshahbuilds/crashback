"""Contemporaneous-document retrieval: point-in-time guard, EDGAR parsing, provenance (STU-65).

All hermetic — no network. The EDGAR fixture stands in for a submissions JSON so the leakage
filter and normalization are tested without hitting SEC.
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl

from crashback.providers.documents import (
    DOCUMENT_SCHEMA,
    DocumentProvider,
    DocumentQuery,
    available_as_of,
    prediction_cutoff,
)
from crashback.providers.edgar_provider import EDGARProvider

# A crash on 2022-02-03; prediction cutoff = 2022-02-03 16:00 ET.
_CRASH = date(2022, 2, 3)
_CUTOFF = prediction_cutoff(_CRASH)

# Fixture: an earnings 8-K filed the evening BEFORE the crash (available), a 10-K filed months
# before (available but outside a short lookback), and an 8-K filed AFTER the crash (a leak).
_SUBMISSIONS = {
    "cik": 320193,
    "filings": {"recent": {
        "form": ["8-K", "8-K", "10-K"],
        "accessionNumber": ["0000320193-22-000001", "0000320193-22-000009",
                            "0000320193-21-000105"],
        "acceptanceDateTime": ["2022-02-02T16:30:00.000Z",   # before crash → available
                               "2022-02-10T09:00:00.000Z",   # after crash → LEAK, must drop
                               "2021-10-28T18:01:00.000Z"],  # long before crash
        "reportDate": ["2022-02-02", "2022-02-10", "2021-09-25"],
        "primaryDocDescription": ["8-K", "8-K", "10-K"],
    }},
}
_RETRIEVED = datetime(2026, 8, 12, 0, 0, 0)


def _provider():
    return EDGARProvider("crashback-test test@example.com", retrieved_at=_RETRIEVED)


def test_prediction_cutoff_is_crash_close():
    assert _CUTOFF == datetime(2022, 2, 3, 16, 0)


def test_parse_submissions_matches_schema_and_provenance():
    df = _provider().parse_submissions(_SUBMISSIONS, security_id=42, company_id=7,
                                       retrieved_at=_RETRIEVED)
    assert list(df.columns) == list(DOCUMENT_SCHEMA)
    assert df.height == 3
    r = df.filter(pl.col("doc_id") == "0000320193-22-000001").row(0, named=True)
    assert r["source"] == "sec_edgar" and r["source_type"] == "8-K"
    assert r["security_id"] == 42 and r["company_id"] == 7
    assert r["available_at"] == datetime(2022, 2, 2, 16, 30)      # timestamp preserved
    assert r["url"].startswith("https://www.sec.gov/Archives/edgar/data/320193/")
    assert r["retrieved_at"] == _RETRIEVED                        # retrieval metadata preserved


def test_future_filing_excluded_by_construction():
    df = _provider().parse_submissions(_SUBMISSIONS, retrieved_at=_RETRIEVED)
    kept = available_as_of(df, _CUTOFF)
    ids = set(kept["doc_id"].to_list())
    assert "0000320193-22-000001" in ids          # filed 02-02, before crash close → kept
    assert "0000320193-22-000009" not in ids       # filed 02-10, after crash → excluded (leak)
    assert kept.height == 2                          # the two pre-cutoff docs


def test_get_documents_applies_pit_and_lookback():
    # A fetch-stubbed provider (no network): return the fixture, then base class filters.
    class _Stub(EDGARProvider):
        def _fetch(self, query):
            return self.parse_submissions(_SUBMISSIONS, retrieved_at=_RETRIEVED)

    prov = _Stub("crashback-test test@example.com", retrieved_at=_RETRIEVED)
    # 30-day lookback: drops the Oct-2021 10-K and the future 8-K, keeps only the 02-02 8-K.
    q = DocumentQuery(as_of=_CUTOFF, lookback_days=30)
    df = prov.get_documents(q)
    assert df.height == 1
    assert df.row(0, named=True)["doc_id"] == "0000320193-22-000001"
    # documents come back newest-first and never after the cutoff
    assert df["available_at"].max() <= _CUTOFF


def test_null_availability_never_admitted():
    subs = {"cik": 1, "filings": {"recent": {
        "form": ["8-K"], "accessionNumber": ["x"], "acceptanceDateTime": [None],
        "reportDate": ["2022-01-01"], "primaryDocDescription": ["8-K"]}}}
    df = _provider().parse_submissions(subs, retrieved_at=_RETRIEVED)
    assert df.height == 1 and df.row(0, named=True)["available_at"] is None
    assert available_as_of(df, _CUTOFF).height == 0       # unprovable availability → dropped


def test_retrieve_raw_paginates_older_shards():
    # submissions 'recent' holds only a 2026 filing; the 2022 crash filing lives in an older
    # shard listed under filings.files → retrieve_raw must fetch and concatenate it.
    recent = {"filings": {
        "recent": {"form": ["4"], "accessionNumber": ["r-1"],
                   "acceptanceDateTime": ["2026-01-05T10:00:00.000Z"],
                   "reportDate": ["2026-01-05"], "primaryDocDescription": ["4"]},
        "files": [{"name": "CIK0000320193-submissions-001.json",
                   "filingFrom": "2021-01-01", "filingTo": "2022-06-30"}]},
        "cik": 320193}
    shard = {"form": ["8-K"], "accessionNumber": ["s-1"],
             "acceptanceDateTime": ["2022-02-02T16:30:00.000Z"],
             "reportDate": ["2022-02-02"], "primaryDocDescription": ["8-K"]}

    class _Stub(EDGARProvider):
        def _get_json(self, url):                       # no network
            return shard if "submissions-001" in url else recent

    prov = _Stub("crashback-test test@example.com", retrieved_at=_RETRIEVED, min_interval=0.0)
    raw = prov.retrieve_raw("320193", start=datetime(2022, 1, 1), end=datetime(2022, 6, 30))
    ids = set(raw["doc_id"].to_list())
    assert "s-1" in ids                                 # older shard was fetched + parsed
    # and the base point-in-time filter then admits only the pre-crash filing
    kept = prov.get_documents(DocumentQuery(as_of=_CUTOFF, cik="320193", lookback_days=90))
    assert set(kept["doc_id"].to_list()) == {"s-1"}


def test_provider_is_abstract_without_fetch():
    assert DocumentProvider.__abstractmethods__ == frozenset({"_fetch"})


def test_edgar_requires_contact_user_agent():
    import pytest
    with pytest.raises(ValueError, match="User-Agent"):
        EDGARProvider("no-email-here")
