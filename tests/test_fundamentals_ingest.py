"""Unrestated fundamentals normalization: field mapping, derived fields, availability dates."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from crashback.fundamentals.ingest import (
    FUNDAMENTALS_SCHEMA,
    RDQ_FALLBACK_DAYS,
    normalize_urqus,
)


def _raw():
    # Two firm-quarters: one with rdq present, one with rdq missing (needs the fallback).
    return pl.DataFrame(
        {
            "gvkey": ["001690", "000002"],
            "datadate": [date(2020, 3, 31), date(2019, 12, 31)],
            "rdq": [date(2020, 4, 30), None],
            "fqtr": [2, 4],
            "fyrq": [2020, 2019],
            "saleq": [100.0, 50.0],
            "cogsq": [60.0, 30.0],
            "oibdpq": [30.0, 12.0],
            "dpq": [10.0, 4.0],
            "niq": [20.0, 5.0],
            "epspxq": [1.5, 0.4],
            "xintq": [2.0, 1.0],
            "atq": [500.0, 200.0],
            "ltq": [300.0, 120.0],
            "cheq": [80.0, 30.0],
            "dlttq": [100.0, 40.0],
            "dlcq": [20.0, 10.0],
            "actq": [150.0, 60.0],
            "lctq": [90.0, 40.0],
            "seqq": [200.0, 80.0],
            "ceqq": [180.0, 75.0],
            "cshoq": [17.0, 8.0],
        },
        schema_overrides={"rdq": pl.Date, "datadate": pl.Date},
    )


def _rows():
    df = normalize_urqus(_raw())
    return {r["company_id"]: r for r in df.iter_rows(named=True)}


def test_schema_and_field_mapping():
    df = normalize_urqus(_raw())
    assert df.columns == list(FUNDAMENTALS_SCHEMA.keys())
    r = _rows()[1690]
    assert r["revenue"] == 100.0 and r["cogs"] == 60.0
    assert r["net_income"] == 20.0 and r["total_assets"] == 500.0
    assert r["fiscal_quarter"] == 2 and r["freq"] == "Q"


def test_derived_fields():
    r = _rows()[1690]
    assert r["gross_profit"] == 40.0                 # saleq - cogsq
    assert r["operating_income"] == 20.0             # oibdpq - dpq
    assert r["total_debt"] == 120.0                  # dlttq + dlcq
    assert r["ebitda"] == 30.0                       # oibdpq


def test_availability_date_uses_rdq_when_present():
    r = _rows()[1690]
    assert r["rdq_available"] is True
    assert r["public_date"] == date(2020, 4, 30)     # the actual report date


def test_availability_date_falls_back_when_rdq_missing():
    r = _rows()[2]
    assert r["rdq_available"] is False
    # period_end + 60 days, flagged as an estimate
    assert r["public_date"] == date(2019, 12, 31) + timedelta(days=RDQ_FALLBACK_DAYS)
