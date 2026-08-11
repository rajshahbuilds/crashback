"""CIZ / Compustat -> canonical normalization (pure, no live WRDS)."""
from __future__ import annotations

from datetime import date

import polars as pl

from crashback.providers import normalize as norm
from crashback.providers.schemas import (
    CORPORATE_ACTION_SCHEMA,
    DAILY_PRICE_SCHEMA,
    FUNDAMENTALS_SCHEMA,
    SECURITY_MASTER_SCHEMA,
)


def test_normalize_ciz_prices_maps_columns_and_adjusts_for_splits():
    # Row 2 has a 2:1 split (cum factor 2.0) and a -15% total return.
    ciz = pl.DataFrame(
        {
            "permno": [14593, 14593],
            "dlycaldt": [date(2008, 9, 26), date(2008, 9, 29)],
            "dlyopen": [130.0, 120.0],
            "dlyhigh": [131.0, 122.0],
            "dlylow": [126.0, 100.0],
            "dlyclose": [128.0, 200.0],   # raw close after split
            "dlyret": [-0.02, -0.15],     # total return (crash basis)
            "dlyretx": [-0.02, -0.15],
            "dlyvol": [1e6, 5e6],
            "dlycumfacpr": [1.0, 2.0],
            "dlycumfacshr": [1.0, 2.0],
            "shrout": [9e8, 9e8],
        }
    )
    out = norm.normalize_ciz_prices(ciz)
    assert out.columns == list(DAILY_PRICE_SCHEMA.keys())
    assert out.schema["date"] == pl.Date
    # daily_return carried through verbatim (never recomputed from close deltas).
    assert out["daily_return"].to_list() == [-0.02, -0.15]
    # adjusted_close = close / cum_factor_price → split does not distort the level.
    assert out["adjusted_close"].to_list() == [128.0, 100.0]
    assert out["security_id"].to_list() == [14593, 14593]


def test_normalize_ciz_delistings_to_corporate_actions():
    dsedelist = pl.DataFrame(
        {"permno": [80599], "dlstdt": [date(2008, 9, 17)], "dlstcd": [574], "dlret": [-0.6]}
    )
    out = norm.normalize_ciz_delistings(dsedelist)
    assert out.columns == list(CORPORATE_ACTION_SCHEMA.keys())
    row = out.row(0, named=True)
    assert row["action_type"] == "DELISTING"
    assert row["value"] == -0.6
    assert row["code"] == 574
    assert row["details"] == "dlstcd=574"


def test_normalize_ciz_security_master_joins_delisting():
    names = pl.DataFrame(
        {
            "permno": [80599],
            "permco": [20678],
            "namedt": [date(1994, 1, 1)],
            "nameenddt": [date(2008, 9, 18)],
            "ticker": ["LEH"],
            "primaryexch": ["N"],
            "securitytype": ["EQTY"],
            "siccd": [6211],
            "securitybegdt": [date(1994, 1, 1)],
            "securityenddt": [date(2008, 9, 18)],
        }
    )
    delist = pl.DataFrame({"permno": [80599], "dlstcd": [574], "dlret": [-0.6]})
    out = norm.normalize_ciz_security_master(names, delist)
    assert out.columns == list(SECURITY_MASTER_SCHEMA.keys())
    row = out.row(0, named=True)
    assert row["security_id"] == 80599
    assert row["exchange"] == "NYSE"          # 'N' mapped to a canonical label
    assert row["delisting_code"] == 574
    assert row["delisting_return"] == -0.6


def test_normalize_compustat_fundamentals_quarterly_with_link():
    fundq = pl.DataFrame(
        {
            "gvkey": [1690],
            "datadate": [date(2020, 12, 31)],
            "rdq": [date(2021, 1, 27)],
            "fyearq": [2020],
            "fqtr": [1],
            "revtq": [111.0],
            "niq": [28.0],
            "atq": [354.0],
            "ltq": [287.0],
            "cheq": [77.0],
            "dlttq": [99.0],
            "dlcq": [13.0],
            "cshoq": [17.0],
        }
    )
    links = pl.DataFrame({"gvkey": [1690], "lpermno": [14593]})
    out = norm.normalize_compustat_fundamentals(fundq, freq="Q", links=links)
    assert out.columns == list(FUNDAMENTALS_SCHEMA.keys())
    row = out.row(0, named=True)
    assert row["public_date"] == date(2021, 1, 27)   # rdq = availability date
    assert row["security_id"] == 14593               # linked permno
    assert row["total_debt"] == 112.0                # dlttq + dlcq
    assert row["freq"] == "Q"


def test_sic_division_ranges():
    assert norm.sic_division(3571) == "Manufacturing"
    assert norm.sic_division(6021) == "Finance, Insurance & Real Estate"
    assert norm.sic_division(None) == "Unknown"
    assert norm.sic_division(9999) == "Nonclassifiable"
