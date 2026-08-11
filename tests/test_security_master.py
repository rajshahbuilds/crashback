"""Universe filter, security-master builder, and versioned-artifact writer (hermetic)."""
from __future__ import annotations

import json

import polars as pl

from crashback import load_config
from crashback.providers import SyntheticProvider, UniverseFilter
from crashback.providers.wrds_provider import _universe_where
from crashback.securities.master import build_security_master, summarize_master
from crashback.storage.artifacts import write_versioned_parquet


def test_universe_filter_from_config_matches_default():
    uf = UniverseFilter.from_config(load_config().universe)
    assert uf.share_types == ("NS",)
    assert uf.security_types == ("EQTY",)
    assert uf.security_subtypes == ("COM",)
    assert uf.exchanges == ("N", "A", "Q")
    assert uf.us_incorporated_only is True


def test_universe_where_builds_ciz_sql():
    uf = UniverseFilter(
        share_types=("NS",), security_types=("EQTY",), security_subtypes=("COM",),
        exchanges=("N", "A", "Q"), us_incorporated_only=True,
    )
    where = _universe_where(uf)
    assert "sharetype IN ('NS')" in where
    assert "securitytype IN ('EQTY')" in where
    assert "securitysubtype IN ('COM')" in where
    assert "primaryexch IN ('N','A','Q')" in where
    assert "usincflg = 'Y'" in where
    assert _universe_where(None) == ""


def test_build_security_master_preserves_active_and_delisted():
    df = build_security_master(SyntheticProvider.example())
    # sorted by security_id
    assert df["security_id"].to_list() == sorted(df["security_id"].to_list())
    s = summarize_master(df)
    assert s["securities"] == 2
    assert s["active"] == 1          # 1001 has no delisting
    assert s["with_delisting"] == 1  # 2002 delisted (code 574)


def test_write_versioned_parquet_roundtrip_and_sidecar(tmp_path):
    df = pl.DataFrame({"security_id": [1, 2], "ticker": ["AAA", "BBB"]})
    path = write_versioned_parquet(
        df, tmp_path, "sec_master", "v1", meta={"filter": "test"}, git_commit="abc123"
    )
    assert path.name == "sec_master_v1.parquet"
    assert pl.read_parquet(path).equals(df)

    sidecar = json.loads((tmp_path / "sec_master_v1.meta.json").read_text())
    assert sidecar["row_count"] == 2
    assert sidecar["version"] == "v1"
    assert sidecar["git_commit"] == "abc123"
    assert sidecar["columns"] == ["security_id", "ticker"]
    assert len(sidecar["sha256"]) == 64
    assert sidecar["meta"] == {"filter": "test"}
