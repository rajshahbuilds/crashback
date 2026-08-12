"""Base-rate helpers: Wilson CI and grouped rates."""
from __future__ import annotations

import polars as pl
import pytest

from crashback.evaluation.descriptive import add_wilson, grouped_rate, overall_rate


def test_wilson_ci_matches_known_value():
    # 50/100 -> Wilson 95% CI is ~ (0.404, 0.596), center 0.5
    r = add_wilson(pl.DataFrame({"hits": [50], "n": [100]})).row(0, named=True)
    assert r["rate"] == pytest.approx(0.5)
    assert r["ci_low"] == pytest.approx(0.4038, abs=1e-3)
    assert r["ci_high"] == pytest.approx(0.5962, abs=1e-3)


def test_wilson_ci_narrows_with_n():
    small = add_wilson(pl.DataFrame({"hits": [5], "n": [10]})).row(0, named=True)
    large = add_wilson(pl.DataFrame({"hits": [5000], "n": [10000]})).row(0, named=True)
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


def test_overall_rate_ignores_null_labels():
    df = pl.DataFrame({"hit_10pct_20d": [1, 0, 1, None, None]})
    r = overall_rate(df, "hit_10pct_20d")
    assert r["n"] == 3 and r["hits"] == 2
    assert r["rate"] == pytest.approx(2 / 3)


def test_grouped_rate_by_bucket():
    df = pl.DataFrame({
        "bucket": ["a", "a", "a", "b", "b"],
        "hit_10pct_20d": [1, 1, 0, 0, None],   # a: 2/3, b: 0/1 determined
    })
    g = grouped_rate(df, "bucket", "hit_10pct_20d")
    d = {r["bucket"]: r for r in g.iter_rows(named=True)}
    assert d["a"]["n"] == 3 and d["a"]["hits"] == 2 and d["a"]["rate"] == pytest.approx(2 / 3)
    assert d["b"]["n"] == 1 and d["b"]["hits"] == 0
