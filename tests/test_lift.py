"""Decile lift table: equal-count buckets, monotonic lift, top-decile helper (STU-62)."""
from __future__ import annotations

import numpy as np

from crashback.evaluation.lift import decile_table, top_decile_lift


def test_equal_count_buckets_and_coverage():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    table, base = decile_table(y, p, q=10)
    assert table.height == 10
    assert table["n"].sum() == 1000
    assert table["n"].min() >= 99 and table["n"].max() <= 101   # equal to within one
    assert abs(base - y.mean()) < 1e-12


def test_lift_is_monotone_when_scores_are_informative():
    # observed recovery should rise with predicted-probability decile
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p).astype(int)   # P(y=1) increases with p
    table, base = decile_table(y, p, q=10)
    obs = table.sort("bucket")["observed_rate"].to_list()
    assert obs[-1] > obs[0]                      # top decile recovers more than bottom
    # broadly increasing: top-3 mean > bottom-3 mean
    assert np.mean(obs[-3:]) > np.mean(obs[:3])
    assert table.filter(table["bucket"] == 10).row(0, named=True)["lift"] > 1.0


def test_top_decile_helper_matches_table():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    table, base = decile_table(y, p, q=10)
    top = table.filter(table["bucket"] == 10).row(0, named=True)
    tdl = top_decile_lift(y, p, q=10)
    assert tdl["top_decile_recovery_rate"] == top["observed_rate"]
    assert abs(tdl["top_decile_lift"] - top["observed_rate"] / base) < 1e-12
    assert tdl["base_rate"] == base
