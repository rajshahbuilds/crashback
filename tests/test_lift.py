"""Decile lift table: equal-count buckets, monotonic lift, top-decile helper (STU-62)."""
from __future__ import annotations

import numpy as np

from crashback.evaluation.lift import confidence_bands, decile_table, top_decile_lift


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


def test_confidence_bands_counts_and_returns():
    # p in {0.15, 0.55, 0.85}; check band counts, observed rate, and return conditioning
    p = np.array([0.15, 0.15, 0.55, 0.55, 0.85, 0.85])
    y = np.array([0, 0, 1, 0, 1, 1])
    ret = np.array([-0.05, -0.03, 0.12, -0.20, 0.15, 0.11])
    dd = np.array([-0.10, -0.08, -0.02, -0.30, -0.01, -0.03])
    t = confidence_bands(p, y, ret=ret, dd=dd, width=0.1).sort("band")
    rows = {round(r["band"], 1): r for r in t.iter_rows(named=True)}
    assert rows[0.1]["n"] == 2 and rows[0.5]["n"] == 2 and rows[0.8]["n"] == 2
    assert t["n"].sum() == 6
    assert abs(rows[0.5]["observed_rate"] - 0.5) < 1e-9
    # 0.5 band: one win (+0.12) one lose (-0.20) → mean_return -0.04, win/lose conditioned
    assert abs(rows[0.5]["mean_return"] - (-0.04)) < 1e-9
    assert abs(rows[0.5]["mean_return_win"] - 0.12) < 1e-9
    assert abs(rows[0.5]["mean_return_lose"] - (-0.20)) < 1e-9
    assert abs(rows[0.1]["frac"] - 2 / 6) < 1e-9


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
