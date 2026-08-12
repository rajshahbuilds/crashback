"""Recent-crash history features: fresh shocks, second legs, post-rebound, no look-ahead."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from crashback.features.recent_crash import build_recent_crash_features

# One security. Crashes (daily_return <= -10%) on days 2, 5, 7.
#   day:  0    1    2*    3    4    5*    6    7*
#   close 100  100  85    90   95   80    82   70
_CLOSE = [100.0, 100.0, 85.0, 90.0, 95.0, 80.0, 82.0, 70.0]
_RET = [0.0, 0.0, -0.15, 0.0588, 0.0556, -0.1579, 0.025, -0.1463]
_CRASH_DAYS = [2, 5, 7]


def _dates(n):
    return [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]


def _prices():
    n = len(_CLOSE)
    return pl.DataFrame(
        {
            "security_id": [1] * n,
            "date": _dates(n),
            "close": _CLOSE,
            "high": [c + 1 for c in _CLOSE],
            "low": [c - 1 for c in _CLOSE],
            "open": _CLOSE,
            "volume": [1000.0] * n,
            "daily_return": _RET,
        }
    )


def _events(crash_days=None):
    crash_days = crash_days if crash_days is not None else _CRASH_DAYS
    ds = _dates(len(_CLOSE))
    return pl.DataFrame(
        {
            "event_id": [f"1_{d}" for d in crash_days],
            "security_id": [1] * len(crash_days),
            "crash_date": [ds[d] for d in crash_days],
        }
    )


def _feat(crash_days=None):
    df = build_recent_crash_features(_events(crash_days), _prices())
    return {r["event_id"]: r for r in df.iter_rows(named=True)}


def test_fresh_first_crash_has_zero_counts_and_null_previous():
    r = _feat()["1_2"]
    assert r["prior_crash_count_5d"] == 0
    assert r["prior_crash_count_20d"] == 0 and r["prior_crash_count_60d"] == 0
    assert r["days_since_previous_crash"] is None
    assert r["previous_crash_return"] is None
    assert r["return_since_previous_crash"] is None
    assert r["max_rebound_since_previous_crash"] is None


def test_second_leg_after_partial_rebound():
    r = _feat()["1_5"]
    assert r["prior_crash_count_5d"] == 1                       # only day 2, not the current
    assert r["days_since_previous_crash"] == 3                  # day 5 - day 2
    assert r["previous_crash_return"] == pytest.approx(-0.15)
    assert r["return_since_previous_crash"] == pytest.approx(80 / 85 - 1)
    assert r["max_rebound_since_previous_crash"] == pytest.approx(95 / 85 - 1)  # peak 95


def test_third_crash_counts_two_priors():
    r = _feat()["1_7"]
    assert r["prior_crash_count_5d"] == 2                       # days 2 and 5 within 5 tds
    assert r["prior_crash_count_20d"] == 2
    assert r["days_since_previous_crash"] == 2                  # day 7 - day 5
    assert r["previous_crash_return"] == pytest.approx(-0.1579)
    assert r["return_since_previous_crash"] == pytest.approx(70 / 80 - 1)
    assert r["max_rebound_since_previous_crash"] == pytest.approx(82 / 80 - 1)


def test_current_crash_not_counted_and_no_lookahead():
    # Features for day-5 crash must be identical whether or not the later day-7 crash exists.
    with_future = _feat([2, 5, 7])["1_5"]
    without_future = _feat([2, 5])["1_5"]
    for k in ("prior_crash_count_5d", "days_since_previous_crash", "previous_crash_return",
              "return_since_previous_crash", "max_rebound_since_previous_crash"):
        assert with_future[k] == without_future[k], k
