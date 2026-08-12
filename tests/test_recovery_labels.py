"""Recovery labels verified against hand-calculated price paths."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from crashback.labels.outcomes import build_labels

# Consecutive calendar days act as trading days; the seq index handles ordering.
_D = [date(2020, 1, d) for d in range(1, 8)]

# sec1 active full recovery; sec2 active censored at H=3; sec3 delisted bankruptcy;
# sec4 delisted merger (recovers via terminal); sec5 active where intraday high hits, close does not
_PRICES = pl.DataFrame(
    {
        "security_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
        "date": [
            _D[0], _D[1], _D[2], _D[3],      # sec1: 100,106,103,112
            _D[0], _D[1], _D[2],             # sec2: 100,102,90
            _D[0], _D[1], _D[2],             # sec3: 100,80,60 -> terminal 6
            _D[0], _D[1],                    # sec4: 100,108 -> terminal 113.4
            _D[0], _D[1], _D[2],             # sec5: 100,(104/111),(103/105)
        ],
        "close": [100.0, 106.0, 103.0, 112.0, 100.0, 102.0, 90.0, 100.0, 80.0, 60.0,
                  100.0, 108.0, 100.0, 104.0, 103.0],
        "high": [100.0, 106.0, 103.0, 112.0, 100.0, 102.0, 90.0, 100.0, 80.0, 60.0,
                 100.0, 108.0, 100.0, 111.0, 105.0],
    }
)
_MASTER = pl.DataFrame(
    {
        "security_id": [1, 2, 3, 4, 5],
        "delisting_date": [None, None, date(2020, 1, 4), date(2020, 1, 3), None],
        "delisting_return": [None, None, -0.9, 0.05, None],
    },
    schema_overrides={"delisting_date": pl.Date, "delisting_return": pl.Float64},
)
_EVENTS = pl.DataFrame(
    {
        "event_id": ["1", "2", "3", "4", "5"],
        "security_id": [1, 2, 3, 4, 5],
        "crash_date": [_D[0]] * 5,
        "crash_close": [100.0] * 5,
    }
)

H = (2, 3)
T = ((5, 0.05), (10, 0.10))


def _labels():
    df = build_labels(_EVENTS, _PRICES, _MASTER, horizons=H, thresholds=T)
    return {r["security_id"]: r for r in df.iter_rows(named=True)}


def test_active_full_recovery():
    r = _labels()[1]
    assert r["hit_5pct_2d"] == 1 and r["hit_10pct_2d"] == 0   # max 106: >=105, <110
    assert r["hit_10pct_3d"] == 1                              # max 112 >= 110
    assert r["censored_2d"] is False and r["censored_3d"] is False
    assert r["return_3d"] == pytest.approx(0.12)
    assert r["max_rebound_3d"] == pytest.approx(0.12)
    assert r["max_drawdown_3d"] == pytest.approx(0.03)


def test_active_censored_at_data_edge():
    r = _labels()[2]
    # H=2 determined (2 forward days); H=3 censored (only 2 days, still active)
    assert r["censored_2d"] is False
    assert r["hit_5pct_2d"] == 0
    assert r["censored_3d"] is True
    assert r["hit_5pct_3d"] is None      # censored -> null, NOT a failure
    assert r["hit_10pct_3d"] is None
    assert r["return_3d"] is None


def test_delisted_bankruptcy_is_determined_not_censored():
    r = _labels()[3]
    # never rebounded, then delisted at ~ -90% -> hit=0 (determined), big drawdown via terminal
    assert r["censored_3d"] is False
    assert r["hit_5pct_3d"] == 0 and r["hit_10pct_3d"] == 0
    assert r["return_3d"] == pytest.approx(-0.94)     # terminal 6 vs P0 100
    assert r["max_drawdown_3d"] == pytest.approx(-0.94)


def test_delisted_merger_recovers_via_terminal():
    r = _labels()[4]
    assert r["censored_3d"] is False
    assert r["hit_5pct_3d"] == 1 and r["hit_10pct_3d"] == 1   # terminal 113.4 >= 110
    assert r["return_3d"] == pytest.approx(0.134)
    assert r["max_drawdown_3d"] == pytest.approx(0.08)


def test_intraday_high_variant_distinct_from_close():
    r = _labels()[5]
    # close peaks at 104 (< +10%) but intraday high hits 111 (>= +10%)
    assert r["hit_10pct_2d"] == 0
    assert r["hit_10pct_2d_hi"] == 1
    assert r["hit_5pct_2d"] == 0 and r["hit_5pct_2d_hi"] == 1


def test_default_horizons_expose_primary_target_column():
    # With default horizons/thresholds the primary close-based target exists and is named
    # distinctly from its intraday variant.
    df = build_labels(_EVENTS, _PRICES, _MASTER)
    assert "hit_10pct_20d" in df.columns
    assert "hit_10pct_20d_hi" in df.columns
    assert "hit_10pct_20d_hi" != "hit_10pct_20d"
