"""Crash-event detection: threshold boundary, consecutive events, no false/ dropped events."""
from __future__ import annotations

from datetime import date

import polars as pl

from crashback.events.detect import (
    CRASH_EVENT_SCHEMA,
    build_crash_events,
    detect_crashes,
)

# security 1: two consecutive crashes, then a near-miss and an up day
# security 2: one deep crash at a sub-$5 price
# security 3: a split-like day (price halves, but daily_return ~0) -> NOT a crash
# security 4: a crash on a date not covered by any master name-period
_PRICES = pl.DataFrame(
    {
        "security_id": [1, 1, 1, 1, 2, 3, 4],
        "date": [
            date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7),
            date(2020, 1, 2), date(2020, 1, 2), date(1995, 6, 1),
        ],
        "daily_return": [-0.10, -0.15, -0.0999, 0.05, -0.20, 0.031, -0.30],
        "close": [90.0, 76.5, 76.4, 80.0, 3.0, 45.0, 12.0],
        "volume": [1e6, 2e6, 1e6, 1e6, 5e5, 9e6, 1e5],
    }
)
_MASTER = pl.DataFrame(
    {
        "security_id": [1, 2, 3, 4],
        "company_id": [11, 22, 33, 44],
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "ticker_start": [date(2010, 1, 1)] * 4,
        "ticker_end": [date(2099, 1, 1)] * 4,  # security 4's crash (1995) is NOT covered
    }
)


def test_threshold_boundary_and_no_false_split_event():
    crashes = detect_crashes(_PRICES, -0.10)
    keys = set(zip(crashes["security_id"].to_list(), crashes["crash_date"].to_list(), strict=True))
    # -0.10 (inclusive) and -0.15 and -0.20 and -0.30 qualify
    assert (1, date(2020, 1, 2)) in keys
    assert (1, date(2020, 1, 3)) in keys
    assert (2, date(2020, 1, 2)) in keys
    assert (4, date(1995, 6, 1)) in keys
    # -0.0999 (just misses), +0.05 (up), and +0.031 split-like day do NOT
    assert (1, date(2020, 1, 6)) not in keys
    assert (1, date(2020, 1, 7)) not in keys
    assert (3, date(2020, 1, 2)) not in keys
    assert crashes.height == 4


def test_consecutive_crashes_are_separate_events():
    events = build_crash_events(_PRICES, _MASTER, threshold=-0.10, min_price=5.0)
    s1 = events.filter(pl.col("security_id") == 1).sort("crash_date")
    assert s1.height == 2  # two consecutive days -> two distinct events
    assert s1["event_id"].to_list() == ["1_20200102", "1_20200103"]
    assert s1["crash_date"].to_list() == [date(2020, 1, 2), date(2020, 1, 3)]


def test_schema_and_identifiers_and_min_price_flag():
    events = build_crash_events(_PRICES, _MASTER, threshold=-0.10, min_price=5.0)
    assert events.columns == list(CRASH_EVENT_SCHEMA.keys())
    row1 = events.filter(pl.col("event_id") == "1_20200102").row(0, named=True)
    assert row1["company_id"] == 11
    assert row1["ticker_as_of_event"] == "AAA"
    assert row1["crash_close"] == 90.0
    assert row1["passes_min_price"] is True
    assert row1["in_universe_at_event"] is True
    # security 2 crashed at $3 -> fails the $5 liquidity flag (but is NOT dropped)
    row2 = events.filter(pl.col("security_id") == 2).row(0, named=True)
    assert row2["passes_min_price"] is False


def test_crash_without_covering_name_period_is_kept_with_null_ticker():
    events = build_crash_events(_PRICES, _MASTER, threshold=-0.10, min_price=5.0)
    row4 = events.filter(pl.col("security_id") == 4).row(0, named=True)
    assert row4["ticker_as_of_event"] is None   # 1995 crash not covered by 2010+ period
    assert row4["company_id"] is None
    assert row4["in_universe_at_event"] is False  # not in a clean universe name-period
    assert row4["event_id"] == "4_19950601"     # event still exists — never dropped


def test_configurable_threshold_changes_event_set():
    strict = build_crash_events(_PRICES, _MASTER, threshold=-0.20, min_price=5.0)
    keys = set(zip(strict["security_id"].to_list(), strict["crash_date"].to_list(), strict=True))
    assert (2, date(2020, 1, 2)) in keys   # -0.20 qualifies at -0.20 threshold
    assert (4, date(1995, 6, 1)) in keys   # -0.30 qualifies
    assert (1, date(2020, 1, 2)) not in keys  # -0.10 no longer qualifies
