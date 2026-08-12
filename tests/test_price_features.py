"""Crash-day / pre-crash price features: hand-calc, sparse history, and no look-ahead."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from crashback.features.price import FEATURE_NAMES, build_price_features


def _prices(closes, *, highs=None, lows=None, opens=None, vols=None, rets=None, sid=1):
    n = len(closes)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "security_id": [sid] * n,
            "date": dates,
            "open": opens or closes,
            "high": highs or [c + 1 for c in closes],
            "low": lows or [c - 1 for c in closes],
            "close": closes,
            "volume": vols or [1000.0] * n,
            "daily_return": rets or [0.01] * n,
        }
    )


def _event(day_index, sid=1):
    d = date(2020, 1, 1) + timedelta(days=day_index)
    return pl.DataFrame({"event_id": ["e"], "security_id": [sid], "crash_date": [d]})


def _build(prices, day_index):
    return build_price_features(_event(day_index), prices).row(0, named=True)


def test_features_match_hand_calculated_values():
    # 25 rising days (close = 100..124), then a crash on day 25.
    closes = [100.0 + i for i in range(25)] + [110.0]
    highs = [c + 1 for c in closes[:25]] + [125.0]
    lows = [c - 1 for c in closes[:25]] + [100.0]
    opens = closes[:25] + [124.0]
    vols = [1000.0] * 25 + [5000.0]
    rets = [0.01] * 25 + [-0.1129]
    p = _prices(closes, highs=highs, lows=lows, opens=opens, vols=vols, rets=rets)
    r = _build(p, 25)

    assert r["crash_return"] == pytest.approx(-0.1129)
    assert r["opening_gap"] == pytest.approx(124 / 124 - 1)               # 0.0
    assert r["intraday_range"] == pytest.approx((125 - 100) / 124)        # 0.2016
    assert r["close_vs_low"] == pytest.approx(110 / 100 - 1)              # 0.10
    assert r["close_vs_open"] == pytest.approx(110 / 124 - 1)             # -0.1129
    assert r["relative_volume_20d"] == pytest.approx(5000 / 1000)        # 5.0 (prior 20 avg)
    assert r["volatility_20d"] == pytest.approx(0.0)                      # prior returns const -> 0
    assert r["return_5d_pre"] == pytest.approx(124 / 119 - 1)            # close[24]/close[19]
    assert r["return_20d_pre"] == pytest.approx(124 / 104 - 1)           # close[24]/close[4]
    assert r["distance_from_20d_high"] == pytest.approx(110 / 125 - 1)   # -0.12
    assert r["drawdown_20d"] == pytest.approx(110 / 124 - 1)             # -0.1129


def test_sparse_history_is_null_for_long_windows_but_crashday_defined():
    closes = [100.0, 105.0, 96.0, 90.0]      # crash on day 3, only 3 prior days
    r = _build(_prices(closes), 3)
    assert r["opening_gap"] is not None       # needs only 1 prior day
    assert r["close_vs_open"] is not None
    for long_feat in ("relative_volume_20d", "volatility_20d", "return_20d_pre",
                      "return_252d_pre", "distance_from_52w_high", "drawdown_60d"):
        assert r[long_feat] is None, long_feat


def test_no_lookahead_leakage():
    # Compute features at the crash day, then append wild FUTURE rows and recompute.
    closes = [100.0 + i for i in range(25)] + [110.0]
    p = _prices(closes)
    before = _build(p, 25)

    future = _prices([999.0, 0.01, 500.0], sid=1)  # dates continue after the crash day
    future = future.with_columns(
        (pl.col("date") + pl.duration(days=26)).alias("date")
    )
    after = _build(pl.concat([p, future]), 25)

    for f in FEATURE_NAMES:
        assert before[f] == after[f] or (before[f] is None and after[f] is None), f
