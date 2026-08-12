"""Market/sector context features: equal-weight aggregates, focal exclusion, no look-ahead."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from crashback.features.market_sector import build_market_sector_features

# 4 securities: sec1,2 in sector 36 (SIC 3600); sec3,4 in sector 60 (SIC 6000). 6 days.
# Focal = sec1, crash on day 5.
_RETS = {
    1: [0.02, 0.01, 0.00, 0.01, -0.01, -0.15],
    2: [0.01, 0.00, 0.01, 0.00, 0.00, -0.09],
    3: [0.00, 0.01, 0.00, 0.02, 0.01, -0.04],
    4: [0.01, -0.01, 0.02, 0.00, 0.00, -0.06],
}
_SIC = {1: 3600, 2: 3600, 3: 6000, 4: 6000}
_DATES = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]


def _prices(rets=None):
    rets = rets or _RETS
    rows = {"security_id": [], "date": [], "daily_return": []}
    for sid, series in rets.items():
        for i, r in enumerate(series):
            rows["security_id"].append(sid)
            rows["date"].append(date(2020, 1, 1) + timedelta(days=i))
            rows["daily_return"].append(r)
    return pl.DataFrame(rows)


_MASTER = pl.DataFrame({"security_id": [1, 2, 3, 4], "sic_code": [3600, 3600, 6000, 6000]})
_EVENTS = pl.DataFrame(
    {"event_id": ["e1"], "security_id": [1], "crash_date": [_DATES[5]], "crash_return": [-0.15]}
)


def _cum(series):
    out = 1.0
    for r in series:
        out *= 1 + r
    return out - 1.0


def test_market_and_sector_equal_weight_with_focal_exclusion():
    r = build_market_sector_features(_EVENTS, _prices(), _MASTER).row(0, named=True)

    # market EW return on the crash day = mean of all 4 securities' day-5 returns
    mkt_d5 = sum(_RETS[s][5] for s in _RETS) / 4
    assert r["market_return_1d"] == pytest.approx(mkt_d5)                 # -0.085

    # sector 36 on day 5 EXCLUDING the focal (sec1) == sec2 alone
    assert r["sector_return_1d"] == pytest.approx(_RETS[2][5])            # -0.09
    assert r["sector_n_members"] == 2

    # trailing 5d cumulative market return (days 1..5, inclusive)
    mkt_series = [sum(_RETS[s][i] for s in _RETS) / 4 for i in range(6)]
    assert r["market_return_5d"] == pytest.approx(_cum(mkt_series[1:6]))

    # sector_return_5d uses the FULL sector (incl focal): sector 36 EW days 1..5
    sec36 = [(_RETS[1][i] + _RETS[2][i]) / 2 for i in range(6)]
    assert r["sector_return_5d"] == pytest.approx(_cum(sec36[1:6]))

    assert r["market_return_20d"] is None      # only 6 days of history
    assert r["market_volatility_20d"] is None


def test_no_lookahead_from_future_market_moves():
    before = build_market_sector_features(_EVENTS, _prices(), _MASTER).row(0, named=True)
    # append two wild post-crash days for every security
    fut = {s: series + [0.5, -0.5] for s, series in _RETS.items()}
    after = build_market_sector_features(_EVENTS, _prices(fut), _MASTER).row(0, named=True)
    for k in ("market_return_1d", "market_return_5d", "sector_return_1d", "sector_return_5d"):
        assert before[k] == pytest.approx(after[k]), k
