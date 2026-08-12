"""As-of fundamentals join: publication-date boundary, leakage injection, derived features."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from crashback.fundamentals.features import build_fundamental_features

# gvkey 1690 / permno 14593: five quarters, each published ~30 days after period end.
# columns: period_end, rdq, rev, ni, gp, oi, ebitda, intx, ta, cash, debt, ca, cl, se, eps, sh
_Q = [
    (date(2019, 3, 31), date(2019, 4, 30),
     100, 10, 40, 20, 25, 2, 500, 50, 100, 150, 100, 200, 1.0, 50),
    (date(2019, 6, 30), date(2019, 7, 30),
     110, 11, 44, 22, 27, 2, 510, 55, 100, 155, 100, 205, 1.1, 50),
    (date(2019, 9, 30), date(2019, 10, 30),
     120, 12, 48, 24, 29, 2, 520, 60, 100, 160, 100, 210, 1.2, 50),
    (date(2019, 12, 31), date(2020, 1, 30),
     130, 13, 52, 26, 31, 2, 530, 65, 100, 165, 100, 215, 1.3, 50),
    (date(2020, 3, 31), date(2020, 4, 30),
     140, 14, 56, 28, 33, 2, 540, 70, 100, 170, 100, 220, 1.4, 50),
]


def _fundamentals(quarters=_Q, company_id=1690):
    cols = ["period_end", "public_date", "revenue", "net_income", "gross_profit",
            "operating_income", "ebitda", "interest_expense", "total_assets", "cash",
            "total_debt", "current_assets", "current_liabilities", "stockholders_equity",
            "eps", "shares_outstanding"]
    data = {c: [q[i] for q in quarters] for i, c in enumerate(cols)}
    data["company_id"] = [company_id] * len(quarters)
    data["rdq_available"] = [True] * len(quarters)
    return pl.DataFrame(data).with_columns(
        pl.col("period_end").cast(pl.Date), pl.col("public_date").cast(pl.Date),
        *[pl.col(c).cast(pl.Float64) for c in cols[2:]],
    )


_LINK = pl.DataFrame(
    {"gvkey": ["001690"], "lpermno": [14593], "linktype": ["LC"], "linkprim": ["P"],
     "linkdt": [date(2000, 1, 1)], "linkenddt": [None]},
    schema_overrides={"linkdt": pl.Date, "linkenddt": pl.Date},
)


def _events(rows):
    return pl.DataFrame(
        {"event_id": [r[0] for r in rows], "security_id": [r[1] for r in rows],
         "crash_date": [r[2] for r in rows], "crash_close": [float(r[3]) for r in rows]},
        schema_overrides={"crash_date": pl.Date},
    )


def _feat(events):
    df = build_fundamental_features(events, _fundamentals(), _LINK)
    return {r["event_id"]: r for r in df.iter_rows(named=True)}


def test_asof_excludes_quarter_published_after_the_crash():
    # Crash 2020-04-15: Q5 period ended 2020-03-31 but was published 2020-04-30 (AFTER) ->
    # must attach Q4 (published 2020-01-30), not Q5.
    r = _feat(_events([("e", 14593, date(2020, 4, 15), 20)]))["e"]
    assert r["fundamentals_available"] is True
    # days since the attached period_end: Q4 (2019-12-31) -> 106, NOT Q5 (2020-03-31) -> 15
    assert r["days_since_fundamental"] == 106


def test_injecting_a_future_fundamental_does_not_change_the_join():
    events = _events([("e", 14593, date(2020, 5, 15), 20)])
    base = build_fundamental_features(events, _fundamentals(), _LINK).row(0, named=True)
    # inject a far-future quarter (published 2021) — must be ignored for the 2020-05-15 crash
    future = _Q + [(date(2020, 6, 30), date(2021, 1, 30),
                    200, 40, 80, 60, 70, 2, 600, 90, 100, 200, 100, 260, 4.0, 50)]
    inj = build_fundamental_features(events, _fundamentals(future), _LINK).row(0, named=True)
    for k in ("days_since_fundamental", "net_margin", "market_cap", "pe"):
        assert base[k] == pytest.approx(inj[k]), k


def test_derived_features_hand_calc():
    # Crash 2020-05-15 attaches Q5 (published 2020-04-30). TTM = sum of Q2..Q5.
    r = _feat(_events([("e", 14593, date(2020, 5, 15), 20)]))["e"]
    assert r["days_since_fundamental"] == 45           # 2020-05-15 - 2020-03-31
    assert r["gross_margin"] == pytest.approx(200 / 500)
    assert r["net_margin"] == pytest.approx(50 / 500)
    assert r["roe"] == pytest.approx(50 / 220)
    assert r["current_ratio"] == pytest.approx(1.70)
    assert r["net_debt_to_ebitda"] == pytest.approx(30 / 120)
    assert r["interest_coverage"] == pytest.approx(100 / 8)
    assert r["market_cap"] == pytest.approx(20 * 50)   # crash_close x shares
    assert r["pe"] == pytest.approx(1000 / 50)
    assert r["price_to_sales"] == pytest.approx(1000 / 500)
    assert r["ev_to_ebitda"] == pytest.approx((1000 + 30) / 120)
    assert r["revenue_growth_yoy"] == pytest.approx(140 / 100 - 1)
    assert r["eps_growth_yoy"] == pytest.approx(1.4 / 1.0 - 1)
    assert r["fundamentals_stale"] is False


def test_missing_link_and_pre_link_crash_have_no_fundamentals():
    rows = [
        ("no_link", 99999, date(2020, 5, 15), 20),      # permno not in CCM
        ("pre_link", 14593, date(1999, 1, 1), 20),      # crash before linkdt (2000-01-01)
    ]
    f = _feat(_events(rows))
    assert f["no_link"]["fundamentals_available"] is False
    assert f["no_link"]["market_cap"] is None
    assert f["pre_link"]["fundamentals_available"] is False


def test_stale_flag_when_fundamentals_are_old():
    # Crash well over STALE_DAYS after the last available quarter (2020-03-31).
    r = _feat(_events([("e", 14593, date(2021, 6, 1), 20)]))["e"]
    assert r["fundamentals_available"] is True
    assert r["fundamentals_stale"] is True             # ~427 days > 200
