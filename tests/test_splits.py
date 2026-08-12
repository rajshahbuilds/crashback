"""Chronological splits: date assignment, embargo boundary, no multi-split membership."""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl

from crashback.datasets.splits import assign_splits

_CAL = pl.Series("date", [date(2000, 1, 1) + timedelta(days=i) for i in range(200)])
_CFG = SimpleNamespace(
    train=(date(2000, 1, 6), date(2000, 2, 1)),        # days 5..31
    validation=(date(2000, 2, 2), date(2000, 3, 1)),   # days 32..60
    test=(date(2000, 3, 2), date(2000, 4, 1)),         # days 61..91
    embargo_trading_days=10,
)

# (event_id, crash_date, expected split)
_CASES = [
    ("train_ok", date(2000, 1, 15), "train"),        # 17 td before train end -> train
    ("train_emb", date(2000, 1, 28), "embargo"),     # 4 td before train end -> embargoed
    ("val_ok", date(2000, 2, 10), "validation"),
    ("val_emb", date(2000, 2, 27), "embargo"),       # 3 td before val end -> embargoed
    ("test_ok", date(2000, 3, 15), "test"),          # no end-embargo on test
    ("before", date(2000, 1, 2), "none"),            # before train start
    ("after", date(2000, 4, 15), "none"),            # after test end
]


def _assign():
    ev = pl.DataFrame(
        {"event_id": [c[0] for c in _CASES], "crash_date": [c[1] for c in _CASES]}
    )
    df = assign_splits(ev, _CFG, _CAL)
    return {r["event_id"]: r["split"] for r in df.iter_rows(named=True)}


def test_split_assignment_and_embargo():
    got = _assign()
    for eid, _, expected in _CASES:
        assert got[eid] == expected, f"{eid}: {got[eid]} != {expected}"


def test_each_event_gets_exactly_one_split():
    ev = pl.DataFrame(
        {"event_id": [c[0] for c in _CASES], "crash_date": [c[1] for c in _CASES]}
    )
    df = assign_splits(ev, _CFG, _CAL)
    assert df.height == len(_CASES)
    assert df["event_id"].n_unique() == len(_CASES)   # one row per event
    assert set(df["split"].unique()) <= {"train", "validation", "test", "embargo", "none"}


def test_no_train_event_outcome_window_reaches_validation():
    # The last train-labeled event must be >= embargo trading days before the train end,
    # so its 10-td outcome window cannot cross into validation.
    ev = pl.DataFrame(
        {"event_id": [f"e{i}" for i in range(40)],
         "crash_date": [date(2000, 1, 6) + timedelta(days=i) for i in range(40)]}
    )
    df = assign_splits(ev, _CFG, _CAL).filter(pl.col("split") == "train")
    # train window is days 5..31; embargo 10 => last train event on/before day 21 (2000-01-22)
    assert df["crash_date"].max() <= date(2000, 1, 22)
