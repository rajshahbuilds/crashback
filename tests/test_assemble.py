"""Assembly of events_v1: one row per event, feature/outcome separation, validation."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from crashback.datasets.assemble import (
    _FUND_FEATURES,
    _FUND_META,
    _MARKET,
    _PRICE,
    _RECENT,
    FEATURE_COLS,
    OUTCOME_COLS,
    OUTCOME_META_COLS,
    PRIMARY_TARGET,
    assemble_events,
    validate_events,
)


def _fill(ids, cols, value=0.0):
    return pl.DataFrame({"event_id": ids, **{c: [value] * len(ids) for c in cols}})


def _inputs():
    ids = ["a", "b"]
    crash_events = pl.DataFrame(
        {
            "event_id": ids, "security_id": [1, 2], "company_id": [10, 20],
            "ticker_as_of_event": ["AAA", "BBB"], "crash_date": [date(2020, 1, 2)] * 2,
            "in_universe_at_event": [True, True], "passes_min_price": [True, False],
            "crash_close": [50.0, 8.0],
        }
    )
    price = _fill(ids, _PRICE).with_columns(pl.lit(-0.15).alias("crash_return"))
    recent = _fill(ids, _RECENT)
    market = _fill(ids, _MARKET)
    fund = _fill(ids, [*_FUND_FEATURES, *_FUND_META])
    labels = _fill(ids, OUTCOME_COLS).with_columns(pl.lit(1).alias(PRIMARY_TARGET))
    labels = labels.with_columns(_fill(ids, OUTCOME_META_COLS).drop("event_id"))
    return crash_events, labels, price, recent, market, fund


def test_feature_and_outcome_columns_are_disjoint():
    assert set(FEATURE_COLS).isdisjoint(set(OUTCOME_COLS))
    assert PRIMARY_TARGET in OUTCOME_COLS
    assert PRIMARY_TARGET not in FEATURE_COLS
    # each family contributes its features
    for fam in (_PRICE, _RECENT, _MARKET, _FUND_FEATURES):
        assert set(fam) <= set(FEATURE_COLS)


def test_assemble_one_row_per_event_with_all_groups():
    ce, lab, price, recent, market, fund = _inputs()
    df = assemble_events(ce, lab, price, recent, market, fund)
    assert df.height == 2
    assert df["event_id"].n_unique() == 2
    assert df.columns[:5] == ["event_id", "security_id", "company_id",
                              "ticker_as_of_event", "crash_date"]
    for c in (*FEATURE_COLS, *OUTCOME_COLS, "crash_close"):
        assert c in df.columns
    rep = validate_events(df)
    assert rep["rows"] == 2 and rep["primary_target"] == PRIMARY_TARGET


def test_validate_rejects_duplicate_event_id():
    ce, lab, price, recent, market, fund = _inputs()
    df = assemble_events(ce, lab, price, recent, market, fund)
    dup = pl.concat([df, df.head(1)])
    with pytest.raises(ValueError, match="duplicate event_id"):
        validate_events(dup)


def test_validate_rejects_non_binary_hit_label():
    ce, lab, price, recent, market, fund = _inputs()
    df = assemble_events(ce, lab, price, recent, market, fund).with_columns(
        pl.lit(5).alias(PRIMARY_TARGET)
    )
    with pytest.raises(ValueError, match="non-binary"):
        validate_events(df)
