"""Assemble the canonical V1 event dataset (events_v1): one row per crash event.

Joins the crash-event spine with the four feature families and the recovery labels, keeping
FEATURE columns strictly separated from OUTCOME columns so modeling can never train on a
label. The column groups below are the contract downstream code uses (``FEATURE_COLS`` for
model inputs, ``OUTCOME_COLS`` for targets); they are disjoint by construction.
"""
from __future__ import annotations

import polars as pl

from crashback.features.market_sector import FEATURE_NAMES as _MARKET
from crashback.features.price import FEATURE_NAMES as _PRICE
from crashback.features.recent_crash import FEATURE_NAMES as _RECENT
from crashback.fundamentals.features import DERIVED_FEATURES as _FUND_ALL
from crashback.labels.outcomes import (
    HORIZONS,
    THRESHOLDS,
    censored_col,
    drawdown_col,
    hit_col,
    nforward_col,
    rebound_col,
    return_col,
)

# --- column groups (the dataset contract) -----------------------------------------
ID_COLS: tuple[str, ...] = ("event_id", "security_id", "company_id", "ticker_as_of_event",
                            "crash_date")
ELIGIBILITY_COLS: tuple[str, ...] = ("in_universe_at_event", "passes_min_price")
ANCHOR_COLS: tuple[str, ...] = ("crash_close",)

# fundamentals: split derived ratios (features) from provenance/flags (meta)
_FUND_META: tuple[str, ...] = (
    "fundamentals_available", "fundamentals_period_end", "fundamentals_public_date",
    "days_since_fundamental", "fundamentals_stale", "fundamentals_rdq_available",
)
_FUND_FEATURES: tuple[str, ...] = tuple(c for c in _FUND_ALL if c not in _FUND_META)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": tuple(_PRICE),
    "recent_crash": tuple(_RECENT),
    "market_sector": tuple(_MARKET),
    "fundamentals": _FUND_FEATURES,
}
FEATURE_COLS: tuple[str, ...] = tuple(c for grp in FEATURE_GROUPS.values() for c in grp)
FEATURE_META_COLS: tuple[str, ...] = _FUND_META

OUTCOME_COLS: tuple[str, ...] = tuple(
    [hit_col(p, h) for h in HORIZONS for p, _ in THRESHOLDS]
    + [hit_col(p, h, hi=True) for h in HORIZONS for p, _ in THRESHOLDS]
    + [f(h) for h in HORIZONS for f in (return_col, rebound_col, drawdown_col)]
)
OUTCOME_META_COLS: tuple[str, ...] = tuple(
    [censored_col(h) for h in HORIZONS]
    + [nforward_col(h) for h in HORIZONS]
    + ["no_close_data", "delisting_return_missing"]
)

PRIMARY_TARGET = "hit_10pct_20d"


def assemble_events(
    crash_events: pl.DataFrame,
    labels: pl.DataFrame,
    price: pl.DataFrame,
    recent: pl.DataFrame,
    market_sector: pl.DataFrame,
    fundamentals: pl.DataFrame,
) -> pl.DataFrame:
    """One row per event_id: ids + eligibility + anchor + features + fund-meta + outcomes."""
    spine = crash_events.select(*ID_COLS, *ELIGIBILITY_COLS, *ANCHOR_COLS)
    df = (
        spine.join(price.select("event_id", *_PRICE), on="event_id", how="left")
        .join(recent.select("event_id", *_RECENT), on="event_id", how="left")
        .join(market_sector.select("event_id", *_MARKET), on="event_id", how="left")
        .join(
            fundamentals.select("event_id", *_FUND_FEATURES, *_FUND_META),
            on="event_id", how="left",
        )
        .join(
            labels.select("event_id", *OUTCOME_COLS, *OUTCOME_META_COLS),
            on="event_id", how="left",
        )
    )
    order = [*ID_COLS, *ELIGIBILITY_COLS, *ANCHOR_COLS, *FEATURE_COLS,
             *FEATURE_META_COLS, *OUTCOME_COLS, *OUTCOME_META_COLS]
    return df.select(order).sort("event_id")


def validate_events(df: pl.DataFrame, *, crash_threshold: float = -0.10) -> dict:
    """Structural / sanity checks. Raises on hard violations; returns a report."""
    problems: list[str] = []

    if df["event_id"].n_unique() != df.height:
        problems.append("duplicate event_id rows")

    # feature and outcome columns must be disjoint (no label leaks into the feature list)
    overlap = set(FEATURE_COLS) & set(OUTCOME_COLS)
    if overlap:
        problems.append(f"feature/outcome overlap: {sorted(overlap)}")

    missing = [c for c in (*FEATURE_COLS, *OUTCOME_COLS) if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")

    # every event is a crash: crash_return <= threshold (allow tiny float slack)
    if "crash_return" in df.columns:
        worst = df["crash_return"].max()
        if worst is not None and worst > crash_threshold + 1e-9:
            problems.append(f"crash_return above threshold: max={worst}")

    # binary labels must be in {0,1} (nulls allowed = censored/no-data)
    for c in df.columns:
        if c.startswith("hit_"):
            vals = set(df[c].drop_nulls().unique().to_list())
            if not vals <= {0, 1}:
                problems.append(f"{c} has non-binary values: {sorted(vals)[:5]}")

    if problems:
        raise ValueError("dataset validation failed: " + "; ".join(problems))

    return {
        "rows": df.height,
        "unique_events": df["event_id"].n_unique(),
        "n_features": len(FEATURE_COLS),
        "n_outcomes": len(OUTCOME_COLS),
        "primary_target": PRIMARY_TARGET,
    }
