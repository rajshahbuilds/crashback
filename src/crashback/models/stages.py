"""Incremental model stages (CLAUDE.md §18): each stage adds one information source so the
research can measure what *actually* improves recovery prediction.

- ``model0`` — historical base rate (no features)
- ``model1`` — crash-day + pre-crash price path + recent-crash history
- ``model2`` — model1 + market / sector context
- ``model3`` — model2 + point-in-time fundamentals

Feature columns come straight from ``crashback.datasets.assemble.FEATURE_GROUPS`` so the model
contract and the dataset contract can never drift apart. Stages are nested by construction
(each is a superset of the previous), which is what makes the M0->M1->M2->M3 comparison a clean
test of incremental value.
"""
from __future__ import annotations

from crashback.datasets.assemble import FEATURE_GROUPS

# Ordered feature-group membership per stage (nested information sets).
STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "model0": (),
    "model1": ("price", "recent_crash"),
    "model2": ("price", "recent_crash", "market_sector"),
    "model3": ("price", "recent_crash", "market_sector", "fundamentals"),
}

STAGE_DESCRIPTIONS: dict[str, str] = {
    "model0": "historical base rate",
    "model1": "price / crash / recent-crash",
    "model2": "+ market & sector context",
    "model3": "+ point-in-time fundamentals",
}


def stage_features(stage: str) -> list[str]:
    """Ordered feature columns for a stage (empty list for the base-rate model)."""
    return [c for g in STAGE_GROUPS[stage] for c in FEATURE_GROUPS[g]]
