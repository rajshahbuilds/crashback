"""Encode a crash-cause assessment into model-ready numeric features (Model 4, STU-67).

Compact by design (~8 columns) so a bounded pilot doesn't overfit: ordinal impact/uncertainty/
damage encodings plus flags. Missing (no cause filing, or an ``unclear`` damage read) is left as
None — the tree model handles it natively, and ``cc_has_filing`` distinguishes "no document" (often
a macro/sector crash) from a real filing. These features attach to the best V1 set (Model 3) to
form Model 4.
"""
from __future__ import annotations

from crashback.extraction.schema import CrashCauseAssessment

_IMPACT = {"none": 0, "low": 1, "moderate": 2, "high": 3, "severe": 4}
_DAMAGE = {"temporary": 0, "mixed": 1, "structural": 2}   # 'unclear' -> None
_UNCERTAINTY = {"low": 0, "medium": 1, "high": 2}

CAUSE_FEATURE_COLS: tuple[str, ...] = (
    "cc_has_filing",          # 1 if a contemporaneous cause filing was found, else 0
    "cc_revenue_impact",      # ordinal 0..4
    "cc_margin_impact",
    "cc_balance_sheet_impact",
    "cc_max_impact",          # worst of the three impact axes
    "cc_thesis_changed",      # 1/0
    "cc_damage",              # temporary=0, mixed=1, structural=2 (the core V2 signal)
    "cc_uncertainty",         # 0..2
)


def assessment_to_features(assessment: CrashCauseAssessment | None) -> dict:
    """Numeric feature dict from an assessment; all-None (except cc_has_filing=0) if absent."""
    if assessment is None:
        return {c: (0 if c == "cc_has_filing" else None) for c in CAUSE_FEATURE_COLS}
    ri = _IMPACT[assessment.revenue_impact]
    mi = _IMPACT[assessment.margin_impact]
    bi = _IMPACT[assessment.balance_sheet_impact]
    return {
        "cc_has_filing": 1,
        "cc_revenue_impact": ri,
        "cc_margin_impact": mi,
        "cc_balance_sheet_impact": bi,
        "cc_max_impact": max(ri, mi, bi),
        "cc_thesis_changed": int(assessment.business_thesis_changed),
        "cc_damage": _DAMAGE.get(assessment.temporary_vs_structural),   # None for 'unclear'
        "cc_uncertainty": _UNCERTAINTY[assessment.uncertainty],
    }
