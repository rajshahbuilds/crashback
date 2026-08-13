"""Crash-cause → model-feature encoding (STU-67)."""
from __future__ import annotations

from crashback.extraction.features import CAUSE_FEATURE_COLS, assessment_to_features
from crashback.extraction.schema import CrashCauseAssessment


def _assessment(**over):
    base = dict(
        event_type="earnings", primary_cause="guidance_cut",
        revenue_impact="moderate", margin_impact="high", balance_sheet_impact="none",
        business_thesis_changed=True, temporary_vs_structural="structural", uncertainty="medium",
        rationale="x", evidence=[
            {"supports": "primary_cause", "doc_id": "d", "quote": "q"},
            {"supports": "temporary_vs_structural", "doc_id": "d", "quote": "q"}])
    base.update(over)
    return CrashCauseAssessment.model_validate(base)


def test_missing_assessment_flags_no_filing():
    f = assessment_to_features(None)
    assert set(f) == set(CAUSE_FEATURE_COLS)
    assert f["cc_has_filing"] == 0
    assert all(f[c] is None for c in CAUSE_FEATURE_COLS if c != "cc_has_filing")


def test_ordinal_encodings_and_max_impact():
    f = assessment_to_features(_assessment())
    assert f["cc_has_filing"] == 1
    assert f["cc_revenue_impact"] == 2 and f["cc_margin_impact"] == 3
    assert f["cc_balance_sheet_impact"] == 0
    assert f["cc_max_impact"] == 3                    # max(2, 3, 0)
    assert f["cc_thesis_changed"] == 1
    assert f["cc_damage"] == 2                         # structural
    assert f["cc_uncertainty"] == 1                    # medium


def test_unclear_damage_is_null():
    f = assessment_to_features(_assessment(temporary_vs_structural="unclear"))
    assert f["cc_damage"] is None                      # 'unclear' → missing, not a fake ordinal
    f2 = assessment_to_features(_assessment(temporary_vs_structural="temporary"))
    assert f2["cc_damage"] == 0
