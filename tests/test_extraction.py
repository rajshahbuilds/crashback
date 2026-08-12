"""Crash-cause extraction: schema + evidence validation, prompt safety, harness (STU-66).

Hermetic — a stub LLMClient stands in for the model, so validation/evidence-grounding are tested
with no API key or network.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from crashback.extraction.extract import Extraction, LLMClient, extract_crash_cause
from crashback.extraction.prompt import build_messages, contains_price_prohibition
from crashback.extraction.schema import (
    SCHEMA_VERSION,
    ValidationError,
    json_schema,
    validate_assessment,
)

_GOOD = {
    "schema_version": SCHEMA_VERSION,
    "event_type": "earnings",
    "primary_cause": "customer_or_subscriber_loss",
    "revenue_impact": "moderate", "margin_impact": "low", "balance_sheet_impact": "none",
    "business_thesis_changed": False,
    "temporary_vs_structural": "mixed",
    "uncertainty": "medium",
    "rationale": "Reported net subscriber decline and softer guidance; core service intact.",
    "evidence": [
        {"supports": "primary_cause", "doc_id": "acc-1", "quote": "lost 200,000 subscribers"},
        {"supports": "temporary_vs_structural", "doc_id": "acc-1",
         "quote": "expect to resume growth in H2"},
    ],
}
_ALLOWED = {"acc-1", "acc-2"}
_DOCS = [{"doc_id": "acc-1", "source_type": "8-K", "available_at": "2022-04-19T20:03:19",
          "text": "Q1 results: lost 200,000 subscribers. We expect to resume growth in H2."}]


class _Stub(LLMClient):
    model = "stub-model"

    def __init__(self, text: str):
        self._text = text

    def complete(self, system: str, user: str) -> str:
        return self._text


# --- schema + evidence validation --------------------------------------------------
def test_valid_assessment_passes():
    a = validate_assessment(_GOOD, _ALLOWED)
    assert a.event_type == "earnings"                 # use_enum_values → plain strings
    assert a.temporary_vs_structural == "mixed"
    assert len(a.evidence) == 2


def test_missing_evidence_for_required_field_rejected():
    bad = {**_GOOD, "evidence": [_GOOD["evidence"][0]]}   # only primary_cause cited
    with pytest.raises(ValidationError, match="temporary_vs_structural"):
        validate_assessment(bad, _ALLOWED)


def test_hallucinated_doc_id_rejected():
    bad = {**_GOOD, "evidence": [
        {"supports": "primary_cause", "doc_id": "acc-1", "quote": "x"},
        {"supports": "temporary_vs_structural", "doc_id": "acc-999", "quote": "y"}]}
    with pytest.raises(ValidationError, match="non-retrieved doc_id"):
        validate_assessment(bad, _ALLOWED)


def test_wrong_schema_version_rejected():
    with pytest.raises(ValidationError, match="schema_version"):
        validate_assessment({**_GOOD, "schema_version": "crash_cause.v0"}, _ALLOWED)


def test_unknown_enum_value_rejected():
    with pytest.raises(ValidationError):
        validate_assessment({**_GOOD, "primary_cause": "aliens"}, _ALLOWED)


def test_json_schema_is_machine_readable():
    js = json_schema()
    assert js["title"] == "CrashCauseAssessment"
    assert "primary_cause" in js["properties"] and "evidence" in js["properties"]


# --- prompt safety -----------------------------------------------------------------
def test_prompt_forbids_price_predictions():
    assert contains_price_prohibition()
    system, user = build_messages(ticker="NFLX", crash_date=date(2022, 4, 20),
                                  crash_return=-0.35, documents=_DOCS)
    assert "do not predict future price" in system.lower()
    assert "acc-1" in user and "lost 200,000 subscribers" in user    # doc text embedded
    assert SCHEMA_VERSION in user


# --- harness -----------------------------------------------------------------------
def test_extract_returns_validated_extraction_with_provenance():
    ex = extract_crash_cause(
        _Stub(json.dumps(_GOOD)), event_id="89393_20220420", ticker="NFLX",
        crash_date=date(2022, 4, 20), crash_return=-0.35, documents=_DOCS)
    assert isinstance(ex, Extraction)
    assert ex.model == "stub-model" and ex.prompt_version and ex.schema_version == SCHEMA_VERSION
    row = ex.to_row()
    assert row["primary_cause"] == "customer_or_subscriber_loss" and row["n_evidence"] == 2
    assert row["cited_doc_ids"] == "acc-1"


def test_extract_tolerates_markdown_fences():
    ex = extract_crash_cause(
        _Stub("```json\n" + json.dumps(_GOOD) + "\n```"), event_id="e", ticker="NFLX",
        crash_date=date(2022, 4, 20), crash_return=-0.35, documents=_DOCS)
    assert ex.assessment.uncertainty == "medium"


def test_extract_rejects_hallucinated_source_from_model():
    poisoned = {**_GOOD, "evidence": [
        {"supports": "primary_cause", "doc_id": "acc-1", "quote": "x"},
        {"supports": "temporary_vs_structural", "doc_id": "made-up", "quote": "y"}]}
    with pytest.raises(ValidationError, match="non-retrieved"):
        extract_crash_cause(_Stub(json.dumps(poisoned)), event_id="e", ticker="NFLX",
                            crash_date=date(2022, 4, 20), crash_return=-0.35, documents=_DOCS)


def test_extract_rejects_non_json_output():
    with pytest.raises(ValidationError, match="no JSON object"):
        extract_crash_cause(_Stub("I cannot help with that."), event_id="e", ticker="NFLX",
                            crash_date=date(2022, 4, 20), crash_return=-0.35, documents=_DOCS)
