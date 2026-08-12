"""Versioned, machine-readable crash-cause / fundamental-damage extraction schema (STU-66).

The LLM is used as a **structured event-understanding extractor**, never as a price oracle
(CLAUDE.md §24). It reads only the contemporaneous documents (STU-65) and emits this schema; the
output is Pydantic-validated before any downstream use, and **every substantive judgment must
cite evidence** that references a retrieved document by ``doc_id`` — so an assessment can always
be audited back to source text, and hallucinated sources are rejected.

``SCHEMA_VERSION`` is embedded in every record and in the prompt; bump it on any field change so
extractions remain reproducible and comparable across versions.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "crash_cause.v1"

# Fields whose judgment must be grounded in at least one cited document.
REQUIRED_EVIDENCE_FIELDS = ("primary_cause", "temporary_vs_structural")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class EventType(StrEnum):
    earnings = "earnings"
    guidance = "guidance"
    regulatory = "regulatory"
    legal = "legal"
    mna = "mna"
    management_change = "management_change"
    product = "product"
    capital_action = "capital_action"     # raise / dilution / buyback change
    macro_sector = "macro_sector"
    other = "other"


class PrimaryCause(StrEnum):
    revenue_miss = "revenue_miss"
    guidance_cut = "guidance_cut"
    margin_compression = "margin_compression"
    eps_miss = "eps_miss"
    demand_weakness = "demand_weakness"
    customer_or_subscriber_loss = "customer_or_subscriber_loss"
    regulatory_action = "regulatory_action"
    litigation = "litigation"
    clinical_or_trial_failure = "clinical_or_trial_failure"
    mna_news = "mna_news"
    dilution_or_raise = "dilution_or_raise"
    management_departure = "management_departure"
    competitive_threat = "competitive_threat"
    macro_or_sector_selloff = "macro_or_sector_selloff"
    accounting_or_restatement = "accounting_or_restatement"
    other = "other"


class ImpactLevel(StrEnum):
    none = "none"
    low = "low"
    moderate = "moderate"
    high = "high"
    severe = "severe"


class DamageAssessment(StrEnum):
    temporary = "temporary"       # overreaction / transient shock
    mixed = "mixed"
    structural = "structural"     # genuine long-term fundamental damage
    unclear = "unclear"


class Uncertainty(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class EvidenceRef(BaseModel):
    """A supporting excerpt tying an assessment field to a retrieved document.

    Extra keys are ignored (not forbidden): LLM tool-use sometimes adds helper fields like a
    context note; we keep only the contracted fields. The top-level assessment stays strict.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    supports: str = Field(..., description="Which assessment field this evidence supports.")
    doc_id: str = Field(..., description="doc_id of a RETRIEVED document (no invented sources).")
    quote: str = Field(..., min_length=1, max_length=500,
                       description="Short verbatim excerpt from that document.")


class CrashCauseAssessment(_Base):
    """Structured understanding of *why* a crash happened — auditable, non-predictive."""

    schema_version: str = SCHEMA_VERSION
    event_type: EventType
    primary_cause: PrimaryCause
    revenue_impact: ImpactLevel
    margin_impact: ImpactLevel
    balance_sheet_impact: ImpactLevel
    business_thesis_changed: bool
    temporary_vs_structural: DamageAssessment
    uncertainty: Uncertainty
    rationale: str = Field(..., min_length=1, max_length=2000,
                           description="Brief reasoning grounded in the documents (no price/return "
                                       "forecast, no buy/sell).")
    evidence: list[EvidenceRef] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _core_fields_are_grounded(self) -> CrashCauseAssessment:
        supported = {e.supports for e in self.evidence}
        missing = [f for f in REQUIRED_EVIDENCE_FIELDS if f not in supported]
        if missing:
            raise ValueError(f"missing evidence for required field(s): {missing}")
        return self


class ValidationError(ValueError):
    """Raised when an LLM extraction fails schema or evidence-grounding validation."""


def validate_assessment(raw: dict, allowed_doc_ids: set[str]) -> CrashCauseAssessment:
    """Validate a raw LLM dict against the schema AND ground evidence to retrieved documents.

    Rejects (a) anything that violates the Pydantic schema, (b) any evidence citing a ``doc_id``
    not in ``allowed_doc_ids`` (a hallucinated / non-retrieved source), and (c) a mismatched
    ``schema_version``.
    """
    try:
        obj = CrashCauseAssessment.model_validate(raw)
    except Exception as e:  # noqa: BLE001 - normalize to our error type
        raise ValidationError(f"schema validation failed: {e}") from e
    if obj.schema_version != SCHEMA_VERSION:
        raise ValidationError(f"schema_version {obj.schema_version!r} != {SCHEMA_VERSION!r}")
    bad = sorted({e.doc_id for e in obj.evidence} - set(allowed_doc_ids))
    if bad:
        raise ValidationError(f"evidence cites non-retrieved doc_id(s): {bad}")
    return obj


def json_schema() -> dict:
    """The machine-readable JSON Schema for the assessment (for prompts + auditing)."""
    return CrashCauseAssessment.model_json_schema()


def _inline_refs(node, defs):
    """Recursively inline ``$ref`` → ``$defs`` so the schema is self-contained (no refs)."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(x, defs) for x in node]
    return node


def tool_schema() -> dict:
    """Dereferenced, self-contained JSON Schema for LLM tool-use (structured output).

    Tool-use is far more reliable than parsing free-text JSON, but nested ``$ref``/``$defs`` can
    make models skip fields — so we inline everything into one flat schema.
    """
    s = json_schema()
    return _inline_refs(s, s.get("$defs", {}))
