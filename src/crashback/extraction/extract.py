"""Crash-cause extraction harness: prompt → LLM client → schema-validated assessment (STU-66).

The LLM call is behind an injectable ``LLMClient`` interface, so the parse+validate pipeline is
unit-tested with a stub (no network, no API key) and a real ``AnthropicClient`` can be dropped in
when a key is available. Extraction always runs the raw model output through
``schema.validate_assessment`` (schema + evidence grounding) before returning — invalid or
hallucinated-source output is rejected, never used.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from crashback.extraction.prompt import PROMPT_VERSION, build_messages
from crashback.extraction.schema import (
    SCHEMA_VERSION,
    CrashCauseAssessment,
    ValidationError,
    validate_assessment,
)


class LLMClient(ABC):
    """Minimal completion interface: (system, user) -> raw text (expected to be JSON)."""

    model: str = "unknown"

    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient(LLMClient):
    """Real client (guarded): needs the ``anthropic`` SDK and ``ANTHROPIC_API_KEY``.

    Default model is Sonnet — a capable, cost-reasonable choice for structured extraction. Kept
    import-guarded so the rest of the pipeline imports and tests without the SDK/key present.
    """

    def __init__(self, model: str = "claude-sonnet-5", *, max_tokens: int = 1500,
                 api_key: str | None = None, tool_schema: dict | None = None,
                 tool_name: str = "emit_assessment"):
        try:
            import anthropic  # noqa: PLC0415 - optional dependency, imported on use
        except ImportError as e:  # pragma: no cover - depends on env
            raise RuntimeError("anthropic SDK not installed (pip install anthropic)") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.model = model
        self.max_tokens = max_tokens
        # tool_schema forces structured (tool-use) output → the SDK returns already-parsed JSON,
        # eliminating free-text JSON parse failures (e.g. unescaped quotes in evidence excerpts).
        self._tool_schema = tool_schema
        self._tool_name = tool_name
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - needs live API
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]}
        if self._tool_schema is not None:
            kwargs["tools"] = [{"name": self._tool_name,
                                "description": "Emit the structured crash-cause assessment.",
                                "input_schema": self._tool_schema}]
            kwargs["tool_choice"] = {"type": "tool", "name": self._tool_name}
        msg = self._client.messages.create(**kwargs)
        for b in msg.content:
            if getattr(b, "type", None) == "tool_use":
                return json.dumps(b.input)
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _strip_json(text: str) -> str:
    """Extract the JSON object from a model reply (tolerates ```json fences / stray prose)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValidationError("no JSON object found in model output")
    return t[start:end + 1]


@dataclass(frozen=True)
class Extraction:
    """A validated assessment plus its full provenance (versions, model, event key)."""

    event_id: str
    assessment: CrashCauseAssessment
    schema_version: str
    prompt_version: str
    model: str
    doc_ids: tuple[str, ...]

    def to_row(self) -> dict:
        a = self.assessment
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "event_type": a.event_type,
            "primary_cause": a.primary_cause,
            "revenue_impact": a.revenue_impact,
            "margin_impact": a.margin_impact,
            "balance_sheet_impact": a.balance_sheet_impact,
            "business_thesis_changed": a.business_thesis_changed,
            "temporary_vs_structural": a.temporary_vs_structural,
            "uncertainty": a.uncertainty,
            "n_evidence": len(a.evidence),
            "cited_doc_ids": ",".join(sorted({e.doc_id for e in a.evidence})),
        }


def extract_crash_cause(
    client: LLMClient, *, event_id: str, ticker: str, crash_date: date, crash_return: float,
    documents: list[dict],
) -> Extraction:
    """Run one extraction: build prompt → call client → parse → schema+evidence validate."""
    if not documents:
        raise ValidationError("no contemporaneous documents to extract from")
    system, user = build_messages(
        ticker=ticker, crash_date=crash_date, crash_return=crash_return, documents=documents)
    raw_text = client.complete(system, user)
    raw = json.loads(_strip_json(raw_text))
    raw.setdefault("schema_version", SCHEMA_VERSION)
    allowed = {d["doc_id"] for d in documents}
    assessment = validate_assessment(raw, allowed)
    return Extraction(
        event_id=event_id, assessment=assessment, schema_version=SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION, model=client.model,
        doc_ids=tuple(d["doc_id"] for d in documents))
