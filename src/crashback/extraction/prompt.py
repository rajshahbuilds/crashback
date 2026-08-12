"""Versioned, reproducible extraction prompt for crash-cause understanding (STU-66).

The prompt (a) forbids any price/return/recommendation output — the LLM explains *what happened*,
never *what to do* (CLAUDE.md §24); (b) requires the model to read ONLY the supplied
contemporaneous documents; and (c) requires an evidence citation (doc_id + verbatim quote) for
every substantive judgment. ``PROMPT_VERSION`` is stored with each extraction so results are
reproducible and comparable; bump it on any wording change.
"""
from __future__ import annotations

import json
from datetime import date

from crashback.extraction.schema import SCHEMA_VERSION, json_schema

PROMPT_VERSION = "crash_cause_prompt.2026-08-12.v1"

SYSTEM = f"""You are a financial-filing analyst acting as a STRUCTURED EVENT EXTRACTOR.
You explain WHY a stock crashed based only on contemporaneous documents. You output a single
JSON object conforming to the crash-cause schema ({SCHEMA_VERSION}).

HARD RULES:
- Do NOT predict future price, returns, recovery, or direction. Do NOT give buy/sell/hold or any
  investment recommendation. You assess the *nature and cause* of the event only.
- Use ONLY the provided documents. Do not use outside or after-the-fact knowledge. If the
  documents are insufficient, say so via higher `uncertainty` and `unclear`/`other` values.
- Ground every substantive judgment in EVIDENCE: each `evidence` item gives the `doc_id`, a short
  verbatim `quote` from that document, and the field it `supports`. `primary_cause` and
  `temporary_vs_structural` MUST each be supported by at least one evidence item.
- `temporary_vs_structural` is your read of whether the new information implies transient
  (overreaction) versus genuine long-term fundamental damage — an assessment of the *information*,
  NOT a market/price forecast.
- Output ONLY the JSON object. No prose before or after, no markdown fences.
"""


def _doc_block(documents: list[dict]) -> str:
    parts = []
    for d in documents:
        parts.append(
            f"[doc_id: {d['doc_id']}] source={d.get('source', 'sec_edgar')} "
            f"type={d.get('source_type')} available_at={d.get('available_at')}\n"
            f"{d.get('text', '').strip()}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    *, ticker: str, crash_date: date, crash_return: float, documents: list[dict],
) -> tuple[str, str]:
    """Return (system, user) message strings for the extraction call.

    ``documents`` are the point-in-time-admissible docs (STU-65), each a dict with doc_id,
    source_type, available_at, and text. The crash magnitude is contemporaneous context; no future
    information is included.
    """
    schema_json = json.dumps(json_schema(), indent=2)
    allowed = ", ".join(d["doc_id"] for d in documents) or "(none)"
    user = f"""EVENT: {ticker} fell {crash_return:.1%} on {crash_date} (crash prediction timestamp
= that day's close). Explain the cause using only the documents below.

Valid doc_ids you may cite (evidence must reference one of these): {allowed}

JSON SCHEMA (output must validate against this):
{schema_json}

CONTEMPORANEOUS DOCUMENTS:
{_doc_block(documents)}

Return ONLY the JSON object for schema_version "{SCHEMA_VERSION}"."""
    return SYSTEM, user


def contains_price_prohibition(system: str = SYSTEM) -> bool:
    """True iff the prompt explicitly forbids price/recommendation output (guard for tests)."""
    s = system.lower()
    return ("do not predict future price" in s) and ("buy/sell" in s)
