# STU-66 — Structured Crash-Cause Extraction (LLM)

The LLM is used as a **structured event-understanding extractor**, never a price oracle
(CLAUDE.md §24): it reads only the contemporaneous documents (STU-65) and emits a schema-validated
JSON assessment of *why* the crash happened, with evidence grounded to retrieved documents.

- **Schema:** `crash_cause.v1` — machine-readable JSON Schema committed at
  `src/crashback/extraction/crash_cause_schema_v1.json`; Pydantic models in
  `crashback.extraction.schema`.
- **Prompt:** `crash_cause_prompt.2026-08-12.v1` (`crashback.extraction.prompt`) — forbids any price/return/
  buy-sell output, restricts to the supplied documents, and requires evidence (doc_id + verbatim
  quote) for every substantive judgment; `primary_cause` and `temporary_vs_structural` must each
  be cited.
- **Validation before use:** `validate_assessment` enforces the schema, a matching
  `schema_version`, and that **every evidence `doc_id` is a retrieved document** — hallucinated
  sources are rejected, never stored.

## Schema fields

`event_type`, `primary_cause`, `revenue_impact` / `margin_impact` / `balance_sheet_impact`
(none→severe), `business_thesis_changed` (bool), **`temporary_vs_structural`** (temporary / mixed
/ structural / unclear — the core "overreaction vs real damage" judgment, §24), `uncertainty`,
`rationale`, and `evidence[]`.

## Extraction inputs (real filing text, fetched live)

Nearest pre-crash 8-K per crash (the likely cause filing), text fetched from EDGAR:

| ticker | crash date | crash | cause 8-K (doc_id) | text chars |
|---|---|---|---|---|
| NFLX | 2022-04-20 | -35.1% | 0001065280-22-000144 | 28,784 |
| META | 2022-10-27 | -24.6% | 0001326801-22-000105 | 40,000 |
| DOCU | 2022-06-10 | -24.5% | 0001261333-22-000103 | 29,531 |
| ROKU | 2024-02-16 | -23.8% | 0001428439-24-000007 | 40,000 |
| PYPL | 2022-02-02 | -24.6% | 0001633917-22-000021 | 40,000 |
| SNAP | 2022-05-24 | -43.1% | 0001193125-22-157565 | 40,000 |

Input bundles saved to `data/documents/crash_cause_inputs_v1.json` (ready for a live run).

## LLM run

**Status: skipped (RuntimeError: anthropic SDK not installed (pip install anthropic)).** The pipeline is complete and unit-tested; the live model call needs an
`ANTHROPIC_API_KEY` (and the `anthropic` SDK), absent in this environment. When set, rerun this
script to populate `data/documents/crash_cause_extractions_v1.parquet`.

_Not run in this environment (no API key)._

## Human reference extraction (illustrative, schema-validated)

Authored from the NFLX 8-K to show expected output and to prove the schema captures a real event
(**not** model output). It validates against `crash_cause.v1` and grounds to the real 8-K doc_id:

```json
{
  "schema_version": "crash_cause.v1",
  "event_type": "earnings",
  "primary_cause": "customer_or_subscriber_loss",
  "revenue_impact": "moderate",
  "margin_impact": "low",
  "balance_sheet_impact": "none",
  "business_thesis_changed": true,
  "temporary_vs_structural": "mixed",
  "uncertainty": "medium",
  "rationale": "Q1 2022 reported a net loss of subscribers with soft Q2 guidance \u2014 a demand/competitive signal touching the growth thesis, though the core service and balance sheet are intact. Human reference, authored from the 8-K exhibit.",
  "evidence": [
    {
      "supports": "primary_cause",
      "doc_id": "0001065280-22-000144",
      "quote": "loss of subscribers reported for the first quarter"
    },
    {
      "supports": "temporary_vs_structural",
      "doc_id": "0001065280-22-000144",
      "quote": "guidance and commentary on competition and account sharing"
    }
  ]
}
```

## Manual-review methodology & failure modes

The intended review: run each event 3× (temperature 0 + 2 higher) and check **consistency** of
`primary_cause` and `temporary_vs_structural`; spot-check that quotes are verbatim and on-point.
Anticipated / guarded failure modes:

- **Fabricated sources** → rejected by construction (evidence `doc_id` must be retrieved).
- **Non-JSON / fenced output** → tolerated (fence-stripping) or rejected cleanly.
- **Missing grounding** on a core field → rejected (required-evidence validator).
- **Over-reading "structural"** from negative tone, or mislabeling a **macro/sector** sympathy
  move as company-specific — the prompt mitigates via the documents-only rule and an explicit
  `macro_or_sector_selloff` cause + `unclear` option; these are the qualitative checks the manual
  review targets once a key is available.
- **Price leakage** in reasoning → prohibited in the prompt (tested) and excluded from the schema.

Artifacts: schema JSON (committed), `crash_cause_inputs_v1.json`, and (when run)
`crash_cause_extractions_v1.parquet`.
