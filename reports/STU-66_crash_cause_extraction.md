# STU-66 — Structured Crash-Cause Extraction (LLM)

The LLM is a **structured event-understanding extractor**, never a price oracle (CLAUDE.md §24):
it reads only the contemporaneous documents (STU-65) and emits a schema-validated JSON assessment
of *why* the crash happened, with evidence grounded to retrieved documents.

- **Schema** `crash_cause.v1` — machine-readable (committed JSON at
  `src/crashback/extraction/crash_cause_schema_v1.json`; Pydantic in `crashback.extraction.schema`).
  Fields: `event_type`, `primary_cause`, `revenue_/margin_/balance_sheet_impact` (none→severe),
  `business_thesis_changed`, **`temporary_vs_structural`** (overreaction vs real damage, §24),
  `uncertainty`, `rationale`, `evidence[]`.
- **Prompt** `crash_cause_prompt.2026-08-12.v1` — forbids any price/return/buy-sell output, documents-only, requires
  evidence (doc_id + verbatim quote) per substantive judgment.
- **Structured output via tool-use** (forced), so JSON is always well-formed; then
  `validate_assessment` enforces the schema, matching `schema_version`, and that every evidence
  `doc_id` is a retrieved document (hallucinated sources rejected).

## Inputs (real 8-K text, fetched live)

| ticker | crash date | crash | cause 8-K (doc_id) | text chars |
|---|---|---|---|---|
| SNAP | 2022-05-24 | -43.1% | 0001193125-22-157565 | 40,000 |
| DOCU | 2022-06-10 | -24.5% | 0001261333-22-000103 | 29,531 |
| NFLX | 2022-04-20 | -35.1% | 0001065280-22-000144 | 28,784 |
| ROKU | 2024-02-16 | -23.8% | 0001428439-24-000007 | 40,000 |
| META | 2022-10-27 | -24.6% | 0001326801-22-000105 | 40,000 |
| PYPL | 2022-02-02 | -24.6% | 0001633917-22-000021 | 40,000 |

## Extraction results — **ran ~3× on 6/6 events with claude-sonnet-5 (2 run(s) rejected+retried)**

Model output (modal value across runs), reproducible via `scripts/extract_crash_cause.py`:

| ticker | event_type | primary_cause | temp/struct | thesis Δ |
|---|---|---|---|---|
| SNAP | guidance | guidance_cut | mixed | False |
| DOCU | earnings | demand_weakness | mixed | True |
| NFLX | earnings | customer_or_subscriber_loss | mixed | True |
| ROKU | guidance | guidance_cut | mixed | False |
| META | earnings | margin_compression | mixed | True |
| PYPL | guidance | guidance_cut | mixed | True |

## Consistency across 3 runs per event

Agreement = modal-value fraction across independent runs (default temperature). Mean
`primary_cause` agreement **100%**, `temporary_vs_structural` **94%**.

| ticker | primary_cause | temp/struct | event_type | thesis Δ |
|---|---|---|---|---|
| SNAP | 100% | 100% | 100% | 100% |
| DOCU | 100% | 100% | 67% | 100% |
| NFLX | 100% | 100% | 100% | 100% |
| ROKU | 100% | 67% | 100% | 100% |
| META | 100% | 100% | 100% | 100% |
| PYPL | 100% | 100% | 100% | 100% |

## Review & failure modes

- **Grounding works:** quotes are verbatim from the filings (spot-checked), and the doc_id filter
  makes fabricated sources impossible by construction.
- **The `temporary_vs_structural` judgment is the V2 signal** STU-67 will test: whether this
  read adds recovery-prediction value beyond price/context/fundamentals.
- Guarded failure modes (unit-tested): fabricated source → rejected; malformed/non-JSON →
  rejected (tool-use largely eliminates this); missing grounding on a core field → rejected;
  unknown enum value → rejected. Residual risk to watch in review: over-reading "structural" from
  negative tone, or labeling a macro/sector sympathy move as company-specific (mitigated by the
  `macro_or_sector_selloff` cause + `unclear` option).

Artifacts: schema JSON (committed); `crash_cause_inputs_v1.json`,
`crash_cause_extractions_v1.parquet` (all runs), `crash_cause_consistency_v1.parquet`.
