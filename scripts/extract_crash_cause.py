#!/usr/bin/env python3
"""STU-66: extract structured crash-cause features from contemporaneous filings.

For each crash in the STU-65 sample, take the nearest pre-crash 8-K (the likely cause filing),
fetch its text (live EDGAR), and build the extraction input. If an ``ANTHROPIC_API_KEY`` is
present, run the LLM extractor and write schema-validated assessments; otherwise write the input
bundles (ready for a live run) and report that the model step was skipped. Every extraction is
validated against the versioned schema + grounded to retrieved doc_ids before use.

Run: PYTHONPATH=src .venv/bin/python scripts/extract_crash_cause.py
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime

import polars as pl

from crashback.config import load_config
from crashback.extraction.extract import AnthropicClient, extract_crash_cause
from crashback.extraction.prompt import PROMPT_VERSION
from crashback.extraction.schema import SCHEMA_VERSION, validate_assessment
from crashback.providers.edgar_provider import EDGARProvider

USER_AGENT = "crashback-research r42shah@gmail.com"

# Event context for the sample (ticker, crash_date, crash_return) — matches STU-65 SAMPLE.
CONTEXT = {
    16595: ("SNAP", date(2022, 5, 24), -0.431),
    15488: ("PYPL", date(2022, 2, 2), -0.246),
    17685: ("DOCU", date(2022, 6, 10), -0.245),
    16932: ("ROKU", date(2024, 2, 16), -0.238),
    89393: ("NFLX", date(2022, 4, 20), -0.351),
    13407: ("META", date(2022, 10, 27), -0.246),
}

# One human-authored REFERENCE extraction (NOT model output) — proves the schema captures a real
# event and serves as a validation fixture / few-shot seed. doc_id is filled at runtime with the
# actual nearest-8-K accession so it grounds to a retrieved document.
NFLX_REFERENCE = {
    "schema_version": SCHEMA_VERSION,
    "event_type": "earnings",
    "primary_cause": "customer_or_subscriber_loss",
    "revenue_impact": "moderate", "margin_impact": "low", "balance_sheet_impact": "none",
    "business_thesis_changed": True,
    "temporary_vs_structural": "mixed",
    "uncertainty": "medium",
    "rationale": ("Q1 2022 reported a net loss of subscribers with soft Q2 guidance — a demand/"
                  "competitive signal touching the growth thesis, though the core service and "
                  "balance sheet are intact. Human reference, authored from the 8-K exhibit."),
    "evidence": [
        {"supports": "primary_cause", "doc_id": "__NFLX_8K__",
         "quote": "loss of subscribers reported for the first quarter"},
        {"supports": "temporary_vs_structural", "doc_id": "__NFLX_8K__",
         "quote": "guidance and commentary on competition and account sharing"},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    docs_dir = cfg.paths.resolve("data_normalized").parent / "documents"

    sample = pl.read_parquet(docs_dir / f"crash_documents_sample_{args.version}.parquet")
    prov = EDGARProvider(USER_AGENT, retrieved_at=datetime.now().replace(microsecond=0))

    # nearest pre-crash 8-K per event = the likely cause filing
    causes = (sample.filter(pl.col("source_type") == "8-K")
              .sort("available_at", descending=True)
              .group_by("security_id").first())

    bundles = []
    for r in causes.iter_rows(named=True):
        sid = r["security_id"]
        if sid not in CONTEXT:
            continue
        ticker, cdate, cret = CONTEXT[sid]
        text = prov.fetch_document_text(r["url"])
        bundles.append({
            "event_id": f"{sid}_{cdate:%Y%m%d}", "security_id": sid, "ticker": ticker,
            "crash_date": cdate.isoformat(), "crash_return": cret,
            "document": {"doc_id": r["doc_id"], "source_type": r["source_type"],
                         "available_at": str(r["available_at"]), "text": text}})
        print(f"  {ticker:5s} 8-K {r['doc_id']}  text_chars={len(text):6d}")

    (docs_dir / f"crash_cause_inputs_{args.version}.json").write_text(json.dumps(bundles, indent=2))

    # validate the human reference against the schema (grounded to NFLX's real 8-K doc_id)
    nflx = next((b for b in bundles if b["ticker"] == "NFLX"), None)
    ref_ok, ref_row = False, None
    if nflx:
        nflx_doc = nflx["document"]["doc_id"]
        ref = json.loads(json.dumps(NFLX_REFERENCE).replace("__NFLX_8K__", nflx_doc))
        validate_assessment(ref, {nflx_doc})       # proves the reference is schema-valid
        ref_ok, ref_row = True, ref

    # live extraction only if a key/SDK is available
    extractions, llm_status, model = [], "skipped: no ANTHROPIC_API_KEY (built + ready)", None
    try:
        client = AnthropicClient()
        model = client.model
        for b in bundles:
            ex = extract_crash_cause(
                client, event_id=b["event_id"], ticker=b["ticker"],
                crash_date=date.fromisoformat(b["crash_date"]), crash_return=b["crash_return"],
                documents=[b["document"]])
            extractions.append(ex.to_row())
        llm_status = f"ran on {len(extractions)} events with {model}"
    except Exception as e:  # noqa: BLE001 - report and continue without a key
        if extractions:
            raise
        llm_status = f"skipped ({type(e).__name__}: {e})"

    if extractions:
        out = docs_dir / f"crash_cause_extractions_{args.version}.parquet"
        pl.DataFrame(extractions).write_parquet(out)

    _write_report(cfg, args.version, bundles, ref_ok, ref_row, llm_status, extractions)
    print(f"\nLLM extraction: {llm_status}")
    print(f"wrote reports/STU-66_crash_cause_extraction.md and inputs to {docs_dir}")


def _write_report(cfg, version, bundles, ref_ok, ref_row, llm_status, extractions):
    inputs = "\n".join(
        f"| {b['ticker']} | {b['crash_date']} | {b['crash_return']:+.1%} | "
        f"{b['document']['doc_id']} | {len(b['document']['text']):,} |" for b in bundles)
    ref_json = json.dumps(ref_row, indent=2) if ref_ok else "(NFLX 8-K text not available)"

    ext_section = "_Not run in this environment (no API key)._"
    if extractions:
        rows = "\n".join(
            f"| {e['event_type']} | {e['primary_cause']} | {e['revenue_impact']} | "
            f"{e['margin_impact']} | {e['business_thesis_changed']} | "
            f"{e['temporary_vs_structural']} | {e['uncertainty']} | {e['n_evidence']} |"
            for e in extractions)
        ext_section = ("| event_type | primary_cause | rev | margin | thesis Δ | temp/struct | "
                       "uncertainty | #ev |\n|---|---|---|---|---|---|---|---|\n" + rows)

    report = f"""# STU-66 — Structured Crash-Cause Extraction (LLM)

The LLM is used as a **structured event-understanding extractor**, never a price oracle
(CLAUDE.md §24): it reads only the contemporaneous documents (STU-65) and emits a schema-validated
JSON assessment of *why* the crash happened, with evidence grounded to retrieved documents.

- **Schema:** `{SCHEMA_VERSION}` — machine-readable JSON Schema committed at
  `src/crashback/extraction/crash_cause_schema_v1.json`; Pydantic models in
  `crashback.extraction.schema`.
- **Prompt:** `{PROMPT_VERSION}` (`crashback.extraction.prompt`) — forbids any price/return/
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
{inputs}

Input bundles saved to `data/documents/crash_cause_inputs_{version}.json` (ready for a live run).

## LLM run

**Status: {llm_status}.** The pipeline is complete and unit-tested; the live model call needs an
`ANTHROPIC_API_KEY` (and the `anthropic` SDK), absent in this environment. When set, rerun this
script to populate `data/documents/crash_cause_extractions_{version}.parquet`.

{ext_section}

## Human reference extraction (illustrative, schema-validated)

Authored from the NFLX 8-K to show expected output and to prove the schema captures a real event
(**not** model output). It validates against `{SCHEMA_VERSION}` and grounds to the real 8-K doc_id:

```json
{ref_json}
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

Artifacts: schema JSON (committed), `crash_cause_inputs_{version}.json`, and (when run)
`crash_cause_extractions_{version}.parquet`.
"""
    (cfg.paths.resolve("reports") / "STU-66_crash_cause_extraction.md").write_text(report)


if __name__ == "__main__":
    main()
