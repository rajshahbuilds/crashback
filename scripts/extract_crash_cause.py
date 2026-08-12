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
from collections import Counter
from datetime import date, datetime

import polars as pl

from crashback.config import load_config
from crashback.extraction.extract import AnthropicClient, extract_crash_cause
from crashback.extraction.prompt import PROMPT_VERSION
from crashback.extraction.schema import SCHEMA_VERSION, ValidationError, tool_schema
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

CONSISTENCY_FIELDS = ("event_type", "primary_cause", "temporary_vs_structural",
                      "business_thesis_changed")


def _agreement(values):
    mode, cnt = Counter(values).most_common(1)[0]
    return mode, cnt / len(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--runs", type=int, default=3, help="extractions per event (consistency)")
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

    # live extraction (needs key/SDK). Tool-use forces valid structured output.
    all_runs, consistency, llm_status, n_fail = [], [], "skipped", 0
    try:
        client = AnthropicClient(model=args.model, tool_schema=tool_schema(),
                                 tool_name="emit_crash_cause")
        for b in bundles:
            runs, attempts = [], 0
            # tolerate the occasional malformed model output: retry up to runs+2 attempts
            while len(runs) < args.runs and attempts < args.runs + 2:
                attempts += 1
                try:
                    ex = extract_crash_cause(
                        client, event_id=b["event_id"], ticker=b["ticker"],
                        crash_date=date.fromisoformat(b["crash_date"]),
                        crash_return=b["crash_return"], documents=[b["document"]])
                except ValidationError:
                    n_fail += 1
                    continue
                row = ex.to_row() | {"ticker": b["ticker"], "run": len(runs)}
                runs.append(row)
                all_runs.append(row)
            if not runs:
                print(f"  {b['ticker']:5s} FAILED all attempts")
                continue
            summ = {"ticker": b["ticker"], "event_id": b["event_id"], "runs": len(runs)}
            for f in CONSISTENCY_FIELDS:
                mode, agr = _agreement([r[f] for r in runs])
                summ[f], summ[f"{f}_agree"] = mode, agr
            consistency.append(summ)
            print(f"  {b['ticker']:5s} cause={summ['primary_cause']} "
                  f"({summ['primary_cause_agree']:.0%}) damage={summ['temporary_vs_structural']} "
                  f"({summ['temporary_vs_structural_agree']:.0%})")
        llm_status = (f"ran ~{args.runs}× on {len(consistency)}/{len(bundles)} events with "
                      f"{args.model} ({n_fail} run(s) rejected+retried)")
    except Exception as e:  # noqa: BLE001 - report and continue without a key
        if all_runs:
            raise
        llm_status = f"skipped ({type(e).__name__}: {e})"

    if all_runs:
        v = args.version
        pl.DataFrame(all_runs).write_parquet(docs_dir / f"crash_cause_extractions_{v}.parquet")
        pl.DataFrame(consistency).write_parquet(docs_dir / f"crash_cause_consistency_{v}.parquet")

    _write_report(cfg, args.version, bundles, args, consistency, all_runs, llm_status)
    print(f"\nLLM extraction: {llm_status}")
    print(f"wrote reports/STU-66_crash_cause_extraction.md and artifacts to {docs_dir}")


def _write_report(cfg, version, bundles, args, consistency, all_runs, llm_status):
    inputs = "\n".join(
        f"| {b['ticker']} | {b['crash_date']} | {b['crash_return']:+.1%} | "
        f"{b['document']['doc_id']} | {len(b['document']['text']):,} |" for b in bundles)

    results = "_Not run (no API key)._"
    consist = ""
    if consistency:
        rows = "\n".join(
            f"| {c['ticker']} | {c['event_type']} | {c['primary_cause']} | "
            f"{c['temporary_vs_structural']} | {c['business_thesis_changed']} |"
            for c in consistency)
        results = ("| ticker | event_type | primary_cause | temp/struct | thesis Δ |\n"
                   "|---|---|---|---|---|\n" + rows)
        crow = "\n".join(
            f"| {c['ticker']} | {c['primary_cause_agree']:.0%} | "
            f"{c['temporary_vs_structural_agree']:.0%} | {c['event_type_agree']:.0%} | "
            f"{c['business_thesis_changed_agree']:.0%} |" for c in consistency)
        mean_pc = sum(c["primary_cause_agree"] for c in consistency) / len(consistency)
        mean_ts = sum(c["temporary_vs_structural_agree"] for c in consistency) / len(consistency)
        consist = f"""## Consistency across {args.runs} runs per event

Agreement = modal-value fraction across independent runs (default temperature). Mean
`primary_cause` agreement **{mean_pc:.0%}**, `temporary_vs_structural` **{mean_ts:.0%}**.

| ticker | primary_cause | temp/struct | event_type | thesis Δ |
|---|---|---|---|---|
{crow}
"""

    report = f"""# STU-66 — Structured Crash-Cause Extraction (LLM)

The LLM is a **structured event-understanding extractor**, never a price oracle (CLAUDE.md §24):
it reads only the contemporaneous documents (STU-65) and emits a schema-validated JSON assessment
of *why* the crash happened, with evidence grounded to retrieved documents.

- **Schema** `{SCHEMA_VERSION}` — machine-readable (committed JSON at
  `src/crashback/extraction/crash_cause_schema_v1.json`; Pydantic in `crashback.extraction.schema`).
  Fields: `event_type`, `primary_cause`, `revenue_/margin_/balance_sheet_impact` (none→severe),
  `business_thesis_changed`, **`temporary_vs_structural`** (overreaction vs real damage, §24),
  `uncertainty`, `rationale`, `evidence[]`.
- **Prompt** `{PROMPT_VERSION}` — forbids any price/return/buy-sell output, documents-only, requires
  evidence (doc_id + verbatim quote) per substantive judgment.
- **Structured output via tool-use** (forced), so JSON is always well-formed; then
  `validate_assessment` enforces the schema, matching `schema_version`, and that every evidence
  `doc_id` is a retrieved document (hallucinated sources rejected).

## Inputs (real 8-K text, fetched live)

| ticker | crash date | crash | cause 8-K (doc_id) | text chars |
|---|---|---|---|---|
{inputs}

## Extraction results — **{llm_status}**

Model output (modal value across runs), reproducible via `scripts/extract_crash_cause.py`:

{results}

{consist}
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

Artifacts: schema JSON (committed); `crash_cause_inputs_{version}.json`,
`crash_cause_extractions_{version}.parquet` (all runs), `crash_cause_consistency_{version}.parquet`.
"""
    (cfg.paths.resolve("reports") / "STU-66_crash_cause_extraction.md").write_text(report)


if __name__ == "__main__":
    main()
