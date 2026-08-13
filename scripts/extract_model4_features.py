#!/usr/bin/env python3
"""STU-67 stage 1: extract crash-cause features for a bounded pilot of crash events.

Selects ~N large crashes from the CLEAN determined pool across the train and test splits,
retrieves each event's point-in-time contemporaneous 8-K (STU-65), runs the LLM extractor
(STU-66), and encodes the result into model-ready features (STU-67). Checkpoints to parquet every
few events and is resumable (skips already-processed event_ids). Records a failure reason per
event so bad retrieval vs bad extraction can be told apart.

Run (loads the key):
    set -a; . ./.env; set +a
    PYTHONPATH=src .venv/bin/python scripts/extract_model4_features.py
"""
from __future__ import annotations

import argparse
from datetime import date

import polars as pl

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.extraction.extract import AnthropicClient, extract_crash_cause
from crashback.extraction.features import assessment_to_features
from crashback.extraction.schema import ValidationError, tool_schema
from crashback.providers.documents import DocumentQuery, prediction_cutoff
from crashback.providers.edgar_provider import EDGARProvider

USER_AGENT = "crashback-research r42shah@gmail.com"
CHECKPOINT = 20


def _select(clean: pl.DataFrame, split: str, n: int, min_crash: float, seed: int) -> pl.DataFrame:
    pool = clean.filter(
        (pl.col("split") == split) & (pl.col("crash_return") <= min_crash)
        & pl.col("ticker_as_of_event").is_not_null())
    if pool.height <= n:
        return pool
    return pool.sample(n=n, seed=seed)


def _empty_row(ev, reason):
    return {"event_id": ev["event_id"], "security_id": ev["security_id"], "split": ev["split"],
            "ticker": ev["ticker_as_of_event"], "crash_date": ev["crash_date"],
            "cause_doc_id": None, "cause_available_at": None, "failure_reason": reason,
            "primary_cause": None, "event_type": None, "damage": None,
            **assessment_to_features(None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--n-train", type=int, default=280)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--min-crash", type=float, default=-0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    docs_dir = cfg.paths.resolve("data_normalized").parent / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / f"model4_cause_features_{args.version}.parquet"
    target = PRIMARY_TARGET

    events = pl.read_parquet(proc / "events_v1.parquet")
    splits = pl.read_parquet(proc / "splits_v1.parquet").select("event_id", "split")
    clean = events.join(splits, on="event_id", how="left").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
        & ~pl.col("censored_20d") & pl.col(target).is_not_null())

    pilot = pl.concat([
        _select(clean, "train", args.n_train, args.min_crash, args.seed),
        _select(clean, "test", args.n_test, args.min_crash, args.seed),
    ]).select("event_id", "security_id", "split", "ticker_as_of_event", "crash_date",
              "crash_return")

    done, rows = set(), []
    if out.exists():
        prev = pl.read_parquet(out)
        rows = prev.to_dicts()
        done = set(prev["event_id"].to_list())
    todo = pilot.filter(~pl.col("event_id").is_in(list(done)))
    print(f"pilot={pilot.height} (train {args.n_train}/test {args.n_test}), "
          f"done={len(done)}, todo={todo.height}")

    prov = EDGARProvider(USER_AGENT)
    client = AnthropicClient(model=args.model, tool_schema=tool_schema(),
                             tool_name="emit_crash_cause")

    for i, ev in enumerate(todo.iter_rows(named=True), 1):
        row = _process(prov, client, ev)
        rows.append(row)
        if i % 10 == 0 or i == todo.height:
            print(f"  [{i}/{todo.height}] {ev['ticker_as_of_event']:5s} {ev['crash_date']} "
                  f"-> {row['failure_reason']}/{row['primary_cause']}")
        if i % CHECKPOINT == 0:
            pl.DataFrame(rows).write_parquet(out)

    pl.DataFrame(rows).write_parquet(out)
    summary = pl.DataFrame(rows)["failure_reason"].value_counts().sort("failure_reason")
    print("\nfailure_reason breakdown:")
    print(summary)
    print(f"\nwrote {out} ({len(rows)} events)")


def _process(prov, client, ev) -> dict:
    try:
        as_of = prediction_cutoff(ev["crash_date"])
        q = DocumentQuery(as_of=as_of, ticker=ev["ticker_as_of_event"],
                          security_id=ev["security_id"], lookback_days=90)
        docs = prov.get_documents(q)
    except Exception:  # noqa: BLE001 - network/parse issue → record + continue
        return _empty_row(ev, "fetch_error")
    if docs.height == 0:
        return _empty_row(ev, "no_cik_or_docs")
    cause = docs.filter(pl.col("source_type") == "8-K").head(1)
    if cause.height == 0:
        return _empty_row(ev, "no_cause_8k")
    c = cause.row(0, named=True)
    text = prov.fetch_document_text(c["url"])
    if not text:
        return {**_empty_row(ev, "empty_text"), "cause_doc_id": c["doc_id"]}

    doc = {"doc_id": c["doc_id"], "source_type": c["source_type"],
           "available_at": str(c["available_at"]), "text": text}
    assessment = None
    for _ in range(2):   # one retry on the occasional malformed extraction
        try:
            ex = extract_crash_cause(
                client, event_id=ev["event_id"], ticker=ev["ticker_as_of_event"],
                crash_date=ev["crash_date"] if isinstance(ev["crash_date"], date)
                else date.fromisoformat(str(ev["crash_date"])),
                crash_return=ev["crash_return"], documents=[doc])
            assessment = ex.assessment
            break
        except ValidationError:
            continue
    if assessment is None:
        return {**_empty_row(ev, "extraction_failed"), "cause_doc_id": c["doc_id"]}

    return {"event_id": ev["event_id"], "security_id": ev["security_id"], "split": ev["split"],
            "ticker": ev["ticker_as_of_event"], "crash_date": ev["crash_date"],
            "cause_doc_id": c["doc_id"], "cause_available_at": str(c["available_at"]),
            "failure_reason": "ok", "primary_cause": assessment.primary_cause,
            "event_type": assessment.event_type, "damage": assessment.temporary_vs_structural,
            **assessment_to_features(assessment)}


if __name__ == "__main__":
    main()
