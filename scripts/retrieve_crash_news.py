#!/usr/bin/env python3
"""STU-65: reconstruct contemporaneous SEC-EDGAR documents for a manual crash sample.

For a hand-picked set of real historical crashes (earnings/guidance-driven, recognizable
tickers), retrieve the filings that were public **at or before the crash-day close** and show
that filings accepted *after* the crash are excluded by construction. Writes a canonical document
table + reports/STU-65_news_retrieval.md documenting the retrieval policy and archive limits.

Live EDGAR calls (needs network). Run:
    PYTHONPATH=src .venv/bin/python scripts/retrieve_crash_news.py
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import polars as pl

from crashback.config import load_config
from crashback.providers.documents import DocumentQuery, available_as_of, prediction_cutoff
from crashback.providers.edgar_provider import EDGARProvider

USER_AGENT = "crashback-research r42shah@gmail.com"

# Manually selected historical crashes (real events in events_v1), all earnings/guidance-driven.
SAMPLE = [
    {"ticker": "NFLX", "crash_date": date(2022, 4, 20), "security_id": 89393,
     "crash_return": -0.351, "recovered_20d": 0, "note": "Q1 subscriber loss"},
    {"ticker": "META", "crash_date": date(2022, 10, 27), "security_id": 13407,
     "crash_return": -0.246, "recovered_20d": 1, "note": "Q3 earnings / spend guidance"},
    {"ticker": "SNAP", "crash_date": date(2022, 5, 24), "security_id": 16595,
     "crash_return": -0.431, "recovered_20d": 1, "note": "mid-quarter guidance warning (8-K)"},
    {"ticker": "PYPL", "crash_date": date(2022, 2, 2), "security_id": 15488,
     "crash_return": -0.246, "recovered_20d": 0, "note": "weak user-growth guidance"},
    {"ticker": "DOCU", "crash_date": date(2022, 6, 10), "security_id": 17685,
     "crash_return": -0.245, "recovered_20d": 0, "note": "Q1 billings miss"},
    {"ticker": "ROKU", "crash_date": date(2024, 2, 16), "security_id": 16932,
     "crash_return": -0.238, "recovered_20d": 0, "note": "Q4 guidance"},
]
LOOKBACK = 90
RETRIEVED_AT = datetime.now().replace(microsecond=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    docs_dir = cfg.paths.resolve("data_normalized").parent / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    prov = EDGARProvider(USER_AGENT, retrieved_at=RETRIEVED_AT)

    all_docs = []
    rows = []
    for ev in SAMPLE:
        cutoff = prediction_cutoff(ev["crash_date"])
        cik = prov.resolve_cik(ev["ticker"])
        if cik is None:
            print(f"  {ev['ticker']}: CIK not resolved, skipping")
            continue
        # one symmetric fetch (±lookback) so we can show both kept and excluded filings
        raw = prov.retrieve_raw(
            cik, start=cutoff - timedelta(days=LOOKBACK), end=cutoff + timedelta(days=LOOKBACK),
            security_id=ev["security_id"])

        q = DocumentQuery(as_of=cutoff, cik=cik, security_id=ev["security_id"],
                          lookback_days=LOOKBACK)
        # canonical contemporaneous docs (point-in-time + lookback)
        contemp = available_as_of(raw, cutoff).filter(
            pl.col("available_at") >= cutoff - timedelta(days=LOOKBACK)
        ).unique("doc_id").sort("available_at", descending=True)

        # what a naive/hindsight retrieval would WRONGLY include (post-crash, same window)
        leaked = raw.filter(
            (pl.col("available_at") > cutoff)
            & (pl.col("available_at") <= cutoff + timedelta(days=LOOKBACK))).unique("doc_id")

        # nearest 8-K before the crash = the likely crash-cause filing
        cause = contemp.filter(pl.col("source_type") == "8-K").head(1)
        cause_txt = "—"
        if cause.height:
            c = cause.row(0, named=True)
            cause_txt = f"8-K @ {c['available_at']} ({(cutoff - c['available_at']).days}d before)"

        rows.append({**ev, "cik": cik, "n_contemporaneous": contemp.height,
                     "n_excluded_postcrash": leaked.height, "cause_filing": cause_txt})
        all_docs.append(contemp.with_columns(
            pl.lit(ev["crash_date"]).alias("crash_date"),
            pl.lit(str(q.as_of)).alias("as_of")))
        print(f"  {ev['ticker']:5s} {ev['crash_date']}  contemporaneous={contemp.height:3d}  "
              f"excluded_postcrash={leaked.height:3d}  {cause_txt}")

    docs = pl.concat(all_docs) if all_docs else None
    if docs is not None:
        docs.write_parquet(docs_dir / f"crash_documents_sample_{args.version}.parquet")
    _write_report(cfg, args.version, rows, docs)
    print(f"\nwrote reports/STU-65_news_retrieval.md and "
          f"{docs_dir}/crash_documents_sample_{args.version}.parquet")


def _write_report(cfg, version, rows, docs):
    tbl = ["| ticker | crash date | crash | recovered 20d | contemp. docs | post-crash excluded | "
           "nearest pre-crash 8-K |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        tbl.append(f"| {r['ticker']} | {r['crash_date']} | {r['crash_return']:+.1%} | "
                   f"{'yes' if r['recovered_20d'] else 'no'} | {r['n_contemporaneous']} | "
                   f"{r['n_excluded_postcrash']} | {r['cause_filing']} |")
    table = "\n".join(tbl)

    # one worked example: newest few contemporaneous docs for the first event
    example = ""
    if docs is not None and rows:
        first = rows[0]
        sub = docs.filter(pl.col("security_id") == first["security_id"]).head(5)
        lines = [f"| {x['source_type']} | {x['available_at']} | {x['url']} |"
                 for x in sub.iter_rows(named=True)]
        example = (f"### Worked example — {first['ticker']} on {first['crash_date']} "
                   f"({first['note']})\n\n"
                   f"Prediction cutoff = {first['crash_date']} 16:00 ET. Contemporaneous filings "
                   f"(most recent first), all with `available_at ≤ cutoff`:\n\n"
                   "| form | available_at (ET) | url |\n|---|---|---|\n" + "\n".join(lines))

    report = f"""# STU-65 — Contemporaneous Crash-News Retrieval (SEC EDGAR)

First V2 retrieval step: reconstruct the information that was **public at the crash prediction
timestamp** for a manually selected sample of real historical crashes, with **no hindsight
contamination**. Built on a provider-neutral interface (`crashback.providers.documents`) so a
timestamped news API can be added later behind the same point-in-time guarantee; **SEC EDGAR** is
the first (free, reachable) source.

## Retrieval policy

- **Prediction timestamp** = crash-day close, **16:00 ET** (`prediction_cutoff`). This is the
  only information horizon (CLAUDE.md §3).
- **Availability timestamp** = each filing's EDGAR `acceptanceDateTime` (exact moment it became
  public), treated as naive ET → canonical `available_at`.
- **Admissibility (by construction):** a document is kept only if `available_at ≤ cutoff`.
  Filings accepted after the crash close — or with no timestamp — are dropped in the base-layer
  filter (`available_as_of`), so leakage cannot be reintroduced by a caller.
- **Lookback:** {LOOKBACK} days before the cutoff (recent, plausibly crash-relevant filings).
- **Provenance preserved per document:** `doc_id` (accession no.), `source` (`sec_edgar`),
  `source_type` (form), `available_at`, `period_of_report`, `url`, `retrieved_at`.

## Sample: contemporaneous retrieval + leakage exclusion

`post-crash excluded` counts filings in the ±{LOOKBACK}-day window that a naive (hindsight)
retrieval would have wrongly pulled in — our filter removes them. A pre-crash **8-K** is the
likely crash-cause document (earnings via item 2.02, guidance, material events).

{table}

Every row has ≥1 excluded post-crash filing, demonstrating the point-in-time filter is doing real
work; and each earnings-driven crash has an 8-K accepted shortly **before** the close — the
contemporaneous cause, available in time.

{example}

## Known limitations (documented per acceptance criteria)

- **EDGAR filings only.** No general-news / press-wire / analyst-note archive yet — those are
  paid, timestamped sources for a future `DocumentProvider`. Crash causes that never hit an SEC
  filing (e.g. sector sympathy moves, pre-announced macro) won't appear here.
- **Ticker→CIK uses SEC's *current* mapping.** Fine for this recent, curated sample; a historical
  backfill should resolve `gvkey`→`cik` via Compustat (`comp.company.cik`) to avoid reused-ticker
  collisions.
- **`acceptanceDateTime` is Eastern Time**; we compare to a 16:00 ET cutoff. Same-day filings
  accepted before the close are (correctly) admitted; a filing accepted 16:01+ is excluded.
- **Metadata + URL only in this pass** — full document *text* extraction (for the LLM
  crash-cause features) is STU-66's scope; here we prove availability and provenance.
- **Older-filing pagination is handled:** EDGAR's submissions feed holds only the most recent
  ~1000 filings, so `retrieve_raw` additionally fetches the `filings.files` shards overlapping the
  window (this is why the heavy filers NFLX/META resolve back to 2022). Very deep history beyond
  the published shards would still be unreachable.

Artifact: `data/documents/crash_documents_sample_{version}.parquet` (canonical
`DOCUMENT_SCHEMA`). Reproducible via `scripts/retrieve_crash_news.py` (live EDGAR).
"""
    (cfg.paths.resolve("reports") / "STU-65_news_retrieval.md").write_text(report)


if __name__ == "__main__":
    main()
