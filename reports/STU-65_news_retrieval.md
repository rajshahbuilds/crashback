# STU-65 — Contemporaneous Crash-News Retrieval (SEC EDGAR)

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
- **Lookback:** 90 days before the cutoff (recent, plausibly crash-relevant filings).
- **Provenance preserved per document:** `doc_id` (accession no.), `source` (`sec_edgar`),
  `source_type` (form), `available_at`, `period_of_report`, `url`, `retrieved_at`.

## Sample: contemporaneous retrieval + leakage exclusion

`post-crash excluded` counts filings in the ±90-day window that a naive (hindsight)
retrieval would have wrongly pulled in — our filter removes them. A pre-crash **8-K** is the
likely crash-cause document (earnings via item 2.02, guidance, material events).

| ticker | crash date | crash | recovered 20d | contemp. docs | post-crash excluded | nearest pre-crash 8-K |
|---|---|---|---|---|---|---|
| NFLX | 2022-04-20 | -35.1% | no | 62 | 63 | 8-K @ 2022-04-19 20:03:19 (0d before) |
| META | 2022-10-27 | -24.6% | yes | 36 | 38 | 8-K @ 2022-10-26 20:19:09 (0d before) |
| SNAP | 2022-05-24 | -43.1% | yes | 23 | 26 | 8-K @ 2022-05-23 21:06:05 (0d before) |
| PYPL | 2022-02-02 | -24.6% | no | 21 | 36 | 8-K @ 2022-02-01 21:28:10 (0d before) |
| DOCU | 2022-06-10 | -24.5% | no | 40 | 27 | 8-K @ 2022-06-09 20:07:53 (0d before) |
| ROKU | 2024-02-16 | -23.8% | no | 26 | 35 | 8-K @ 2024-02-15 21:08:40 (0d before) |

Every row has ≥1 excluded post-crash filing, demonstrating the point-in-time filter is doing real
work; and each earnings-driven crash has an 8-K accepted shortly **before** the close — the
contemporaneous cause, available in time.

### Worked example — NFLX on 2022-04-20 (Q1 subscriber loss)

Prediction cutoff = 2022-04-20 16:00 ET. Contemporaneous filings (most recent first), all with `available_at ≤ cutoff`:

| form | available_at (ET) | url |
|---|---|---|
| 8-K | 2022-04-19 20:03:19 | https://www.sec.gov/Archives/edgar/data/1065280/000106528022000144/0001065280-22-000144-index.htm |
| PRE 14A | 2022-04-08 20:10:10 | https://www.sec.gov/Archives/edgar/data/1065280/000119312522100334/0001193125-22-100334-index.htm |
| 3 | 2022-04-07 20:24:07 | https://www.sec.gov/Archives/edgar/data/1065280/000106528022000143/0001065280-22-000143-index.htm |
| 4 | 2022-04-04 21:58:47 | https://www.sec.gov/Archives/edgar/data/1065280/000106299322009534/0001062993-22-009534-index.htm |
| 4 | 2022-04-04 21:22:30 | https://www.sec.gov/Archives/edgar/data/1065280/000106528022000138/0001065280-22-000138-index.htm |

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

Artifact: `data/documents/crash_documents_sample_v1.parquet` (canonical
`DOCUMENT_SCHEMA`). Reproducible via `scripts/retrieve_crash_news.py` (live EDGAR).
