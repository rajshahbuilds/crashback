# STU-54 — Point-in-Time Fundamentals (Ingest & Normalize)

**Status:** ✅ complete. Historical fundamentals ingested from the **unrestated** Compustat
quarterly file, normalized with a defensible availability date, written to versioned Parquet.

**Build:** `.venv/bin/python scripts/ingest_fundamentals.py --username r43shah`
**Artifact:** `data/normalized/fundamentals/fundamentals_v1.parquet` (gitignored). Module:
`crashback.fundamentals.ingest`.

## Source decision (Option B — leakage-safe)

- **Full-vintage snapshot** (`compsamp_snapshot.wrds_csq_pit`, with `pitdate1/2` windows) was
  probed and **rejected**: it's a *sample* — only **2,376 gvkeys** (Apple isn't in it) — so it
  can't cover our universe.
- **Chosen: `comp_urq.urqus`** (Compustat **unrestated** quarterly): full coverage
  (**29,473 gvkeys, 1982→2026**), one row per firm-quarter, values **as originally reported**.

## Restatement & availability policy (documented)

- **Restatements:** the unrestated file stores originally-reported values, so a later
  restatement of an old quarter is never reflected — **no future revision leaks into an earlier
  date**. (True full-vintage reconstruction wasn't available at universe scale.)
- **Availability date** (`public_date`): the earnings report date `rdq` where present
  (**80%**); otherwise a **conservative `period_end + 60d` fallback** flagged with
  `rdq_available = False`, so the as-of join (STU-55) never assumes availability too early.

## Fields (26)

Identifiers/dates: `company_id (gvkey)`, `period_end`, `public_date`, `rdq_available`, `freq`,
`fiscal_year`, `fiscal_quarter`. Income: `revenue` (saleq), `cogs`, `gross_profit`, `ebitda`
(oibdpq), `depreciation`, `operating_income` (oibdpq−dpq), `net_income`, `eps`,
`interest_expense`. Balance sheet: `total_assets`, `total_liabilities`, `cash`,
`debt_long_term`, `debt_current`, `total_debt`, `current_assets`, `current_liabilities`,
`stockholders_equity`, `common_equity`, `shares_outstanding`.

`total_debt` is null only when **both** debt components are missing (never masks missing data
as zero — 151k such rows).

### Known gap — free cash flow

The unrestated file carries **no cash-flow-statement items** (`oancfy`/`capxy` absent), so
**FCF features (`fcf_margin`, `fcf_yield`) are unavailable** from this source. Documented; can
be back-filled from restated `comp.fundq` later, isolating restatement risk to those fields.

## Result

- **1,354,311** rows; **29,473** companies; `period_end` 1982-03-31 → 2026-06-30.
- `rdq_available` **80.0%** (20% use the +60d fallback).
- Coverage: revenue **97.4%**, net_income 97.4%, eps 93.4%, total_assets 89.4%,
  stockholders_equity 89.3%, ebitda 87.3%.

## Validation

- **Unit tests** (`tests/test_fundamentals_ingest.py`): field mapping, derived fields
  (`gross_profit`, `operating_income`, `total_debt`), and availability-date logic (rdq present
  → `public_date = rdq`; rdq missing → `period_end + 60d`, `rdq_available = False`). 50 tests
  passing.
- **Company-timeline hand-check** (AAPL, gvkey 1690) vs the real earnings calendar: quarter
  ending 2025-12-31 → **2026-01-29** (revenue $143.8B), 2025-09-30 → 2025-10-30 — matches
  Apple's actual release dates and figures.

## Acceptance criteria

- ✅ Every record has a defensible availability date (`rdq`, or a flagged +60d fallback).
- ✅ Restatement policy documented and non-leaking (unrestated = originally reported).
- ✅ Core raw fields available; missingness (incl. the FCF gap) documented.
- ✅ Sample company timeline manually checked against historical release dates.
- ✅ Versioned normalized fundamentals written to Parquet.

Next: **STU-55** — as-of join fundamentals to crash events (permno→gvkey via CCM; attach the
latest quarter with `public_date ≤ crash_date`), with dedicated leakage tests.
