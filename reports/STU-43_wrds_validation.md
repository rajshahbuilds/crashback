# STU-43 — WRDS Access & Historical Data Coverage Validation

**Status:** ✅ PASS — WRDS access confirmed; CRSP + Compustat NA entitled; survivorship-safe
universe and point-in-time fundamentals are achievable.

**Date:** 2026-08-11
**Account:** `r43shah` (WRDS PostgreSQL, `wrds-pgdata.wharton.upenn.edu:9737`)
**Client:** `wrds` 3.5.0 on Python 3.14 (isolated `.venv`)
**Machine-readable results:** [`reports/wrds_validation.json`](./wrds_validation.json)
**Reproduce:** `.venv/bin/python scripts/validate_wrds.py --username r43shah`

---

## Results against the 7 acceptance criteria (CLAUDE.md §5)

| # | Check | Result |
|---|-------|--------|
| 1 | Programmatic Python access | ✅ Connected via `wrds` 3.5.0 + `~/.pgpass`; 261 libraries entitled |
| 2 | CRSP entitlement | ✅ `crsp` (433 tables) + `crsp_a_ccm`, `crsp_a_stock`, `crsp_a_indexes` |
| 3 | Compustat North America entitlement | ✅ `comp` (293 tables) + `comp_na_daily_all`, `comp_urq`, `compseg` |
| 4 | Delisted-security coverage | ✅ `crsp.dsedelist` = 38,872 delisting records, 1926→2024, 25+ codes |
| 5 | Exact historical date coverage | ✅ SIZ→2024-12-31; CIZ (`dsf_v2`)→2025-12-31; Compustat→2026-07-31 |
| 6 | Compustat Point-in-Time availability | ✅ Achievable — see below |
| 7 | Precise tables/columns needed | ✅ Identified — see canonical mapping below |

---

## Coverage detail

### CRSP (prices, security master, delistings)
- **`crsp.dsf`** (daily stock file): **1925-12-31 → 2024-12-31**, ~107.7M rows,
  **38,872 distinct `permno`**. Cols: `permno, permco, date, prc, ret, retx, vol,
  shrout, cfacpr, cfacshr, openprc, bidlo, askhi, cusip, hexcd, hsiccd`.
- **`crsp.stocknames`** / **`crsp.dsenames`**: full ticker/name history, 38,872 permnos.
  Cols: `permno, namedt, nameenddt, shrcd, exchcd, siccd, ncusip, ticker, comnam,
  shrcls, permco, cusip`. → `shrcd`, `exchcd`, `siccd` give us the common-stock /
  exchange / sector filters for the §4 universe.
- **`crsp.dsedelist`**: 38,872 records, **1926-02-24 → 2024-12-31**. Delisting codes
  span mergers (200s), liquidations (400s), and drops (500s: bankruptcy, price/cap
  minimums, delinquent filings). Confirms the universe includes companies that later
  died → **survivorship-safe**.

### Compustat NA (fundamentals)
- **`comp.funda`** (annual): **1950-06-30 → 2026-07-31**, ~941k rows, 47,189 gvkeys.
- **`comp.fundq`** (quarterly): **1961-03-31 → 2026-07-31**, ~2.13M rows, 46,514 gvkeys.
- **`comp.company`**: 58,222 gvkeys.

### CRSP ↔ Compustat link (CCM) — needed to join fundamentals to prices
- `crsp.ccmxpf_lnkhist` (123,388 rows) — **recommended link-history table**
  (`gvkey ↔ permno` with `linktype`, `linkprim`, `linkdt`, `linkenddt`).
- Also present: `crsp.ccmxpf_linktable` (92,711), `crsp.ccm_lookup` (109,912).

---

## Point-in-time fundamentals (criterion 6) — how we'll stay leakage-free

Three complementary mechanisms are available, all entitled:

1. **`comp.fundq.rdq`** (earnings *report date*): populated for **1,387,577 / 2,133,402**
   quarterly rows (~65%), range **1971 → 2026-08-10**. This is the primary
   filing-availability timestamp for the as-of join (§15 / §5 rule: use *public
   availability*, not fiscal period end).
2. **`compsamp_snapshot`** — Compustat **Snapshot / Point-in-Time** product is entitled
   (true unrestated point-in-time vintages).
3. **`comp_urq`** — Compustat **unrestated quarterly** (guards against later restatements
   being treated as historically known).
   Plus `funda`/`fundq` carry `pdate` (preliminary) and `fdate` (final) date columns.

**Convention to document later:** where only `datadate`/`rdq` dates exist (no intraday
timestamp), a fundamental is considered available at end-of-day on `rdq`; rows with a
null `rdq` require a fallback lag policy — to be decided in STU-54/55.

---

## ⚠️ Material caveats to carry forward

1. **CRSP price front-edge depends on which format we use — RESOLVED via CIZ.**
   - Classic **SIZ** `crsp.dsf` (cols `prc`/`ret`) ends **2024-12-31**.
   - Modern **CIZ** `crsp.dsf_v2` / `crsp.stkdlysecuritydata` (date col `dlycaldt`) ends
     **2025-12-31** — a full year more current, and CRSP's actively-maintained format.
   - **Decision for STU-48:** adopt the **CIZ (`_v2`) tables** as the canonical price
     source (`crsp.dsf_v2` + `crsp.stocknames_v2`), which caps the study period at
     **2025-12-31** (Compustat runs to 2026-07-31, so prices are the binding constraint).
     CIZ uses different column names than SIZ (`dlyprc, dlyret, dlyvol, dlycaldt`, plus
     built-in delisting handling) — the provider adapter (STU-45) must normalize these to
     the canonical price schema. This sets the realistic TEST ceiling near end-2025, not
     the 2026 sketched illustratively in §20.
2. **Identifier model:** CRSP keys on `permno` (security) / `permco` (company); Compustat
   keys on `gvkey` (company) / `iid` (security). Map to the canonical schema (§9) as
   `security_id ← permno`, `company_id ← permco`/`gvkey` (via CCM). Never key on ticker.
3. **Common-stock filter:** use `shrcd IN (10,11)` and `exchcd IN (1,2,3)` (NYSE/AMEX/
   Nasdaq) to exclude ETFs/funds/preferred/warrants per §4.

---

## Canonical table/column mapping (feeds STU-45 / STU-47 / STU-48 / STU-54)

| Canonical need | WRDS source | Key columns |
|---|---|---|
| Daily prices | `crsp.dsf` | `permno, date, prc, ret, retx, vol, shrout, cfacpr, cfacshr, openprc, bidlo, askhi` |
| Security master / ticker history | `crsp.stocknames` (+`dsenames`) | `permno, permco, namedt, nameenddt, ticker, comnam, shrcd, exchcd, siccd, ncusip, cusip` |
| Delistings | `crsp.dsedelist` | `permno, dlstdt, dlstcd, dlret` |
| Fundamentals (annual/qtrly) | `comp.funda`, `comp.fundq` | `gvkey, datadate, rdq, pdate, fdate, fyearq, fqtr, …` |
| Company metadata | `comp.company` | `gvkey, conm, sic, gind, …` |
| CRSP↔Compustat link | `crsp.ccmxpf_lnkhist` | `gvkey, lpermno, linktype, linkprim, linkdt, linkenddt` |

---

## Recommendation

STU-43 is **complete and passing**. Proceed to **STU-44** (repo init) and **STU-45**
(provider-neutral interfaces) with WRDS as the confirmed primary source. Open the one
follow-up above as a small task: **probe for a CRSP price product more current than
2024-12-31** before locking the chronological TEST window.
