# STU-55 — Fundamentals As-Of Join (leakage-free) + Derived Features

**Status:** ✅ complete. Fundamentals attached to each crash event by **availability date**, with
derived §15 features, verified leakage-free at scale.

**Build:** `.venv/bin/python scripts/build_fundamental_features.py --username r43shah`
**Artifact:** `data/processed/features_fundamentals_v1.parquet` (keyed by `event_id`;
gitignored). Module: `crashback.fundamentals.features`.

## The join (the leakage-critical part)

1. **Point-in-time permno → gvkey link** via `crsp.ccmxpf_lnkhist` (linktype LU/LC, linkprim
   P/C), using the link valid on the crash date. Events with no valid link keep a null gvkey
   (never dropped).
2. **Backward `join_asof` on `public_date`** by gvkey: attach the most recent fundamentals with
   `public_date ≤ crash_date`. A quarter whose fiscal period ended before the crash but was
   *published* after it is **excluded** (uses the prior quarter).

**Leakage verified globally:** across **861,675** matched events, **0** have
`fundamentals_public_date > crash_date`.

## Derived features (formulas documented)

TTM = trailing-twelve-month sum of the last 4 quarters (per company, backward).

- **Growth:** `revenue_growth_yoy` (rev / rev[-4q] − 1), `revenue_growth_qoq` (÷ rev[-1q]),
  `eps_growth_yoy`.
- **Profitability (TTM):** `gross_margin` = gross_profit_ttm/revenue_ttm; `operating_margin`;
  `net_margin`; `roa` = ni_ttm/total_assets; `roe` = ni_ttm/stockholders_equity.
- **Balance sheet:** `net_debt` = total_debt − cash; `current_ratio`; `debt_to_assets`;
  `net_debt_to_ebitda` = net_debt/ebitda_ttm; `interest_coverage` = operating_income_ttm/interest_ttm.
- **Valuation** (event price × latest known shares): `market_cap` = crash_close × shares;
  `pe` = market_cap/ni_ttm; `price_to_sales` = market_cap/revenue_ttm;
  `ev_to_sales`, `ev_to_ebitda` with `ev` = market_cap + net_debt.
- **Meta/provenance:** `fundamentals_available`, `fundamentals_period_end`,
  `fundamentals_public_date`, `days_since_fundamental`, `fundamentals_stale` (> 200 days),
  `fundamentals_rdq_available`.

Non-finite ratios (division by ~0) are set to null. **FCF ratios are omitted** — the unrestated
source (STU-54) has no cash-flow items.

## Coverage

- `fundamentals_available`: **74.7%** (all events), **79.5%** (CLEAN pool). The ~20% without a
  match are non-Compustat filers or crashes before a company's first filing — expected.
- Only **1.0%** of attached fundamentals are stale (> 200 days), so the as-of join usually finds
  a recent quarter.
- CLEAN coverage of key ratios: net_margin 0.72, roe 0.74, debt_to_assets 0.78,
  revenue_growth_yoy 0.74, pe 0.74, ev_to_ebitda 0.65 (lower where TTM/growth need several
  quarters).

## Validation

- **Unit tests** (`tests/test_fundamentals_asof.py`): as-of publication boundary (a quarter
  published *after* the crash is excluded), **future-record injection** (a far-future quarter
  never changes the join), full derived-feature hand-calc, point-in-time link (crash before
  linkdt → no fundamentals), missing-link and stale flags. 56 tests passing.
- **Global leakage sweep:** 0 leaks over 861,675 matched events.
- **Earnings-boundary spot check:** AAPL crash 2013-01-24 → attached the quarter ending
  2012-12-31, **published 2013-01-23 (one day before the crash)** — correct as-of behavior at a
  one-day boundary (market_cap ≈ $423B, PE ≈ 10). COVID crash 2020-03-16 → 2019-12-31 quarter.

## Acceptance criteria

- ✅ No event receives fundamentals first published after the crash (0 leaks / 861,675).
- ✅ Derived feature formulas documented (above).
- ✅ Missing/stale fundamentals explicitly flagged (`fundamentals_available`, `_stale`,
  `days_since_fundamental`, `_rdq_available`).
- ✅ Manual spot checks across earnings-release boundaries confirm correct as-of behavior.
- ✅ Automated leakage tests inject future records and verify exclusion.

**M5 (point-in-time fundamentals) complete.** Next: STU-56 (assemble the master `events_v1`
dataset — events + labels + all feature families), then STU-57 (descriptive base rates).
