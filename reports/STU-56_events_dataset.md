# STU-56 — Canonical V1 Event Dataset (`events_v1`)

**Status:** ✅ complete. All event / feature / fundamentals / outcome tables assembled into one
versioned research dataset — one row per crash prediction event.

**Build:** `.venv/bin/python scripts/build_events_dataset.py --version v1`
**Artifact:** `data/processed/events_v1.parquet` (+ `.meta.json` sidecar; gitignored). Module:
`crashback.datasets.assemble`. Registered in Supabase `dataset_versions` + `artifacts`.

## Shape

- **1,153,414 rows** (exactly one per `event_id`), **101 columns**, 1926 → 2025. **CLEAN pool
  268,452**. ~209 MB.

## Column contract (features strictly separated from outcomes)

| group | n | columns |
|---|---|---|
| ids | 5 | event_id, security_id, company_id, ticker_as_of_event, crash_date |
| eligibility | 2 | in_universe_at_event, passes_min_price |
| anchor | 1 | crash_close (P0) |
| **features** | **52** | price (19) + recent_crash (7) + market_sector (8) + fundamentals (18) |
| feature meta | 6 | fundamentals_available / period_end / public_date / days_since / stale / rdq_available |
| **outcomes** | **27** | 9 close-based `hit_*` + 9 intraday `hit_*_hi` + 9 continuous (return/rebound/drawdown) |
| outcome meta | 6+2 | censored_{5,20,60}d, n_forward_{5,20,60}d, no_close_data, delisting_return_missing |

`FEATURE_COLS` (model inputs) and `OUTCOME_COLS` (targets) are **disjoint by construction** and
validation fails if they ever overlap — so modeling can never train on a label. Primary target:
**`hit_10pct_20d`**. The exact lists live in `crashback.datasets.assemble` and in the sidecar.

## Validation (all pass; hard failures raise)

- **Unique**: `event_id` unique == row count (no duplicates).
- **Impossible values**: every `crash_return ≤ -0.10` (all rows are crashes); every `hit_*` ∈ {0,1,null}.
- **Leakage indicator**: `FEATURE_COLS ∩ OUTCOME_COLS = ∅`; all required columns present.
- **Missingness**: per-feature null fraction (CLEAN pool) recorded in the sidecar — e.g.
  `market_return_1d` 0.00, `volatility_20d` 0.03, `net_margin` 0.28, `distance_from_52w_high` 0.47.
  This is expected/explicit missingness (short history, non-Compustat filers), not corruption.

Base-rate sanity: primary target on the CLEAN determined pool = **0.5649** (matches STU-50).

## Provenance & reproducibility

The sidecar records the full column contract (feature groups, feature/outcome lists), the
per-feature missingness, the crash-date range, the **resolved config snapshot**, `git_commit`,
`sha256`, and size — so the dataset is self-describing. It is **regenerable from raw inputs** by
re-running the build (no new data pulls; a pure join of committed pipeline outputs). A
`dataset_versions` row (+ an `artifacts` pointer) is written to Supabase (STU-46) — the first use
of the metadata store.

## Acceptance criteria

- ✅ One unique row per `event_id`.
- ✅ Required V1 columns present with documented types/units (schemas + sidecar).
- ✅ Outcome columns isolated from model feature lists (disjoint, validated).
- ✅ Validation checks duplicates, impossible values, leakage indicators, and missingness.
- ✅ Versioned `events_v1` Parquet regenerable from raw inputs.

Next: STU-57 — establish recovery base rates and descriptive slices on `events_v1` (the CLEAN pool).
