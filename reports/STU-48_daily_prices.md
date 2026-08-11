# STU-48 — Historical Daily Prices (canonical, normalized)

**Status:** ✅ complete. The canonical daily price history for the common-stock universe
(active + delisted, 1925→2025) is materialized to Hive-partitioned Parquet.

**Build:** `.venv/bin/python scripts/ingest_daily_prices.py --username r43shah --start-year 1925 --end-year 2025`
**Artifact:** `data/normalized/daily_prices/year=YYYY/prices.parquet` (+ `manifest.json`;
gitignored — bulk data). Read via `crashback.ingestion.prices.scan_daily_prices()` or
DuckDB `read_parquet('.../year=*/prices.parquet')`.

## Schema (canonical `daily_price`)

`date, security_id, open, high, low, close, adjusted_close, volume, daily_return,
daily_return_ex_div, cum_factor_price, cum_factor_shares, shares_outstanding`.
Source: CRSP CIZ `crsp.dsf_v2`, normalized in `providers/normalize.py`. One row per
(security_id, trading_date); the universe is filtered server-side via a permno subquery on
the STU-47 common-stock filter.

## Adjustment & corporate-action policy (tested)

- **Returns use `daily_return`** (CRSP `dlyret`) — total return, already adjusted for splits
  and dividends. Crash detection uses this, **never** close-to-close deltas, so a split can
  never look like a crash.
- **`adjusted_close`** = `close / cum_factor_price` (raw `close` retained separately).
- **No forward-fill** — gaps are genuine non-trading days or halts; `daily_return` spans them.
- **Delisting preserved, not dropped** — the last real trading bar is kept; the terminal
  delisting return lives in the security master (`delisting_return`, dsedelist code ≥ 200),
  so label logic (STU-50) can apply it without corrupting the price series.
- **Uniqueness enforced** — ingestion rejects any duplicate (security_id, date) row.

### Split-correctness verification (manual, acceptance criterion)

| Event | Raw close | `daily_return` on split day | Verdict |
|---|---|---|---|
| AAPL 4:1 (2020-08-31) | $499 → $129 (−74% optical) | **+3.4%** | ✅ not a false crash |
| NVDA 10:1 (2024-06-10) | $1209 → $122 (−90% optical) | **+0.7%** | ✅ not a false crash |

`adjusted_close` stays continuous across both; `cum_factor_price` steps (e.g. 4.0→1.0).

## Result

| metric | value |
|---|---|
| total rows | **85,302,997** |
| years with data | 101 (1925 → 2025) |
| securities | **27,363** (matches the security master exactly) |
| on-disk size | 1.2 GB, 101 `year=YYYY/prices.parquet` partitions |
| duplicate (security_id, date) rows | **0** |

### Coverage & correctness (DuckDB verification over all 85.3M rows)

- **Survivorship-safe coverage:** all **4,171 active** and all **23,192 delisted** securities
  from the master have price bars (100%).
- **AAPL** (14593): 1980-12-12 → 2025-12-31, 11,355 bars (worst day −51.9%, Sept 2000).
- **Lehman** (80599): 1994-05-31 → **2008-09-18**, 3,604 bars (worst −94.25%) — delisted
  history terminates at the delisting; the terminal bar is preserved, not dropped.
- Crash-day observations (`daily_return ≤ −10%`): **1,153,414** (the raw pool STU-49 turns
  into deduped crash events).

## Partitioning

Hive-partitioned by calendar **year** for predicate pushdown on date in DuckDB/Polars scans.
Full rebuild (not incremental) for V1; re-runnable and idempotent per year.

## Tests

`tests/test_ingestion_prices.py` (hermetic): duplicate-(security_id,date) rejection,
partitioned write + manifest, round-trip via `scan_daily_prices`, empty-year skip.
Split correctness covered in `tests/test_normalization.py`. 25 tests passing.
