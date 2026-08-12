# STU-50 — Recovery Labels & Continuous Outcomes

**Status:** ✅ complete. The full recovery-label grid + continuous outcomes are computed for
every crash event, anchored to the crash-day close (P0), with rigorous censoring and
delisting handling. Written to versioned Parquet.

**Build:** `.venv/bin/python scripts/build_recovery_labels.py --version v1`
**Artifact:** `data/events/recovery_labels_v1.parquet` (gitignored). Module:
`crashback.labels.outcomes`. Keyed by `event_id` (joins to crash events / features downstream).

## Schema

- **Binary grid, close-based (9):** `hit_{5,10,20}pct_{5,20,60}d`. **Primary = `hit_10pct_20d`.**
- **Binary grid, intraday-high (9):** same names with a **`_hi`** suffix — kept strictly
  separate so they can never be confused with the close-based primaries.
- **Continuous (3 × horizon):** `return_{h}d`, `max_rebound_{h}d`, `max_drawdown_{h}d`.
- **Meta:** `censored_{h}d`, `n_forward_{h}d`, `crash_close`, `delisting_return_missing`.

All measured over forward **trading-day** windows via a per-security sequence index (not
calendar days).

## Policy (the parts that determine correctness)

- **Anchor:** every outcome is relative to `crash_close` (P0). `hit_Xpct_Hd` = max forward
  *close* over the next H trading days ≥ (1+X)·P0.
- **Censoring vs delisting:**
  - *Active* security whose window runs past the data edge → **censored** (label null), never
    a failure.
  - *Delisted* security → **determined**: the price path is augmented with a terminal node at
    `last_close · (1 + delisting_return)`, so a bankruptcy that never rebounded is a real
    `hit=0` and its drawdown reflects the loss. Censoring delisted names would re-introduce
    survivorship bias.
- **Missing CRSP delisting returns** are flagged (`delisting_return_missing`), not imputed.
- **Never dropped:** events with no valid close (CRSP no-trade days) can't have close-based
  labels, so they are censored/null — flagged, not treated as failures.

## Result

- **1,153,414** label rows; **CLEAN pool (in-universe & ≥ $5): 268,452**.
- Censoring is small and genuine within CLEAN: 3,583 / 6,090 / 8,981 at 5/20/60d
  (~2.3% at 20d), all with valid closes. (Raw censored_20d = 32,617 includes 6,456 no-close
  illiquid events + 26,161 data-edge; none of the CLEAN pool has a null close.)

### Base rates (CLEAN, determined events) — a preview; full analysis is STU-57

| threshold | 5d | 20d | 60d |
|---|---|---|---|
| +5% | 0.545 | 0.723 | 0.819 |
| **+10%** | 0.344 | **0.564** | 0.705 |
| +20% | 0.123 | 0.312 | 0.493 |

So **P(+10% close within 20 trading days | crash ≤ −10%) ≈ 56%** on the liquid universe —
monotonic in horizon (↑) and threshold (↓), as expected. Including penny stocks pushes the
primary to ~68% (low-priced names bounce +10% off a depressed base far more often — exactly
why the ≥ $5 filter matters).

## Validation

- **Hand-calculated unit tests** (`tests/test_recovery_labels.py`): active full recovery,
  data-edge censoring (label null, not failure), delisted bankruptcy (determined `hit=0`,
  drawdown via terminal), delisted merger (recovers via terminal), intraday-vs-close
  distinction, primary-target column presence. 36 tests passing.
- **Spot checks at scale:** Lehman 2008-09-12 → `hit_10pct_20d=0`, `return_20d=−98.6%`
  (determined, terminal folds in the collapse); AAPL 2000-09-29 → `hit=0` at all horizons
  (`max_rebound_60d=−5.8%`), matching history.

## Acceptance criteria

- ✅ All 9 binary labels generated for events with sufficient forward history.
- ✅ Continuous outcomes computed consistently from the crash close.
- ✅ Insufficient horizon → censored/flagged, never counted as a failure.
- ✅ Close-based vs intraday variants are un-confusable (`_hi` suffix, separate columns).
- ✅ Unit tests verify labels against hand-calculated price paths.

Next: STU-51/52/53 (features) and STU-56/57 (assemble the event dataset + descriptive base
rates on the CLEAN pool).
