# STU-52 — Recent-Crash History Features

**Status:** ✅ complete. A first-class recent-crash feature family, point-in-time, written to
versioned Parquet.

**Build:** `.venv/bin/python scripts/build_recent_crash_features.py --version v1`
**Artifact:** `data/processed/features_recent_crash_v1.parquet` (keyed by `event_id`;
gitignored). Module: `crashback.features.recent_crash`.

## Features (7)

- `prior_crash_count_{5,20,60}d` — crashes in the N trading days **strictly before** the
  current crash day (today's crash never counts as its own prior).
- `days_since_previous_crash` — trading days since the most recent prior crash (null if none).
- `previous_crash_return` — the prior crash's daily return (null if none).
- `return_since_previous_crash` — `crash_close / previous_crash_close − 1` (null if none).
- `max_rebound_since_previous_crash` — highest close between the previous crash and now,
  relative to the previous crash close (null if none).

`cumulative_return_{N}d` and `drawdown_from_{N}d_high` (also listed under §12) are already
produced point-in-time by STU-51 (`return_{N}d_pre`, `distance_from_{N}d_high`) and are not
duplicated here.

## Point-in-time correctness

All computed via per-security window expressions over a (security_id, date)-sorted frame.
Prior-crash counts use `rolling_sum(N).shift(1)` (window ends **yesterday**); previous-crash
markers use `when(is_crash).then(...).shift(1).forward_fill()` — so the current crash is never
included, and no value references a date after the crash. Verified by a **no-look-ahead test**
(the day-5 crash's features are identical whether or not a later day-7 crash exists).

A fresh (first-ever, or no-recent) crash has **zero** prior counts and **null** previous-crash
fields.

## Result

- **1,153,414** feature rows.
- **Fresh crashes** (0 prior in 60 td): 199,103 = **17.3%** overall; **40.3%** in the CLEAN
  pool (liquid names repeat-crash far less than penny stocks). Median days-since-previous-crash
  in CLEAN ≈ **29** trading days.
- `prior_crash_count_20d` distribution (all events): 0→384k, 1→245k, 2→172k, 3→124k, … — a long
  repeat-crasher tail.

Invariants verified over all 1.15M rows (0 violations): counts monotonic
(`5d ≤ 20d ≤ 60d`); any event with a prior crash in 60 td has a non-null
`days_since_previous_crash`.

## Validation

Hand-calculated unit tests (`tests/test_recent_crash_features.py`): fresh first crash (zero
counts, null previous fields), second-leg crash after a partial rebound (counts,
`days_since`, `previous_crash_return`, `return_since`, `max_rebound_since`), third crash with
two priors, and current-crash-not-counted / no-look-ahead. 44 tests passing overall.

## Acceptance criteria

- ✅ Current-day crash is never counted as a prior crash.
- ✅ First-ever crash → zero prior counts + null previous-crash fields.
- ✅ Consecutive crash days produce distinct rows with correctly updated history.
- ✅ No feature references data after the current crash timestamp (leakage test).
- ✅ Unit tests cover fresh shocks, second-leg crashes, and crashes after rebounds.

Next: STU-53 (market/sector context features).
