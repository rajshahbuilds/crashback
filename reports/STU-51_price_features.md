# STU-51 — Crash-Day & Pre-Crash Price Features

**Status:** ✅ complete. 19 point-in-time price-action features per crash event, verified
leakage-free, written to versioned Parquet.

**Build:** `.venv/bin/python scripts/build_price_features.py --version v1`
**Artifact:** `data/processed/features_price_v1.parquet` (keyed by `event_id`; gitignored).
Module: `crashback.features.price`.

## Features (all available by the crash-day close)

- **Crash day:** `crash_return`, `opening_gap` (open/prev_close−1), `intraday_range`
  ((high−low)/prev_close), `close_vs_low`, `close_vs_open`, `crash_volume`,
  `relative_volume_20d` (volume ÷ prior-20-day average).
- **Trailing volatility** (std of daily return over the prior N days, **excluding** the
  crash day): `volatility_20d`, `volatility_60d`.
- **Pre-crash returns** (window ends the day **before** the crash): `return_{5,20,60,252}d_pre`.
- **Distance below trailing high** (**includes** the crash day): `distance_from_{20d,60d,52w}_high`.
- **Drawdown from trailing close peak** (includes the crash day): `drawdown_{20d,60d,252d}`.

## Point-in-time guarantee

Every feature is a per-security window expression (`.over("security_id")`) over a
(security_id, date)-sorted frame using only observations with date ≤ crash_date. Pre-crash
momentum/volatility windows `shift(1)` to exclude the crash day so it can't contaminate them.
A dedicated **no-look-ahead test** appends extreme future rows and asserts every crash-day
feature is unchanged.

**Missing history is explicit:** a rolling window shorter than its size, or a shift before
the series start, yields null — never a partial or forward-filled value.

## Coverage (fraction of events with a non-null value)

| feature | ALL | CLEAN |
|---|---|---|
| volatility_20d | 0.98 | 0.97 |
| relative_volume_20d | 0.93 | 0.97 |
| intraday_range | 0.74 | 0.94 |
| return_20d_pre | 0.70 | 0.89 |
| return_252d_pre | 0.63 | 0.74 |
| distance_from_52w_high | 0.32 | 0.53 |
| drawdown_60d | 0.43 | 0.70 |

The 252-day / 52-week features have lower coverage because they require a full year of
non-null history — genuinely absent for recently-listed or short-history securities (and for
illiquid stretches with null highs). This is honest, explicit missingness; how to handle it
(missingness indicators, sample restriction) is a dataset-assembly decision (STU-56).

## Validation

- **Hand-calculated unit tests** (`tests/test_price_features.py`): every crash-day feature +
  `return_5d/20d_pre` + `distance_from_20d_high` + `drawdown_20d` on a known path;
  sparse-history nulls; and **no-look-ahead leakage**. 40 tests passing.
- **Spot check at scale** — AAPL 2000-09-29 crash, feature vs independent recompute:
  `return_20d_pre = −0.10084` (match), `distance_from_52w_high = −0.82876` (match) — the
  dot-com collapse, closing ~83% below its 52-week high.

## Acceptance criteria

- ✅ All defined features generated for eligible events.
- ✅ Missing-history behavior explicit and consistent (null on insufficient window).
- ✅ Rolling windows verified to use dates ≤ crash date (leakage test).
- ✅ Hand-checked sample events match independent calculations.
- ✅ Unit tests cover window boundaries and sparse histories.

Next: STU-52 (recent-crash history features) and STU-53 (market/sector context).
