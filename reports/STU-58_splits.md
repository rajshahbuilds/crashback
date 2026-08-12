# STU-58 — Chronological Train / Validation / Test Splits

Assigned by `scripts/build_splits.py` (module `crashback.datasets.splits`) over `events_v1`.
Splits are by `crash_date` only — **never random row sampling** — so training is always on the
past relative to validation/test. Definitions are versioned in `configs/default.yaml`.

## Split definitions (from config)

- **train**: 2003-01-01 → 2018-12-31
- **validation**: 2019-01-01 → 2021-12-31
- **test**: 2022-01-01 → 2025-12-31
- **embargo**: 60 trading days

`none` = events outside all configured ranges (chiefly pre-2003 history,
retained only for robustness work, not primary modeling).

## Embargo (outcome-window boundary treatment)

Outcomes look up to **60 trading days** forward, so an event in
the last 60 trading days of train (or validation) has an outcome
window that spills into the next split. Those events are reassigned to **`embargo`** and dropped
from modeling, measured against the **real trading calendar** (not calendar days). This
guarantees no training event's outcome overlaps validation/test — the primary leakage risk of
chronological splitting with forward-looking labels.

## Summary (CLEAN pool)

| split | events | determined | date range | securities | sectors | base rate |
|---|---|---|---|---|---|---|
| train | 52,535 | 52,510 | 2003-01-02 → 2018-10-03 | 6,468 | 11 | 0.5111 |
| validation | 19,481 | 19,480 | 2019-01-02 → 2021-10-06 | 3,800 | 11 | 0.6005 |
| test | 19,181 | 18,856 | 2022-01-03 → 2025-12-31 | 3,487 | 11 | 0.4954 |
| embargo | 2,592 | 2,592 | 2018-10-04 → 2021-12-31 | 1,481 | 11 | 0.4441 |
| none | 174,663 | 174,108 | 1926-01-06 → 2002-12-31 | 14,696 | 11 | 0.5865 |

- **No event appears in multiple splits** — each `event_id` gets exactly one label.
- **Primary training never sees test-period events or outcomes** — test is strictly later and
  the embargo removes boundary-crossing outcome windows.
- Target prevalence (base rate) is reported per split; drift across splits reflects genuine
  regime differences, and is why we evaluate on the held-out test period rather than in-sample.
