# STU-49 — Crash-Event Detection

**Status:** ✅ complete. Every qualifying daily crash (`daily_return ≤ threshold`) is one
event — no cooldown, no future-dependent suppression — written to versioned Parquet.

**Build:** `.venv/bin/python scripts/build_crash_events.py --version v1`
**Artifact:** `data/events/crash_events_v1.parquet` (+ sidecar; gitignored). Source: STU-48
daily prices + STU-47 security master. Module: `crashback.events.detect`.

## Definition & policy

- **Crash** = `daily_return ≤ crash.threshold` (config, default −0.10). Uses CRSP
  `daily_return` (split/dividend adjusted) so corporate actions never create false events.
- **One event per qualifying (security_id, date).** No cooldown; consecutive crash days are
  separate events; nearby crashes are all kept (dependence is an evaluation concern, §22).
- **Threshold is configurable** — a stricter threshold yields a strict subset (tested).
- **Nothing is dropped.** Penny-stock and out-of-universe crashes are retained and *flagged*,
  not filtered, at construction (CLAUDE.md principle 10). Filtering is deferred to descriptive
  analysis / dataset assembly (STU-56/57).

## Schema (`CRASH_EVENT_SCHEMA`)

`event_id` (`{security_id}_{YYYYMMDD}`), `security_id`, `company_id`, `ticker_as_of_event`,
`crash_date`, `crash_return`, `crash_close` (= P0, the label anchor), `crash_volume`,
`in_universe_at_event`, `passes_min_price`.

Identifiers (`company_id`, `ticker_as_of_event`) come from a **point-in-time** join to the
security master name-period covering the crash date.

## Result

| metric | value |
|---|---|
| crash events (all) | **1,153,414** |
| securities | 25,323 |
| date range | 1926-01-04 → 2025-12-31 |
| `in_universe_at_event` | 1,138,955 (98.7%) |
| `passes_min_price` (≥ $5) | 270,712 |
| **CLEAN (in-universe AND ≥ $5)** | **268,452** ← primary study pool |

## Data-quality findings (documented, non-blocking)

1. **`crash_close` null on CRSP no-trade days** (183,354 events, all illiquid): CRSP leaves
   `dlyclose` null while still computing `daily_return` from the bid/ask midpoint. These all
   fail `passes_min_price`, so the CLEAN pool has a valid P0 anchor throughout. A future
   normalization tweak (coalesce `dlyclose` with `abs(dlyprc)`) could recover price levels for
   illiquid names if ever needed — not required for the ≥ $5 universe.
2. **Point-in-time universe membership.** Daily prices were ingested per *permno ever in the
   universe* (so surrounding bars exist for features/labels), but a crash can land on a date
   when the security was not in a clean universe name-period (minor exchange / other share
   type). `in_universe_at_event` captures this at the event level (~14.5k events are outside;
   filter on the flag downstream). This is the correct layering: complete prices per security,
   universe membership decided per event.

## Acceptance criteria

- ✅ Every qualifying day → exactly one event (unique on `(security_id, crash_date)`).
- ✅ Consecutive crash days stay separate — e.g. security 10000 crashes on 1986-08-14, -08-18,
  -08-19 as three distinct events.
- ✅ Non-qualifying days create nothing (`-9.99%` and up-days excluded; `≤` boundary tested).
- ✅ Split/corporate-action days create no false events (uses `daily_return`; tested).
- ✅ Unit tests: first crash, consecutive crashes, threshold boundary, no-covering-period
  (kept with null identifiers), configurable threshold. `tests/test_crash_events.py`, 30
  tests passing.

Next (STU-50): recovery labels + continuous outcomes anchored to `crash_close` (P0), with
forward-horizon censoring and delisting-return handling.
