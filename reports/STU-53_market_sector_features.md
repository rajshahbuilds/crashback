# STU-53 — Market & Sector Context Features

**Status:** ✅ complete. Market-wide and sector-wide context per crash event, point-in-time,
broad selloffs retained (never filtered).

**Build:** `.venv/bin/python scripts/build_market_sector_features.py --version v1`
**Artifact:** `data/processed/features_market_sector_v1.parquet` (keyed by `event_id`;
gitignored). Module: `crashback.features.market_sector`.

## Features (8)

- `market_return_1d`, `market_return_5d`, `market_return_20d` (trailing 1-month = regime proxy),
  `market_volatility_20d`.
- `sector_return_1d` (**focal company excluded**), `sector_return_5d`, `sector_volatility_20d`,
  `sector_n_members`.

## Methodology (documented & reproducible)

- **Equal-weighted** daily-return series built **from the research universe** — the mean of
  member `daily_return` per day. EW captures *breadth* (how many names are down), which is the
  relevant signal for "a sector-wide shock hit N companies"; value-weighting is a future option.
- **Sector = 2-digit SIC major group**, assigned once per security from the master
  (`sic_code // 100`). SIC drift across a security's life is rare; over sector aggregates of
  many members its impact is negligible.
- **Focal-company exclusion** is implemented for `sector_return_1d`
  (`(Σ_sector r − r_focal) / (n − 1)`) so a stock's own crash cannot masquerade as a
  sector-wide crash. For the multi-day `sector_return_5d` and 20d volatility the focal's impact
  over many members/days is negligible, so the full-sector series is used.
- **Point-in-time:** crash-day values use that day's cross-section (known at close); trailing
  windows use only dates ≤ crash_date. Verified by a no-look-ahead test (future market moves
  never change crash-day features).

## Coverage

market features **1.000**; `sector_return_1d` **0.998** (null only where the focal is the sole
sector member on the day); other sector features **~1.000**. Broad-market and sector-wide
crashes are kept in the dataset, per the acceptance criterion.

## Validation

- **Unit tests** (`tests/test_market_sector_features.py`): equal-weight market/sector aggregates
  and focal exclusion vs independently computed values; no-look-ahead. 46 tests passing overall.
- **Hand-checked market-crash dates** (equal-weighted `market_return_1d`):
  1987-10-19 (Black Monday) **−10.4%**, 2020-03-16 (COVID) **−12.4%**, 2020-03-12 −10.8%,
  2008-10-15 −7.1%, 2001-09-17 (post-9/11 reopen) −4.2% — correct direction and magnitude.
- **Focal exclusion at scale:** 105,295 events where the stock fell ≥ 20% but its ex-focal
  sector (>30 peers) stayed ~flat — confirming the stock's own move is removed.

## Acceptance criteria

- ✅ Market & sector features exist for the majority of eligible events (≥ 99.8%).
- ✅ Sector-wide crashes remain in the dataset (no filtering of broad selloffs).
- ✅ Sector aggregation methodology documented and reproducible (EW, 2-digit SIC).
- ✅ Focal-stock exclusion implemented (`sector_return_1d`); negligible-impact aggregates noted.
- ✅ Hand-checked broad-market/sector-crash dates match expected direction/magnitude.

Next: STU-54/55 (point-in-time fundamentals), then STU-56/57 (assemble `events_v1` + descriptive base rates).
