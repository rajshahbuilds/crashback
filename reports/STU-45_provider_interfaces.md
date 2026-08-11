# STU-45 — Provider-Neutral Market-Data Interfaces

**Status:** ✅ complete (scoped). One `MarketDataProvider` interface, canonical schemas, a
WRDS/CIZ adapter, and a synthetic in-memory provider. Multi-vendor support is intentionally
out of scope (WRDS is confirmed; see the STU-45 scope note).

## The boundary

Downstream stages (crash detection, labels, features, fundamentals joins) import **only**:

- `crashback.providers.MarketDataProvider` — the read interface
- `crashback.providers.schemas` — canonical column names + dtypes

They never see vendor names (`dlyret`, `permno`, `gvkey`, `rdq`). All vendor knowledge lives in:

- `crashback/providers/normalize.py` — pure `raw -> canonical` functions (unit-tested, no WRDS)
- `crashback/providers/wrds_provider.py` — thin SQL layer that calls `normalize`

Design choice: bulk tabular data is **Polars DataFrames validated against Polars schemas**
(the price table is ~110M rows; Polars/DuckDB are the analytics engines). Pydantic is used
for *config* (STU-44). `validate_schema(df, SCHEMA)` enforces columns/dtypes and is applied at
every adapter boundary.

## Interface

```python
class MarketDataProvider(ABC):
    def get_security_master(security_ids=None) -> pl.DataFrame          # schema: security_master
    def get_daily_prices(security_ids, start, end) -> pl.DataFrame      # schema: daily_price
    def get_fundamentals(*, company_ids=None, security_ids=None, freq="Q") -> pl.DataFrame
    def get_corporate_actions(security_ids=None) -> pl.DataFrame        # schema: corporate_action
    def get_sector_metadata(security_ids=None) -> pl.DataFrame          # schema: sector_metadata
```

`security_ids=None` means the whole available universe (pass explicit ids in production).

## Canonical schemas (required fields)

**daily_price** — `date, security_id, open, high, low, close, adjusted_close, volume,
daily_return, daily_return_ex_div, cum_factor_price, cum_factor_shares, shares_outstanding`
- **Crash detection must use `daily_return`** (CRSP total return, already split/dividend
  adjusted) — never close-to-close deltas, so splits never look like crashes.
- `adjusted_close` = `close / cum_factor_price` (falls back to `close` if factor null/0).

**security_master** — `security_id, company_id, ticker, ticker_start, ticker_end, exchange,
security_type, sic_code, listing_date, delisting_date, delisting_code, delisting_return`
- Grain: one row per (security_id, ticker period) — ticker history preserved.

**fundamentals** — `company_id, security_id, period_end, public_date, freq, fiscal_year,
fiscal_quarter, revenue, net_income, total_assets, total_liabilities, cash, total_debt,
shares_outstanding`
- **`public_date`** (Compustat `rdq`/`pdate`) is the point-in-time availability date for
  leakage-free as-of joins (STU-55). `total_debt` = long-term + current debt.

**corporate_action** — `security_id, effective_date, action_type, value, code, details`
- `action_type` ∈ {`DELISTING`, `SPLIT`, `DIVIDEND`}. The WRDS adapter emits `DELISTING`
  (from `dsedelist`); splits/dividends are captured in the price factors + `daily_return`.

**sector_metadata** — `security_id, sic_code, sic_division` (coarse SIC major-division label).

## Identifiers

`security_id` = CRSP **permno**, `company_id` = CRSP **permco** (or Compustat **gvkey** for
fundamentals). permno↔gvkey via `crsp.ccmxpf_lnkhist`. **gvkey is a 6-digit zero-padded
string** in Compustat — filtered with quoted, padded literals (`gvkey IN ('001690')`), not
integers.

## WRDS source mapping (CIZ)

| Canonical | WRDS source | Notes |
|---|---|---|
| daily_price | `crsp.dsf_v2` | `dlyclose→close`, `dlyret→daily_return`, `dlycaldt→date`, … |
| security_master | `crsp.stocknames_v2` (+ `crsp.dsedelist`) | ticker periods; delisting joined |
| corporate_action | `crsp.dsedelist` | `dlstcd→code`, `dlret→value` |
| fundamentals | `comp.fundq` (Q) / `comp.funda` (A) | `rdq/pdate→public_date`; link via CCM |
| sector_metadata | `crsp.stocknames_v2` | `siccd→sic_code` + division |

## Verification

- **Unit tests** (hermetic, no WRDS): `tests/test_provider_schemas.py`,
  `tests/test_normalization.py` — schema validation + every CIZ/Compustat mapping. 17 passing.
- **Live smoke** (AAPL 14593 + Lehman 80599): all 5 methods return schema-conforming frames —
  Lehman −94.25% crash day, delisting code 574, AAPL Q revenue $109.4B with `public_date`=rdq.

## Environment note

Runs on **Python 3.12/3.13**, not 3.14: `wrds` pins `pandas<2.3`, and pandas <2.3 segfaults on
3.14 (`pd.to_datetime`). `requires-python` is capped `>=3.12,<3.14`.
