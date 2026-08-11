# CLAUDE.md — Stock Crash Recovery Prediction

## Project overview

This repository implements a research project that studies **short-term stock-price recovery after large crashes**.

The core question is:

> After a major stock-price decline, can we predict which individual companies will recover, by how much, and how quickly, using only information that was actually available at the time of the crash?

This is **not initially a trading bot**. V1 is an empirical research and calibrated-probability prediction pipeline. The priority is to build a defensible historical dataset, avoid leakage/survivorship bias, establish base rates, and test whether different information sources improve recovery prediction.

The longer-term hypothesis is that there may be a useful distinction between:

- temporary / overreaction-driven price damage, and
- genuine long-term fundamental damage.

V1 tests this using market data and point-in-time fundamentals. V2 may add contemporaneous news, filings, earnings releases, and LLM-derived crash-cause / fundamental-damage features.

---

# 1. Prediction contract

## Crash definition

The default V1 crash threshold is:

```text
daily adjusted return <= -10%
```

This threshold must be configurable.

### Important: every qualifying crash is its own prediction event

There is **no cooldown and no future-dependent event suppression**.

Example:

```text
Monday      -17%   -> Event A
Tuesday     -13%   -> Event B
Wednesday    +2%
Thursday    -11%   -> Event C
```

All three qualifying crash days are valid prediction events.

This is intentional because in live use, at the close of Monday, we do not know whether Tuesday will contain another crash. Historical event construction must reproduce that same information set.

Do **not** use future recovery, later crashes, or any other post-event information to decide whether an event is included.

Nearby crash events may be correlated. That is an evaluation/statistics concern, not a reason to delete valid prediction opportunities.

---

# 2. Primary target and outcome labels

For a crash event at day `t`, anchor the prediction to the stock's **closing price on the crash day**, `P0`.

## Primary modeling target

The primary V1 target is:

> Probability that the stock closes at least **+10% above the crash-day closing price at some point within the next 20 trading days**.

Conceptually:

```text
hit_10pct_20d = 1
if max(close[t+1 : t+20]) >= 1.10 * close[t]
else 0
```

The primary label should be based on **closing prices**, not an intraday touch.

## Full binary recovery grid

Retain all of these labels:

```text
hit_5pct_5d
hit_5pct_20d
hit_5pct_60d

hit_10pct_5d
hit_10pct_20d
hit_10pct_60d

hit_20pct_5d
hit_20pct_20d
hit_20pct_60d
```

Optional intraday-high versions may also be computed, but they must have clearly different names and must never be confused with the primary close-based labels.

## Continuous outcomes

Also retain:

```text
return_5d
return_20d
return_60d

max_rebound_5d
max_rebound_20d
max_rebound_60d

max_drawdown_5d
max_drawdown_20d
max_drawdown_60d
```

Events without enough future trading history for a requested horizon must be **flagged/censored**, not silently labeled as failures.

---

# 3. Point-in-time rule — critical

Every feature used for an event at time `t` must have been knowable by the **prediction timestamp**, currently defined as the close of the crash day.

This project must aggressively prevent look-ahead leakage.

Never use:

- future prices,
- financial statements that were released after the crash,
- later restatements as though they were historically known,
- later analyst revisions,
- later news articles explaining an old event with hindsight,
- future sector composition,
- future company classifications if they were not valid historically.

For fundamentals, the relevant date is **public availability**, not merely the fiscal period end.

Example:

```text
Quarter ended:       Dec 31
Results published:   Feb 8
Crash date:          Jan 25
```

The Dec 31 results are NOT available for the Jan 25 prediction.

Where exact timestamps are available, preserve them. If only dates are available, document the convention used for same-day releases.

Leakage prevention should have dedicated tests.

---

# 4. Historical universe and survivorship bias

The research universe should contain US-listed common stocks, including companies that later:

- delisted,
- went bankrupt,
- were acquired,
- changed ticker,
- changed exchange.

Do not build the universe from today's surviving tickers.

Target exchanges:

- NYSE
- Nasdaq
- NYSE American / AMEX where available

Exclude:

- ETFs
- mutual funds
- preferred shares
- warrants
- other non-common-stock security types

Initial liquidity filters may later include rules such as price >= $5 and minimum trailing dollar volume, but these should be configurable and calculated only from information available before/on the event.

Use stable security/company identifiers wherever possible. Ticker alone is not a stable identifier.

---

# 5. Data-source strategy

## Preferred source

Preferred research-grade source:

```text
WRDS
  + CRSP for securities / historical prices / delistings
  + Compustat for fundamentals
```

WRDS access is **live and validated** (STU-43, 2026-08-11; account `r43shah`). CRSP + Compustat
North America are both entitled. See `reports/STU-43_wrds_validation.md` for the full findings.

The following were validated (all pass); re-run `scripts/validate_wrds.py` to refresh:

1. programmatic Python access — ✅ (`wrds` 3.5.0 via `~/.pgpass`),
2. CRSP entitlement — ✅,
3. Compustat North America entitlement — ✅,
4. delisted-security coverage — ✅ (`crsp.dsedelist`, 1926→2024),
5. exact historical date coverage — ✅ (CIZ prices → 2025-12-31; Compustat → 2026-07-31),
6. Compustat Point-in-Time — ✅ (`fundq.rdq`, `compsamp_snapshot`, `comp_urq`),
7. the precise tables and columns needed — ✅ (documented in the STU-43 report).

**Canonical source decision:** use CRSP **CIZ** tables (`crsp.dsf_v2`, `crsp.stocknames_v2`),
not the legacy SIZ `crsp.dsf`. Identifiers: `permno`/`permco` (CRSP) ↔ `gvkey` (Compustat) via
`crsp.ccmxpf_lnkhist`. The study period is currently price-bound at ~2025-12-31.

## If WRDS is unavailable

The project must remain provider-neutral.

Preferred fallback candidates:

```text
1. Tiingo
2. Massive + SEC EDGAR
3. EODHD + SEC EDGAR
```

For research quality, historical/delisted securities and point-in-time semantics matter more than API convenience.

`yfinance` is acceptable for rapid prototyping and unit-test fixtures, but it should not be the canonical final research universe unless its historical universe/survivorship limitations are explicitly solved.

## SEC EDGAR fallback

SEC EDGAR may be used to reconstruct point-in-time fundamentals from filings and filing availability timestamps.

If using SEC/XBRL, expect normalization work because different issuers can use different tags/extensions.

---

# 6. Provider abstraction

Do not let WRDS-specific schemas leak through the entire codebase.

Implement provider-neutral interfaces, conceptually like:

```python
class MarketDataProvider:
    def get_security_master(...): ...
    def get_daily_prices(...): ...
    def get_fundamentals(...): ...
    def get_corporate_actions(...): ...
    def get_sector_metadata(...): ...
```

Normalize provider-specific output into canonical schemas as early as possible.

Downstream code for:

- crash detection,
- label generation,
- feature engineering,
- modeling,
- evaluation

should not need to know whether the upstream source was WRDS, Tiingo, Massive, or another provider.

---

# 7. Recommended technology stack

## Core language

```text
Python
```

## Bulk data / analytics

```text
Polars
DuckDB
PyArrow
Parquet
```

Use Parquet for large immutable/versioned datasets such as:

- raw daily prices,
- normalized fundamentals,
- derived feature tables,
- model matrices.

Use DuckDB / Polars for high-volume analytical transformations.

Avoid forcing the entire raw historical market database into Postgres.

## Supabase

The owner already uses Supabase for other projects.

Use Supabase/Postgres for durable relational project state such as:

```text
dataset_versions
pipeline_runs
feature_versions
model_runs
metrics
predictions
artifact references
```

Potentially also light security metadata if useful.

Bulk daily bars and wide training tables should normally remain Parquet artifacts.

Supabase Storage may eventually be used for larger artifacts if desired.

## Modeling

Start with:

```text
scikit-learn
LightGBM or XGBoost
SHAP
```

Do not start with neural networks.

## Research / development

```text
Jupyter
pytest
Pydantic
YAML/TOML configuration
Git/GitHub
```

---

# 8. Suggested repository layout

Use something close to:

```text
stock-crash-recovery/
|
|-- pyproject.toml
|-- README.md
|-- CLAUDE.md
|
|-- configs/
|   |-- default.yaml
|   `-- experiments/
|
|-- data/
|   |-- raw/
|   |-- normalized/
|   |-- processed/
|   `-- events/
|
|-- src/
|   |-- ingestion/
|   |-- providers/
|   |-- securities/
|   |-- events/
|   |-- labels/
|   |-- features/
|   |-- fundamentals/
|   |-- datasets/
|   |-- models/
|   |-- evaluation/
|   `-- storage/
|
|-- notebooks/
|   |-- 01_base_rates.ipynb
|   |-- 02_crash_analysis.ipynb
|   |-- 03_fundamentals.ipynb
|   `-- 04_models.ipynb
|
|-- tests/
|
`-- reports/
```

Prefer production transformations in `src/`; notebooks should consume those functions rather than becoming the only implementation.

---

# 9. Canonical security master

Build a historical security master containing, where available:

```text
security_id
company_id
ticker
ticker_start
ticker_end
exchange
security_type
sector
industry
listing_date
delisting_date
```

Requirements:

- include active and delisted companies,
- preserve ticker history,
- avoid treating ticker changes as unrelated securities when stable IDs indicate continuity,
- explicitly document security-type inclusion/exclusion rules.

---

# 10. Canonical daily price data

Daily normalized price records should contain roughly:

```text
date
security_id
open
high
low
close
adjusted_close
volume
daily_return
```

Depending on the source, retain raw and adjusted fields separately where useful.

Corporate-action handling must prevent stock splits or similar adjustments from becoming false crash events.

Delisting behavior must be preserved/documented rather than silently discarded.

There must be at most one canonical record per `(security_id, trading_date)`.

---

# 11. Crash-event table

Every qualifying day creates a crash event.

Minimum fields:

```text
event_id
security_id
company_id
ticker_as_of_event
crash_date
crash_return
crash_close
```

The threshold should live in configuration:

```yaml
crash_threshold: -0.10
```

No cooldown.

No event suppression based on later recovery or later crashes.

---

# 12. Recent-crash history — first-class feature family

The model should explicitly know whether the stock has recently crashed.

For each current crash event, calculate from data strictly **before the current crash day**:

```text
prior_crash_count_5d
prior_crash_count_20d
prior_crash_count_60d

days_since_previous_crash
previous_crash_return

return_since_previous_crash
max_rebound_since_previous_crash

cumulative_return_5d
cumulative_return_20d
cumulative_return_60d

drawdown_from_20d_high
drawdown_from_60d_high
drawdown_from_52w_high
```

Important invariant:

> Today's crash must never count as one of its own prior crashes.

Example:

```text
Day -2: -14%
Day  0: -12%
```

For the Day 0 event:

```text
prior_crash_count_5d = 1
days_since_previous_crash = 2
previous_crash_return = -0.14
```

A fresh first crash should have zero prior counts and null/explicit missing values for previous-crash-specific features.

This feature family is important because a first sudden shock may behave differently from a second or third leg down.

---

# 13. Crash-day and pre-crash price features

Candidate features include:

## Crash day

```text
crash_return
opening_gap
intraday_range
close_vs_low
close_vs_open
volume
relative_volume_20d
volatility_20d
volatility_60d
```

## Pre-crash trajectory

```text
return_5d_pre
return_20d_pre
return_60d_pre
return_252d_pre

distance_from_20d_high
distance_from_60d_high
distance_from_52w_high

drawdown_20d
drawdown_60d
drawdown_252d
```

All rolling calculations must be point-in-time safe.

---

# 14. Market and sector context

Do NOT filter out sector-wide or market-wide crashes.

Those events are valuable.

For each crash event, compute context such as:

```text
market_return_1d
market_return_5d
market_volatility
market_regime

sector_return_1d
sector_return_5d
sector_volatility
```

Where practical, calculate historical sector return from the research universe and exclude the focal company from its own sector aggregate.

The goal is not simply to convert everything to abnormal return. We want the model to know both:

- what happened to the individual company, and
- what happened to its environment.

This allows analysis such as:

> If 30 semiconductor companies crashed in the same shock, which individual companies subsequently recovered and what differentiated them?

---

# 15. Point-in-time fundamentals

Candidate raw/derived feature families:

## Growth

```text
revenue_growth_yoy
revenue_growth_qoq
eps_growth_yoy
```

## Profitability

```text
gross_margin
operating_margin
net_margin
free_cash_flow_margin
ROA
ROE
```

## Balance sheet

```text
cash
total_debt
net_debt
current_ratio
debt_to_assets
net_debt_to_ebitda
interest_coverage
```

## Valuation

```text
market_cap
pe
price_to_sales
ev_to_sales
ev_to_ebitda
fcf_yield
```

Valuation metrics that depend on price should use the appropriate event-date market data plus the latest historically available fundamental inputs.

Missing or stale fundamentals should be explicitly flagged rather than silently imputed with future information.

---

# 16. Master event-level research dataset

The canonical V1 modeling artifact should be something like:

```text
events_v1.parquet
```

with **one row per crash prediction event**.

Conceptually:

```text
event_id
security_id
crash_date

# crash features
...

# recent crash / path features
...

# market / sector context
...

# company metadata
...

# point-in-time fundamentals
...

# outcomes / labels
...
```

Important:

- Feature columns and outcome columns must be explicitly separated.
- Dataset version/provenance must be recorded.
- The artifact should be reproducible from raw inputs + config.
- Validate duplicate rows, impossible values, leakage, missingness, and horizon censoring.

---

# 17. Descriptive analysis before ML

Do not jump directly to machine learning.

First estimate base rates.

Primary example:

```text
P(+10% close-based recovery within 20 trading days | crash <= -10%)
```

Also compute the full target grid.

Slice by:

```text
crash magnitude
recent crash count
days since previous crash
company size
profitability
free cash flow
leverage
valuation
sector
market environment
```

Recent crash history deserves explicit analysis, e.g.:

```text
fresh crash
1 prior crash within 20d
2+ prior crashes within 20d
```

Report sample sizes and uncertainty/confidence intervals.

The purpose is to understand whether a meaningful phenomenon exists before asking an ML model to exploit it.

---

# 18. Modeling progression

Use incremental models so the research can identify **which information actually adds predictive value**.

## Model 0 — base rate

Everyone receives the historical recovery base rate.

## Model 1 — price / crash / recent crash

Use:

```text
crash characteristics
pre-crash price path
recent crash history
```

Question:

> Does price behavior alone discriminate recovery probabilities?

## Model 2 — add market / sector context

Use Model 1 plus:

```text
market movement
sector movement
market/sector volatility and regimes
```

Question:

> Does environmental context improve prediction?

## Model 3 — add fundamentals

Use Model 2 plus:

```text
growth
profitability
balance sheet
cash flow
valuation
company size
```

Question:

> Does knowing what kind of company just crashed materially improve recovery prediction?

This Model 2 -> Model 3 comparison is central to the original hypothesis.

---

# 19. Model families

Start with only two model families.

## Logistic regression

Use for:

- interpretability,
- baseline probability modeling,
- coefficient direction,
- detecting broad linear relationships.

## Gradient-boosted trees

Use LightGBM or XGBoost for nonlinear interactions.

Tune on validation data only.

Do not use test results to select features, hyperparameters, or model family.

Do not add neural networks unless later evidence shows a real need.

---

# 20. Chronological validation

Do **not** randomly split rows for primary results.

Use chronological train / validation / test splits.

Example only:

```text
TRAIN        2003-2018
VALIDATION   2019-2021
TEST         2022-2026
```

Actual periods must be selected based on final data coverage.

Because outcomes extend 5/20/60 trading days into the future, consider an embargo/boundary policy so outcome windows do not improperly cross split boundaries.

Record all split definitions in configuration and model metadata.

The held-out test period should remain untouched until model and feature choices are finalized.

---

# 21. Evaluation

The project is initially about **probability prediction**, not a full trading backtest.

Primary metrics:

```text
Brier score
log loss
ROC-AUC
PR-AUC
calibration curves / calibration error
top-decile recovery rate
top-decile lift over historical base rate
```

A key qualitative result would look like:

```text
Predicted bucket    Actual recovery rate
0-20%               ~17%
20-40%              ~31%
40-60%              ~52%
60-80%              ~69%
80-100%             ~84%
```

The exact numbers are unknown; this is only an illustration.

We care especially about whether predicted probabilities are calibrated.

Also compare:

```text
Model 0
vs Model 1
vs Model 2
vs Model 3
```

and explicitly quantify the incremental value of:

- recent crash history,
- market/sector context,
- fundamentals.

---

# 22. Repeated-crash dependence and robustness

Since each qualifying crash is its own event, observations close together for the same company are correlated.

Do not "fix" this by deleting them from the primary event definition.

Instead, quantify robustness.

Analyze:

```text
fresh crashes
events with 1 prior crash
events with 2+ prior crashes
```

and sensitivity to:

```text
crash threshold = -10%, -15%, -20%, -30%
recovery threshold = +5%, +10%, +20%
horizon = 5d, 20d, 60d
```

Where useful, use clustered/company-aware uncertainty estimates or block-bootstrap approaches.

Document overlapping event windows and dependence caveats.

---

# 23. Explainability

For individual predictions, provide interpretable contributors.

Logistic regression:

- coefficient-based explanation.

Tree model:

- SHAP or equivalent.

Example desired output:

```text
Predicted P(+10% within 20d): 73%

Positive contributors:
- positive FCF
- low leverage
- strong prior momentum
- sector-wide selloff

Negative contributors:
- declining revenue
- high valuation
- repeated recent crashes
```

Do not over-interpret SHAP as causal evidence.

---

# 24. V2 — news / crash-cause / fundamental-damage modeling

V2 begins only after V1 is working.

## Motivation

Fundamentals answer:

> What kind of company was this?

But contemporaneous event information can answer:

> What new information caused today's repricing?

A 25% decline caused by a temporary regulatory delay may have very different implications from a 25% decline caused by a failed pivotal clinical trial.

## Retrieval requirements

For each historical crash, retrieve only documents that were available by the prediction timestamp:

```text
SEC filings
earnings releases
company press releases
timestamped historical news
```

Do not use present-day search results containing hindsight as if they were contemporaneous information.

Preserve:

```text
source_id / URL
publication timestamp
source type
retrieval metadata
```

## LLM role

Do NOT ask the LLM:

> Should we buy this stock?

Use the LLM as a structured event-understanding extractor.

Potential schema:

```json
{
  "event_type": "earnings",
  "primary_cause": "guidance_cut",
  "revenue_impact": "moderate",
  "margin_impact": "low",
  "balance_sheet_impact": "none",
  "business_thesis_changed": false,
  "temporary_vs_structural": "temporary",
  "uncertainty": "medium"
}
```

Every assessment should preserve evidence/source references where possible.

Prompts and schemas must be versioned.

## Model 4

Add LLM-derived event/fundamental-damage features to the best V1 model and measure incremental value using the same chronological methodology.

Research question:

> Does structured understanding of why the crash happened improve recovery prediction beyond prices, context, and fundamentals?

---

# 25. Explicit V1 non-goals

Do not scope-creep V1 into:

- brokerage integration,
- automated execution,
- portfolio optimization,
- position sizing,
- transaction-cost simulation,
- live alerting,
- frontend/web app,
- options strategies,
- neural networks,
- news scraping,
- direct LLM stock recommendations.

Those can be considered only after V1 produces credible evidence.

---

# 26. Linear project

There is a dedicated Linear project:

```text
Stock Crash Recovery Prediction
Team: Studio R42
```

Milestones:

```text
M1 — Data Source & Research Foundation
M2 — Security Master & Historical Prices
M3 — Crash Events & Recovery Labels
M4 — Feature Engineering
M5 — Point-in-Time Fundamentals
M6 — Master Dataset & Descriptive Analysis
M7 — Predictive Modeling
M8 — Evaluation, Robustness & Explainability
M9 — V2 News & Fundamental-Damage Modeling
```

Current implementation issues are numbered approximately:

```text
STU-43 through STU-67
```

Important issues include:

```text
STU-43  Validate WRDS access and historical data coverage
STU-44  Initialize repository and research architecture
STU-45  Define provider-neutral market data interfaces
STU-46  Design Supabase metadata and experiment schema

STU-47  Build survivorship-safe security master
STU-48  Ingest and normalize historical daily prices

STU-49  Implement crash-event detection
STU-50  Generate recovery labels and continuous outcomes

STU-51  Engineer crash-day and pre-crash price features
STU-52  Engineer recent-crash history features
STU-53  Engineer market and sector context features

STU-54  Ingest and normalize point-in-time fundamentals
STU-55  Join fundamentals to crash events without leakage

STU-56  Assemble canonical V1 event dataset
STU-57  Establish recovery base rates and descriptive slices

STU-58  Implement chronological train/validation/test splitting
STU-59  Train baseline and logistic regression models
STU-60  Train gradient-boosted recovery models
STU-61  Compare incremental predictive value across feature stages

STU-62  Evaluate held-out test performance and calibration
STU-63  Run robustness and repeated-crash dependence analysis
STU-64  Add per-event model explainability

STU-65  Define contemporaneous crash-news retrieval pipeline
STU-66  Extract structured crash-cause and fundamental-damage features with an LLM
STU-67  Train and evaluate Model 4 with event-understanding features
```

The Linear issues contain more detailed acceptance criteria.

Recommended initial execution order:

```text
STU-43 -> STU-44 -> STU-45 -> STU-47 -> STU-48
```

STU-43 is complete (WRDS access validated, 2026-08-11). Next up: STU-44 (repo init) and
STU-45 (provider-neutral interfaces — where the CRSP CIZ column normalization lands).

---

# 27. Engineering principles for Claude Code

When working on this repository:

1. **Inspect before editing.**
   - Read existing code, configs, README, and tests first.
   - Do not assume this document exactly matches implementation if the repo has evolved.

2. **Preserve point-in-time correctness.**
   - Treat leakage as a correctness bug, not a modeling nuance.

3. **Prefer deterministic, testable pipelines.**
   - Data transformations should be reproducible from source/version/config.

4. **Keep provider concerns isolated.**
   - Normalize upstream data once; do not spread vendor-specific column names everywhere.

5. **Use stable identifiers.**
   - Never rely on ticker alone when the source provides a better security/company identifier.

6. **Write tests around the dangerous parts.**
   Highest-priority unit/integration tests:
   - corporate-action-adjusted return handling,
   - crash detection,
   - repeated consecutive crash events,
   - recovery-label calculations,
   - censored forward horizons,
   - recent-crash counts,
   - rolling-window boundaries,
   - point-in-time/as-of fundamental joins,
   - chronological split boundaries.

7. **Do not prematurely optimize.**
   - Correctness and auditability are more important than shaving seconds off the first research pipeline.

8. **Prefer reusable source code over notebook-only logic.**
   - Notebooks should call tested library functions.

9. **Make assumptions explicit in config.**
   Examples:
   ```text
   crash threshold
   liquidity filters
   target threshold
   horizon
   sector aggregation rule
   split dates
   missing-data policy
   ```

10. **Never silently drop problematic records.**
    - Log/flag why an event/security was excluded.

11. **Do not build a trading system unless explicitly requested.**
    - The immediate deliverable is research quality and calibrated predictive value.

12. **If a methodological choice is ambiguous, surface it.**
    - Explain the tradeoff before hard-coding a choice that could materially alter results.

---

# 28. Current immediate goal

Unless the repository state indicates otherwise, the immediate goal is:

> Build the research foundation now that WRDS access is validated (STU-43 done).

A sensible next sequence is:

1. Inspect the repository.
2. If not already initialized, implement STU-44.
3. Define canonical Pydantic/schema contracts and provider abstraction from STU-45.
4. Add small synthetic fixtures (or the STU-43 WRDS sample) only for testing the crash/label logic if useful.
5. Do not pretend prototype data is the final research dataset.
6. Implement/validate the WRDS adapter and canonical security/price extraction against the CRSP CIZ tables.
7. If WRDS ever becomes unavailable, evaluate Tiingo vs Massive+SEC vs EODHD+SEC without changing downstream contracts.

---

# 29. Desired project outcome

A successful V1 should be able to answer, defensibly:

> Among historical US stock crashes, what is the base probability of a meaningful short-term rebound, and how much do crash behavior, recent crash history, market/sector context, and company fundamentals improve our ability to predict which individual stocks recover?

The strongest result is not necessarily a profitable trading strategy.

A meaningful finding could also be:

- fundamentals add substantial predictive value,
- recent crash history is surprisingly important,
- sector context matters little,
- simple price features dominate,
- fundamentals add almost no signal,
- probability estimates are poorly calibrated,
- the hypothesis does not survive chronological testing.

Negative results are valid if the methodology is sound.

The project should optimize for **credible evidence**, not for manufacturing an exciting conclusion.
