# STU-57 — Recovery Base Rates & Descriptive Slices

**Generated from code** by `scripts/build_descriptive_analysis.py` over `events_v1`
(module `crashback.evaluation.descriptive`). Population: the **CLEAN pool**
(in-universe & ≥ $5), determined events only (censored / no-close labels excluded).
Confidence intervals are Wilson 95%.

> **Dependence caveat:** crashes of the same security close together in time are correlated
> (§22). Wilson CIs assume independent observations and therefore **understate** true
> uncertainty — treat them as a lower bound on sampling noise, not a full accounting of
> event dependence.

## Headline

**P(+10% close within 20 trading days | crash ≤ −10%) = 0.5649**
(n = 267,546; 95% CI [0.5631, 0.5668]).

## Key findings

1. **Recent-crash history is the strongest descriptive signal.** Recovery rises monotonically
   with the number of recent crashes: fresh shocks (0 prior in 20d) recover **0.504**,
   vs **0.698** for 3+ prior — a ~19 pp spread. A repeat leg-down
   is *more* likely to bounce (these are volatile names that whipsaw), not less.
2. **Broad-market crashes recover more than idiosyncratic ones.** Crashing on a day the market
   fell ≥ 5% → **0.679**; crashing on a flat/up market day → **0.538**
   (~14 pp). Consistent with market-driven drops being more
   overreaction than firm-specific damage.
3. **Smaller is bouncier.** Micro-caps (<$300M) recover **0.585** vs large-caps (≥$10B)
   **0.524** — recovery decreases with size.
4. **Crash magnitude and fundamentals add little on their own.** Recovery is roughly flat
   across −10%→−30% crashes; and profitable vs unprofitable firms are nearly identical
   (**0.565** vs **0.569**). A first hint that for *short-term* rebound, price/context
   signals may dominate fundamentals — the M2→M3 question to test in modeling.

## Full recovery grid (close-based)

| threshold | 5d | 20d | 60d |
|---|---|---|---|
| +5% | 0.545 (n=265,240) | 0.724 (n=267,546) | 0.820 (n=266,954) |
| +10% | 0.344 (n=265,240) | 0.565 (n=267,546) | 0.706 (n=266,954) |
| +20% | 0.122 (n=265,240) | 0.311 (n=267,546) | 0.492 (n=266,954) |

Monotone as expected: recovery probability rises with horizon and falls with threshold.

## By crash magnitude

| group | n | recovery | 95% CI |
|---|---|---|---|
| -10% to -15% | 197,592 | **0.560** | [0.558, 0.562] |
| -15% to -20% | 43,829 | **0.582** | [0.577, 0.586] |
| -20% to -30% | 20,409 | **0.580** | [0.573, 0.587] |
| ≤ -30% | 5,716 | **0.561** | [0.548, 0.573] |

## By recent-crash history (fresh vs repeat legs down)

| group | n | recovery | 95% CI |
|---|---|---|---|
| fresh (0 prior/20d) | 157,901 | **0.504** | [0.502, 0.507] |
| 1 prior | 58,004 | **0.626** | [0.622, 0.630] |
| 2 prior | 26,578 | **0.668** | [0.662, 0.674] |
| 3+ prior | 25,063 | **0.698** | [0.692, 0.703] |

## By company size (market cap)

| group | n | recovery | 95% CI |
|---|---|---|---|
| micro (<$300M) | 116,611 | **0.585** | [0.583, 0.588] |
| small ($0.3–2B) | 67,147 | **0.545** | [0.542, 0.549] |
| mid ($2–10B) | 19,633 | **0.527** | [0.520, 0.534] |
| large (≥$10B) | 5,442 | **0.524** | [0.511, 0.537] |

## By profitability (TTM net margin)

| group | n | recovery | 95% CI |
|---|---|---|---|
| unprofitable (margin<0) | 81,979 | **0.569** | [0.565, 0.572] |
| profitable (margin≥0) | 110,595 | **0.565** | [0.562, 0.568] |

## By leverage (debt / assets)

| group | n | recovery | 95% CI |
|---|---|---|---|
| <0.1 | 95,234 | **0.576** | [0.573, 0.579] |
| 0.1–0.3 | 52,212 | **0.559** | [0.554, 0.563] |
| 0.3–0.5 | 34,789 | **0.553** | [0.548, 0.558] |
| ≥0.5 | 25,771 | **0.555** | [0.549, 0.561] |

## By market environment on the crash day

| group | n | recovery | 95% CI |
|---|---|---|---|
| market ≤ -5% (broad crash) | 29,268 | **0.679** | [0.674, 0.685] |
| -5% to -2% | 38,623 | **0.611** | [0.606, 0.616] |
| -2% to 0% | 112,764 | **0.541** | [0.538, 0.544] |
| market ≥ 0% | 86,891 | **0.538** | [0.534, 0.541] |

## By sector (SIC major division, top by n)

| sector | n | recovery | 95% CI |
|---|---|---|---|
| Manufacturing | 107,699 | **0.562** | [0.559, 0.565] |
| Services | 57,450 | **0.562** | [0.557, 0.566] |
| Finance, Insurance & Real Estate | 36,168 | **0.596** | [0.591, 0.601] |
| Transportation & Public Utilities | 18,248 | **0.581** | [0.574, 0.589] |
| Retail Trade | 16,175 | **0.527** | [0.520, 0.535] |
| Wholesale Trade | 9,700 | **0.542** | [0.532, 0.552] |
| Mining | 9,400 | **0.572** | [0.562, 0.582] |
| Nonclassifiable | 9,100 | **0.547** | [0.536, 0.557] |
| Construction | 2,791 | **0.573** | [0.554, 0.591] |
| Agriculture, Forestry & Fishing | 698 | **0.553** | [0.516, 0.590] |
| Public Administration | 117 | **0.564** | [0.474, 0.651] |

## Caveats

- Base rates use close-based labels anchored to the crash-day close; a +10% rebound off a
  depressed close is partly mechanical (mean reversion / bid-ask bounce), which is why the
  ≥ $5 filter matters (penny-stock inclusion pushes the primary rate to ~0.68).
- Fundamentals slices (size/profitability/leverage) cover only events with a Compustat match
  (~80% of CLEAN); missingness is non-random (small/young firms less covered).
- These are **descriptive** associations, not causal or predictive claims. Chronological
  modeling (M7) tests whether they hold out-of-sample.
