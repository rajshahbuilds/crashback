# STU-61 — Incremental Predictive Value Across Feature Stages

How much does each added information source actually improve recovery prediction for the primary
target **`hit_10pct_20d`**? Evaluated on the **validation** period (STU-58 splits); **test** untouched.
A fine ablation ladder splits price from recent-crash so each source gets a clean marginal
contribution. Uncertainty is a **paired clustered bootstrap** (500 resamples of *securities*,
not events — CLAUDE.md §22), so a delta is "material" only if its 95% CI excludes 0.

Focus is **discrimination + calibration**, not trading P&L. Note (as in STU-59/60) the tree
rungs were tuned on validation, so absolute levels are mildly optimistic; the *deltas* between
nested rungs are the robust quantity here, and the held-out test read is STU-62.

## Logistic — ladder (validation)

| rung | features | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| base | 0 | 0.6889 | 0.2479 | — | 0.6005 | 0.0893 |
| price | 19 | 0.6544 | 0.2297 | 0.6532 | 0.7202 | 0.0762 |
| price+recent | 26 | 0.6499 | 0.2285 | 0.6574 | 0.7247 | 0.0744 |
| +market | 34 | 0.6128 | 0.2133 | 0.6984 | 0.7827 | 0.0437 |
| +fundamentals | 52 | 0.6163 | 0.2140 | 0.6966 | 0.7795 | 0.0402 |

### Marginal value of each source (Δ vs previous rung, 95% CI)

| added source | Δ log loss (95% CI) | Δ ROC-AUC (95% CI) | verdict |
|---|---|---|---|
| price / crash path | -0.0345 ([-0.0402, -0.0278]) | +0.1532 ([+0.1443, +0.1618]) | **improves** |
| recent-crash history | -0.0045 ([-0.0069, -0.0026]) | +0.0041 ([+0.0021, +0.0061]) | **improves** |
| market / sector context | -0.0371 ([-0.0411, -0.0323]) | +0.0411 ([+0.0350, +0.0468]) | **improves** |
| fundamentals | +0.0035 ([+0.0012, +0.0073]) | -0.0019 ([-0.0030, -0.0008]) | **hurts** |

## Lightgbm — ladder (validation)

| rung | features | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| base | 0 | 0.6889 | 0.2479 | — | 0.6005 | 0.0893 |
| price | 19 | 0.6206 | 0.2157 | 0.6959 | 0.7699 | 0.0500 |
| price+recent | 26 | 0.6215 | 0.2160 | 0.6954 | 0.7674 | 0.0509 |
| +market | 34 | 0.6151 | 0.2144 | 0.6985 | 0.7824 | 0.0476 |
| +fundamentals | 52 | 0.6112 | 0.2132 | 0.7070 | 0.7957 | 0.0551 |

### Marginal value of each source (Δ vs previous rung, 95% CI)

| added source | Δ log loss (95% CI) | Δ ROC-AUC (95% CI) | verdict |
|---|---|---|---|
| price / crash path | -0.0683 ([-0.0726, -0.0644]) | +0.1959 ([+0.1880, +0.2041]) | **improves** |
| recent-crash history | +0.0009 ([+0.0003, +0.0015]) | -0.0005 ([-0.0020, +0.0011]) | **hurts** |
| market / sector context | -0.0064 ([-0.0104, -0.0017]) | +0.0030 ([-0.0024, +0.0087]) | **improves** |
| fundamentals | -0.0039 ([-0.0049, -0.0028]) | +0.0085 ([+0.0071, +0.0100]) | **improves** |

## Xgboost — ladder (validation)

| rung | features | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| base | 0 | 0.6889 | 0.2479 | — | 0.6005 | 0.0893 |
| price | 19 | 0.6209 | 0.2158 | 0.6940 | 0.7661 | 0.0486 |
| price+recent | 26 | 0.6209 | 0.2157 | 0.6974 | 0.7703 | 0.0496 |
| +market | 34 | 0.6054 | 0.2115 | 0.7074 | 0.8011 | 0.0563 |
| +fundamentals | 52 | 0.6074 | 0.2123 | 0.7057 | 0.7991 | 0.0570 |

### Marginal value of each source (Δ vs previous rung, 95% CI)

| added source | Δ log loss (95% CI) | Δ ROC-AUC (95% CI) | verdict |
|---|---|---|---|
| price / crash path | -0.0680 ([-0.0723, -0.0640]) | +0.1940 ([+0.1860, +0.2022]) | **improves** |
| recent-crash history | -0.0000 ([-0.0007, +0.0006]) | +0.0034 ([+0.0020, +0.0049]) | no material effect (CI spans 0) |
| market / sector context | -0.0155 ([-0.0199, -0.0107]) | +0.0100 ([+0.0040, +0.0156]) | **improves** |
| fundamentals | +0.0020 ([+0.0014, +0.0025]) | -0.0017 ([-0.0023, -0.0009]) | **hurts** |

## Cross-family synthesis (Δ log loss verdict per source)

| added source | logistic | lightgbm | xgboost |
|---|---|---|---|
| price / crash path | -0.0345 — improves | -0.0683 — improves | -0.0680 — improves |
| recent-crash history | -0.0045 — improves | +0.0009 — hurts | -0.0000 — no material effect |
| market / sector context | -0.0371 — improves | -0.0064 — improves | -0.0155 — improves |
| fundamentals | +0.0035 — hurts | -0.0039 — improves | +0.0020 — hurts |

## Conclusions

- **Δ log loss < 0 = improvement** (lower is better); **Δ ROC-AUC > 0 = improvement**. A source
  is "material" only when its 95% clustered-bootstrap CI excludes 0; the sign of Δ then says
  whether it helps or hurts. Each row is the *marginal* value of that source given everything
  below it on the ladder.
- **Agreement across model families** is the real test: a source carries robust short-term
  recovery signal only if the sign of its effect is consistent whether the model is linear
  (logistic) or nonlinear (LightGBM, XGBoost). Where the families disagree on sign, the effect
  is not a dependable phenomenon — it is model-specific noise.