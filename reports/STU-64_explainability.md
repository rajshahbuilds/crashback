# STU-64 — Per-Event Explainability (XGBoost model2)

Explanations for the locked final model on the held-out test set (18,856 events), primary
target **`hit_10pct_20d`**. Both methods are **additive in log-odds space**: per-feature
contributions + a bias sum to the raw margin, and `sigmoid(margin)` = predicted P(recover).
Positive ⇒ pushes toward recovery, negative ⇒ against.

- **Tree model:** exact **TreeSHAP** via XGBoost's native `pred_contribs` (no external `shap`
  dependency).
- **Logistic model:** per-event term `coef · standardized_value`, shown as a cross-check.
- Values shown are the **exact point-in-time features** recorded at the crash close.

Per-event SHAP for *every* test event is saved to `data/models/test_shap_v1.parquet`
(`shap__<feature>` columns + `shap_bias` + `pred_prob`), so any single prediction can be
decomposed into ranked contributions. SHAP is descriptive attribution, **not causal**.

## Representative cases (most-confident TP / FP / TN / FN)

### True Positive (correctly predicted recovery)

**COIN** on 2022-05-11 — predicted P(recover) = **0.832**, actual =
**recovered** (event `20892_20220511`). Base value (bias) = +0.045 log-odds.

Top drivers **up** (toward recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| market_return_20d | -0.1567 | +0.413 |
| drawdown_60d | -0.7429 | +0.163 |
| market_return_5d | -0.1281 | +0.155 |
| market_volatility_20d | 0.0191 | +0.148 |
| drawdown_20d | -0.6529 | +0.122 |
| sector_n_members | 604 | +0.105 |

Top drivers **down** (against recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| relative_volume_20d | 9.316 | -0.069 |
| market_return_1d | -0.03069 | -0.064 |
| opening_gap | -0.2486 | -0.022 |
| close_vs_low | 0.07119 | -0.016 |
| sector_return_1d | -0.03783 | -0.010 |
| close_vs_open | -0.02051 | -0.004 |

Logistic model's strongest terms for the same event: drawdown_20d (+1.96), distance_from_52w_high (+1.47), distance_from_60d_high (+0.21), distance_from_20d_high (-1.29).

### False Positive (predicted recover, did not)

**BLZE** on 2022-05-11 — predicted P(recover) = **0.806**, actual =
**did not recover** (event `22287_20220511`). Base value (bias) = +0.045 log-odds.

Top drivers **up** (toward recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| market_return_20d | -0.1567 | +0.407 |
| market_return_5d | -0.1281 | +0.162 |
| market_volatility_20d | 0.0191 | +0.121 |
| sector_n_members | 604 | +0.119 |
| drawdown_60d | -0.6105 | +0.109 |
| drawdown_20d | -0.4397 | +0.097 |

Top drivers **down** (against recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| market_return_1d | -0.03069 | -0.085 |
| crash_volume | 1.74e+05 | -0.015 |
| sector_return_1d | -0.03803 | -0.010 |
| close_vs_low | 0.02306 | -0.006 |
| return_252d_pre | — | -0.003 |
| drawdown_252d | — | -0.002 |

Logistic model's strongest terms for the same event: drawdown_20d (+0.92), market_return_5d (+0.20), distance_from_60d_high (+0.14), distance_from_20d_high (-0.60).

### True Negative (correctly predicted no recovery)

**RTX** on 2023-07-25 — predicted P(recover) = **0.135**, actual =
**did not recover** (event `17830_20230725`). Base value (bias) = +0.045 log-odds.

Top drivers **up** (toward recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| previous_crash_return | -0.1448 | +0.001 |
| prior_crash_count_60d | 0 | +0.000 |

Top drivers **down** (against recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| volatility_20d | 0.008142 | -0.286 |
| days_since_previous_crash | 843 | -0.193 |
| relative_volume_20d | 12.31 | -0.190 |
| market_volatility_20d | 0.008467 | -0.175 |
| volatility_60d | 0.01041 | -0.168 |
| market_return_20d | 0.05715 | -0.114 |

Logistic model's strongest terms for the same event: drawdown_252d (+0.95), distance_from_20d_high (+0.53), crash_return (+0.09), distance_from_52w_high (-1.07).

### False Negative (predicted no recovery, but did)

**CLH** on 2024-10-30 — predicted P(recover) = **0.146**, actual =
**recovered** (event `11809_20241030`). Base value (bias) = +0.045 log-odds.

Top drivers **up** (toward recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| previous_crash_return | -0.1058 | +0.001 |
| sector_volatility_20d | 0.009176 | +0.000 |
| prior_crash_count_60d | 0 | +0.000 |

Top drivers **down** (against recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
| volatility_20d | 0.01011 | -0.285 |
| relative_volume_20d | 6.864 | -0.189 |
| days_since_previous_crash | 1153 | -0.182 |
| volatility_60d | 0.01259 | -0.166 |
| market_return_20d | 0.03425 | -0.145 |
| market_volatility_20d | 0.008849 | -0.134 |

Logistic model's strongest terms for the same event: drawdown_252d (+1.14), distance_from_20d_high (+0.53), crash_return (+0.08), distance_from_52w_high (-1.38).


## Global logistic direction (mean |contribution| ranks; sign = direction)

For the linear model, coefficient sign gives a stable global reading of each feature's direction.

| feature | coef (std) | direction |
|---|---|---|
| distance_from_52w_high | -0.681 | ↓ recovery |
| drawdown_20d | -0.560 | ↓ recovery |
| drawdown_252d | +0.580 | ↑ recovery |
| distance_from_20d_high | +0.395 | ↑ recovery |
| volatility_60d | +0.093 | ↑ recovery |
| close_vs_open | -0.141 | ↓ recovery |
| prior_crash_count_20d | -0.122 | ↓ recovery |
| crash_return | +0.127 | ↑ recovery |
| market_volatility_20d | +0.253 | ↑ recovery |
| market_return_1d | -0.144 | ↓ recovery |
| prior_crash_count_60d | +0.099 | ↑ recovery |
| distance_from_60d_high | -0.080 | ↓ recovery |

## Notes

- Reproducible from saved artifacts (`model2_xgboost_v1.joblib`,
  `model2_logistic_v1.joblib`); deterministic.
- Contributions are attributions of *this model's* output, not causal effects on recovery.
- The market-context features dominate individual explanations, consistent with the STU-60
  global importance and the STU-63 finding that broad-market conditions drive most of the signal.
