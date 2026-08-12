# STU-60 — Gradient-Boosted (LightGBM) Recovery Models (M1–M3)

Nonlinear trees on the same staged features as STU-59, primary target **`hit_10pct_20d`**. CLEAN
determined pool; STU-58 splits (fit on **train**, evaluated on **validation**; **test**
untouched). Config-driven (`configs/default.yaml` → `models.lightgbm`), reproducible via
`scripts/train_gbm_models.py`.

## Tuning discipline

The **predeclared** grid (`num_leaves × min_child_samples × feature_fraction × lambda_l2`) is
searched on the **validation** period only, and early stopping watches validation to choose the
boosting-round count. The best model per stage is selected by the predeclared criterion
**validation log_loss** (lower is better) — never test. Class
prevalence is preserved (no oversampling / class weighting). LightGBM handles missing values
natively; `inf` (undefined ratios) is routed to the missing path.

> ⚠️ Because selection *and* reporting both use validation, these validation metrics are mildly
> optimistic. The unbiased estimate is the held-out **test** set (STU-62).

## Validation metrics — LightGBM vs logistic baseline

| stage | features | model | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|
| model1 — price / crash / recent-crash | 26 | LightGBM | 0.6215 | 0.2160 | 0.6954 | 0.7674 | 0.0509 |
| ↳ (logistic STU-59) | 26 | logistic | 0.6499 | 0.2285 | 0.6574 | 0.7247 | 0.0744 |
| model2 — + market & sector context | 34 | LightGBM | 0.6151 | 0.2144 | 0.6985 | 0.7824 | 0.0476 |
| ↳ (logistic STU-59) | 34 | logistic | 0.6128 | 0.2133 | 0.6984 | 0.7827 | 0.0437 |
| model3 — + point-in-time fundamentals | 52 | LightGBM | 0.6112 | 0.2132 | 0.7070 | 0.7957 | 0.0551 |
| ↳ (logistic STU-59) | 52 | logistic | 0.6163 | 0.2140 | 0.6966 | 0.7795 | 0.0402 |

## Selected hyperparameters (per stage, by validation log_loss)

- **model1**: {'num_leaves': 15, 'min_child_samples': 200, 'feature_fraction': 0.7, 'lambda_l2': 0.0}, best_iteration=163
- **model2**: {'num_leaves': 31, 'min_child_samples': 200, 'feature_fraction': 0.7, 'lambda_l2': 0.0}, best_iteration=133
- **model3**: {'num_leaves': 15, 'min_child_samples': 200, 'feature_fraction': 0.7, 'lambda_l2': 0.0}, best_iteration=179

## Feature importance — model3 (best stage), top 15 by gain

Gain = total loss reduction attributed to splits on the feature (normalized); split = number of
times it was used. Importance is descriptive, not causal.

| feature | gain share | # splits |
|---|---|---|
| market_return_20d | 21.0% | 275 |
| market_volatility_20d | 17.3% | 359 |
| market_return_1d | 11.5% | 223 |
| volatility_20d | 7.3% | 64 |
| relative_volume_20d | 5.7% | 48 |
| market_return_5d | 5.5% | 199 |
| volatility_60d | 4.6% | 71 |
| days_since_previous_crash | 3.9% | 53 |
| drawdown_60d | 3.4% | 74 |
| return_20d_pre | 1.6% | 51 |
| opening_gap | 1.5% | 41 |
| drawdown_20d | 1.5% | 48 |
| distance_from_60d_high | 1.4% | 37 |
| close_vs_open | 1.2% | 62 |
| return_60d_pre | 1.2% | 80 |

## Artifacts

Per stage in `data/models/` (gitignored): `{stage}_lightgbm_v1.txt` (portable booster) +
`.joblib`, `{stage}_gbm_importance_v1.parquet`, `{stage}_gbm_trials_v1.parquet`
(every grid point + validation score), `{stage}_gbm_calibration_val_v1.parquet`. Plus
`gbm_metrics_v1.parquet` and `run_meta_gbm_v1.json`. Registered in Supabase
`model_runs` (family=lightgbm) + `metrics`.
