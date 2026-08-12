# STU-60 (follow-on) — XGBoost vs LightGBM vs Logistic

XGBoost cross-check on the same staged features / CLEAN determined pool / STU-58 splits, primary
target **`hit_10pct_20d`**. Same discipline as LightGBM: predeclared grid searched on **validation**
only, early stopping on validation, best per stage by validation
**log_loss**; **test** untouched. XGBoost handles missing natively
(`missing=nan`); `inf` routed to missing; class prevalence preserved.

> ⚠️ Validation is used for both selection and reporting, so these are mildly optimistic. The
> unbiased estimate is the held-out **test** set (STU-62).

## Validation metrics — three families side by side

| stage | model | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| **model1 — price / crash / recent-crash** | **XGBoost** | 0.6209 | 0.2157 | 0.6974 | 0.7703 | 0.0496 |
| | LightGBM | 0.6215 | 0.2160 | 0.6954 | 0.7674 | 0.0509 |
| | logistic | 0.6499 | 0.2285 | 0.6574 | 0.7247 | 0.0744 |
| **model2 — + market & sector context** | **XGBoost** | 0.6054 | 0.2115 | 0.7074 | 0.8011 | 0.0563 |
| | LightGBM | 0.6151 | 0.2144 | 0.6985 | 0.7824 | 0.0476 |
| | logistic | 0.6128 | 0.2133 | 0.6984 | 0.7827 | 0.0437 |
| **model3 — + point-in-time fundamentals** | **XGBoost** | 0.6074 | 0.2123 | 0.7057 | 0.7991 | 0.0570 |
| | LightGBM | 0.6112 | 0.2132 | 0.7070 | 0.7957 | 0.0551 |
| | logistic | 0.6163 | 0.2140 | 0.6966 | 0.7795 | 0.0402 |

## XGBoost selected hyperparameters (per stage, by validation log_loss)

- **model1**: {'max_depth': 3, 'min_child_weight': 10.0, 'colsample_bytree': 0.7, 'reg_lambda': 1.0}, best_iteration=267
- **model2**: {'max_depth': 3, 'min_child_weight': 10.0, 'colsample_bytree': 0.7, 'reg_lambda': 0.0}, best_iteration=199
- **model3**: {'max_depth': 3, 'min_child_weight': 10.0, 'colsample_bytree': 0.7, 'reg_lambda': 0.0}, best_iteration=214

## XGBoost feature importance — model2 (best XGB stage), top 15 by gain

| feature | gain share | # splits |
|---|---|---|
| market_return_20d | 21.9% | 226 |
| market_volatility_20d | 17.0% | 269 |
| market_return_1d | 11.2% | 151 |
| relative_volume_20d | 8.0% | 48 |
| volatility_20d | 6.6% | 56 |
| market_return_5d | 5.1% | 130 |
| days_since_previous_crash | 4.9% | 42 |
| volatility_60d | 4.5% | 62 |
| drawdown_60d | 3.3% | 44 |
| distance_from_60d_high | 2.4% | 38 |
| drawdown_20d | 2.4% | 42 |
| opening_gap | 1.6% | 38 |
| return_20d_pre | 1.4% | 35 |
| close_vs_open | 1.1% | 47 |
| return_60d_pre | 1.1% | 70 |

## Takeaway

XGBoost and LightGBM are the same model class (gradient-boosted trees) and should land close;
any gap is implementation/hyperparameter-grid differences, not a fundamentally different signal.
The three-way table isolates **linear (logistic) vs nonlinear (both GBMs)** as the real
methodological axis. Artifacts per stage in `data/models/`: `{stage}_xgboost_v1.json` +
`.joblib`, `{stage}_xgb_importance/_xgb_trials/_xgb_calibration_val_v1.parquet`, plus
`xgb_metrics_v1.parquet` and `run_meta_xgb_v1.json`. Registered in Supabase
`model_runs` (family=xgboost) + `metrics`.
