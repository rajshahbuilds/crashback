# STU-59 — Baseline & Logistic Recovery Models (M0–M3)

Primary target **`hit_10pct_20d`** = P(close ≥ +10% within 20 trading days of the crash close).
Trained on the CLEAN determined pool (`in_universe_at_event & passes_min_price & not
censored_20d & label present`). Splits from `splits_v1` (STU-58): fit on **train**,
report on **validation** — the **test** period is untouched. Everything is config-driven
(`configs/default.yaml` → `models:`), reproducible via `scripts/train_models.py`.

## Incremental stages (CLAUDE.md §18)

- **model0** — historical base rate (predicts train prevalence for everyone).
- **model1** — crash-day + pre-crash price path + recent-crash history.
- **model2** — model1 + market / sector context.
- **model3** — model2 + point-in-time fundamentals.

Stages are nested (each a superset of the previous), so the row-to-row change isolates the
value of the added information source. Logistic pipeline = median impute (+ missingness
indicator) → standardize → L2 logistic (C=1.0).

## Validation metrics

Base rate (Model 0 prevalence on validation) = **0.6005**. Lower log loss / Brier is
better; higher AUC / PR-AUC is better; ECE closer to 0 is better-calibrated.

| stage | features | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| model0 — historical base rate | 0 | 0.6889 | 0.2479 | — | 0.6005 | 0.0893 |
| model1 — price / crash / recent-crash | 26 | 0.6499 | 0.2285 | 0.6574 | 0.7247 | 0.0744 |
| model2 — + market & sector context | 34 | 0.6128 | 0.2133 | 0.6984 | 0.7827 | 0.0437 |
| model3 — + point-in-time fundamentals | 52 | 0.6163 | 0.2140 | 0.6966 | 0.7795 | 0.0402 |

Δ log loss vs Model 0 quantifies incremental value; the STU-61 ticket formalizes the
M0→M1→M2→M3 comparison. Note the validation base rate differs from the train base rate — the
splits are chronological, so this is genuine regime drift, not leakage.

## Model 3 — top standardized coefficients (|coef|, direction)

Standardized so magnitudes are comparable; sign is the direction of association with recovery.
`<feature>__missing` rows are the informativeness of a feature being absent. Not causal.

| feature | coef (std) | direction |
|---|---|---|
| distance_from_52w_high | -0.684 | decreases P(recover) |
| drawdown_252d | +0.584 | increases P(recover) |
| drawdown_20d | -0.556 | decreases P(recover) |
| close_vs_low | -0.430 | decreases P(recover) |
| distance_from_20d_high | +0.389 | increases P(recover) |
| max_rebound_since_previous_crash | +0.289 | increases P(recover) |
| market_volatility_20d | +0.241 | increases P(recover) |
| ev_to_ebitda__missing | -0.207 | decreases P(recover) |
| net_debt_to_ebitda__missing | +0.188 | increases P(recover) |
| return_since_previous_crash | -0.179 | decreases P(recover) |
| gross_margin__missing | -0.176 | decreases P(recover) |
| operating_margin | +0.149 | increases P(recover) |
| market_return_1d | -0.148 | decreases P(recover) |
| net_margin | -0.146 | decreases P(recover) |
| close_vs_open | -0.138 | decreases P(recover) |

## Artifacts

Per stage in `data/models/` (gitignored): `{stage}_{family}_v1.joblib` (fitted
pipeline / base-rate), `{stage}_coef_v1.parquet` (M1–M3), `{stage}_calibration_val_
v1.parquet`. Plus `model_metrics_v1.parquet` (long-form, all stages × splits)
and `run_meta_v1.json` (config, hyperparams, git commit, pool sizes). Registered in
Supabase `model_runs` + `metrics`.
