# STU-62 — Held-Out Test Evaluation

**First and only** evaluation on the untouched chronological **test** period (crash dates
2022–2025), primary target **`hit_10pct_20d`**. The model was **locked before any test read**: on
validation-only results (STU-59/60/61) the final model is **XGBoost model2** (price +
recent-crash + market/sector, 34 features), selected by validation log loss. Test pool =
18,856 CLEAN determined events. Reproducible from the saved train-only booster
(`data/models/model2_xgboost_v1.joblib`); no retraining.

## Headline: does the validation signal generalize?

| model | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|
| M0 — base rate (reference) | 0.6936 | 0.2502 | — | 0.4954 | 0.0157 |
| **XGBoost model2 (raw)** — locked final | 0.6722 | 0.2399 | 0.6041 | 0.5863 | 0.0198 |
| XGBoost model2 (isotonic, cal. on val) | 0.6774 | 0.2416 | 0.5888 | 0.5528 | 0.0296 |

- **ROC-AUC on test = 0.6041** vs validation 0.7074. The model still
  beats the M0 baseline out-of-sample on every proper score, **but discrimination is modest
  (~0.60) and far below the validation figure — most of the validation edge did not transfer.**
- **Brier skill score vs base rate = 0.041** (4.1% reduction in Brier vs
  M0). Log loss 0.6722 vs M0 0.6936 — a small but genuine improvement.
- **Calibration is a genuine positive**: the raw model's test ECE is 0.0198
  and predicted ≈ observed across the reliability table below.
- Post-hoc isotonic calibration (fit on validation) does not improve test ECE
  (0.0198 → 0.0296) — it was fit on a
  higher-base-rate regime (validation 0.600 vs test 0.495) and did not transfer. The raw
  probabilities are already well-calibrated here, so no correction is applied.

**Regime shift.** The test period (2022–2025) has a lower recovery base rate (0.495) than
train (0.511) or validation (0.600), and the model's dominant features are
market-context (STU-60 importance). Regime-dependent market relationships are the most likely
reason discrimination attenuated out-of-sample.

## Reliability — predicted vs observed recovery (raw model, equal-width bins)

| predicted bucket | n | mean predicted | observed recovery |
|---|---|---|---|
| 0.0–0.1 | 0 | — | — |
| 0.1–0.2 | 145 | 0.181 | 0.172 |
| 0.2–0.3 | 999 | 0.258 | 0.251 |
| 0.3–0.4 | 2,058 | 0.354 | 0.337 |
| 0.4–0.5 | 4,374 | 0.460 | 0.479 |
| 0.5–0.6 | 7,961 | 0.548 | 0.536 |
| 0.6–0.7 | 2,937 | 0.636 | 0.586 |
| 0.7–0.8 | 369 | 0.733 | 0.756 |
| 0.8–0.9 | 13 | 0.813 | 0.923 |
| 0.9–1.0 | 0 | — | — |

Well-calibrated ⇔ mean-predicted ≈ observed within each row.

## Recovery rate by predicted-probability decile (equal count)

Base rate (test) = **0.4954**; historically-available base rate (train) = 0.5111.

| decile (1=low→10=high) | n | mean predicted | observed recovery | lift |
|---|---|---|---|---|
| 1 | 1,886 | 0.277 | 0.264 | 0.53× |
| 2 | 1,886 | 0.384 | 0.385 | 0.78× |
| 3 | 1,885 | 0.448 | 0.459 | 0.93× |
| 4 | 1,886 | 0.486 | 0.511 | 1.03× |
| 5 | 1,885 | 0.511 | 0.523 | 1.05× |
| 6 | 1,886 | 0.533 | 0.531 | 1.07× |
| 7 | 1,886 | 0.555 | 0.559 | 1.13× |
| 8 | 1,885 | 0.579 | 0.537 | 1.08× |
| 9 | 1,886 | 0.610 | 0.530 | 1.07× |
| 10 | 1,885 | 0.673 | 0.655 | 1.32× |

- **Top-decile recovery rate = 0.655**, a
  **1.32× lift** over the 0.495 base rate
  (n=1,885, mean predicted 0.673).
- Observed recovery **broadly increases** across deciles, clearest at the tails (bottom decile
  0.264 → top decile 0.655, a 2.5× spread); the middle
  deciles are muddy. So even with AUC ~0.60 the model still usefully concentrates recoveries in
  its top-scored events and flags the least-likely ones — the practical payoff survives, attenuated.

## Confidence-band distribution (natural bins)

Equal-count deciles hide *how often* the model is actually confident. In its natural 0.1-wide
bands the model's probabilities never leave [0.14, 0.83] — it stays near the base rate on the
bulk of events and only rarely commits. Where it does commit, calibration holds (mean predicted
≈ observed).

| predicted band | n | % of test | mean predicted | observed recovery |
|---|---|---|---|---|
| 0.10–0.20 | 145 | 0.8% | 0.181 | 0.172 |
| 0.20–0.30 | 999 | 5.3% | 0.258 | 0.251 |
| 0.30–0.40 | 2,058 | 10.9% | 0.354 | 0.337 |
| 0.40–0.50 | 4,374 | 23.2% | 0.460 | 0.479 |
| 0.50–0.60 | 7,961 | 42.2% | 0.548 | 0.536 |
| 0.60–0.70 | 2,937 | 15.6% | 0.636 | 0.586 |
| 0.70–0.80 | 369 | 2.0% | 0.733 | 0.756 |
| 0.80–0.90 | 13 | 0.1% | 0.813 | 0.923 |

## Expected return by confidence band ⚠️ (recovery ≠ return)

The recovery *label* ("closes ≥ +10% at some point in 20d") hides the **downside when it fails**.
Joining the continuous `return_20d` outcome shows that **recovery probability is NOT monotonic
with expected return**: the moderately-confident bands are the *worst* economically, because the
losers there fall much harder than the winners rise.

| predicted band | n | P(+10%) | mean return | return if win | return if lose | mean max drawdown |
|---|---|---|---|---|---|---|
| 0.10–0.20 | 145 | 0.172 | -0.7% | +12.7% | -3.5% | -5.6% |
| 0.20–0.30 | 999 | 0.251 | +0.2% | +11.4% | -3.5% | -6.1% |
| 0.30–0.40 | 2,058 | 0.337 | +0.1% | +13.1% | -6.6% | -8.3% |
| 0.40–0.50 | 4,374 | 0.479 | -1.0% | +13.4% | -14.2% | -14.0% |
| 0.50–0.60 | 7,961 | 0.536 | -3.3% | +12.9% | -22.0% | -19.4% |
| 0.60–0.70 | 2,937 | 0.586 | -4.5% | +10.5% | -25.6% | -21.5% |
| 0.70–0.80 | 369 | 0.756 | +8.8% | +18.1% | -19.8% | -11.2% |
| 0.80–0.90 | 13 | 0.923 | +14.6% | +16.0% | -2.2% | +2.4% |

- Worst-EV band is **0.60–0.70** (mean return -4.5%, loser return -25.6%) — *more*
  confident of recovery than average, yet negative expected return. High predicted-recovery names
  are high-volatility names; when they don't bounce they keep falling.
- Only the **extreme top band** turns clearly EV-positive, where the high win-rate finally
  overwhelms the asymmetry. **A confidence-weighted decision must be driven by expected return /
  downside, not by P(recovery)** — the two diverge. (Descriptive, gross of costs; §25 keeps
  position-sizing / trading out of V1 scope.)

## Notes

- Test integrity: every prior script filtered to `split=='validation'`; this is the first read
  of `split=='test'`. Model/feature/hyperparameter choices were fixed beforehand.
- Focus is calibrated probability + discrimination, not trading P&L (§21). Dependence across
  same-security events understates CI width; block-bootstrap robustness is STU-63.
- Artifacts: `data/models/test_predictions_v1.parquet`,
  `test_reliability_raw/cal_v1.parquet`, `test_deciles_v1.parquet`,
  `test_eval_v1.json`. Test metrics registered in Supabase.
