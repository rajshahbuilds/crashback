# STU-63 — Robustness & Repeated-Crash Dependence

All sections evaluate **every staged model** (M0 baseline + logistic/lightgbm/xgboost × M1/M2/M3)
on the held-out **test** period (18,856 CLEAN determined events, base rate 0.495),
reusing saved train-only artifacts. Sample sizes shown for every slice. Full matrices (all 10
models) are in the `data/models/robust_*_v1.parquet` artifacts; report tables show the
baseline + validation-selected family-best models for readability.

## 1. Is the signal solidly above chance? (bootstrap 95% CI, clustered by security)

Clustered bootstrap resamples securities (block bootstrap, §22). "Solidly >0.5" = AUC CI lower
bound above 0.5; "beats M0" = log-loss-delta-vs-baseline CI upper bound below 0.

| model | test AUC (95% CI) | AUC >0.5? | Δ log loss vs M0 (95% CI) | beats M0? |
|---|---|---|---|---|
| xgboost_model1 | 0.6117 ([0.6020, 0.6204]) | **yes** | -0.0197 ([-0.0241, -0.0154]) | **yes** |
| lightgbm_model1 | 0.6105 ([0.6011, 0.6195]) | **yes** | -0.0193 ([-0.0236, -0.0147]) | **yes** |
| xgboost_model3 | 0.6063 ([0.5969, 0.6144]) | **yes** | -0.0218 ([-0.0251, -0.0184]) | **yes** |
| lightgbm_model3 | 0.6044 ([0.5951, 0.6130]) | **yes** | -0.0184 ([-0.0221, -0.0143]) | **yes** |
| xgboost_model2 | 0.6041 ([0.5953, 0.6123]) | **yes** | -0.0214 ([-0.0247, -0.0180]) | **yes** |
| lightgbm_model2 | 0.6035 ([0.5945, 0.6120]) | **yes** | -0.0174 ([-0.0211, -0.0132]) | **yes** |
| logistic_model2 | 0.5950 ([0.5853, 0.6038]) | **yes** | 0.0011 ([-0.0064, 0.0103]) | no |
| logistic_model3 | 0.5944 ([0.5846, 0.6029]) | **yes** | 0.0105 ([-0.0014, 0.0266]) | no |
| logistic_model1 | 0.5840 ([0.5738, 0.5930]) | **yes** | 0.0190 ([0.0097, 0.0292]) | no |
| baseline_model0 | 0.5000 ([0.5000, 0.5000]) | no | — | — |

**Dependence effect.** For the primary model (xgboost model2), the clustered AUC CI
width is 0.0170 vs 0.0172 under a naive iid bootstrap — clustering by security
widens the interval **0.99×**, the concrete cost of repeated-crash dependence. The
honest CI is the clustered one.

## 2. Repeated-crash cohorts (reported separately)

Test-period ROC-AUC by how many prior crashes the security had in the preceding 20 trading days.

| slice | n | base rate | baseline model0 | logistic model2 | lightgbm model3 | xgboost model2 |
|---|---|---|---|---|---|---|
| fresh (0 prior/20d) | 11,459 | 0.457 | 0.500 | 0.610 | 0.622 | 0.622 |
| 1 prior/20d | 3,834 | 0.556 | 0.500 | 0.548 | 0.538 | 0.536 |
| 2+ prior/20d | 3,563 | 0.553 | 0.500 | 0.538 | 0.549 | 0.550 |

## 3. Crash-threshold sensitivity

Same events re-sliced by crash severity (the model was trained on the −10% definition).

| slice | n | base rate | baseline model0 | logistic model2 | lightgbm model3 | xgboost model2 |
|---|---|---|---|---|---|---|
| ≤ -10% (all) | 18,856 | 0.495 | 0.500 | 0.595 | 0.604 | 0.604 |
| ≤ -15% | 6,249 | 0.485 | 0.500 | 0.594 | 0.600 | 0.599 |
| ≤ -20% | 2,682 | 0.493 | 0.500 | 0.578 | 0.583 | 0.580 |
| ≤ -30% | 768 | 0.513 | 0.500 | 0.521 | 0.544 | 0.530 |

## 4. Recovery-definition & horizon transfer

Does the model's score (trained for **+10%/20d**) still rank recoveries under other definitions?
ROC-AUC of the family-best models' scores against each target on test.

| target | n determined | base rate | logistic model2 | lightgbm model3 | xgboost model2 |
|---|---|---|---|---|---|
| +5%/5d | 18,853 | 0.456 | 0.573 | 0.582 | 0.582 |
| +5%/20d | 18,856 | 0.654 | 0.566 | 0.571 | 0.569 |
| +5%/60d | 17,808 | 0.763 | 0.555 | 0.552 | 0.550 |
| +10%/5d | 18,853 | 0.271 | 0.606 | 0.624 | 0.626 |
| +10%/20d ⭐ | 18,856 | 0.495 | 0.595 | 0.604 | 0.604 |
| +10%/60d | 17,808 | 0.648 | 0.578 | 0.583 | 0.582 |
| +20%/5d | 18,853 | 0.104 | 0.636 | 0.654 | 0.666 |
| +20%/20d | 18,856 | 0.281 | 0.625 | 0.640 | 0.645 |
| +20%/60d | 17,808 | 0.460 | 0.610 | 0.617 | 0.619 |

## Robustness verdicts

- **Signal is real but weak — ROBUST (every model's AUC CI clears 0.5).** All 10 models beat
  chance out-of-sample (Part 1); the weak ~0.60 AUC is *statistically solid*, not noise. But only
  the **tree** models beat M0 on log loss (CI excludes 0); the **linear** models do not (delta CI
  spans/exceeds 0) — the linear edge was validation optimism.
- **ROBUST across recovery definition & horizon.** The +10%/20d-trained score ranks *every* target
  in the grid above chance (primary AUC 0.550–0.666, Part 4), and is if anything stronger
  for larger/faster rebounds (+20% & 5d). The finding is not an artifact of the specific target.
- **DEFINITION-SENSITIVE to crash severity.** Discrimination holds from −10% to −20% (~0.60) but
  decays to 0.530 at the −30% slice (n small) — extreme crashes are less predictable.
- **DEFINITION-SENSITIVE to crash recency — the key caveat.** Discrimination is concentrated in
  **fresh** crashes (primary AUC 0.622, n large) and collapses toward chance for
  repeat-crashers (0.536 at 1 prior, 0.550 at 2+). Repeat-crash cohorts recover
  *more often* (higher base rate) but the model cannot discriminate *within* them. Cohorts are
  reported separately, never pooled.
- **Dependence effect is negligible here (surprising).** Clustering by security barely changes the
  primary CI (0.99× vs iid) — with ~5 events/security spread over four years, the
  a-priori §22 concern turns out small for these aggregate metrics. Residual overlap within a
  security remains, so treat *single-event* precision with more caution than the aggregate CIs.

Artifacts: `data/models/robust_metric_ci_v1.parquet`, `robust_delta_ci_v1.parquet`,
`robust_slices_v1.parquet`, `robust_target_grid_v1.parquet`.
