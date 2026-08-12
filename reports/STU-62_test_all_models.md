# STU-62 (follow-on) — Test Evaluation Across All Staged Models

⚠️ **Descriptive robustness view, not a re-selection.** The pre-registered final model is and
remains **XGBoost model2** (STU-62, ⭐ below). These numbers are *diagnostic*: we do **not** pick
a new final model from them — that would turn the held-out test set into a second validation set
and reintroduce the optimism the chronological split exists to prevent. The question here is
whether the validation→test degradation is **systematic** (a regime effect) or **model-specific**
(overfitting), and whether the validation ranking survived out of sample.

All models are the saved **train-only** artifacts (STU-59/60), evaluated on the untouched test
period (18,856 CLEAN determined events, base rate 0.4954). No retraining.

## Every staged model on test (val shown for contrast)

| family | stage | feats | val log loss | test log loss | val AUC | test AUC | AUC drop | test PR-AUC | test ECE | test top-decile lift |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | model0 | 0 | 0.6889 | 0.6936 | — | — | — | 0.4954 | 0.0157 | 0.95× |
| logistic | model1 | 26 | 0.6499 | 0.7126 | 0.6574 | 0.5840 | 0.0734 | 0.5547 | 0.0608 | 1.18× |
| logistic | model2 | 34 | 0.6128 | 0.6947 | 0.6984 | 0.5950 | 0.1034 | 0.5671 | 0.0426 | 1.25× |
| logistic | model3 | 52 | 0.6163 | 0.7041 | 0.6966 | 0.5944 | 0.1022 | 0.5663 | 0.0470 | 1.25× |
| lightgbm | model1 | 26 | 0.6215 | 0.6743 | 0.6954 | 0.6105 | 0.0850 | 0.5823 | 0.0312 | 1.26× |
| lightgbm | model2 | 34 | 0.6151 | 0.6762 | 0.6985 | 0.6035 | 0.0949 | 0.5845 | 0.0336 | 1.28× |
| lightgbm | model3 | 52 | 0.6112 | 0.6752 | 0.7070 | 0.6044 | 0.1026 | 0.5837 | 0.0308 | 1.29× |
| xgboost | model1 | 26 | 0.6209 | 0.6739 | 0.6974 | 0.6117 | 0.0858 | 0.5825 | 0.0279 | 1.26× |
| xgboost ⭐ | model2 | 34 | 0.6054 | 0.6722 | 0.7074 | 0.6041 | 0.1033 | 0.5863 | 0.0198 | 1.32× |
| xgboost | model3 | 52 | 0.6074 | 0.6718 | 0.7057 | 0.6063 | 0.0995 | 0.5866 | 0.0175 | 1.30× |

## What this tells us

- **The degradation is systematic, not model-specific.** Every model's ROC-AUC falls from
  validation to test by a similar amount (AUC drop range +0.073 to +0.103). A
  single overfit model would stand out with a much larger drop; instead the whole cohort moves
  together — the signature of a **regime shift** (test base rate 0.495 vs the higher
  validation regime), consistent with STU-62's read.
- **The validation ranking did not cleanly survive.** Best by validation log loss was
  `xgboost model2`; best by *test* log loss was
  `xgboost model3`. That they differ is exactly why we do **not**
  re-pick from test — small test-set reshuffling among similar models is expected noise, not a
  reason to change the locked choice.
- **Fundamentals (model3) do not rescue generalization** — adding them does not systematically
  reduce the AUC drop, reinforcing STU-61's "no robust incremental value" for the 20-day target.
- Read alongside `reports/STU-62_test_evaluation.md` (the pre-registered primary) — this table
  is context, that report is the result.

Artifact: `data/models/test_all_models_v1.parquet`. Test metrics for each model are added
to the corresponding Supabase `model_runs` row (`split='test'`).
