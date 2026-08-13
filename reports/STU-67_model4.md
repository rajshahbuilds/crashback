# STU-67 — Model 4: Event-Understanding Features (Pilot)

**Bounded pilot** (per the scale/cost decision): can LLM crash-cause features (STU-66) improve
recovery prediction beyond the best V1 model (Model 3)? Model 4 = Model 3's feature set + the
crash-cause features, trained with an **identical fixed XGBoost config** on the **same** pilot
events (so the delta isolates the features), using the official chronological split. Primary
target `hit_10pct_20d`. **Only contemporaneous event information is used** (STU-65 point-in-time
retrieval + STU-66 documents-only extraction).

> ⚠️ **Underpowered by design.** 400 pilot events (280 train / 120
> test) — far smaller than V1's 52k/19k. Treat this as a *directional* read + a working pipeline,
> not a definitive verdict. It complements V1's held-out framework; it does not replace it.

## Incremental value (pilot test set, 120 events, base rate 0.375)

| metric | Model 3 | Model 4 | Δ (M4−M3) |
|---|---|---|---|
| log_loss | 0.7717 | 0.7599 | -0.0119 |
| brier | 0.2750 | 0.2688 | -0.0061 |
| roc_auc | 0.5650 | 0.5861 | +0.0210 |
| pr_auc | 0.4572 | 0.4640 | +0.0068 |
| ece | 0.1685 | 0.1511 | -0.0174 |
| top_decile_lift | 1.3333 | 1.5556 | +0.2222 |

**Δ log loss vs Model 3 (95% clustered-bootstrap CI):** -0.0119
([-0.0457, 0.0204]) — **not** material (CI spans 0).
**Δ ROC-AUC:** 0.0210 ([-0.0125, 0.0560]).

### Verdict

On this pilot the result is **inconclusive but directionally positive**: all 6/6 metrics' point estimates favor Model 4, but the Δ log-loss 95% CI spans 0, so the improvement is **not statistically material** at n=120 test — it cannot be distinguished from noise. Notably this differs from V1 fundamentals (whose effect was *sign-inconsistent* across model families): here the lean is consistently positive, which — combined with the strong test-period retrieval coverage — makes crash-cause features a **credible candidate worth a larger, better-powered run** rather than a dead end. The pilot neither confirms nor refutes added value; it is underpowered by design.

## Failure analysis (retrieval vs extraction vs model)

Coverage: **172/400 events (43.0%)** produced a usable extraction.

| failure_reason | n |
|---|---|
| no_cik_or_docs | 210 |
| ok | 172 |
| extraction_failed | 12 |
| no_cause_8k | 6 |

- **Bad retrieval** (no CIK / no cause 8-K / empty text / fetch error): **216** — the
  dominant loss. Many crashes have no contemporaneous SEC filing (macro/sector moves, illiquid or
  renamed tickers whose current ticker→CIK mapping misses). This is a *coverage* ceiling, not an
  extraction fault.
- **Bad extraction** (LLM output failed schema/grounding even after retry): **12**.
- **Model limitation:** where extraction succeeded, the added features still need to *move* the
  metric. The `temporary_vs_structural` signal's variance among successful events:

| damage | n |
|---|---|
| unclear | 97 |
| mixed | 67 |
| temporary | 6 |
| structural | 2 |

If damage is nearly constant (e.g. mostly `mixed`), the feature carries little information for the
model regardless of extraction quality — a genuine signal limitation, distinct from retrieval.

## Notes

- Model 4 adds only the compact `cc_*` features (has_filing, ordinal impacts, thesis-changed,
  damage, uncertainty); missing (no filing / `unclear`) stays null and the tree handles it.
- Pipeline: `scripts/extract_model4_features.py` (stage 1, checkpointed LLM extraction) →
  `scripts/build_model4.py` (this). Cause features in
  `data/documents/model4_cause_features_v1.parquet`.
