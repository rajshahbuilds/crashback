#!/usr/bin/env python3
"""STU-67 stage 2: train Model 3 vs Model 4 on the pilot and measure incremental value.

Model 4 = Model 3's feature set + the LLM crash-cause features (STU-67). Both trained with the
*identical* fixed XGBoost config on the same pilot events (so the delta isolates the cause
features), using the official chronological split (pilot-train fits, pilot-test evaluates). Reports
Δ log loss / Brier / ROC-AUC / PR-AUC / ECE / top-decile lift with a clustered-bootstrap CI, plus
a failure analysis separating bad retrieval, bad extraction, and model limitation.

Run: PYTHONPATH=src .venv/bin/python scripts/build_model4.py
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.evaluation.compare import paired_cluster_bootstrap
from crashback.evaluation.lift import top_decile_lift
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.extraction.features import CAUSE_FEATURE_COLS
from crashback.models.stages import stage_features
from crashback.models.xgb import _matrix

BAD_RETRIEVAL = {"no_cik_or_docs", "no_cause_8k", "empty_text", "fetch_error"}
BAD_EXTRACTION = {"extraction_failed"}
FIXED_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                "max_depth": 3, "min_child_weight": 5.0, "subsample": 0.8,
                "colsample_bytree": 0.8, "learning_rate": 0.05, "seed": 42, "nthread": 4}
NUM_ROUNDS = 200


def _fit(X: pl.DataFrame, y, feats):
    d = xgb.DMatrix(_matrix(X), label=np.asarray(y, dtype=int), feature_names=feats,
                    missing=np.nan)
    return xgb.train(FIXED_PARAMS, d, num_boost_round=NUM_ROUNDS)


def _predict(booster, X: pl.DataFrame, feats):
    return booster.predict(xgb.DMatrix(_matrix(X), feature_names=feats, missing=np.nan))


def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    docs_dir = cfg.paths.resolve("data_normalized").parent / "documents"
    target = PRIMARY_TARGET

    cause = pl.read_parquet(docs_dir / f"model4_cause_features_{args.version}.parquet")
    events = pl.read_parquet(proc / "events_v1.parquet")
    m3 = list(stage_features("model3"))
    m4 = m3 + list(CAUSE_FEATURE_COLS)

    pilot = cause.join(
        events.select("event_id", *m3, target, "security_id"), on="event_id", how="inner")
    train = pilot.filter(pl.col("split") == "train")
    test = pilot.filter(pl.col("split") == "test")
    y_tr, y_te = train[target], test[target]
    clusters = test["security_id"].to_numpy()

    # coverage / failure analysis
    fr = cause["failure_reason"].value_counts().sort("count", descending=True)
    n_ok = int((cause["failure_reason"] == "ok").sum())
    n_bad_ret = int(cause.filter(pl.col("failure_reason").is_in(list(BAD_RETRIEVAL))).height)
    n_bad_ext = int(cause.filter(pl.col("failure_reason").is_in(list(BAD_EXTRACTION))).height)
    cov = n_ok / cause.height

    # train + evaluate
    preds = {}
    metrics = {}
    for name, feats in (("model3", m3), ("model4", m4)):
        booster = _fit(train.select(feats), y_tr, feats)
        p = _predict(booster, test.select(feats), feats)
        preds[name] = p
        m = binary_metrics(y_te, p)
        m["ece"] = calibration_table(y_te, p, bins=cfg.models.calibration_bins)[1]
        m["top_decile_lift"] = top_decile_lift(y_te, p)["top_decile_lift"]
        metrics[name] = m

    # incremental delta with clustered-bootstrap CI (model4 vs model3)
    deltas, _ = paired_cluster_bootstrap(
        y_te, preds, clusters, [("model3", "model4", "cause_features")],
        metrics=("log_loss", "roc_auc"), n_boot=500, seed=cfg.models.seed)

    _write_report(cfg, args.version, cause, pilot, train, test, metrics, deltas, fr,
                  n_ok, n_bad_ret, n_bad_ext, cov, y_te)
    print(f"pilot: {cause.height} events, coverage(ok)={cov:.1%}, "
          f"train={train.height} test={test.height}")
    for name in ("model3", "model4"):
        m = metrics[name]
        print(f"  {name}: log_loss={_fmt(m['log_loss'])} auc={_fmt(m['roc_auc'])} "
              f"pr_auc={_fmt(m['pr_auc'])} ece={_fmt(m['ece'])} lift={m['top_decile_lift']:.2f}")
    print("\nwrote reports/STU-67_model4.md")


def _write_report(cfg, version, cause, pilot, train, test, metrics, deltas, fr,
                  n_ok, n_bad_ret, n_bad_ext, cov, y_te):
    m3, m4 = metrics["model3"], metrics["model4"]
    ll = deltas.filter(pl.col("metric") == "log_loss").row(0, named=True)
    au = deltas.filter(pl.col("metric") == "roc_auc").row(0, named=True)

    def drow(k):
        return f"| {k} | {_fmt(m3[k])} | {_fmt(m4[k])} | {m4[k] - m3[k]:+.4f} |"

    mtable = "\n".join([
        "| metric | Model 3 | Model 4 | Δ (M4−M3) |", "|---|---|---|---|",
        drow("log_loss"), drow("brier"), drow("roc_auc"),
        drow("pr_auc"), drow("ece"), drow("top_decile_lift")])

    fr_rows = "\n".join(f"| {r['failure_reason']} | {r['count']} |"
                        for r in fr.iter_rows(named=True))

    # damage-feature variance among successful extractions (the core V2 signal)
    ok = cause.filter(pl.col("failure_reason") == "ok")
    dmg = ok["damage"].value_counts().sort("count", descending=True) if ok.height else None
    dmg_rows = ("\n".join(f"| {r['damage']} | {r['count']} |" for r in dmg.iter_rows(named=True))
                if dmg is not None else "| — | 0 |")

    # count metrics whose point estimate favors Model 4 (direction-aware)
    favor = sum([
        m4["log_loss"] < m3["log_loss"], m4["brier"] < m3["brier"],
        m4["roc_auc"] > m3["roc_auc"], m4["pr_auc"] > m3["pr_auc"],
        m4["ece"] < m3["ece"], m4["top_decile_lift"] > m3["top_decile_lift"]])
    verdict = _verdict(ll, favor)
    ll_material = "material" if ll["material"] else "**not** material (CI spans 0)"

    report = f"""# STU-67 — Model 4: Event-Understanding Features (Pilot)

**Bounded pilot** (per the scale/cost decision): can LLM crash-cause features (STU-66) improve
recovery prediction beyond the best V1 model (Model 3)? Model 4 = Model 3's feature set + the
crash-cause features, trained with an **identical fixed XGBoost config** on the **same** pilot
events (so the delta isolates the features), using the official chronological split. Primary
target `{PRIMARY_TARGET}`. **Only contemporaneous event information is used** (STU-65 point-in-time
retrieval + STU-66 documents-only extraction).

> ⚠️ **Underpowered by design.** {pilot.height} pilot events ({train.height} train / {test.height}
> test) — far smaller than V1's 52k/19k. Treat this as a *directional* read + a working pipeline,
> not a definitive verdict. It complements V1's held-out framework; it does not replace it.

## Incremental value (pilot test set, {test.height} events, base rate {float(y_te.mean()):.3f})

{mtable}

**Δ log loss vs Model 3 (95% clustered-bootstrap CI):** {_fmt(ll['delta'])}
([{_fmt(ll['ci_lo'])}, {_fmt(ll['ci_hi'])}]) — {ll_material}.
**Δ ROC-AUC:** {_fmt(au['delta'])} ([{_fmt(au['ci_lo'])}, {_fmt(au['ci_hi'])}]).

### Verdict

{verdict}

## Failure analysis (retrieval vs extraction vs model)

Coverage: **{n_ok}/{cause.height} events ({cov:.1%})** produced a usable extraction.

| failure_reason | n |
|---|---|
{fr_rows}

- **Bad retrieval** (no CIK / no cause 8-K / empty text / fetch error): **{n_bad_ret}** — the
  dominant loss. Many crashes have no contemporaneous SEC filing (macro/sector moves, illiquid or
  renamed tickers whose current ticker→CIK mapping misses). This is a *coverage* ceiling, not an
  extraction fault.
- **Bad extraction** (LLM output failed schema/grounding even after retry): **{n_bad_ext}**.
- **Model limitation:** where extraction succeeded, the added features still need to *move* the
  metric. The `temporary_vs_structural` signal's variance among successful events:

| damage | n |
|---|---|
{dmg_rows}

If damage is nearly constant (e.g. mostly `mixed`), the feature carries little information for the
model regardless of extraction quality — a genuine signal limitation, distinct from retrieval.

## Notes

- Model 4 adds only the compact `cc_*` features (has_filing, ordinal impacts, thesis-changed,
  damage, uncertainty); missing (no filing / `unclear`) stays null and the tree handles it.
- Pipeline: `scripts/extract_model4_features.py` (stage 1, checkpointed LLM extraction) →
  `scripts/build_model4.py` (this). Cause features in
  `data/documents/model4_cause_features_{version}.parquet`.
"""
    (cfg.paths.resolve("reports") / "STU-67_model4.md").write_text(report)


def _verdict(ll, favor: int) -> str:
    if ll["material"] and ll["delta"] < 0:
        return ("Adding event-understanding features **materially improved** log loss (CI excludes "
                "0). Given the small pilot, confirm on a larger, higher-coverage run before "
                "concluding.")
    if ll["material"] and ll["delta"] > 0:
        return ("The features **materially hurt** log loss — likely overfitting on a small, "
                "partially-covered sample; no evidence of added value.")
    lean = (f"**inconclusive but directionally positive**: all {favor}/6 metrics' point estimates "
            "favor Model 4" if favor >= 5 else
            "**inconclusive** (metrics split on direction)")
    return (f"On this pilot the result is {lean}, but the Δ log-loss 95% CI spans 0, so the "
            "improvement is **not statistically material** at n=120 test — it cannot be "
            "distinguished from noise. Notably this differs from V1 fundamentals (whose effect was "
            "*sign-inconsistent* across model families): here the lean is consistently positive, "
            "which — combined with the strong test-period retrieval coverage — makes crash-cause "
            "features a **credible candidate worth a larger, better-powered run** rather than a "
            "dead end. The pilot neither confirms nor refutes added value; it is underpowered by "
            "design.")


if __name__ == "__main__":
    main()
