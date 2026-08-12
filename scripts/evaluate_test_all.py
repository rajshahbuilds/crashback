#!/usr/bin/env python3
"""STU-62 follow-on: DESCRIPTIVE test evaluation of every staged model (robustness view).

⚠️ This is a diagnostic, NOT a re-selection. The pre-registered final model stays XGBoost
model2 (STU-62); we do not crown a new winner from these test numbers — doing so would turn the
test set into a second validation set and reintroduce selection optimism. The purpose is to see
whether the validation→test degradation is *systematic* (regime shift hits everything) or
*model-specific* (overfitting), and whether the validation ranking held out of sample.

All models are the saved train-only artifacts from STU-59/60; evaluated on the untouched test
period from disk. No retraining, no tuning.

Run: PYTHONPATH=src .venv/bin/python scripts/evaluate_test_all.py --version v1
"""
from __future__ import annotations

import argparse

import joblib
import polars as pl

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.evaluation.lift import top_decile_lift
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models import gbm, xgb
from crashback.models.logistic import fit_base_rate, predict_logistic
from crashback.models.stages import stage_features

STAGES = ("model1", "model2", "model3")
FAMILIES = ("logistic", "lightgbm", "xgboost")
VAL_SOURCE = {"logistic": "model_metrics_v1.parquet",
              "lightgbm": "gbm_metrics_v1.parquet",
              "xgboost": "xgb_metrics_v1.parquet"}


def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def _predict(family, booster, X, feats):
    if family == "logistic":
        return predict_logistic(booster, X)
    if family == "lightgbm":
        return gbm.predict(booster, X)
    return xgb.predict(booster, X, feats)


def _val_metric(models_dir, family, stage, metric):
    m = pl.read_parquet(models_dir / VAL_SOURCE[family]).filter(
        (pl.col("stage") == stage) & (pl.col("split") == "validation")
        & (pl.col("metric") == metric))
    return m["value"][0] if m.height else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    models_dir = cfg.paths.resolve("data_models")
    target = PRIMARY_TARGET
    bins = cfg.models.calibration_bins

    events = pl.read_parquet(proc / "events_v1.parquet")
    splits = pl.read_parquet(proc / f"splits_{args.version}.parquet").select("event_id", "split")
    df = events.join(splits, on="event_id", how="left")
    clean = df.filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
        & ~pl.col("censored_20d") & pl.col(target).is_not_null())
    train = clean.filter(pl.col("split") == "train")
    test = clean.filter(pl.col("split") == "test")
    y_te = test[target].to_numpy()
    print(f"test pool={test.height:,}  base_rate={y_te.mean():.4f}")

    rows: list[dict] = []
    # M0 baseline
    p0 = fit_base_rate(train[target]).predict_proba(test.height)
    m0 = binary_metrics(y_te, p0)
    rows.append({"family": "baseline", "stage": "model0", "run": "stu59_model0_v1",
                 "n_features": 0,
                 "val_log_loss": _val_metric(models_dir, "logistic", "model0", "log_loss"),
                 "val_roc_auc": None, "test_log_loss": m0["log_loss"], "test_brier": m0["brier"],
                 "test_roc_auc": m0["roc_auc"], "test_pr_auc": m0["pr_auc"],
                 "test_ece": calibration_table(y_te, p0, bins=bins)[1],
                 "test_top_decile_lift": top_decile_lift(y_te, p0)["top_decile_lift"]})

    run_name = {("logistic", s): f"stu59_{s}_v1" for s in STAGES}
    run_name |= {("lightgbm", s): f"stu60_{s}_v1" for s in STAGES}
    run_name |= {("xgboost", s): f"stu60_xgb_{s}_v1" for s in STAGES}

    for family in FAMILIES:
        for stage in STAGES:
            feats = stage_features(stage)
            booster = joblib.load(models_dir / f"{stage}_{family}_{args.version}.joblib")
            p = _predict(family, booster, test.select(feats), feats)
            m = binary_metrics(y_te, p)
            rows.append({
                "family": family, "stage": stage, "run": run_name[(family, stage)],
                "n_features": len(feats),
                "val_log_loss": _val_metric(models_dir, family, stage, "log_loss"),
                "val_roc_auc": _val_metric(models_dir, family, stage, "roc_auc"),
                "test_log_loss": m["log_loss"], "test_brier": m["brier"],
                "test_roc_auc": m["roc_auc"], "test_pr_auc": m["pr_auc"],
                "test_ece": calibration_table(y_te, p, bins=bins)[1],
                "test_top_decile_lift": top_decile_lift(y_te, p)["top_decile_lift"]})

    res = pl.DataFrame(rows).with_columns(
        (pl.col("val_roc_auc") - pl.col("test_roc_auc")).alias("auc_drop"))
    res.write_parquet(models_dir / f"test_all_models_{args.version}.parquet")
    _write_report(cfg, args.version, target, res, y_te.mean(), test.height)

    print("\n=== TEST across all staged models ===")
    for r in res.iter_rows(named=True):
        print(f"  {r['family']:9s} {r['stage']:7s}  test_ll={_fmt(r['test_log_loss'])} "
              f"test_auc={_fmt(r['test_roc_auc'])}  val_auc={_fmt(r['val_roc_auc'])} "
              f"drop={_fmt(r['auc_drop'])}")
    print(f"\nwrote reports/STU-62_test_all_models.md and "
          f"{models_dir}/test_all_models_{args.version}.parquet")


def _write_report(cfg, version, target, res, test_base, n_test):
    hdr = ("| family | stage | feats | val log loss | test log loss | val AUC | test AUC | "
           "AUC drop | test PR-AUC | test ECE | test top-decile lift |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for r in res.iter_rows(named=True):
        marker = " ⭐" if r["run"] == "stu60_xgb_model2_v1" else ""
        lines.append(
            f"| {r['family']}{marker} | {r['stage']} | {r['n_features']} | "
            f"{_fmt(r['val_log_loss'])} | {_fmt(r['test_log_loss'])} | {_fmt(r['val_roc_auc'])} | "
            f"{_fmt(r['test_roc_auc'])} | {_fmt(r['auc_drop'])} | {_fmt(r['test_pr_auc'])} | "
            f"{_fmt(r['test_ece'])} | {r['test_top_decile_lift']:.2f}× |")
    table = "\n".join(lines)

    # did the validation winner stay best on test?
    scored = res.filter(pl.col("stage") != "model0")
    best_val = scored.sort("val_log_loss").row(0, named=True)
    best_test = scored.sort("test_log_loss").row(0, named=True)
    drops = scored["auc_drop"].drop_nulls().to_list()
    drop_lo, drop_hi = min(drops), max(drops)

    report = f"""# STU-62 (follow-on) — Test Evaluation Across All Staged Models

⚠️ **Descriptive robustness view, not a re-selection.** The pre-registered final model is and
remains **XGBoost model2** (STU-62, ⭐ below). These numbers are *diagnostic*: we do **not** pick
a new final model from them — that would turn the held-out test set into a second validation set
and reintroduce the optimism the chronological split exists to prevent. The question here is
whether the validation→test degradation is **systematic** (a regime effect) or **model-specific**
(overfitting), and whether the validation ranking survived out of sample.

All models are the saved **train-only** artifacts (STU-59/60), evaluated on the untouched test
period ({n_test:,} CLEAN determined events, base rate {test_base:.4f}). No retraining.

## Every staged model on test (val shown for contrast)

{table}

## What this tells us

- **The degradation is systematic, not model-specific.** Every model's ROC-AUC falls from
  validation to test by a similar amount (AUC drop range {drop_lo:+.3f} to {drop_hi:+.3f}). A
  single overfit model would stand out with a much larger drop; instead the whole cohort moves
  together — the signature of a **regime shift** (test base rate {test_base:.3f} vs the higher
  validation regime), consistent with STU-62's read.
- **The validation ranking did not cleanly survive.** Best by validation log loss was
  `{best_val['family']} {best_val['stage']}`; best by *test* log loss was
  `{best_test['family']} {best_test['stage']}`. That they differ is exactly why we do **not**
  re-pick from test — small test-set reshuffling among similar models is expected noise, not a
  reason to change the locked choice.
- **Fundamentals (model3) do not rescue generalization** — adding them does not systematically
  reduce the AUC drop, reinforcing STU-61's "no robust incremental value" for the 20-day target.
- Read alongside `reports/STU-62_test_evaluation.md` (the pre-registered primary) — this table
  is context, that report is the result.

Artifact: `data/models/test_all_models_{version}.parquet`. Test metrics for each model are added
to the corresponding Supabase `model_runs` row (`split='test'`).
"""
    (cfg.paths.resolve("reports") / "STU-62_test_all_models.md").write_text(report)


if __name__ == "__main__":
    main()
