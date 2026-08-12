#!/usr/bin/env python3
"""STU-60 follow-on: train XGBoost recovery models and compare vs LightGBM + logistic.

Mirrors scripts/train_gbm_models.py (same stages / CLEAN pool / STU-58 splits / validation-only
tuning) with XGBoost, then writes reports/STU-60_xgboost_comparison.md putting all three model
families side by side on the validation period.

Run: PYTHONPATH=src .venv/bin/python scripts/train_xgb_models.py --version v1
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import polars as pl

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models.stages import STAGE_DESCRIPTIONS, stage_features
from crashback.models.xgb import importance_table, predict, tune
from crashback.storage.artifacts import current_git_commit

STAGES = ("model1", "model2", "model3")
METRIC_KEYS = ("log_loss", "brier", "roc_auc", "pr_auc", "prevalence", "mean_pred", "n")


def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    models_dir = cfg.paths.resolve("data_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target = PRIMARY_TARGET
    mc = cfg.models
    xg = mc.xgboost
    git = current_git_commit()

    events = pl.read_parquet(proc / "events_v1.parquet")
    splits = pl.read_parquet(proc / f"splits_{args.version}.parquet").select("event_id", "split")
    df = events.join(splits, on="event_id", how="left")
    clean = df.filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
        & ~pl.col("censored_20d") & pl.col(target).is_not_null()
    )
    train = clean.filter(pl.col("split") == "train")
    val = clean.filter(pl.col("split") == "validation")
    n_combos = (len(xg.max_depth) * len(xg.min_child_weight)
                * len(xg.colsample_bytree) * len(xg.reg_lambda))
    print(f"pool: clean+determined={clean.height:,}  train={train.height:,}  val={val.height:,}")
    print(f"grid: {n_combos} combos/stage  select by val {xg.primary_metric}")

    metric_rows: list[dict] = []
    stage_meta: list[dict] = []
    y_tr, y_va = train[target], val[target]

    for stage in STAGES:
        feats = stage_features(stage)
        res = tune(train.select(feats), y_tr, val.select(feats), y_va, xg, mc.seed,
                   feature_names=feats)
        p_tr = predict(res.booster, train.select(feats), feats)
        p_va = predict(res.booster, val.select(feats), feats)

        res.booster.save_model(str(models_dir / f"{stage}_xgboost_{args.version}.json"))
        joblib.dump(res.booster, models_dir / f"{stage}_xgboost_{args.version}.joblib")
        importance_table(res.booster, feats).write_parquet(
            models_dir / f"{stage}_xgb_importance_{args.version}.parquet")
        res.trials.write_parquet(models_dir / f"{stage}_xgb_trials_{args.version}.parquet")

        m_tr = binary_metrics(y_tr, p_tr)
        m_va = binary_metrics(y_va, p_va)
        cal_va, ece_va = calibration_table(y_va, p_va, bins=mc.calibration_bins)
        cal_va.write_parquet(models_dir / f"{stage}_xgb_calibration_val_{args.version}.parquet")

        for split_name, m in (("train", m_tr), ("validation", m_va)):
            for k in METRIC_KEYS:
                metric_rows.append({"stage": stage, "split": split_name,
                                    "metric": k, "value": m[k]})
            metric_rows.append({"stage": stage, "split": split_name, "metric": "ece",
                                "value": ece_va if split_name == "validation" else None})

        stage_meta.append({
            "stage": stage, "family": "xgboost", "n_features": len(feats),
            "best_params": res.best_params, "best_iteration": res.best_iteration,
            "val": {k: m_va[k] for k in METRIC_KEYS}, "val_ece": ece_va,
        })
        print(f"  {stage:7s} ({STAGE_DESCRIPTIONS[stage]:28s}) best={res.best_params} "
              f"iter={res.best_iteration}\n           val log_loss={_fmt(m_va['log_loss'])} "
              f"brier={_fmt(m_va['brier'])} auc={_fmt(m_va['roc_auc'])} "
              f"pr_auc={_fmt(m_va['pr_auc'])} ece={_fmt(ece_va)}")

    pl.DataFrame(metric_rows).write_parquet(models_dir / f"xgb_metrics_{args.version}.parquet")
    run_meta = {
        "target": target, "version": args.version, "family": "xgboost", "git_commit": git,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection": {"criterion": f"validation {xg.primary_metric}", "space": "predeclared grid"},
        "config_xgboost": json.loads(xg.model_dump_json()),
        "pool": {"clean_determined": clean.height, "train": train.height, "val": val.height},
        "stages": stage_meta,
    }
    (models_dir / f"run_meta_xgb_{args.version}.json").write_text(json.dumps(run_meta, indent=2))

    _write_report(cfg, args.version, target, stage_meta, models_dir)
    print(f"\nwrote artifacts to {models_dir} and reports/STU-60_xgboost_comparison.md")


def _val_metrics(models_dir, fname, stage) -> dict:
    p = models_dir / fname
    if not p.exists():
        return {}
    m = pl.read_parquet(p).filter((pl.col("stage") == stage) & (pl.col("split") == "validation"))
    return {r["metric"]: r["value"] for r in m.iter_rows(named=True)}


def _write_report(cfg, version, target, stage_meta, models_dir):
    rows = ["| stage | model | log loss | Brier | ROC-AUC | PR-AUC | ECE |",
            "|---|---|---|---|---|---|---|"]
    for s in stage_meta:
        stage = s["stage"]
        v = s["val"]
        log = _val_metrics(models_dir, "model_metrics_v1.parquet", stage)
        gbm = _val_metrics(models_dir, "gbm_metrics_v1.parquet", stage)
        rows.append(f"| **{stage} — {STAGE_DESCRIPTIONS[stage]}** | **XGBoost** | "
                    f"{_fmt(v['log_loss'])} | {_fmt(v['brier'])} | {_fmt(v['roc_auc'])} | "
                    f"{_fmt(v['pr_auc'])} | {_fmt(s['val_ece'])} |")
        if gbm:
            rows.append(f"| | LightGBM | {_fmt(gbm.get('log_loss'))} | {_fmt(gbm.get('brier'))} | "
                        f"{_fmt(gbm.get('roc_auc'))} | {_fmt(gbm.get('pr_auc'))} | "
                        f"{_fmt(gbm.get('ece'))} |")
        if log:
            rows.append(f"| | logistic | {_fmt(log.get('log_loss'))} | {_fmt(log.get('brier'))} | "
                        f"{_fmt(log.get('roc_auc'))} | {_fmt(log.get('pr_auc'))} | "
                        f"{_fmt(log.get('ece'))} |")
    table = "\n".join(rows)

    best = min(stage_meta, key=lambda s: s["val"]["log_loss"])
    imp_p = models_dir / f"{best['stage']}_xgb_importance_{version}.parquet"
    imp_md = ""
    if imp_p.exists():
        top = pl.read_parquet(imp_p).head(15)
        imp_md = "\n".join([f"| {r['feature']} | {r['gain_frac'] * 100:.1f}% | {int(r['split'])} |"
                            for r in top.iter_rows(named=True)])
    params_md = "\n".join([f"- **{s['stage']}**: {s['best_params']}, "
                           f"best_iteration={s['best_iteration']}" for s in stage_meta])

    report = f"""# STU-60 (follow-on) — XGBoost vs LightGBM vs Logistic

XGBoost cross-check on the same staged features / CLEAN determined pool / STU-58 splits, primary
target **`{target}`**. Same discipline as LightGBM: predeclared grid searched on **validation**
only, early stopping on validation, best per stage by validation
**{cfg.models.xgboost.primary_metric}**; **test** untouched. XGBoost handles missing natively
(`missing=nan`); `inf` routed to missing; class prevalence preserved.

> ⚠️ Validation is used for both selection and reporting, so these are mildly optimistic. The
> unbiased estimate is the held-out **test** set (STU-62).

## Validation metrics — three families side by side

{table}

## XGBoost selected hyperparameters (per stage, by validation {cfg.models.xgboost.primary_metric})

{params_md}

## XGBoost feature importance — {best['stage']} (best XGB stage), top 15 by gain

| feature | gain share | # splits |
|---|---|---|
{imp_md}

## Takeaway

XGBoost and LightGBM are the same model class (gradient-boosted trees) and should land close;
any gap is implementation/hyperparameter-grid differences, not a fundamentally different signal.
The three-way table isolates **linear (logistic) vs nonlinear (both GBMs)** as the real
methodological axis. Artifacts per stage in `data/models/`: `{{stage}}_xgboost_{version}.json` +
`.joblib`, `{{stage}}_xgb_importance/_xgb_trials/_xgb_calibration_val_{version}.parquet`, plus
`xgb_metrics_{version}.parquet` and `run_meta_xgb_{version}.json`. Registered in Supabase
`model_runs` (family=xgboost) + `metrics`.
"""
    (cfg.paths.resolve("reports") / "STU-60_xgboost_comparison.md").write_text(report)


if __name__ == "__main__":
    main()
