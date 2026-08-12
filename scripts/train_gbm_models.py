#!/usr/bin/env python3
"""STU-60: train gradient-boosted (LightGBM) recovery models for the primary target.

Same staged feature sets as STU-59 (model1/2/3), same CLEAN determined pool and STU-58 splits.
For each stage the predeclared grid is searched on VALIDATION only (early stopping also watches
validation); the best booster is picked by the predeclared primary metric. Persists boosters,
feature importance, per-stage best params + trial tables, metrics, calibration, and run metadata,
then writes reports/STU-60_gbm_models.md comparing GBM to the logistic baselines.

Run: PYTHONPATH=src .venv/bin/python scripts/train_gbm_models.py --version v1
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
from crashback.models.gbm import importance_table, predict, tune
from crashback.models.stages import STAGE_DESCRIPTIONS, stage_features
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
    lg = mc.lightgbm
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
    n_combos = (len(lg.num_leaves) * len(lg.min_child_samples)
                * len(lg.feature_fraction) * len(lg.lambda_l2))
    print(f"pool: clean+determined={clean.height:,}  train={train.height:,}  val={val.height:,}")
    print(f"grid: {n_combos} combos/stage  select by val {lg.primary_metric}")

    metric_rows: list[dict] = []
    stage_meta: list[dict] = []
    y_tr, y_va = train[target], val[target]

    for stage in STAGES:
        feats = stage_features(stage)
        res = tune(train.select(feats), y_tr, val.select(feats), y_va, lg, mc.seed,
                   feature_names=feats)
        p_tr = predict(res.booster, train.select(feats))
        p_va = predict(res.booster, val.select(feats))

        res.booster.save_model(str(models_dir / f"{stage}_lightgbm_{args.version}.txt"),
                               num_iteration=res.best_iteration)
        joblib.dump(res.booster, models_dir / f"{stage}_lightgbm_{args.version}.joblib")
        imp = importance_table(res.booster)
        imp.write_parquet(models_dir / f"{stage}_gbm_importance_{args.version}.parquet")
        res.trials.write_parquet(models_dir / f"{stage}_gbm_trials_{args.version}.parquet")

        m_tr = binary_metrics(y_tr, p_tr)
        m_va = binary_metrics(y_va, p_va)
        cal_va, ece_va = calibration_table(y_va, p_va, bins=mc.calibration_bins)
        cal_va.write_parquet(models_dir / f"{stage}_gbm_calibration_val_{args.version}.parquet")

        for split_name, m in (("train", m_tr), ("validation", m_va)):
            for k in METRIC_KEYS:
                metric_rows.append({"stage": stage, "split": split_name,
                                    "metric": k, "value": m[k]})
            metric_rows.append({"stage": stage, "split": split_name, "metric": "ece",
                                "value": ece_va if split_name == "validation" else None})

        stage_meta.append({
            "stage": stage, "family": "lightgbm", "n_features": len(feats),
            "best_params": res.best_params, "best_iteration": res.best_iteration,
            "val": {k: m_va[k] for k in METRIC_KEYS}, "val_ece": ece_va,
        })
        print(f"  {stage:7s} ({STAGE_DESCRIPTIONS[stage]:28s}) best={res.best_params} "
              f"iter={res.best_iteration}\n           val log_loss={_fmt(m_va['log_loss'])} "
              f"brier={_fmt(m_va['brier'])} auc={_fmt(m_va['roc_auc'])} "
              f"pr_auc={_fmt(m_va['pr_auc'])} ece={_fmt(ece_va)}")

    metrics = pl.DataFrame(metric_rows)
    metrics.write_parquet(models_dir / f"gbm_metrics_{args.version}.parquet")

    run_meta = {
        "target": target, "version": args.version, "family": "lightgbm", "git_commit": git,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection": {"criterion": f"validation {lg.primary_metric}", "space": "predeclared grid"},
        "config_lightgbm": json.loads(lg.model_dump_json()),
        "pool": {"clean_determined": clean.height, "train": train.height, "val": val.height},
        "stages": stage_meta,
    }
    (models_dir / f"run_meta_gbm_{args.version}.json").write_text(json.dumps(run_meta, indent=2))

    _write_report(cfg, args.version, target, stage_meta, models_dir, proc)
    print(f"\nwrote artifacts to {models_dir} and reports/STU-60_gbm_models.md")


def _load_logistic_val(models_dir, stage) -> dict:
    """Validation metrics for the STU-59 logistic model of the same stage (for comparison)."""
    p = models_dir / "model_metrics_v1.parquet"
    if not p.exists():
        return {}
    m = pl.read_parquet(p).filter((pl.col("stage") == stage) & (pl.col("split") == "validation"))
    return {r["metric"]: r["value"] for r in m.iter_rows(named=True)}


def _write_report(cfg, version, target, stage_meta, models_dir, proc):
    rows = ["| stage | features | model | log loss | Brier | ROC-AUC | PR-AUC | ECE |",
            "|---|---|---|---|---|---|---|---|"]
    for s in stage_meta:
        stage = s["stage"]
        v = s["val"]
        lgm = _load_logistic_val(models_dir, stage)
        rows.append(
            f"| {stage} — {STAGE_DESCRIPTIONS[stage]} | {s['n_features']} | LightGBM | "
            f"{_fmt(v['log_loss'])} | {_fmt(v['brier'])} | {_fmt(v['roc_auc'])} | "
            f"{_fmt(v['pr_auc'])} | {_fmt(s['val_ece'])} |")
        if lgm:
            rows.append(
                f"| ↳ (logistic STU-59) | {s['n_features']} | logistic | "
                f"{_fmt(lgm.get('log_loss'))} | {_fmt(lgm.get('brier'))} | "
                f"{_fmt(lgm.get('roc_auc'))} | {_fmt(lgm.get('pr_auc'))} | "
                f"{_fmt(lgm.get('ece'))} |")
    table = "\n".join(rows)

    # best stage by validation log loss (predeclared criterion)
    best = min(stage_meta, key=lambda s: s["val"]["log_loss"])
    imp_p = models_dir / f"{best['stage']}_gbm_importance_{version}.parquet"
    imp_md = ""
    if imp_p.exists():
        top = pl.read_parquet(imp_p).head(15)
        imp_md = "\n".join(
            [f"| {r['feature']} | {r['gain_frac'] * 100:.1f}% | {int(r['split'])} |"
             for r in top.iter_rows(named=True)])

    params_md = "\n".join(
        [f"- **{s['stage']}**: {s['best_params']}, best_iteration={s['best_iteration']}"
         for s in stage_meta])

    report = f"""# STU-60 — Gradient-Boosted (LightGBM) Recovery Models (M1–M3)

Nonlinear trees on the same staged features as STU-59, primary target **`{target}`**. CLEAN
determined pool; STU-58 splits (fit on **train**, evaluated on **validation**; **test**
untouched). Config-driven (`configs/default.yaml` → `models.lightgbm`), reproducible via
`scripts/train_gbm_models.py`.

## Tuning discipline

The **predeclared** grid (`num_leaves × min_child_samples × feature_fraction × lambda_l2`) is
searched on the **validation** period only, and early stopping watches validation to choose the
boosting-round count. The best model per stage is selected by the predeclared criterion
**validation {cfg.models.lightgbm.primary_metric}** (lower is better) — never test. Class
prevalence is preserved (no oversampling / class weighting). LightGBM handles missing values
natively; `inf` (undefined ratios) is routed to the missing path.

> ⚠️ Because selection *and* reporting both use validation, these validation metrics are mildly
> optimistic. The unbiased estimate is the held-out **test** set (STU-62).

## Validation metrics — LightGBM vs logistic baseline

{table}

## Selected hyperparameters (per stage, by validation {cfg.models.lightgbm.primary_metric})

{params_md}

## Feature importance — {best['stage']} (best stage), top 15 by gain

Gain = total loss reduction attributed to splits on the feature (normalized); split = number of
times it was used. Importance is descriptive, not causal.

| feature | gain share | # splits |
|---|---|---|
{imp_md}

## Artifacts

Per stage in `data/models/` (gitignored): `{{stage}}_lightgbm_{version}.txt` (portable booster) +
`.joblib`, `{{stage}}_gbm_importance_{version}.parquet`, `{{stage}}_gbm_trials_{version}.parquet`
(every grid point + validation score), `{{stage}}_gbm_calibration_val_{version}.parquet`. Plus
`gbm_metrics_{version}.parquet` and `run_meta_gbm_{version}.json`. Registered in Supabase
`model_runs` (family=lightgbm) + `metrics`.
"""
    (cfg.paths.resolve("reports") / "STU-60_gbm_models.md").write_text(report)


if __name__ == "__main__":
    main()
