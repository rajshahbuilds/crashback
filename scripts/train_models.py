#!/usr/bin/env python3
"""STU-59: train Model 0 (base rate) + Models 1-3 (logistic) for the primary target.

Loads events_v1 + splits_v1, restricts to the CLEAN determined pool, trains each incremental
stage on the TRAIN split, evaluates on TRAIN and VALIDATION (test is never touched here), and
persists per-stage model artifacts, standardized coefficients, per-split metrics, calibration
tables, and a run-metadata sidecar. Writes reports/STU-59_logistic_models.md.

Run: PYTHONPATH=src .venv/bin/python scripts/train_models.py --version v1
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
from crashback.models.logistic import (
    coefficient_table,
    fit_base_rate,
    fit_logistic,
    predict_logistic,
)
from crashback.models.stages import STAGE_DESCRIPTIONS, STAGE_GROUPS, stage_features
from crashback.storage.artifacts import current_git_commit

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
    git = current_git_commit()

    events = pl.read_parquet(proc / "events_v1.parquet")
    splits = pl.read_parquet(proc / f"splits_{args.version}.parquet").select("event_id", "split")
    df = events.join(splits, on="event_id", how="left")

    # CLEAN determined pool for the primary target (§ dataset contract).
    clean = df.filter(
        pl.col("in_universe_at_event")
        & pl.col("passes_min_price")
        & ~pl.col("censored_20d")
        & pl.col(target).is_not_null()
    )
    train = clean.filter(pl.col("split") == "train")
    val = clean.filter(pl.col("split") == "validation")
    print(f"pool: clean+determined={clean.height:,}  train={train.height:,}  val={val.height:,}")

    metric_rows: list[dict] = []
    stage_meta: list[dict] = []

    for stage, groups in STAGE_GROUPS.items():
        feats = stage_features(stage)
        y_tr, y_va = train[target], val[target]

        if stage == "model0":
            model = fit_base_rate(y_tr)
            p_tr = model.predict_proba(train.height)
            p_va = model.predict_proba(val.height)
            family, hyper = "baseline", {"rate": model.rate}
            joblib.dump(model, models_dir / f"{stage}_baseline_{args.version}.joblib")
            coef_path = None
        else:
            pipe = fit_logistic(train.select(feats), y_tr, C=mc.logistic.C,
                                max_iter=mc.logistic.max_iter, seed=mc.seed)
            p_tr = predict_logistic(pipe, train.select(feats))
            p_va = predict_logistic(pipe, val.select(feats))
            family = "logistic"
            hyper = {"C": mc.logistic.C, "max_iter": mc.logistic.max_iter, "seed": mc.seed,
                     "intercept": float(pipe.named_steps["clf"].intercept_[0])}
            joblib.dump(pipe, models_dir / f"{stage}_logistic_{args.version}.joblib")
            coef = coefficient_table(pipe, feats)
            coef_path = models_dir / f"{stage}_coef_{args.version}.parquet"
            coef.write_parquet(coef_path)

        m_tr = binary_metrics(y_tr, p_tr)
        m_va = binary_metrics(y_va, p_va)
        cal_va, ece_va = calibration_table(y_va, p_va, bins=mc.calibration_bins)
        cal_va.write_parquet(models_dir / f"{stage}_calibration_val_{args.version}.parquet")

        for split_name, m in (("train", m_tr), ("validation", m_va)):
            for k in METRIC_KEYS:
                metric_rows.append({"stage": stage, "split": split_name,
                                    "metric": k, "value": m[k]})
            ece = ece_va if split_name == "validation" else None
            metric_rows.append({"stage": stage, "split": split_name,
                                "metric": "ece", "value": ece})

        stage_meta.append({
            "stage": stage, "family": family, "n_features": len(feats),
            "feature_groups": list(groups), "hyperparams": hyper,
            "val": {k: m_va[k] for k in METRIC_KEYS}, "val_ece": ece_va,
            "coef_artifact": coef_path.name if coef_path else None,
        })
        print(f"  {stage:7s} ({STAGE_DESCRIPTIONS[stage]:28s}) "
              f"val log_loss={_fmt(m_va['log_loss'])} brier={_fmt(m_va['brier'])} "
              f"auc={_fmt(m_va['roc_auc'])} pr_auc={_fmt(m_va['pr_auc'])} ece={_fmt(ece_va)}")

    metrics = pl.DataFrame(metric_rows)
    metrics.write_parquet(models_dir / f"model_metrics_{args.version}.parquet")

    run_meta = {
        "target": target, "version": args.version, "git_commit": git,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_models": json.loads(cfg.models.model_dump_json()),
        "splits": {"train": [str(d) for d in cfg.splits.train],
                   "validation": [str(d) for d in cfg.splits.validation],
                   "test": [str(d) for d in cfg.splits.test],
                   "embargo_trading_days": cfg.splits.embargo_trading_days},
        "pool": {"clean_determined": clean.height, "train": train.height, "val": val.height},
        "stages": stage_meta,
    }
    (models_dir / f"run_meta_{args.version}.json").write_text(json.dumps(run_meta, indent=2))

    _write_report(cfg, args.version, target, stage_meta, metrics, val, models_dir)
    print(f"\nwrote artifacts to {models_dir} and reports/STU-59_logistic_models.md")


def _write_report(cfg, version, target, stage_meta, metrics, val, models_dir):
    base = next(s for s in stage_meta if s["stage"] == "model0")["val"]["prevalence"]

    hdr = ("| stage | features | log loss | Brier | ROC-AUC | PR-AUC | ECE |\n"
           "|---|---|---|---|---|---|---|")
    lines = [hdr]
    for s in stage_meta:
        v = s["val"]
        lines.append(
            f"| {s['stage']} — {STAGE_DESCRIPTIONS[s['stage']]} | {s['n_features']} | "
            f"{_fmt(v['log_loss'])} | {_fmt(v['brier'])} | {_fmt(v['roc_auc'])} | "
            f"{_fmt(v['pr_auc'])} | {_fmt(s['val_ece'])} |")
    table = "\n".join(lines)

    # top standardized coefficients for the fullest model
    m3 = models_dir / f"model3_coef_{version}.parquet"
    coef_md = ""
    if m3.exists():
        top = pl.read_parquet(m3).head(15)
        coef_md = "\n".join(
            [f"| {r['feature']} | {r['coef_std']:+.3f} | {r['direction']} |"
             for r in top.iter_rows(named=True)])

    report = f"""# STU-59 — Baseline & Logistic Recovery Models (M0–M3)

Primary target **`{target}`** = P(close ≥ +10% within 20 trading days of the crash close).
Trained on the CLEAN determined pool (`in_universe_at_event & passes_min_price & not
censored_20d & label present`). Splits from `splits_{version}` (STU-58): fit on **train**,
report on **validation** — the **test** period is untouched. Everything is config-driven
(`configs/default.yaml` → `models:`), reproducible via `scripts/train_models.py`.

## Incremental stages (CLAUDE.md §18)

- **model0** — historical base rate (predicts train prevalence for everyone).
- **model1** — crash-day + pre-crash price path + recent-crash history.
- **model2** — model1 + market / sector context.
- **model3** — model2 + point-in-time fundamentals.

Stages are nested (each a superset of the previous), so the row-to-row change isolates the
value of the added information source. Logistic pipeline = median impute (+ missingness
indicator) → standardize → L2 logistic (C={cfg.models.logistic.C}).

## Validation metrics

Base rate (Model 0 prevalence on validation) = **{base:.4f}**. Lower log loss / Brier is
better; higher AUC / PR-AUC is better; ECE closer to 0 is better-calibrated.

{table}

Δ log loss vs Model 0 quantifies incremental value; the STU-61 ticket formalizes the
M0→M1→M2→M3 comparison. Note the validation base rate differs from the train base rate — the
splits are chronological, so this is genuine regime drift, not leakage.

## Model 3 — top standardized coefficients (|coef|, direction)

Standardized so magnitudes are comparable; sign is the direction of association with recovery.
`<feature>__missing` rows are the informativeness of a feature being absent. Not causal.

| feature | coef (std) | direction |
|---|---|---|
{coef_md}

## Artifacts

Per stage in `data/models/` (gitignored): `{{stage}}_{{family}}_{version}.joblib` (fitted
pipeline / base-rate), `{{stage}}_coef_{version}.parquet` (M1–M3), `{{stage}}_calibration_val_
{version}.parquet`. Plus `model_metrics_{version}.parquet` (long-form, all stages × splits)
and `run_meta_{version}.json` (config, hyperparams, git commit, pool sizes). Registered in
Supabase `model_runs` + `metrics`.
"""
    (cfg.paths.resolve("reports") / "STU-59_logistic_models.md").write_text(report)


if __name__ == "__main__":
    main()
