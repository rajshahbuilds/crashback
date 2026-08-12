#!/usr/bin/env python3
"""STU-61: incremental predictive value across feature stages, with bootstrap uncertainty.

Builds a fine ablation ladder (base → price → +recent-crash → +market/sector → +fundamentals)
for the logistic and XGBoost families on the primary target, reusing the already-trained
model1/2/3 rungs and fitting only the new price-only rung. Attributes a clean marginal
contribution to each information source and puts a clustered-bootstrap 95% CI on every delta
(resampling by security, per CLAUDE.md §22). Evaluated on VALIDATION; test untouched.

Run: PYTHONPATH=src .venv/bin/python scripts/compare_stages.py --version v1
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import polars as pl

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.evaluation.compare import paired_cluster_bootstrap
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models import gbm, xgb
from crashback.models.logistic import fit_base_rate, fit_logistic, predict_logistic
from crashback.models.stages import (
    ABLATION_LADDER,
    INCREMENTS,
    LADDER_TO_STAGE,
    ladder_features,
)

RUNGS = list(ABLATION_LADDER)
N_BOOT = 500


def _fit_price(family, X_tr, y_tr, X_val, y_val, feats, cfg):
    """Fit the one new ladder rung (price-only) for a family and return its validation preds."""
    if family == "logistic":
        pipe = fit_logistic(X_tr, y_tr, C=cfg.models.logistic.C,
                            max_iter=cfg.models.logistic.max_iter, seed=cfg.models.seed)
        return predict_logistic(pipe, X_val)
    if family == "lightgbm":
        res = gbm.tune(X_tr, y_tr, X_val, y_val, cfg.models.lightgbm, cfg.models.seed,
                       feature_names=feats)
        return gbm.predict(res.booster, X_val)
    res = xgb.tune(X_tr, y_tr, X_val, y_val, cfg.models.xgboost, cfg.models.seed,
                   feature_names=feats)
    return xgb.predict(res.booster, X_val, feats)


def _predict_reloaded(family, booster, X_val, feats):
    if family == "logistic":
        return predict_logistic(booster, X_val)
    if family == "lightgbm":
        return gbm.predict(booster, X_val)
    return xgb.predict(booster, X_val, feats)


def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def _signed(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:+.{nd}f}"


def _ladder_preds(family, train, val, target, models_dir, cfg, version):
    """Predicted P(hit) on validation for each ladder rung of one family."""
    y_tr, y_va = train[target], val[target]
    preds: dict[str, np.ndarray] = {}
    # base rung = historical base rate (family-independent, but kept per family for paired deltas)
    preds["base"] = fit_base_rate(y_tr).predict_proba(val.height)

    # price-only rung: the one model not already trained by STU-59/60
    feats = ladder_features("price")
    preds["price"] = _fit_price(family, train.select(feats), y_tr, val.select(feats), y_va,
                                feats, cfg)

    # remaining rungs coincide with trained staged models → reload and predict
    for rung, stage in LADDER_TO_STAGE.items():
        feats = ladder_features(rung)
        booster = joblib.load(models_dir / f"{stage}_{family}_{version}.joblib")
        preds[rung] = _predict_reloaded(family, booster, val.select(feats), feats)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--families", default="logistic,lightgbm,xgboost")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    models_dir = cfg.paths.resolve("data_models")
    target = PRIMARY_TARGET
    families = args.families.split(",")

    events = pl.read_parquet(proc / "events_v1.parquet")
    splits = pl.read_parquet(proc / f"splits_{args.version}.parquet").select("event_id", "split")
    df = events.join(splits, on="event_id", how="left")
    clean = df.filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
        & ~pl.col("censored_20d") & pl.col(target).is_not_null()
    )
    train = clean.filter(pl.col("split") == "train")
    val = clean.filter(pl.col("split") == "validation")
    clusters = val["security_id"].to_numpy()
    y_va = val[target]
    print(f"pool: train={train.height:,}  val={val.height:,}  "
          f"val securities={val['security_id'].n_unique():,}  bootstrap={N_BOOT}")

    all_rungs: list[dict] = []
    all_deltas: list[pl.DataFrame] = []
    for family in families:
        preds = _ladder_preds(family, train, val, target, models_dir, cfg, args.version)
        for rung in RUNGS:
            m = binary_metrics(y_va, preds[rung])
            _, ece = calibration_table(y_va, preds[rung], bins=cfg.models.calibration_bins)
            all_rungs.append({"family": family, "rung": rung,
                              "n_features": len(ladder_features(rung)),
                              "log_loss": m["log_loss"], "brier": m["brier"],
                              "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "ece": ece})
        deltas, _ = paired_cluster_bootstrap(
            y_va, preds, clusters, INCREMENTS,
            metrics=("log_loss", "roc_auc"), n_boot=N_BOOT, seed=cfg.models.seed)
        all_deltas.append(deltas.with_columns(pl.lit(family).alias("family")))
        print(f"  {family}: ladder + bootstrap done")

    rungs_df = pl.DataFrame(all_rungs)
    deltas_df = pl.concat(all_deltas)
    rungs_df.write_parquet(models_dir / f"stage_ladder_{args.version}.parquet")
    deltas_df.write_parquet(models_dir / f"stage_increments_{args.version}.parquet")

    _write_report(cfg, args.version, target, families, rungs_df, deltas_df)
    print(f"\nwrote {models_dir}/stage_increments_{args.version}.parquet and "
          "reports/STU-61_incremental_value.md")


def _verdict(row) -> str:
    if not row["material"]:
        return "no material effect (CI spans 0)"
    improves = (row["delta"] < 0) if row["metric"] == "log_loss" else (row["delta"] > 0)
    return "**improves**" if improves else "**hurts**"


def _write_report(cfg, version, target, families, rungs_df, deltas_df):
    parts = [f"""# STU-61 — Incremental Predictive Value Across Feature Stages

How much does each added information source actually improve recovery prediction for the primary
target **`{target}`**? Evaluated on the **validation** period (STU-58 splits); **test** untouched.
A fine ablation ladder splits price from recent-crash so each source gets a clean marginal
contribution. Uncertainty is a **paired clustered bootstrap** ({N_BOOT} resamples of *securities*,
not events — CLAUDE.md §22), so a delta is "material" only if its 95% CI excludes 0.

Focus is **discrimination + calibration**, not trading P&L. Note (as in STU-59/60) the tree
rungs were tuned on validation, so absolute levels are mildly optimistic; the *deltas* between
nested rungs are the robust quantity here, and the held-out test read is STU-62."""]

    for family in families:
        rr = rungs_df.filter(pl.col("family") == family)
        lad = ["\n## " + family.capitalize() + " — ladder (validation)\n",
               "| rung | features | log loss | Brier | ROC-AUC | PR-AUC | ECE |",
               "|---|---|---|---|---|---|---|"]
        for r in rr.iter_rows(named=True):
            lad.append(f"| {r['rung']} | {r['n_features']} | {_fmt(r['log_loss'])} | "
                       f"{_fmt(r['brier'])} | {_fmt(r['roc_auc'])} | {_fmt(r['pr_auc'])} | "
                       f"{_fmt(r['ece'])} |")
        parts.append("\n".join(lad))

        dd = deltas_df.filter(pl.col("family") == family)
        inc = ["\n### Marginal value of each source (Δ vs previous rung, 95% CI)\n",
               "| added source | Δ log loss (95% CI) | Δ ROC-AUC (95% CI) | verdict |",
               "|---|---|---|---|"]
        for src in [i[2] for i in INCREMENTS]:
            sd = dd.filter(pl.col("source") == src)
            ll = sd.filter(pl.col("metric") == "log_loss").row(0, named=True)
            au = sd.filter(pl.col("metric") == "roc_auc").row(0, named=True)
            inc.append(
                f"| {src} | {_signed(ll['delta'])} "
                f"([{_signed(ll['ci_lo'])}, {_signed(ll['ci_hi'])}]) | "
                f"{_signed(au['delta'])} ([{_signed(au['ci_lo'])}, {_signed(au['ci_hi'])}]) | "
                f"{_verdict(ll)} |")
        parts.append("\n".join(inc))

    # cross-family synthesis on the primary proper score (log loss)
    syn = ["\n## Cross-family synthesis (Δ log loss verdict per source)\n",
           "| added source | " + " | ".join(families) + " |",
           "|---|" + "---|" * len(families)]
    ll_all = deltas_df.filter(pl.col("metric") == "log_loss")
    for src in [i[2] for i in INCREMENTS]:
        cells = []
        for fam in families:
            fd = ll_all.filter((pl.col("family") == fam) & (pl.col("source") == src))
            r = fd.row(0, named=True)
            tag = _verdict(r).replace("**", "")
            cells.append(f"{_signed(r['delta'])} — {tag.split(' (')[0]}")
        syn.append(f"| {src} | " + " | ".join(cells) + " |")
    parts.append("\n".join(syn))

    parts.append("""
## Conclusions

- **Δ log loss < 0 = improvement** (lower is better); **Δ ROC-AUC > 0 = improvement**. A source
  is "material" only when its 95% clustered-bootstrap CI excludes 0; the sign of Δ then says
  whether it helps or hurts. Each row is the *marginal* value of that source given everything
  below it on the ladder.
- **Agreement across model families** is the real test: a source carries robust short-term
  recovery signal only if the sign of its effect is consistent whether the model is linear
  (logistic) or nonlinear (LightGBM, XGBoost). Where the families disagree on sign, the effect
  is not a dependable phenomenon — it is model-specific noise.""")

    (cfg.paths.resolve("reports") / "STU-61_incremental_value.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
