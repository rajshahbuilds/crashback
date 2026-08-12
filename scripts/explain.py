#!/usr/bin/env python3
"""STU-64: per-event explainability for the locked final model (XGBoost model2).

Computes exact TreeSHAP contributions for every test event (so any prediction can be accompanied
by ranked feature contributions), documents representative TP / TN / FP / FN cases with their
strongest positive/negative drivers and the exact point-in-time feature values, and cross-checks
against per-event logistic (coefficient) contributions. Reproducible from saved artifacts.

Run: PYTHONPATH=src .venv/bin/python scripts/explain.py --version v1
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import polars as pl

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.evaluation.explain import (
    logistic_contributions,
    sigmoid,
    top_contributors,
    tree_shap,
)
from crashback.models.stages import stage_features

STAGE = "model2"
THRESHOLD = 0.5
K = 6


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg.paths.resolve("data_processed")
    models_dir = cfg.paths.resolve("data_models")
    target = PRIMARY_TARGET
    feats = stage_features(STAGE)

    events = pl.read_parquet(proc / "events_v1.parquet")
    splits = pl.read_parquet(proc / f"splits_{args.version}.parquet").select("event_id", "split")
    df = events.join(splits, on="event_id", how="left")
    test = df.filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
        & ~pl.col("censored_20d") & pl.col(target).is_not_null()
        & (pl.col("split") == "test"))

    xgb_booster = joblib.load(models_dir / f"{STAGE}_xgboost_{args.version}.joblib")
    logi_pipe = joblib.load(models_dir / f"{STAGE}_logistic_{args.version}.joblib")

    contribs, bias = tree_shap(xgb_booster, test.select(feats), feats)
    margin = contribs.sum(axis=1) + bias
    prob = sigmoid(margin)
    y = test[target].to_numpy()

    # full per-event SHAP artifact (log-odds contributions + bias + prob)
    shap_df = pl.DataFrame({"event_id": test["event_id"]}).with_columns(
        [pl.Series(f"shap__{f}", contribs[:, j]) for j, f in enumerate(feats)]
        + [pl.Series("shap_bias", bias), pl.Series("pred_prob", prob),
           pl.Series("actual", y)])
    shap_df.write_parquet(models_dir / f"test_shap_{args.version}.parquet")

    # logistic per-event contributions (for cross-check)
    lc, lintercept, lnames = logistic_contributions(logi_pipe, test.select(feats), feats)

    # representative cases: most-confident TP / TN / FP / FN at threshold 0.5
    pred_pos = prob >= THRESHOLD
    ninf = -np.inf
    cases = {
        "True Positive (correctly predicted recovery)": np.where((y == 1) & pred_pos, prob, ninf),
        "False Positive (predicted recover, did not)": np.where((y == 0) & pred_pos, prob, ninf),
        "True Negative (correctly predicted no recovery)":
            np.where((y == 0) & ~pred_pos, -prob, ninf),
        "False Negative (predicted no recovery, but did)":
            np.where((y == 1) & ~pred_pos, -prob, ninf),
    }
    fvals = test.select(feats).to_numpy()
    meta = test.select("event_id", "ticker_as_of_event", "crash_date").to_dicts()

    _write_report(cfg, args.version, target, feats, contribs, bias, prob, y, lc, lnames,
                  cases, fvals, meta, logi_pipe, test.height)
    print(f"per-event SHAP for {test.height:,} test events -> test_shap_{args.version}.parquet")
    print("representative cases (idx, ticker, pred, actual):")
    for label, score in cases.items():
        i = int(np.argmax(score))
        m = meta[i]
        print(f"  {label[:24]:24s} {m['ticker_as_of_event'] or '?':6s} "
              f"p={prob[i]:.3f} y={int(y[i])}")
    print("\nwrote reports/STU-64_explainability.md")


def _case_block(label, i, feats, contribs, bias, prob, y, fvals, meta, lc, lnames):
    m = meta[i]
    pos, neg = top_contributors(contribs[i], list(feats), k=K)
    val = {f: fvals[i, j] for j, f in enumerate(feats)}

    def rows(items):
        out = []
        for name, c in items:
            v = val.get(name)
            vtxt = _fmt(v) if v is not None else "(missing-flag)"
            out.append(f"| {name} | {vtxt} | {c:+.3f} |")
        return "\n".join(out) if out else "| — | — | — |"

    # logistic top driver agreement
    lpos, lneg = top_contributors(lc[i], lnames, k=3)
    ltop = (lpos + lneg)
    ltop_txt = ", ".join(f"{n} ({c:+.2f})" for n, c in ltop[:4]) if ltop else "—"

    tick = m['ticker_as_of_event'] or '?'
    outcome = 'recovered' if y[i] == 1 else 'did not recover'
    return f"""### {label}

**{tick}** on {m['crash_date']} — predicted P(recover) = **{prob[i]:.3f}**, actual =
**{outcome}** (event `{m['event_id']}`). Base value (bias) = {bias[i]:+.3f} log-odds.

Top drivers **up** (toward recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
{rows(pos)}

Top drivers **down** (against recovery):

| feature | value at crash | SHAP (log-odds) |
|---|---|---|
{rows(neg)}

Logistic model's strongest terms for the same event: {ltop_txt}.
"""


def _write_report(cfg, version, target, feats, contribs, bias, prob, y, lc, lnames,
                  cases, fvals, meta, logi_pipe, n_test):
    blocks = []
    for label, score in cases.items():
        i = int(np.argmax(score))
        if not np.isfinite(score[i]):
            continue
        blocks.append(_case_block(label, i, feats, contribs, bias, prob, y, fvals, meta,
                                  lc, lnames))

    # global logistic direction reference (mean |contribution| ranks importance)
    mean_abs = np.abs(lc).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]
    coef = logi_pipe.named_steps["clf"].coef_.ravel()
    glob = "\n".join(
        f"| {lnames[j]} | {coef[j]:+.3f} | {'↑ recovery' if coef[j] > 0 else '↓ recovery'} |"
        for j in order)

    report = f"""# STU-64 — Per-Event Explainability (XGBoost model2)

Explanations for the locked final model on the held-out test set ({n_test:,} events), primary
target **`{target}`**. Both methods are **additive in log-odds space**: per-feature
contributions + a bias sum to the raw margin, and `sigmoid(margin)` = predicted P(recover).
Positive ⇒ pushes toward recovery, negative ⇒ against.

- **Tree model:** exact **TreeSHAP** via XGBoost's native `pred_contribs` (no external `shap`
  dependency).
- **Logistic model:** per-event term `coef · standardized_value`, shown as a cross-check.
- Values shown are the **exact point-in-time features** recorded at the crash close.

Per-event SHAP for *every* test event is saved to `data/models/test_shap_{version}.parquet`
(`shap__<feature>` columns + `shap_bias` + `pred_prob`), so any single prediction can be
decomposed into ranked contributions. SHAP is descriptive attribution, **not causal**.

## Representative cases (most-confident TP / FP / TN / FN)

{chr(10).join(blocks)}

## Global logistic direction (mean |contribution| ranks; sign = direction)

For the linear model, coefficient sign gives a stable global reading of each feature's direction.

| feature | coef (std) | direction |
|---|---|---|
{glob}

## Notes

- Reproducible from saved artifacts (`model2_xgboost_{version}.joblib`,
  `model2_logistic_{version}.joblib`); deterministic.
- Contributions are attributions of *this model's* output, not causal effects on recovery.
- The market-context features dominate individual explanations, consistent with the STU-60
  global importance and the STU-63 finding that broad-market conditions drive most of the signal.
"""
    (cfg.paths.resolve("reports") / "STU-64_explainability.md").write_text(report)


if __name__ == "__main__":
    main()
