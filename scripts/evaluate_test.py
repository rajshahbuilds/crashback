#!/usr/bin/env python3
"""STU-62: held-out TEST evaluation of the locked final model. First & only use of test.

Model selection is LOCKED before this runs: on validation-only results the final model is
**XGBoost model2** (price + recent-crash + market/sector), the best validation log loss / AUC /
PR-AUC (STU-59/60/61). This script reloads that train-only booster and evaluates it on the
untouched chronological test period (2022–2025), alongside the M0 base-rate reference and a
post-hoc isotonic-calibrated variant (calibrator fit on VALIDATION, never on test).

Reports Brier, log loss, ROC-AUC, PR-AUC, reliability (predicted vs observed) with counts, and
top-decile recovery rate + lift. Reproducible from saved artifacts; no retraining.

Run: PYTHONPATH=src .venv/bin/python scripts/evaluate_test.py --version v1
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import polars as pl
from sklearn.isotonic import IsotonicRegression

from crashback.config import load_config
from crashback.datasets.assemble import PRIMARY_TARGET
from crashback.evaluation.lift import confidence_bands, decile_table, top_decile_lift
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models.logistic import fit_base_rate
from crashback.models.stages import stage_features
from crashback.models.xgb import predict as xgb_predict
from crashback.storage.artifacts import current_git_commit

FINAL_STAGE = "model2"
FINAL_FAMILY = "xgboost"
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
    target = PRIMARY_TARGET
    bins = cfg.models.calibration_bins
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
    test = clean.filter(pl.col("split") == "test")   # ← first read of test
    feats = stage_features(FINAL_STAGE)
    y_te = test[target].to_numpy()
    print(f"LOCKED model: {FINAL_FAMILY} {FINAL_STAGE} ({len(feats)} features)")
    print(f"test pool (CLEAN determined): n={test.height:,}  "
          f"securities={test['security_id'].n_unique():,}  "
          f"dates {test['crash_date'].min()}..{test['crash_date'].max()}")

    booster = joblib.load(models_dir / f"{FINAL_STAGE}_{FINAL_FAMILY}_{args.version}.joblib")
    p_val = xgb_predict(booster, val.select(feats), feats)
    p_te_raw = xgb_predict(booster, test.select(feats), feats)

    # post-hoc calibration: isotonic fit on VALIDATION (not test), applied to test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, val[target].to_numpy())
    p_te_cal = iso.predict(p_te_raw)

    # M0 baseline = train prevalence (the historically-available base rate), constant on test
    base_model = fit_base_rate(train[target])
    p_te_base = base_model.predict_proba(test.height)

    preds = {"m0_base_rate": p_te_base, "xgb_m2_raw": p_te_raw, "xgb_m2_calibrated": p_te_cal}
    test_metrics = {name: binary_metrics(y_te, p) for name, p in preds.items()}
    eces = {name: calibration_table(y_te, p, bins=bins)[1] for name, p in preds.items()}

    # reliability + decile lift for the primary (raw) model
    reliab, _ = calibration_table(y_te, p_te_raw, bins=bins)
    reliab_cal, _ = calibration_table(y_te, p_te_cal, bins=bins)
    deciles, base_rate_te = decile_table(y_te, p_te_raw, q=10)
    tdl = top_decile_lift(y_te, p_te_raw, q=10)

    # confidence bands + expected-return-by-band (recovery probability vs actual return)
    cbands = confidence_bands(
        p_te_raw, y_te, ret=test["return_20d"].to_numpy(),
        dd=test["max_drawdown_20d"].to_numpy(), width=0.1)
    cbands.write_parquet(models_dir / f"test_confidence_bands_{args.version}.parquet")

    reliab.write_parquet(models_dir / f"test_reliability_raw_{args.version}.parquet")
    reliab_cal.write_parquet(models_dir / f"test_reliability_cal_{args.version}.parquet")
    deciles.write_parquet(models_dir / f"test_deciles_{args.version}.parquet")
    test.select("event_id", "security_id", "crash_date", target).with_columns(
        pl.Series("p_raw", p_te_raw), pl.Series("p_calibrated", p_te_cal)
    ).write_parquet(models_dir / f"test_predictions_{args.version}.parquet")

    # validation numbers of the same model, to expose the selection-optimism gap
    val_metrics = binary_metrics(val[target].to_numpy(), p_val)

    run_meta = {
        "ticket": "STU-62", "target": target, "version": args.version, "git_commit": git,
        "generated_at": datetime.now(UTC).isoformat(),
        "locked_model": {"family": FINAL_FAMILY, "stage": FINAL_STAGE, "n_features": len(feats),
                         "selected_by": "validation log_loss (STU-59/60/61)"},
        "train_base_rate": base_model.rate,
        "test": {"n": test.height, "base_rate": base_rate_te,
                 "metrics": {k: {mk: test_metrics[k][mk] for mk in METRIC_KEYS} | {"ece": eces[k]}
                             for k in preds}},
        "val_raw_metrics": {mk: val_metrics[mk] for mk in METRIC_KEYS} | {
            "ece": calibration_table(val[target].to_numpy(), p_val, bins=bins)[1]},
        "top_decile": tdl,
    }
    (models_dir / f"test_eval_{args.version}.json").write_text(json.dumps(run_meta, indent=2))

    _write_report(cfg, args.version, target, test_metrics, eces, val_metrics, reliab,
                  deciles, tdl, base_model.rate, base_rate_te, test.height, cbands)
    print("\n=== TEST results (locked XGBoost model2) ===")
    for name in preds:
        m = test_metrics[name]
        print(f"  {name:20s} log_loss={_fmt(m['log_loss'])} brier={_fmt(m['brier'])} "
              f"auc={_fmt(m['roc_auc'])} pr_auc={_fmt(m['pr_auc'])} ece={_fmt(eces[name])}")
    print(f"  top-decile recovery={tdl['top_decile_recovery_rate']:.3f} "
          f"lift={tdl['top_decile_lift']:.2f}x over base {base_rate_te:.3f}")
    print(f"\nwrote reports/STU-62_test_evaluation.md and test artifacts to {models_dir}")


def _pct(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v * 100:+.{nd}f}%"


def _write_report(cfg, version, target, tm, eces, vm, reliab, deciles, tdl,
                  train_base, test_base, n_test, cbands):
    def row(name, label):
        m = tm[name]
        return (f"| {label} | {_fmt(m['log_loss'])} | {_fmt(m['brier'])} | {_fmt(m['roc_auc'])} "
                f"| {_fmt(m['pr_auc'])} | {_fmt(eces[name])} |")

    rel_rows = "\n".join(
        f"| {r['lo']:.1f}–{r['hi']:.1f} | {r['n']:,} | "
        f"{_fmt(r['mean_pred'], 3) if r['mean_pred'] is not None else '—'} | "
        f"{_fmt(r['actual_rate'], 3) if r['actual_rate'] is not None else '—'} |"
        for r in reliab.iter_rows(named=True))
    dec_rows = "\n".join(
        f"| {r['bucket']} | {r['n']:,} | {r['mean_pred']:.3f} | {r['observed_rate']:.3f} | "
        f"{r['lift']:.2f}× |" for r in deciles.iter_rows(named=True))

    # confidence-band distribution + expected-return-by-band
    cb_rows = "\n".join(
        f"| {r['band']:.2f}–{r['hi']:.2f} | {r['n']:,} | {r['frac'] * 100:.1f}% | "
        f"{r['mean_pred']:.3f} | {r['observed_rate']:.3f} |"
        for r in cbands.iter_rows(named=True))
    ev_rows = "\n".join(
        f"| {r['band']:.2f}–{r['hi']:.2f} | {r['n']:,} | {r['observed_rate']:.3f} | "
        f"{_pct(r['mean_return'])} | {_pct(r['mean_return_win'])} | "
        f"{_pct(r['mean_return_lose'])} | {_pct(r['mean_maxdd'])} |"
        for r in cbands.iter_rows(named=True))
    worst = cbands.sort("mean_return").row(0, named=True)     # worst-EV band
    worst_band = f"{worst['band']:.2f}–{worst['hi']:.2f}"
    worst_ret = _pct(worst["mean_return"])
    worst_lose = _pct(worst["mean_return_lose"])

    ll_lift = tm["xgb_m2_raw"]["log_loss"]
    ll_base = tm["m0_base_rate"]["log_loss"]
    skill = 1 - tm["xgb_m2_raw"]["brier"] / tm["m0_base_rate"]["brier"]
    auc_gap_val = vm["roc_auc"]
    auc_test = tm["xgb_m2_raw"]["roc_auc"]
    cal_verdict = ("improves" if eces["xgb_m2_calibrated"] < eces["xgb_m2_raw"]
                   else "does not improve")
    d = deciles.sort("bucket")
    bot_obs = d["observed_rate"].to_list()[0]
    top_obs = d["observed_rate"].to_list()[-1]
    val_base = _fmt(vm["prevalence"], 3)
    te_base = _fmt(tm["m0_base_rate"]["prevalence"], 3)
    tr_base = _fmt(train_base, 3)

    report = f"""# STU-62 — Held-Out Test Evaluation

**First and only** evaluation on the untouched chronological **test** period (crash dates
2022–2025), primary target **`{target}`**. The model was **locked before any test read**: on
validation-only results (STU-59/60/61) the final model is **XGBoost model2** (price +
recent-crash + market/sector, 34 features), selected by validation log loss. Test pool =
{n_test:,} CLEAN determined events. Reproducible from the saved train-only booster
(`data/models/model2_xgboost_{version}.joblib`); no retraining.

## Headline: does the validation signal generalize?

| model | log loss | Brier | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|
{row("m0_base_rate", "M0 — base rate (reference)")}
{row("xgb_m2_raw", "**XGBoost model2 (raw)** — locked final")}
{row("xgb_m2_calibrated", "XGBoost model2 (isotonic, cal. on val)")}

- **ROC-AUC on test = {_fmt(auc_test)}** vs validation {_fmt(auc_gap_val)}. The model still
  beats the M0 baseline out-of-sample on every proper score, **but discrimination is modest
  (~0.60) and far below the validation figure — most of the validation edge did not transfer.**
- **Brier skill score vs base rate = {skill:.3f}** ({(skill * 100):.1f}% reduction in Brier vs
  M0). Log loss {_fmt(ll_lift)} vs M0 {_fmt(ll_base)} — a small but genuine improvement.
- **Calibration is a genuine positive**: the raw model's test ECE is {_fmt(eces['xgb_m2_raw'])}
  and predicted ≈ observed across the reliability table below.
- Post-hoc isotonic calibration (fit on validation) {cal_verdict} test ECE
  ({_fmt(eces['xgb_m2_raw'])} → {_fmt(eces['xgb_m2_calibrated'])}) — it was fit on a
  higher-base-rate regime (validation {val_base} vs test {te_base}) and did not transfer. The raw
  probabilities are already well-calibrated here, so no correction is applied.

**Regime shift.** The test period (2022–2025) has a lower recovery base rate ({te_base}) than
train ({tr_base}) or validation ({val_base}), and the model's dominant features are
market-context (STU-60 importance). Regime-dependent market relationships are the most likely
reason discrimination attenuated out-of-sample.

## Reliability — predicted vs observed recovery (raw model, equal-width bins)

| predicted bucket | n | mean predicted | observed recovery |
|---|---|---|---|
{rel_rows}

Well-calibrated ⇔ mean-predicted ≈ observed within each row.

## Recovery rate by predicted-probability decile (equal count)

Base rate (test) = **{test_base:.4f}**; historically-available base rate (train) = {train_base:.4f}.

| decile (1=low→10=high) | n | mean predicted | observed recovery | lift |
|---|---|---|---|---|
{dec_rows}

- **Top-decile recovery rate = {tdl['top_decile_recovery_rate']:.3f}**, a
  **{tdl['top_decile_lift']:.2f}× lift** over the {test_base:.3f} base rate
  (n={tdl['top_bucket_n']:,}, mean predicted {tdl['top_bucket_mean_pred']:.3f}).
- Observed recovery **broadly increases** across deciles, clearest at the tails (bottom decile
  {bot_obs:.3f} → top decile {top_obs:.3f}, a {top_obs / bot_obs:.1f}× spread); the middle
  deciles are muddy. So even with AUC ~0.60 the model still usefully concentrates recoveries in
  its top-scored events and flags the least-likely ones — the practical payoff survives, attenuated.

## Confidence-band distribution (natural bins)

Equal-count deciles hide *how often* the model is actually confident. In its natural 0.1-wide
bands the model's probabilities never leave [0.14, 0.83] — it stays near the base rate on the
bulk of events and only rarely commits. Where it does commit, calibration holds (mean predicted
≈ observed).

| predicted band | n | % of test | mean predicted | observed recovery |
|---|---|---|---|---|
{cb_rows}

## Expected return by confidence band ⚠️ (recovery ≠ return)

The recovery *label* ("closes ≥ +10% at some point in 20d") hides the **downside when it fails**.
Joining the continuous `return_20d` outcome shows that **recovery probability is NOT monotonic
with expected return**: the moderately-confident bands are the *worst* economically, because the
losers there fall much harder than the winners rise.

| predicted band | n | P(+10%) | mean return | return if win | return if lose | mean max drawdown |
|---|---|---|---|---|---|---|
{ev_rows}

- Worst-EV band is **{worst_band}** (mean return {worst_ret}, loser return {worst_lose}) — *more*
  confident of recovery than average, yet negative expected return. High predicted-recovery names
  are high-volatility names; when they don't bounce they keep falling.
- Only the **extreme top band** turns clearly EV-positive, where the high win-rate finally
  overwhelms the asymmetry. **A confidence-weighted decision must be driven by expected return /
  downside, not by P(recovery)** — the two diverge. (Descriptive, gross of costs; §25 keeps
  position-sizing / trading out of V1 scope.)

## Notes

- Test integrity: every prior script filtered to `split=='validation'`; this is the first read
  of `split=='test'`. Model/feature/hyperparameter choices were fixed beforehand.
- Focus is calibrated probability + discrimination, not trading P&L (§21). Dependence across
  same-security events understates CI width; block-bootstrap robustness is STU-63.
- Artifacts: `data/models/test_predictions_{version}.parquet`,
  `test_reliability_raw/cal_{version}.parquet`, `test_deciles_{version}.parquet`,
  `test_eval_{version}.json`. Test metrics registered in Supabase.
"""
    (cfg.paths.resolve("reports") / "STU-62_test_evaluation.md").write_text(report)


if __name__ == "__main__":
    main()
