#!/usr/bin/env python3
"""Headline figures + metrics for the recovery model: the pooled expanding-window walk-forward.

Reads the pooled out-of-sample predictions (scripts/train_production.py ->
data/processed/pred_walkforward_oos.parquet) and the saved production model, and renders three
figures in the paper's house style plus the metrics for the LaTeX tables:
  paper/figures/model_vs_m0.pdf       --- actual recovery rate per predicted decile vs base rate
  paper/figures/model_reliability.pdf --- pooled reliability (predicted vs actual)
  paper/figures/model_importance.pdf  --- production-model gain importance

Model 0 is time-aware: for each year Y it forecasts the recovery base rate of the data available
to train on then (crash_date <= Y-2), so the comparison is honest.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_walkforward_results.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashback.analysis.recovery import one_year_returns  # noqa: E402
from crashback.analysis.recovery_model import BASE_FEATURES, REGIME_FEATURES  # noqa: E402
from crashback.config import load_config  # noqa: E402
from crashback.evaluation.metrics import binary_metrics, calibration_table  # noqa: E402
from crashback.models.xgb import importance_table  # noqa: E402

BAR, INK, MUTED, GREEN, RED = "#4C72B0", "#222222", "#8A8A8A", "#55A868", "#C44E52"
COLS = BASE_FEATURES + REGIME_FEATURES
FIGDIR = Path("paper/figures")
PRETTY = {
    "log_market_cap": "log market cap", "crash_return": "crash severity",
    "prior_crash_count_20d": "prior crashes (20d)", "pe": "P/E",
    "return_252d_pre": "pre-crash 1yr return", "beta": "beta (252d)",
    "ebitda_margin": "EBITDA margin", "revenue_growth_yoy": "revenue growth YoY",
    "market_return_20d": "market 20d return", "rel_sector": "return vs sector",
    "mkt_ret_126d": "market 126d return", "mkt_ret_252d": "market 252d return",
    "mkt_drawdown_252d": "market drawdown (52w)", "mkt_vol_60d": "market volatility (60d)",
    "crash_breadth_20d": "crash breadth (20d)",
}


def m0_forecast(cfg, pooled: pl.DataFrame) -> np.ndarray:
    """Time-aware Model 0: each year's forecast is the base rate of data trainable then (<= Y-2)."""
    allev = one_year_returns(cfg).with_columns(y=(pl.col("ret") > 0).cast(pl.Int8))
    prev = {}
    for Y in sorted(pooled["year"].unique().to_list()):
        prev[Y] = float(allev.filter(pl.col("crash_date") <= date(Y - 2, 12, 31))["y"].mean())
    return pooled["year"].replace_strict(prev).cast(pl.Float64).to_numpy()


def decile_fig(pooled, base, out: Path):
    p, y = pooled["p"].to_numpy(), pooled["y"].to_numpy().astype(float)
    groups = np.array_split(np.argsort(p), 10)
    rates = np.array([y[g].mean() for g in groups])
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.bar(np.arange(1, 11), rates, color=BAR, edgecolor="white", linewidth=0.4, zorder=2)
    ax.axhline(base, color=RED, lw=1.6, ls="--", zorder=3)
    ax.text(0.6, base + 0.012, f"base rate = {base:.0%}", color=RED, fontsize=8,
            ha="left", va="bottom")
    ax.text(0.98, 0.03, f"bottom decile {rates[0]:.0%}  |  top decile {rates[-1]:.0%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylabel("Actual recovery rate", fontsize=9, color=INK)
    ax.set_xlabel("Predicted-probability decile (low $\\rightarrow$ high)", fontsize=9, color=INK)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def reliability_fig(pooled, cfg, out: Path):
    y, p = pooled["y"].to_numpy(), pooled["p"].to_numpy()
    tbl, ece = calibration_table(y, p, cfg.models.calibration_bins)
    c = tbl.filter(pl.col("actual_rate").is_not_null() & (pl.col("n") > 0)).sort("mean_pred")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK, lw=1.0, ls="--", zorder=1)
    ax.text(0.97, 0.99, "perfect calibration", color=INK, fontsize=7.5,
            rotation=45, ha="right", va="top", rotation_mode="anchor")
    ax.plot(c["mean_pred"].to_numpy(), c["actual_rate"].to_numpy(), color=BAR, lw=1.8,
            marker="o", ms=5, zorder=3, label=f"walk-forward   ECE={ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(earn money in a year)", fontsize=9, color=INK)
    ax.set_ylabel("Actual recovery rate", fontsize=9, color=INK)
    ax.grid(color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def importance_fig(cfg, out: Path):
    booster = xgb.Booster()
    booster.load_model(str(cfg.paths.resolve("data_models") / "recovery_prod.json"))
    imp = importance_table(booster, COLS).sort("gain_frac")
    feats = imp["feature"].to_list()
    colors = [GREEN if f in REGIME_FEATURES else BAR for f in feats]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(range(len(feats)), imp["gain_frac"].to_numpy(), color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([PRETTY.get(f, f) for f in feats], fontsize=8, color=INK)
    ax.set_xlabel("Importance (fraction of total gain)", fontsize=9, color=INK)
    ax.grid(axis="x", color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAR), plt.Rectangle((0, 0), 1, 1, color=GREEN)]
    ax.legend(handles, ["per-stock features", "market-regime features"],
              loc="lower right", fontsize=8, frameon=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    cfg = load_config()
    pooled = pl.read_parquet(cfg.paths.resolve("data_processed") / "pred_walkforward_oos.parquet")
    y, p = pooled["y"].to_numpy(), pooled["p"].to_numpy()
    m = binary_metrics(y, p)
    _, ece = calibration_table(y, p, cfg.models.calibration_bins)
    m0p = m0_forecast(cfg, pooled)
    m0 = binary_metrics(y, m0p)
    _, m0_ece = calibration_table(y, m0p, cfg.models.calibration_bins)

    print(f"pooled OOS n={m['n']}  base(actual)={m['prevalence']:.3f}")
    print(f"{'metric':10s}{'Model 0':>12s}{'walk-fwd':>12s}")
    for k in ("log_loss", "brier", "roc_auc", "pr_auc"):
        print(f"{k:10s}{m0[k]:12.4f}{m[k]:12.4f}")
    print(f"{'ece':10s}{m0_ece:12.4f}{ece:12.4f}")

    # decile earn / median / mean ROI (for the LaTeX table)
    groups = np.array_split(np.argsort(p), 10)
    ret = pooled["ret"].to_numpy()
    print("\ndecile  earn   median_roi  mean_roi")
    for i, g in enumerate(groups, 1):
        print(f"  {i:2d}  {(ret[g] > 0).mean():.3f}  "
              f"{np.median(ret[g]):+.3f}  {ret[g].mean():+.3f}")

    decile_fig(pooled, float(m["prevalence"]), FIGDIR / "model_vs_m0.pdf")
    reliability_fig(pooled, cfg, FIGDIR / "model_reliability.pdf")
    importance_fig(cfg, FIGDIR / "model_importance.pdf")


if __name__ == "__main__":
    main()
