#!/usr/bin/env python3
"""Figures + metrics for the recovery-model section.

Fits the model under three organizations (chronological base / chronological +regime / security-
level) via crashback.analysis.recovery_model, prints a metrics table for the LaTeX writeup, and
renders two figures in the paper's house style:
  paper/figures/model_reliability.pdf  --- reliability diagram, chrono base vs +regime
  paper/figures/model_importance.pdf   --- gain importance of the 15 features (chrono +regime)

Run: PYTHONPATH=src .venv/bin/python scripts/fig_model_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashback.analysis.recovery_model import (  # noqa: E402
    REGIME_FEATURES,
    assemble,
    fit_predict,
)
from crashback.config import load_config  # noqa: E402

BAR, INK, MUTED = "#4C72B0", "#222222", "#8A8A8A"
GREEN = "#55A868"
RED = "#C44E52"
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


def reliability_fig(base, regime, out: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK, lw=1.0, ls="--", zorder=1)
    ax.text(0.97, 0.99, "perfect calibration", color=INK, fontsize=7.5,
            rotation=45, ha="right", va="top", rotation_mode="anchor")
    for r, color, name in ((base, MUTED, "base (10 feat)"), (regime, BAR, "+ regime (15 feat)")):
        c = r.calib.filter(pl_notnull()).sort("mean_pred")
        x = c["mean_pred"].to_numpy()
        y = c["actual_rate"].to_numpy()
        ax.plot(x, y, color=color, lw=1.8, marker="o", ms=5, zorder=3,
                label=f"{name}   ECE={r.ece:.3f}")
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


def _decile_panel(ax, r, title):
    """Bars = actual recovery rate per predicted-probability decile.

    Two references: the realized test base rate (the no-skill line the staircase should straddle)
    and Model 0's flat forecast (the train base rate, which over-shoots the test period).
    """
    p = r.test["p"].to_numpy()
    y = r.test["y"].to_numpy().astype(float)
    order = np.argsort(p)
    groups = np.array_split(order, 10)  # equal-count deciles, lowest predicted first
    rates = np.array([y[g].mean() for g in groups])
    base = r.metrics["prevalence"]     # realized test-period recovery rate (no-skill reference)
    m0 = r.m0["mean_pred"]             # Model 0's flat forecast (train base rate)

    x = np.arange(1, 11)
    ax.bar(x, rates, color=BAR, edgecolor="white", linewidth=0.4, zorder=2)
    ax.axhline(base, color=RED, lw=1.6, ls="--", zorder=3)
    ax.text(0.6, base + 0.012, f"base rate = {base:.0%}", color=RED, fontsize=8,
            ha="left", va="bottom")
    ax.axhline(m0, color=MUTED, lw=1.4, ls=":", zorder=3)
    ax.text(10.4, m0 + 0.012, f"Model 0 forecast = {m0:.0%}", color=MUTED, fontsize=8,
            ha="right", va="bottom")
    ax.text(0.98, 0.03, f"bottom decile {rates[0]:.0%}  |  top decile {rates[-1]:.0%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("Predicted-probability decile (low $\\rightarrow$ high)", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def decile_fig(chrono, out: Path):
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    _decile_panel(ax, chrono, "Chronological (out-of-time)")
    ax.set_ylabel("Actual recovery rate", fontsize=9, color=INK)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def importance_fig(regime, out: Path):
    imp = regime.importance.sort("gain_frac")
    feats = imp["feature"].to_list()
    vals = imp["gain_frac"].to_numpy()
    colors = [GREEN if f in REGIME_FEATURES else BAR for f in feats]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(range(len(feats)), vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([PRETTY.get(f, f) for f in feats], fontsize=8, color=INK)
    ax.set_xlabel("Importance (fraction of total gain)", fontsize=9, color=INK)
    ax.grid(axis="x", color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAR),
               plt.Rectangle((0, 0), 1, 1, color=GREEN)]
    ax.legend(handles, ["per-stock features", "market-regime features"],
              loc="lower right", fontsize=8, frameon=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# tiny helper: keep only reliability bins that actually have events
def pl_notnull():
    import polars as pl
    return pl.col("actual_rate").is_not_null() & (pl.col("n") > 0)


def metrics_row(name, r):
    m, m0 = r.metrics, r.m0
    print(f"{name:22s} n={m['n']:6d}  logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
          f"auc={m['roc_auc']:.4f}  prauc={m['pr_auc']:.4f}  ece={r.ece:.4f}  "
          f"| M0 logloss={m0['log_loss']:.4f} brier={m0['brier']:.4f} prauc={m0['pr_auc']:.4f}")


def main():
    cfg = load_config()
    print("assembling features ...")
    df = assemble(cfg)
    base = fit_predict(df, cfg, regime=False)
    regime = fit_predict(df, cfg, regime=True)

    print("\n===== metrics for the LaTeX table =====")
    metrics_row("chrono base", base)
    metrics_row("chrono +regime", regime)

    reliability_fig(base, regime, FIGDIR / "model_reliability.pdf")
    importance_fig(regime, FIGDIR / "model_importance.pdf")
    decile_fig(regime, FIGDIR / "model_vs_m0.pdf")


if __name__ == "__main__":
    main()
