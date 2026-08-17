#!/usr/bin/env python3
"""Figure: one-year recovery rate by market-cap decile (feature-correlation section).

Joins the survivorship-safe one-year outcome to each event's point-in-time market cap (crash-day
close x latest shares), buckets into deciles, and plots P(earn money) per decile against the
unconditional base rate. Reveals a U-shaped relationship: nano-caps and large caps recover more
often than the mid-caps between them.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_recovery_by_marketcap.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashback.analysis.recovery import one_year_returns  # noqa: E402
from crashback.config import load_config  # noqa: E402

BAR, INK, MUTED, REF = "#4C72B0", "#222222", "#8A8A8A", "#C44E52"


def _sz(m):  # market cap ($M) -> short label
    return f"${m / 1000:.1f}B" if m >= 1000 else f"${m:.0f}M"


def main():
    cfg = load_config()
    ret = one_year_returns(cfg)
    mc = pl.read_parquet(cfg.paths.resolve("data_processed") / "events_v1.parquet").select(
        "event_id", "market_cap")
    d = ret.join(mc, on="event_id").filter(
        pl.col("market_cap").is_not_null() & (pl.col("market_cap") > 0))
    base = float((d["ret"] > 0).mean())
    d = d.with_columns(dec=(pl.col("market_cap").rank("ordinal") / pl.len() * 10)
                       .ceil().clip(1, 10).cast(pl.Int64))
    g = (d.group_by("dec").agg(n=pl.len(), p_earn=(pl.col("ret") > 0).mean(),
                               med_ret=pl.col("ret").median(),
                               med_mc=pl.col("market_cap").median()).sort("dec"))
    dec = g["dec"].to_numpy()
    pe = g["p_earn"].to_numpy() * 100
    print(f"events={d.height:,}  base P(earn)={base:.3f}")
    for r in g.iter_rows(named=True):
        print(f"  d{r['dec']:2d} n={r['n']:5d}  size~{_sz(r['med_mc']):>7s}  "
              f"P(earn)={r['p_earn']:.3f}  med_ret={r['med_ret']:+.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.bar(dec, pe, color=BAR, edgecolor="white", width=0.8, zorder=2)
    ax.axhline(base * 100, color=REF, lw=1.3, ls="--", zorder=3)
    ax.text(10.4, base * 100, f"base rate {base:.0%}", color=REF, fontsize=8.5,
            va="center", ha="left")

    ax.set_xticks(dec)
    ax.set_xticklabels([str(i) for i in dec])
    ax.set_xlabel("Market-cap decile (1 = smallest, 10 = largest)")
    ax.set_ylabel("P(earn money in one year)  (%)")
    ax.set_ylim(40, 54)
    ax.annotate(_sz(g["med_mc"][0]), xy=(1, pe[0]), xytext=(1, 41.2),
                ha="center", fontsize=7.5, color=MUTED)
    ax.annotate(_sz(g["med_mc"][-1]), xy=(10, pe[-1]), xytext=(10, 41.2),
                ha="center", fontsize=7.5, color=MUTED)
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_xlim(0.4, 11.4)
    fig.tight_layout()
    out = Path("paper/figures/recovery_by_marketcap.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
