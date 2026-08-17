#!/usr/bin/env python3
"""Figure: distribution of one-year returns after a crash (Model 0 section).

Reuses the survivorship-safe one-year return from base_rate_1yr (compounded total daily returns,
delisting-inclusive, recent crashes censored) and renders a single-series histogram to a vector
PDF for the paper. Right tail is clipped at +200% into an overflow bin (2.9% of events; max
return exceeds +3000%) so the shape near the mass is legible; break-even, median, and mean are
marked directly.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_m0_return_hist.py
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashback.config import load_config  # noqa: E402
from crashback.ingestion.prices import scan_daily_prices  # noqa: E402

HORIZON = 252
BAR = "#4C72B0"      # single muted hue (one series → one color)
INK = "#222222"
MUTED = "#8A8A8A"


def one_year_returns(cfg) -> np.ndarray:
    norm = cfg.paths.resolve("data_normalized")
    events_dir = cfg.paths.resolve("data_events")
    ev = pl.read_parquet(events_dir / "crash_events_v1.parquet").filter(
        pl.col("in_universe_at_event") & pl.col("passes_min_price")
    ).select("security_id", "crash_date")
    sids = ev["security_id"].unique().to_list()
    px = (scan_daily_prices(norm / "daily_prices")
          .filter(pl.col("security_id").is_in(sids))
          .select("security_id", "date", "daily_return").sort(["security_id", "date"])
          .with_columns(
              td_idx=pl.int_range(pl.len()).over("security_id"),
              cumlog=(pl.col("daily_return").fill_null(0.0) + 1.0).log().cum_sum()
              .over("security_id"))
          .collect())
    last = px.group_by("security_id").agg(last_idx=pl.col("td_idx").max(),
                                          last_date=pl.col("date").max())
    idx = px.select("security_id", "td_idx", "cumlog")
    edge = px["date"].max() - timedelta(days=17)
    e = (ev.join(px.select("security_id", "date", "td_idx", "cumlog"),
                 left_on=["security_id", "crash_date"], right_on=["security_id", "date"],
                 how="inner")
         .rename({"td_idx": "k0", "cumlog": "cl0"})
         .join(last, on="security_id")
         .with_columns(end_idx=pl.min_horizontal(pl.col("k0") + HORIZON, pl.col("last_idx")),
                       fwd=pl.col("last_idx") - pl.col("k0"))
         .join(idx.rename({"td_idx": "end_idx", "cumlog": "cl_end"}),
               on=["security_id", "end_idx"], how="left")
         .with_columns(ret1y=(pl.col("cl_end") - pl.col("cl0")).exp() - 1.0,
                       censored=(pl.col("fwd") < HORIZON) & (pl.col("last_date") >= edge)))
    return e.filter(~pl.col("censored") & pl.col("ret1y").is_not_null())["ret1y"].to_numpy()


def main():
    cfg = load_config()
    r = one_year_returns(cfg)
    med, mean = float(np.median(r)), float(r.mean())
    p_earn = float((r > 0).mean())

    lo, hi = -1.0, 2.0                      # display window; clip the long right tail
    rc = np.clip(r, lo, hi)
    bins = np.arange(lo, hi + 1e-9, 0.10)   # 10% bins

    red, green = "#C44E52", "#55A868"
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(rc, bins=bins, weights=np.full(len(rc), 100.0 / len(rc)),
            color=BAR, edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)     # headroom for callouts
    ytop = ax.get_ylim()[1]

    # three reference lines, labelled with offset callouts so they don't collide near 0
    ax.axvline(0.0, color=INK, lw=1.4)
    ax.axvline(med, color=red, lw=1.4, ls="--")
    ax.axvline(mean, color=green, lw=1.4, ls="--")
    ax.text(0.0, ytop * 0.995, "break-even", color=INK, fontsize=8, ha="center", va="top")
    ax.annotate(f"median {med:+.0%}", xy=(med, ytop * 0.62), xytext=(-0.78, ytop * 0.80),
                color=red, fontsize=8.5, va="center",
                arrowprops={"arrowstyle": "-", "color": red, "lw": 0.8})
    ax.annotate(f"mean {mean:+.0%}", xy=(mean, ytop * 0.55), xytext=(0.42, ytop * 0.88),
                color=green, fontsize=8.5, va="center",
                arrowprops={"arrowstyle": "-", "color": green, "lw": 0.8})
    ax.text(0.97, 0.74, f"earns money: {p_earn:.0%}\nloses money: {1 - p_earn:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)

    ax.set_xlim(lo, hi)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["-100%", "-50%", "0", "+50%", "+100%", "+150%", "≥+200%"])
    ax.set_xlabel("One-year total return after the crash")
    ax.set_ylabel("Share of crash events (%)")
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)

    fig.tight_layout()
    out = Path("paper/figures/m0_return_hist.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"n={len(r):,}  median={med:+.3f}  mean={mean:+.3f}  P(earn)={p_earn:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
