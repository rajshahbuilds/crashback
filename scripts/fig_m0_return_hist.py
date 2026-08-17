#!/usr/bin/env python3
"""Figures: distribution of post-crash returns at 60 trading days and one year (Model 0).

Survivorship-safe returns from the crash-day close (compounded total daily returns, so splits,
dividends, and delisting/bankruptcy terminal returns are all included); crashes without a full
forward window before the data edge are censored. One histogram per horizon, rendered to vector
PDF; break-even, median, and mean are marked. Long right tails are clipped into an overflow bin.

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

BAR, INK, MUTED = "#4C72B0", "#222222", "#8A8A8A"
RED, GREEN = "#C44E52", "#55A868"

# horizon (trading days) -> display config
HORIZONS = {
    60: {"label": "60-day", "file": "m0_return_hist_60d.pdf", "lo": -1.0, "hi": 1.0,
         "bin": 0.05, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
         "ticklabels": ["-100%", "-50%", "0", "+50%", "≥+100%"]},
    252: {"label": "one-year", "file": "m0_return_hist.pdf", "lo": -1.0, "hi": 2.0,
          "bin": 0.10, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
          "ticklabels": ["-100%", "-50%", "0", "+50%", "+100%", "+150%", "≥+200%"]},
}


def _prices(cfg):
    """Per-security trading-day index + cumulative log total return, over crashed securities."""
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
    return ev, px


def forward_returns(ev, px, horizon: int) -> np.ndarray:
    last = px.group_by("security_id").agg(last_idx=pl.col("td_idx").max(),
                                          last_date=pl.col("date").max())
    idx = px.select("security_id", "td_idx", "cumlog")
    edge = px["date"].max() - timedelta(days=17)
    e = (ev.join(px.select("security_id", "date", "td_idx", "cumlog"),
                 left_on=["security_id", "crash_date"], right_on=["security_id", "date"],
                 how="inner")
         .rename({"td_idx": "k0", "cumlog": "cl0"})
         .join(last, on="security_id")
         .with_columns(end_idx=pl.min_horizontal(pl.col("k0") + horizon, pl.col("last_idx")),
                       fwd=pl.col("last_idx") - pl.col("k0"))
         .join(idx.rename({"td_idx": "end_idx", "cumlog": "cl_end"}),
               on=["security_id", "end_idx"], how="left")
         .with_columns(ret=(pl.col("cl_end") - pl.col("cl0")).exp() - 1.0,
                       censored=(pl.col("fwd") < horizon) & (pl.col("last_date") >= edge)))
    return e.filter(~pl.col("censored") & pl.col("ret").is_not_null())["ret"].to_numpy()


def render(r: np.ndarray, cfg_h: dict, out: Path):
    med, mean, p_earn = float(np.median(r)), float(r.mean()), float((r > 0).mean())
    lo, hi = cfg_h["lo"], cfg_h["hi"]
    rc = np.clip(r, lo, hi)
    bins = np.arange(lo, hi + 1e-9, cfg_h["bin"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(rc, bins=bins, weights=np.full(len(rc), 100.0 / len(rc)),
            color=BAR, edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ytop = ax.get_ylim()[1]
    ax.axvline(0.0, color=INK, lw=1.4)
    ax.axvline(med, color=RED, lw=1.4, ls="--")
    ax.axvline(mean, color=GREEN, lw=1.4, ls="--")
    ax.text(0.0, ytop * 0.995, "break-even", color=INK, fontsize=8, ha="center", va="top")
    ax.annotate(f"median {med:+.0%}", xy=(med, ytop * 0.62),
                xytext=(lo * 0.78, ytop * 0.80), color=RED, fontsize=8.5, va="center",
                arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8})
    ax.annotate(f"mean {mean:+.0%}", xy=(mean, ytop * 0.55),
                xytext=(hi * 0.30, ytop * 0.88), color=GREEN, fontsize=8.5, va="center",
                arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.8})
    ax.text(0.97, 0.74, f"earns money: {p_earn:.0%}\nloses money: {1 - p_earn:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlim(lo, hi)
    ax.set_xticks(cfg_h["ticks"])
    ax.set_xticklabels(cfg_h["ticklabels"])
    ax.set_xlabel(f"{cfg_h['label'].capitalize()} total return after the crash")
    ax.set_ylabel("Share of crash events (%)")
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return med, mean, p_earn, len(r)


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ev, px = _prices(cfg)
    for h, hc in HORIZONS.items():
        r = forward_returns(ev, px, h)
        med, mean, p_earn, n = render(r, hc, figdir / hc["file"])
        print(f"{hc['label']:8s} (h={h}): n={n:,}  P(earn)={p_earn:.3f}  "
              f"median={med:+.3f}  mean={mean:+.3f}  -> {hc['file']}")


if __name__ == "__main__":
    main()
