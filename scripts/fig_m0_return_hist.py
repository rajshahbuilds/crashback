#!/usr/bin/env python3
"""Figures: distribution of post-crash returns at 60 trading days and one year (Model 0).

Survivorship-safe returns from the crash-day close (compounded total daily returns, so splits,
dividends, and delisting/bankruptcy terminal returns are all included); crashes without a full
forward window before the data edge are censored. One histogram per horizon (60-day, full-history
year, 2010-onward year), rendered via the shared baseline histogram.

Run: PYTHONPATH=src .venv/bin/python scripts/fig_m0_return_hist.py
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from crashback.analysis.plots import ONE_YEAR, return_histogram
from crashback.config import load_config
from crashback.ingestion.prices import scan_daily_prices

# figures to render: horizon (trading days), optional crash-date floor, display config
SPECS = [
    {"horizon": 60, "since": None, "label": "60-day", "file": "m0_return_hist_60d.pdf",
     "lo": -1.0, "hi": 1.0, "bin": 0.05, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
     "ticklabels": ["-100%", "-50%", "0", "+50%", "≥+100%"]},
    {"horizon": 252, "since": None, "label": "one-year", "file": "m0_return_hist.pdf", **ONE_YEAR},
    {"horizon": 252, "since": date(2010, 1, 1), "label": "one-year (2010–2025)",
     "file": "m0_return_hist_2010.pdf", **ONE_YEAR},
]


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


def forward_returns(ev, px, horizon: int, since: date | None = None) -> np.ndarray:
    if since is not None:
        ev = ev.filter(pl.col("crash_date") >= since)
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


def main():
    cfg = load_config()
    figdir = Path("paper/figures")
    ev, px = _prices(cfg)
    for spec in SPECS:
        r = forward_returns(ev, px, spec["horizon"], spec["since"])
        med, mean, p_earn, n = return_histogram(r, spec, figdir / spec["file"])
        print(f"{spec['label']:22s} (h={spec['horizon']}): n={n:,}  P(earn)={p_earn:.3f}  "
              f"median={med:+.3f}  mean={mean:+.3f}  -> {spec['file']}")


if __name__ == "__main__":
    main()
