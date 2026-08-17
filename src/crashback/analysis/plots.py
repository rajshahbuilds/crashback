"""Shared return-distribution histogram (the Model 0 baseline style).

Single source of the histogram structure used across the writeup so every return histogram---the
Model 0 baselines and the per-feature breakdowns---looks identical: single-hue bars, a break-even
line, dashed median/mean with offset callouts, an earns/loses-money annotation, and a clipped
overflow bin. Callers supply the display window; the marks are fixed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BAR, INK, MUTED = "#4C72B0", "#222222", "#8A8A8A"
RED, GREEN = "#C44E52", "#55A868"


def return_histogram(r: np.ndarray, spec: dict, out: Path, title: str | None = None):
    """Render one return histogram to ``out``. ``spec`` keys: label, lo, hi, bin, ticks,
    ticklabels. Optional ``title`` labels the panel (e.g. a cap range). Returns
    (median, mean, p_earn, n)."""
    med, mean, p_earn = float(np.median(r)), float(r.mean()), float((r > 0).mean())
    lo, hi = spec["lo"], spec["hi"]
    rc = np.clip(r, lo, hi)
    bins = np.arange(lo, hi + 1e-9, spec["bin"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(rc, bins=bins, weights=np.full(len(rc), 100.0 / len(rc)),
            color=BAR, edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ytop = ax.get_ylim()[1]
    ax.axvline(0.0, color=INK, lw=1.4)
    ax.axvline(med, color=RED, lw=1.4, ls="--")
    ax.axvline(mean, color=GREEN, lw=1.4, ls="--")
    ax.text(0.0, ytop * 0.995, "break-even", color=INK, fontsize=8, ha="center", va="top")
    ax.annotate(f"median {med:+.0%}", xy=(med, ytop * 0.62), xytext=(lo * 0.78, ytop * 0.80),
                color=RED, fontsize=8.5, va="center",
                arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8})
    ax.annotate(f"mean {mean:+.0%}", xy=(mean, ytop * 0.55), xytext=(hi * 0.30, ytop * 0.88),
                color=GREEN, fontsize=8.5, va="center",
                arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.8})
    ax.text(0.97, 0.74, f"earns money: {p_earn:.0%}\nloses money: {1 - p_earn:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlim(lo, hi)
    ax.set_xticks(spec["ticks"])
    ax.set_xticklabels(spec["ticklabels"])
    ax.set_xlabel(f"{spec['label'].capitalize()} total return after the crash")
    ax.set_ylabel("Share of crash events (%)")
    if title is not None:
        ax.set_title(title, fontsize=10, color=INK)
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


ONE_YEAR = {"label": "one-year", "lo": -1.0, "hi": 2.0, "bin": 0.10,
            "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            "ticklabels": ["-100%", "-50%", "0", "+50%", "+100%", "+150%", "≥+200%"]}

SIXTY_DAY = {"label": "60-day", "lo": -1.0, "hi": 1.0, "bin": 0.05,
             "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
             "ticklabels": ["-100%", "-50%", "0", "+50%", "≥+100%"]}
