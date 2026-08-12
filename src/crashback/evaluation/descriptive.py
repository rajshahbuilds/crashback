"""Descriptive base-rate analysis helpers (recovery rates + Wilson confidence intervals).

Rates are computed only over *determined* events for a target — the label is null when the
event is censored or has no close, so filtering `target.is_not_null()` excludes them
automatically. Confidence intervals use the Wilson score interval (better than normal
approximation for proportions near 0/1 and small n).

Caveat carried into the report: crash events for the same security close together in time are
correlated (CLAUDE.md §22), so Wilson CIs — which assume independent observations — understate
the true uncertainty. They bound *sampling* noise, not event dependence.
"""
from __future__ import annotations

import polars as pl

Z = 1.96  # ~95%


def add_wilson(df: pl.DataFrame, hits: str = "hits", n: str = "n") -> pl.DataFrame:
    """Add rate, ci_low, ci_high (Wilson score interval) from hit-count and n columns."""
    nn = pl.col(n)
    p = pl.col(hits) / nn
    z2 = Z * Z
    denom = 1.0 + z2 / nn
    center = (p + z2 / (2 * nn)) / denom
    half = (Z / denom) * (p * (1 - p) / nn + z2 / (4 * nn * nn)).sqrt()
    return df.with_columns(
        p.alias("rate"),
        pl.when(nn > 0).then(center - half).alias("ci_low"),
        pl.when(nn > 0).then(center + half).alias("ci_high"),
    )


def overall_rate(df: pl.DataFrame, target: str) -> dict:
    """Base rate for a single target over determined events (non-null label)."""
    sub = df.filter(pl.col(target).is_not_null())
    if sub.height == 0:
        return {"n": 0, "hits": 0, "rate": None, "ci_low": None, "ci_high": None}
    agg = add_wilson(
        pl.DataFrame({"hits": [sub[target].sum()], "n": [sub.height]})
    ).row(0, named=True)
    return {"n": sub.height, "hits": int(sub[target].sum()), **{k: agg[k] for k in
            ("rate", "ci_low", "ci_high")}}


def grouped_rate(df: pl.DataFrame, by: str, target: str) -> pl.DataFrame:
    """Recovery rate by a grouping column, with n and Wilson CI, over determined events."""
    g = (
        df.filter(pl.col(target).is_not_null())
        .group_by(by)
        .agg(pl.col(target).sum().alias("hits"), pl.len().alias("n"))
    )
    return add_wilson(g).sort(by)


def qcut_labels(df: pl.DataFrame, col: str, q: int, prefix: str) -> pl.Series:
    """Quantile-bucket a numeric column into q labeled groups (ties handled by qcut)."""
    return df[col].qcut(q, labels=[f"{prefix}{i + 1}" for i in range(q)], allow_duplicates=True)
