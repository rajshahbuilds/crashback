"""Decile lift: recovery rate by predicted-probability bucket, and top-decile lift (STU-62).

Complements the equal-width reliability table (``metrics.calibration_table``) with equal-COUNT
buckets, which is the standard framing for "does the model concentrate recoveries in its
top-scored events?". Lift = a bucket's observed recovery rate divided by the overall base rate;
top-decile lift > 1 means the model's most-confident decile recovers more often than average.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def decile_table(y, p, q: int = 10) -> tuple[pl.DataFrame, float]:
    """Equal-count buckets by predicted probability, ascending.

    Returns (table, base_rate). Table columns: bucket (1=lowest p … q=highest p), n, mean_pred,
    observed_rate, lift (observed_rate / base_rate). Ties are broken by ordinal rank so bucket
    counts are equal to within one.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = y.shape[0]
    df = pl.DataFrame({"y": y, "p": p})
    # ordinal rank 0..n-1 → bucket 0..q-1 with (near-)equal counts, then 1-index ascending
    df = df.with_columns(
        (((pl.col("p").rank("ordinal") - 1) * q // n)).clip(0, q - 1).alias("bucket")
    )
    base = float(y.mean())
    g = (
        df.group_by("bucket")
        .agg(
            pl.len().alias("n"),
            pl.col("p").mean().alias("mean_pred"),
            pl.col("y").mean().alias("observed_rate"),
        )
        .with_columns(((pl.col("bucket") + 1)).alias("bucket"))
        .sort("bucket")
    )
    g = g.with_columns((pl.col("observed_rate") / base).alias("lift"))
    return g, base


def confidence_bands(p, y, ret=None, dd=None, width: float = 0.1) -> pl.DataFrame:
    """Fixed-width predicted-probability bands with counts, observed recovery, and (optionally)
    the return distribution per band.

    Unlike ``decile_table`` (equal-count), these are the model's *natural* confidence bins, so
    empty/sparse high-confidence bands are visible. When ``ret`` (e.g. return_20d) and ``dd``
    (max_drawdown_20d) are given, adds mean_return, mean_return_win / _lose (conditioned on the
    recovery label), and mean_maxdd — exposing that recovery probability ≠ expected return.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = p.shape[0]
    lo_edges = np.arange(0.0, 1.0, width)
    cols = {"lo": p, "y": y}
    if ret is not None:
        cols["ret"] = np.asarray(ret, dtype=float)
    if dd is not None:
        cols["dd"] = np.asarray(dd, dtype=float)
    df = pl.DataFrame(cols).with_columns(
        (pl.col("lo") // width * width).clip(0.0, lo_edges[-1]).alias("band")
    )
    aggs = [pl.len().alias("n"), pl.col("lo").mean().alias("mean_pred"),
            pl.col("y").mean().alias("observed_rate")]
    if ret is not None:
        aggs += [
            pl.col("ret").mean().alias("mean_return"),
            pl.col("ret").filter(pl.col("y") == 1).mean().alias("mean_return_win"),
            pl.col("ret").filter(pl.col("y") == 0).mean().alias("mean_return_lose"),
        ]
    if dd is not None:
        aggs.append(pl.col("dd").mean().alias("mean_maxdd"))
    g = df.group_by("band").agg(aggs).sort("band")
    return g.with_columns(
        (pl.col("band") + width).alias("hi"),
        (pl.col("n") / n).alias("frac"),
    )


def top_decile_lift(y, p, q: int = 10) -> dict:
    """Observed recovery rate and lift for the highest-predicted-probability bucket."""
    table, base = decile_table(y, p, q=q)
    top = table.filter(pl.col("bucket") == q).row(0, named=True)
    return {
        "base_rate": base,
        "top_bucket_n": int(top["n"]),
        "top_bucket_mean_pred": float(top["mean_pred"]),
        "top_decile_recovery_rate": float(top["observed_rate"]),
        "top_decile_lift": float(top["lift"]),
    }
