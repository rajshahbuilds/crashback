"""Reusable fitting for the paper's one-year recovery model (P(earn money in a year)).

Assembles the feature matrix once (assemble), then fits/evaluates XGBoost under a chosen split
and feature set (fit_predict). Shared by scripts/train_1yr_model.py (CLI) and
scripts/fig_model_results.py (figures) so both use one code path. See train_1yr_model.py for the
split and feature definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from crashback.analysis.extra_features import (
    event_beta,
    event_ebitda_margin,
    market_regime,
)
from crashback.analysis.recovery import one_year_returns
from crashback.evaluation.metrics import binary_metrics, calibration_table
from crashback.models import xgb as xgbm

BASE_FEATURES = [
    "log_market_cap", "crash_return", "prior_crash_count_20d", "pe", "return_252d_pre",
    "beta", "ebitda_margin", "revenue_growth_yoy", "market_return_20d", "rel_sector",
]
REGIME_FEATURES = [
    "mkt_ret_126d", "mkt_ret_252d", "mkt_drawdown_252d", "mkt_vol_60d", "crash_breadth_20d",
]

# chronological split with a one-year embargo (outcome window is 252 trading days)
TRAIN_END = date(2017, 12, 31)
VAL = (date(2019, 1, 1), date(2020, 12, 31))
TEST_START = date(2022, 1, 1)


def assemble(cfg) -> pl.DataFrame:
    """One row per clean crash event with the target y and all candidate features."""
    proc = cfg.paths.resolve("data_processed")
    ret = one_year_returns(cfg)
    ev = pl.read_parquet(proc / "events_v1.parquet").select(
        "event_id", "market_cap", "crash_return", "prior_crash_count_20d", "pe",
        "return_252d_pre", "revenue_growth_yoy", "market_return_20d", "sector_return_1d")
    beta = event_beta(cfg)
    ebm = event_ebitda_margin(cfg)
    reg = market_regime(cfg)
    return (ret.join(ev, on="event_id").join(beta, on="event_id", how="left")
            .join(ebm, on="event_id", how="left")
            .join(reg, left_on="crash_date", right_on="date", how="left")
            .with_columns(
                y=(pl.col("ret") > 0.0).cast(pl.Int8),
                log_market_cap=pl.when(pl.col("market_cap") > 0)
                .then(pl.col("market_cap").log10()).otherwise(None),
                rel_sector=pl.col("crash_return") - pl.col("sector_return_1d"),
                prior_crash_count_20d=pl.col("prior_crash_count_20d").fill_null(0)))


def split_chrono(df: pl.DataFrame) -> pl.DataFrame:
    in_val = (pl.col("crash_date") >= VAL[0]) & (pl.col("crash_date") <= VAL[1])
    return df.with_columns(split=pl.when(pl.col("crash_date") <= TRAIN_END).then(pl.lit("train"))
                           .when(in_val).then(pl.lit("validation"))
                           .when(pl.col("crash_date") >= TEST_START).then(pl.lit("test"))
                           .otherwise(pl.lit("embargo")))


@dataclass
class FitResult:
    label: str
    features: list[str]
    test: pl.DataFrame       # event_id, y, p
    metrics: dict            # XGBoost test metrics
    m0: dict                 # Model 0 (constant train prevalence) test metrics
    calib: pl.DataFrame      # reliability table
    ece: float
    importance: pl.DataFrame
    sizes: pl.DataFrame      # split, n, p_earn, min_date, max_date
    best_params: dict
    best_iteration: int


def fit_predict(df: pl.DataFrame, cfg, *, regime: bool, seed: int = 42) -> FitResult:
    """Fit XGBoost for one feature set on the chronological split; evaluate once on test."""
    cols = BASE_FEATURES + (REGIME_FEATURES if regime else [])
    d = split_chrono(df)
    parts = {s: d.filter(pl.col("split") == s) for s in ("train", "validation", "test")}

    sizes = pl.DataFrame([{
        "split": s, "n": p.height, "p_earn": float(p["y"].mean()),
        "min_date": p["crash_date"].min(), "max_date": p["crash_date"].max()}
        for s, p in parts.items()])

    def xy(p):
        return p.select(cols), p["y"].to_numpy()

    X_tr, y_tr = xy(parts["train"])
    X_val, y_val = xy(parts["validation"])
    X_te, y_te = xy(parts["test"])

    res = xgbm.tune(X_tr, y_tr, X_val, y_val, cfg.models.xgboost, seed=seed, feature_names=cols)
    p_te = xgbm.predict(res.booster, X_te, cols)

    m = binary_metrics(y_te, p_te)
    base = float(y_tr.mean())
    m0 = binary_metrics(y_te, np.full(y_te.shape[0], base))
    tbl, ece = calibration_table(y_te, p_te, cfg.models.calibration_bins)
    imp = xgbm.importance_table(res.booster, cols)

    label = f"chrono{'+regime' if regime else ''}"
    test = parts["test"].select("event_id", "crash_date", "y").with_columns(p=pl.Series(p_te))
    return FitResult(label, cols, test, m, m0, tbl, ece, imp, sizes,
                     res.best_params, res.best_iteration)
