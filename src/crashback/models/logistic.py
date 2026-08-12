"""Baseline (base-rate) and logistic-regression recovery models.

Logistic pipeline: **median imputation with missingness indicators** (missingness is itself
informative — e.g. absent fundamentals) -> **standardization** -> **L2-regularized logistic
regression**. Standardizing makes coefficients directly comparable in magnitude, and their
sign gives the direction of association with recovery. The base-rate model (Model 0) predicts
the training prevalence for every event — the yardstick every other stage must beat.

``keep_empty_features=True`` keeps all-missing columns (imputed to 0) so the fitted
coefficient vector stays aligned with the input feature list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaseRateModel:
    """Model 0: predict the historical (training) recovery rate for every event."""

    rate: float

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(int(n), self.rate, dtype=float)


def fit_base_rate(y_train) -> BaseRateModel:
    return BaseRateModel(rate=float(np.asarray(y_train, dtype=float).mean()))


def build_logistic(*, C: float, max_iter: int, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=seed)),
    ])


def _matrix(X: pl.DataFrame) -> np.ndarray:
    """To a float matrix, routing non-finite values (inf from undefined ratios like a
    zero-denominator interest_coverage) into the missing path so the imputer handles and
    flags them rather than sklearn rejecting the input."""
    a = X.to_numpy().astype(float)
    a[~np.isfinite(a)] = np.nan
    return a


def fit_logistic(X_train: pl.DataFrame, y_train, *, C: float, max_iter: int, seed: int) -> Pipeline:
    pipe = build_logistic(C=C, max_iter=max_iter, seed=seed)
    pipe.fit(_matrix(X_train), np.asarray(y_train, dtype=int))
    return pipe


def predict_logistic(pipe: Pipeline, X: pl.DataFrame) -> np.ndarray:
    return pipe.predict_proba(_matrix(X))[:, 1]


def coefficient_table(pipe: Pipeline, feature_cols: list[str]) -> pl.DataFrame:
    """Standardized logistic coefficients with direction, sorted by |coef|.

    Handles the imputer's added missingness indicators — their names are ``<feature>__missing``
    for the subset of features that had missing values in training (from ``indicator_``).
    """
    imp: SimpleImputer = pipe.named_steps["impute"]
    names = list(feature_cols)
    if getattr(imp, "indicator_", None) is not None:
        names += [f"{feature_cols[i]}__missing" for i in imp.indicator_.features_]

    clf: LogisticRegression = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    return (
        pl.DataFrame({"feature": names, "coef_std": coefs})
        .with_columns(
            pl.col("coef_std").abs().alias("abs_coef"),
            pl.when(pl.col("coef_std") > 0).then(pl.lit("increases P(recover)"))
            .when(pl.col("coef_std") < 0).then(pl.lit("decreases P(recover)"))
            .otherwise(pl.lit("none")).alias("direction"),
        )
        .sort("abs_coef", descending=True)
    )
