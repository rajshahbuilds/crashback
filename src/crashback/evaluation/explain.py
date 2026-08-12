"""Per-event explainability: TreeSHAP for the tree model, coefficient terms for logistic (STU-64).

Both explanations are **additive in log-odds (margin) space**: the per-feature contributions plus
a bias term sum to the model's raw margin, whose sigmoid is the predicted P(recover). Positive
contribution ⇒ pushes the probability up, negative ⇒ down. Magnitudes are directly comparable
within an event.

- **Tree (XGBoost):** exact TreeSHAP via the booster's native ``pred_contribs`` — no external
  ``shap`` dependency. Returns one contribution per feature plus the bias.
- **Logistic:** contribution of feature j for event i is ``coef_j * standardized_x_ij`` (the
  term's push on the log-odds), with the intercept as bias. Standardized features are ~0-mean, so
  contributions are naturally centered on the average event.

Explanations use the exact point-in-time feature values recorded for the crash event.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb

from crashback.models.logistic import _matrix as _logi_matrix
from crashback.models.xgb import _matrix as _xgb_matrix


def tree_shap(
    booster: xgb.Booster, X: pl.DataFrame, feats: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP contributions (log-odds). Returns (contribs [n, F], bias [n]).

    Uses the same ``best_iteration`` tree count as prediction, so contributions + bias reproduce
    the model's predicted probability exactly.
    """
    d = xgb.DMatrix(_xgb_matrix(X), feature_names=feats, missing=np.nan)
    c = booster.predict(d, pred_contribs=True,
                        iteration_range=(0, booster.best_iteration + 1))  # [n, F+1]; last=bias
    return c[:, :-1], c[:, -1]


def logistic_contributions(
    pipe, X: pl.DataFrame, feats: list[str],
) -> tuple[np.ndarray, float, list[str]]:
    """Per-event logistic contributions (log-odds). Returns (contribs [n, F'], intercept, names).

    ``names`` is ``feats`` plus ``<feature>__missing`` for imputer indicator columns, matching the
    fitted coefficient vector.
    """
    imp = pipe.named_steps["impute"]
    sc = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    x_std = sc.transform(imp.transform(_logi_matrix(X)))
    contribs = x_std * clf.coef_.ravel()          # broadcast [n, F'] * [F']
    names = list(feats)
    if getattr(imp, "indicator_", None) is not None:
        names += [f"{feats[i]}__missing" for i in imp.indicator_.features_]
    return contribs, float(clf.intercept_[0]), names


def top_contributors(contrib_row: np.ndarray, names: list[str], k: int = 6):
    """Return (positive, negative) lists of (name, contribution), each sorted by |contribution|."""
    order = np.argsort(contrib_row)
    neg = [(names[i], float(contrib_row[i])) for i in order if contrib_row[i] < 0][:k]
    pos = [(names[i], float(contrib_row[i])) for i in order[::-1] if contrib_row[i] > 0][:k]
    return pos, neg


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
