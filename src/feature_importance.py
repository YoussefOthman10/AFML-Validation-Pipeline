"""
Stage 6 — Feature importance (AFML Ch 8).

Three complementary methods:
  - MDI: in-sample mean decrease in impurity, averaged across trees.
  - MDA: out-of-sample mean decrease in accuracy / log-loss when a feature
         column is permuted at test time. Computed over a purged CV.
  - SFI: single-feature CV score — train the model on each feature alone.

MDA uses `neg_log_loss` by default per de Prado: with binary labels and
small samples, accuracy is too coarse to reliably reflect a permuted-column
degradation, while log-loss responds smoothly to changes in predicted
probabilities.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone

try:
    from cross_validation import cv_score, weighted_score
except ImportError:
    from src.cross_validation import cv_score, weighted_score


def feat_imp_MDI(clf, feature_names: Iterable[str]) -> pd.DataFrame:
    """
    Mean Decrease in Impurity, with per-tree std.

    Expects a fitted forest-style classifier exposing `estimators_`,
    each with `.feature_importances_`. Importances are normalised per tree
    to sum to 1.
    """
    feature_names = list(feature_names)
    per_tree = np.array(
        [tree.feature_importances_ for tree in clf.estimators_]
    )
    # Re-normalise per tree (RF in sklearn already normalises, but be safe).
    row_sums = per_tree.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    per_tree = per_tree / row_sums

    df = pd.DataFrame(
        {
            "mean": per_tree.mean(axis=0),
            "std":  per_tree.std(axis=0, ddof=1) / np.sqrt(per_tree.shape[0]),
        },
        index=feature_names,
    )
    return df.sort_values("mean", ascending=False)


def feat_imp_MDA(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    sample_weight: pd.Series = None,
    scoring: str = "neg_log_loss",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Mean Decrease in Accuracy (or any score) under permutation, over folds
    of a purged CV.

    For each fold:
      1. Fit `clf` on train (sample-weighted if provided).
      2. baseline = weighted_score(test).
      3. For each feature j: permute column j of test, re-score with the
         same test-fold weights.
         delta_j_fold = baseline - permuted_score (positive => feature helps).
    Both the baseline and the permuted scores honour the test-fold
    sample weights so the MDA delta is computed on a like-for-like basis.
    """
    rng = np.random.default_rng(random_state)
    feature_names = list(X.columns)

    baselines: list[float] = []
    deltas: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if sample_weight is not None:
            sw_tr = sample_weight.iloc[train_idx].values
            sw_te = sample_weight.iloc[test_idx].values
        else:
            sw_tr = sw_te = None

        candidate = clone(clf)
        if sw_tr is not None:
            candidate.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            candidate.fit(X_tr, y_tr)

        baseline = weighted_score(candidate, X_te, y_te,
                                  sample_weight=sw_te, scoring=scoring)
        baselines.append(baseline)

        fold_delta = {}
        for col in feature_names:
            X_te_perm = X_te.copy()
            X_te_perm[col] = rng.permutation(X_te_perm[col].values)
            permuted_score = weighted_score(candidate, X_te_perm, y_te,
                                            sample_weight=sw_te, scoring=scoring)
            fold_delta[col] = baseline - permuted_score
        deltas.append(fold_delta)

    delta_df = pd.DataFrame(deltas, columns=feature_names)
    out = pd.DataFrame(
        {
            "mean": delta_df.mean(axis=0),
            "std":  delta_df.std(axis=0, ddof=1) / np.sqrt(len(delta_df)),
        }
    )
    out.attrs["baseline_per_fold"] = np.asarray(baselines)
    return out.sort_values("mean", ascending=False)


def feat_imp_SFI(
    clf_template,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    sample_weight: pd.Series = None,
    scoring: str = "neg_log_loss",
) -> pd.DataFrame:
    """
    Single-feature importance: purged-CV score of `clf_template` trained on
    each feature alone. Mean and std are computed across the CV folds.
    """
    rows = {}
    for col in X.columns:
        scores = cv_score(
            clf=clf_template,
            X=X[[col]],
            y=y,
            sample_weight=sample_weight,
            scoring=scoring,
            cv=cv,
        )
        rows[col] = {
            "mean": float(scores.mean()),
            "std":  float(scores.std(ddof=1) / np.sqrt(len(scores))),
        }
    return pd.DataFrame(rows).T.sort_values("mean", ascending=False)


def plot_feature_importance(
    imp: pd.DataFrame,
    title: str,
    color: str = "#2c7fb8",
    figsize: tuple = (8, 6),
):
    """Sorted horizontal bar chart with std error bars."""
    df = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df.index, df["mean"], xerr=df["std"], color=color,
            edgecolor="black", alpha=0.85, error_kw={"elinewidth": 1})
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel(title)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
