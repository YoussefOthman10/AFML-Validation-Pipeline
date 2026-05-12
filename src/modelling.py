import numpy as np
import pandas as pd
from sklearn.base import clone

from src.cross_validation import cv_score


def train_and_evaluate(clf, X, y, sample_weight, cv, scoring="accuracy"):
    """
    Run purged CV with sample weights and refit the classifier on the full dataset.

    Returns
    -------
    dict with keys:
        mean_score   : float
        std_score    : float
        fold_scores  : pd.Series  (one score per fold)
        fitted_clf   : estimator fit on the full dataset
    """
    fold_scores = cv_score(
        clf=clf,
        X=X,
        y=y,
        sample_weight=sample_weight,
        scoring=scoring,
        cv=cv,
    )

    fitted_clf = clone(clf)
    if sample_weight is not None:
        fitted_clf.fit(X, y, sample_weight=np.asarray(sample_weight))
    else:
        fitted_clf.fit(X, y)

    return {
        "mean_score": float(fold_scores.mean()),
        "std_score": float(fold_scores.std()),
        "fold_scores": fold_scores,
        "fitted_clf": fitted_clf,
    }
