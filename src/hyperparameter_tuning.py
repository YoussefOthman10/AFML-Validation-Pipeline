"""
Stage 5 — Hyperparameter tuning with Purged K-Fold CV.

Implements:
- purged_grid_search:    exhaustive grid search over a small param grid.
- purged_random_search:  RandomizedSearchCV-style sampling over distributions.
- log_trials:            tidy long-form trial table.
- probabilistic_sharpe_ratio / deflated_sharpe_ratio_for_trials:
                         AFML Ch 14 multiple-testing correction applied to
                         CV-trial Sharpe-like statistics (mean / std of
                         per-fold scores).

The trial "Sharpe" used here is the standard AFML convention:
    SR_trial = mean(fold_scores) / std(fold_scores)
i.e. the in-sample CV Sharpe of a hyperparameter trial. The Deflated Sharpe
formula then asks: given N trials, how impressive is the best of those SRs?
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.cross_validation import cv_score


GAMMA_EM = 0.5772156649015329  # Euler-Mascheroni constant


def _evaluate_params(
    clf,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    cv,
    scoring: str,
) -> Dict[str, Any]:
    candidate = clone(clf)
    candidate.set_params(**params)
    fold_scores = cv_score(
        clf=candidate,
        X=X,
        y=y,
        sample_weight=sample_weight,
        scoring=scoring,
        cv=cv,
    )
    fs = fold_scores.values.astype(float)
    sr = float(fs.mean() / fs.std(ddof=1)) if fs.std(ddof=1) > 0 else np.nan
    return {
        "params": dict(params),
        "fold_scores": fs,
        "mean_score": float(fs.mean()),
        "std_score": float(fs.std(ddof=1)),
        "sr_trial": sr,
    }


def purged_grid_search(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Mapping[str, Sequence],
    cv,
    sample_weight: pd.Series = None,
    scoring: str = "accuracy",
) -> Dict[str, Any]:
    """Exhaustive grid search using a purged-CV `cv` splitter."""
    trials: List[Dict[str, Any]] = []
    for params in ParameterGrid(dict(param_grid)):
        trials.append(_evaluate_params(clf, params, X, y, sample_weight, cv, scoring))
    return _summarise(trials)


def purged_random_search(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    param_dist: Mapping[str, Sequence],
    cv,
    n_iter: int = 25,
    sample_weight: pd.Series = None,
    scoring: str = "accuracy",
    random_state: int | None = None,
) -> Dict[str, Any]:
    """Randomised search over a param distribution using a purged-CV splitter."""
    sampler = ParameterSampler(
        dict(param_dist), n_iter=n_iter, random_state=random_state
    )
    trials: List[Dict[str, Any]] = []
    for params in sampler:
        trials.append(_evaluate_params(clf, params, X, y, sample_weight, cv, scoring))
    return _summarise(trials)


def _summarise(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = log_trials(trials)
    best_idx = int(df["mean_score"].idxmax())
    return {
        "trials": trials,
        "log": df,
        "best_idx": best_idx,
        "best_params": trials[best_idx]["params"],
        "best_mean_score": float(df.loc[best_idx, "mean_score"]),
        "best_std_score": float(df.loc[best_idx, "std_score"]),
        "best_fold_scores": trials[best_idx]["fold_scores"],
    }


def log_trials(trials: List[Dict[str, Any]]) -> pd.DataFrame:
    """Tidy DataFrame: one row per trial, params expanded into columns."""
    rows = []
    for i, t in enumerate(trials):
        row = {"trial": i, "mean_score": t["mean_score"],
               "std_score": t["std_score"], "sr_trial": t["sr_trial"]}
        for k, v in t["params"].items():
            row[f"param_{k}"] = v
        for j, s in enumerate(t["fold_scores"]):
            row[f"fold_{j+1}"] = s
        rows.append(row)
    return pd.DataFrame(rows)


def probabilistic_sharpe_ratio(
    sr_hat: float,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """
    PSR(SR*) = Φ( (SR̂ - SR*) √(n-1) / √(1 - γ₃ SR̂ + (γ₄-1)/4 SR̂²) )

    `kurt` is the *non-excess* kurtosis (γ₄, normal = 3).
    """
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2))
    z = (sr_hat - sr_benchmark) * np.sqrt(max(1, n - 1)) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio_for_trials(
    trial_fold_scores: Sequence[np.ndarray],
    best_idx: int | None = None,
) -> Dict[str, float]:
    """
    Deflated Sharpe Ratio for a hyperparameter sweep (AFML Ch 14, Bailey & LdP 2014).

    Given N hyperparameter trials, each with its own array of per-fold CV
    scores, treats each trial's mean/std as a Sharpe-like statistic and
    computes the probability that the best trial's true SR exceeds the
    expected maximum SR under the null of "no skill".

    Parameters
    ----------
    trial_fold_scores : list of arrays, one per trial.
    best_idx : optional, defaults to argmax of trial mean scores.

    Returns
    -------
    {
      'n_trials', 'best_sr', 'expected_max_sr', 'dsr',
      'best_mean', 'best_std', 'best_skew', 'best_kurt'
    }
    """
    sr = np.array(
        [(s.mean() / s.std(ddof=1)) if s.std(ddof=1) > 0 else 0.0
         for s in trial_fold_scores]
    )
    n_trials = int(len(sr))
    if best_idx is None:
        best_idx = int(np.argmax([s.mean() for s in trial_fold_scores]))

    fs_best = np.asarray(trial_fold_scores[best_idx], dtype=float)
    n_obs = len(fs_best)
    sr_best = float(sr[best_idx])

    var_sr = float(np.var(sr, ddof=1)) if n_trials > 1 else 0.0

    # Expected maximum SR under null (Bailey & LdP 2014):
    expected_max_sr = float(
        np.sqrt(max(0.0, var_sr))
        * (
            (1.0 - GAMMA_EM) * norm.ppf(1.0 - 1.0 / max(n_trials, 2))
            + GAMMA_EM * norm.ppf(1.0 - 1.0 / (max(n_trials, 2) * np.e))
        )
    )

    # Higher moments of the *best trial's* fold scores (used by PSR).
    # ddof=1 is the convention for sample skew/kurt here; with only ~5
    # folds these are noisy but still preferable to ignoring them.
    if n_obs >= 3:
        m = fs_best.mean()
        s = fs_best.std(ddof=0)
        skew = float(np.mean(((fs_best - m) / s) ** 3)) if s > 0 else 0.0
        kurt = float(np.mean(((fs_best - m) / s) ** 4)) if s > 0 else 3.0
    else:
        skew, kurt = 0.0, 3.0

    dsr = probabilistic_sharpe_ratio(
        sr_hat=sr_best, n=n_obs, skew=skew, kurt=kurt,
        sr_benchmark=expected_max_sr,
    )

    return {
        "n_trials": n_trials,
        "best_sr": sr_best,
        "expected_max_sr": expected_max_sr,
        "dsr": float(dsr),
        "best_mean": float(fs_best.mean()),
        "best_std": float(fs_best.std(ddof=1)),
        "best_skew": skew,
        "best_kurt": kurt,
    }
