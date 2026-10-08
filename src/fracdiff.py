"""
Stage 3 — Fractional differentiation (AFML Ch 5, snippets 5.1–5.4).

The binomial weight recursion is:
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

The fractionally differentiated series at time t is:
    X̃_t = sum_{k=0}^{K} w_k * X_{t-k}

That sum applies w_0 to the *most recent* observation (X_t) and w_K to the
*oldest* observation in the window (X_{t-K}). Two equivalent ways to compute
it on a window:

(a) AFML's dot-product: build the weights array in oldest-first order,
    `w_dot = [w_K, ..., w_1, w_0]`, and dot it with the chronological window
    `[X_{t-K}, ..., X_{t-1}, X_t]`.

(b) numpy convolution: `np.convolve(a, v)[n] = sum_m a[m] * v[n-m]`. With
    `v = [w_0, w_1, ..., w_K]` (newest-first) and a chronological signal,
    `np.convolve(a, v, mode='valid')[t] = sum_k v[k] * a[t-k]` is exactly
    the fracdiff sum we want.

`get_weights_ffd` returns `[w_K, ..., w_0]` (oldest-first) so it is ready
for AFML-style dot products. `frac_diff_ffd` reverses to newest-first
before passing to `np.convolve`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def get_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Variable-width binomial weights, truncated when |w_k| < threshold.
    Returned in oldest-first order: [w_K, ..., w_1, w_0], shape (n, 1).
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Fixed-width binomial weights: keep all w_k with |w_k| >= threshold.
    Returned in oldest-first order: [w_K, ..., w_1, w_0], shape (n, 1).
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Fixed-width fractional differentiation of a chronological series.

    Internally we forward-fill then drop NaNs to get a clean signal, and
    derive the output index from *that* cleaned series so an output value
    is never silently mismatched to a date that was dropped.

    The convolution kernel is the FFD weights with the *oldest-first*
    array reversed to *newest-first* — that combination produces
    `X̃_t = sum_k w_k X_{t-k}` with w_0 multiplying the most recent
    observation, which is the fracdiff convention.
    """
    w = get_weights_ffd(d, threshold)
    width = len(w)

    s = series.ffill().dropna()
    if len(s) < width:
        return pd.Series(index=series.index, dtype=float)

    # Reverse oldest-first weights to newest-first for np.convolve
    # (see module docstring for the derivation).
    kernel = w[::-1].flatten()
    res = np.convolve(s.values, kernel, mode="valid")

    # mode='valid' gives len(s) - width + 1 outputs starting at s.index[width - 1].
    out_index = s.index[width - 1:]
    return pd.Series(res, index=out_index, name=f"fracdiff_d{d:.2f}")


def _sweep(
    series: pd.Series,
    d_range: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for d in d_range:
        s = frac_diff_ffd(series, d, threshold=threshold)
        if len(s) < 10:
            continue
        adf = adfuller(s, maxlag=1, regression="c", autolag=None)
        corr = float(s.corr(series.loc[s.index]))
        n_weights = len(get_weights_ffd(d, threshold))
        rows.append(
            {
                "d": round(float(d), 2),
                "adf_stat": float(adf[0]),
                "p_value": float(adf[1]),
                "correlation": corr,
                "n_obs": int(len(s)),
                "window_length": int(n_weights),
            }
        )
    return pd.DataFrame(rows).set_index("d")


def find_min_d(
    series: pd.Series,
    d_range: np.ndarray = np.arange(0.0, 1.05, 0.05),
    threshold: float = 1e-5,
    p_threshold: float = 0.05,
    corr_threshold: float = 0.9,
) -> Dict:
    """
    Sweep d, return the smallest d that makes the FFD series stationary
    while preserving high correlation with the original.

    Selection rule:
      strict   — smallest d with ADF p-value < p_threshold AND
                 correlation with aligned original >= corr_threshold.
      fallback — if no d satisfies the strict rule but at least one d is
                 stationary, pick the stationary d with the highest
                 correlation (preserves as much memory as possible while
                 still rejecting the unit root).
      hard fallback — if no d in the sweep is stationary, return d = 1.0
                      (first differencing).

    Returns
    -------
    {
      'd_star':            float,
      'sweep':             pd.DataFrame indexed by d with columns
                           [adf_stat, p_value, correlation, n_obs, window_length],
      'rule_applied':      str describing which branch was taken,
      'strict_satisfied':  bool,
      'p_threshold':       float,
      'corr_threshold':    float,
    }
    """
    sweep = _sweep(series, d_range, threshold)

    stat_mask = sweep["p_value"] < p_threshold
    mem_mask = sweep["correlation"] >= corr_threshold
    both = stat_mask & mem_mask

    if both.any():
        d_star = float(sweep.index[both].min())
        rule = (
            f"strict: smallest d with p<{p_threshold} and corr>={corr_threshold}"
        )
        strict = True
    elif stat_mask.any():
        d_star = float(sweep.loc[stat_mask, "correlation"].idxmax())
        rule = (
            f"fallback: no d met both criteria; chose the stationary d "
            f"(p<{p_threshold}) with the highest correlation to preserve memory"
        )
        strict = False
    else:
        d_star = 1.0
        rule = (
            "hard fallback: no d in the sweep was stationary; returning "
            "d=1.0 (first differences)"
        )
        strict = False

    return {
        "d_star": d_star,
        "sweep": sweep,
        "rule_applied": rule,
        "strict_satisfied": strict,
        "p_threshold": p_threshold,
        "corr_threshold": corr_threshold,
    }


def plot_min_ffd(
    series: pd.Series,
    d_range: np.ndarray = np.arange(0.0, 1.05, 0.05),
    threshold: float = 1e-5,
    p_threshold: float = 0.05,
    corr_threshold: float = 0.9,
):
    """
    Dual-axis plot:
      left y  — ADF statistic with 1% / 5% critical lines.
      right y — correlation of the FFD series with the aligned original.
    A vertical line marks the selected d*; markdown describes which rule fired.
    """
    selection = find_min_d(
        series,
        d_range=d_range,
        threshold=threshold,
        p_threshold=p_threshold,
        corr_threshold=corr_threshold,
    )
    out = selection["sweep"]
    d_star = selection["d_star"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(out.index, out["adf_stat"], color="#2c7fb8",
             marker="o", label="ADF statistic")
    ax1.axhline(-3.432, color="red",    linestyle="--", alpha=0.7, label="1% critical")
    ax1.axhline(-2.862, color="orange", linestyle="--", alpha=0.7, label="5% critical")
    ax1.set_xlabel("d")
    ax1.set_ylabel("ADF statistic", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")

    ax2 = ax1.twinx()
    ax2.plot(out.index, out["correlation"], color="#31a354",
             marker="s", label=f"corr with log price")
    ax2.axhline(corr_threshold, color="#31a354", linestyle=":", alpha=0.6,
                label=f"corr threshold = {corr_threshold:.2f}")
    ax2.set_ylabel("correlation with original", color="#31a354")
    ax2.tick_params(axis="y", labelcolor="#31a354")
    ax2.set_ylim(-0.1, 1.05)

    ax1.axvline(d_star, color="black", linestyle=":", alpha=0.8,
                label=f"d* = {d_star:.2f}")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")

    ax1.set_title(
        "Fractional differentiation: stationarity vs memory  "
        f"({'strict' if selection['strict_satisfied'] else 'fallback'} rule)"
    )
    fig.tight_layout()
    return fig, selection
