"""
Entropy features — AFML Chapter 18.

Standalone implementations of entropy estimators.  The rolling versions
used in the modelling pipeline live in src/features.py; these functions
are for standalone use in analysis notebooks and the report.

Implements:
  shannon_entropy         : plug-in Shannon entropy  [AFML §18.2]
  lempel_ziv_complexity   : LZ-76 complexity         [Snippet 18.3]
  kontoyiannis_entropy    : match-length estimator   [AFML §18.2]
  rolling_entropy         : rolling Shannon entropy on a return series
  rolling_lz              : rolling LZ complexity
"""
import numpy as np
import pandas as pd


def shannon_entropy(msg, base: float = 2.0) -> float:
    """
    Shannon entropy of a discrete message.

    Parameters
    ----------
    msg  : iterable of hashable symbols (int, str, …)
    base : logarithm base (2 → bits, e → nats)

    Returns
    -------
    float  H = -Σ p(x) log_base p(x)
    """
    from collections import Counter
    counts = Counter(msg)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    probs = np.array([v / n for v in counts.values()], dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs) / np.log(base)))


def lempel_ziv_complexity(binary_sequence, normalize: bool = True) -> float:
    """
    Lempel-Ziv (LZ-76) complexity.  AFML Snippet 18.3.

    Implements the Kaspar-Schuster (1987) production-rule algorithm.
    Normalised by n / log2(n) — the asymptotic upper bound for a uniformly
    random binary string — so the value is ≈1 for a random sequence and
    approaches 0 for a highly repetitive one.

    Parameters
    ----------
    binary_sequence : 1-D iterable coercible to {0, 1}
    normalize       : divide by n / log₂(n) if True

    Returns
    -------
    float
    """
    s = "".join(str(int(x)) for x in binary_sequence)
    n = len(s)
    if n == 0:
        return 0.0 if normalize else 0

    i, c, u, v, vmax = 0, 1, 1, 1, 1
    while u + i <= n:
        if s[i + u - 1] == s[v + u - 1]:
            u += 1
            if v + u > n:
                c += 1
                break
        else:
            if u > vmax:
                vmax = u
            i += 1
            if i == v:
                c += 1
                v += vmax
                if v + 1 > n:
                    break
                i, u, vmax = 0, 1, 1
            else:
                u = 1

    if normalize:
        if n <= 1:
            return float(c)
        return float(c) / (n / np.log2(n))
    return c


def kontoyiannis_entropy(msg, window: int = None) -> float:
    """
    Kontoyiannis entropy estimator based on match lengths.  AFML §18.2.

    Ĥ = (n / Σ_i L_i) * log₂(n)

    where L_i is the length of the longest prefix of S[i:] that already
    appears in S[:i] (or in the last `window` symbols if window is set).

    Parameters
    ----------
    msg    : 1-D iterable of hashable symbols
    window : restrict history search to last `window` symbols (speeds up)

    Returns
    -------
    float, estimated entropy in bits per symbol
    """
    s = list(msg)
    n = len(s)
    if n <= 1:
        return 0.0

    match_lengths = []
    for i in range(1, n):
        start = max(0, i - window) if window else 0
        history = s[start:i]
        max_len = 0
        for l in range(1, n - i + 1):
            pattern = s[i:i + l]
            found = any(history[k:k + l] == pattern
                        for k in range(len(history) - l + 1))
            if found:
                max_len = l
            else:
                break
        match_lengths.append(max_len + 1)

    total = sum(match_lengths)
    if total == 0:
        return 0.0
    return float((n / total) * np.log2(n))


def rolling_entropy(returns: pd.Series, window: int = 50,
                    n_bins: int = 10) -> pd.Series:
    """
    Rolling Shannon entropy of discretised log returns.

    Parameters
    ----------
    returns : pd.Series of log returns
    window  : rolling window length
    n_bins  : number of histogram bins

    Returns
    -------
    pd.Series of entropy values (bits), NaN for first window-1 observations
    """
    def _h(x):
        hist, _ = np.histogram(x, bins=n_bins)
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    return returns.rolling(window=window).apply(_h, raw=True)


def rolling_lz(returns: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling normalised Lempel-Ziv complexity of sign-encoded returns.

    Parameters
    ----------
    returns : pd.Series of log returns
    window  : rolling window length

    Returns
    -------
    pd.Series of normalised LZ complexity values
    """
    signs = (returns > 0).astype(int)
    return signs.rolling(window=window).apply(
        lambda x: lempel_ziv_complexity(x.astype(int), normalize=True),
        raw=True,
    )
