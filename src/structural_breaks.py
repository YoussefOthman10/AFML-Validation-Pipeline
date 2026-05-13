"""
Structural break detection — AFML Chapter 17.

Implements:
  get_bsadf  : SADF path (expanding right-tailed ADF)    [Snippet 17.1]
  get_gsadf  : GSADF statistic (double-recursion)        [AFML §17.2]
  cv_sadf    : Monte Carlo critical values               [AFML §17.3]

ADF model in all tests:
    Δy_t = α + β·y_{t-1} + Σ_{j=1}^{lags} γ_j·Δy_{t-j} + ε_t
    H₀: β = 0   H₁: β > 0  (right-tailed — explosive / bubble test)

The t-statistic on β is returned by every helper.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core OLS ADF t-statistic
# ---------------------------------------------------------------------------

def _ols_tstat(X: np.ndarray, y: np.ndarray, col: int = 1) -> float:
    """OLS t-statistic for coefficient `col` in X @ beta = y."""
    n, k = X.shape
    if n <= k + 1:
        return np.nan
    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < k:
            return np.nan
        resid = y - X @ beta
        s2 = float(np.dot(resid, resid)) / (n - k)
        cov = s2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(max(0.0, cov[col, col])))
        return float(beta[col] / se) if se > 0 else np.nan
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def _adf_tstat(y_sub: np.ndarray, lags: int) -> float:
    """
    ADF t-statistic on sub-window y_sub.

    Model: Δy_t = α + β·y_{t-1} + Σ_{j=1}^{lags} γ_j·Δy_{t-j} + ε
    Returns t-stat on the y_{t-1} coefficient (index 1 of design matrix).
    """
    dy = np.diff(y_sub)
    n = len(dy)
    if n < lags + 2:
        return np.nan

    dy_t = dy[lags:]          # dependent variable
    y_lag = y_sub[lags:-1]    # y_{t-1}
    m = len(dy_t)
    if m < 3:
        return np.nan

    cols = [np.ones(m), y_lag]
    for j in range(1, lags + 1):
        cols.append(dy[lags - j: n - j])
    X = np.column_stack(cols)
    return _ols_tstat(X, dy_t, col=1)


# ---------------------------------------------------------------------------
# SADF path
# ---------------------------------------------------------------------------

def get_bsadf(log_p: pd.Series, min_sl: int, lags: int = 1) -> pd.Series:
    """
    SADF path: for each end-point r2 in [min_sl, T], fit ADF on log_p[0:r2+1]
    and record the right-tailed t-statistic.  AFML Snippet 17.1.

    Parameters
    ----------
    log_p   : log-price series with DatetimeIndex
    min_sl  : minimum window length (e.g. 63 ≈ one calendar quarter)
    lags    : number of lagged differences in the ADF regression

    Returns
    -------
    pd.Series indexed like log_p; NaN for the first min_sl - 1 observations.
    The supremum of this series is the SADF statistic.
    """
    y = log_p.values.astype(float)
    T = len(y)
    stats = np.full(T, np.nan)

    for r2 in range(min_sl - 1, T):
        stats[r2] = _adf_tstat(y[:r2 + 1], lags)

    return pd.Series(stats, index=log_p.index, name='bsadf')


# ---------------------------------------------------------------------------
# GSADF statistic
# ---------------------------------------------------------------------------

def get_gsadf(log_p: pd.Series, min_sl: int, lags: int = 1) -> float:
    """
    GSADF: supremum ADF over all sub-windows [r1, r2].

    More powerful than SADF for detecting multiple bubble episodes within
    the same series.  AFML §17.2.

    Warning
    -------
    O(T²) — only practical for series shorter than ~1000 observations.
    For long series, downsample or increase min_sl before calling.

    Returns
    -------
    float, the GSADF statistic (NaN if no valid window found).
    """
    y = log_p.values.astype(float)
    T = len(y)
    max_stat = -np.inf

    for r2 in range(min_sl - 1, T):
        for r1 in range(0, r2 - min_sl + 2):
            if (r2 - r1 + 1) < min_sl:
                continue
            t_stat = _adf_tstat(y[r1:r2 + 1], lags)
            if not np.isnan(t_stat):
                max_stat = max(max_stat, t_stat)

    return float(max_stat) if max_stat > -np.inf else np.nan


# ---------------------------------------------------------------------------
# Monte Carlo critical values
# ---------------------------------------------------------------------------

def cv_sadf(n: int, min_sl: int, lags: int = 1,
            reps: int = 200, seed: int = 42) -> dict:
    """
    Monte Carlo critical values for the SADF statistic.

    Simulates `reps` random walks of length `n`, computes the SADF maximum
    for each, and returns the 90%/95%/99% quantiles of the distribution.

    Parameters
    ----------
    n       : length of each simulated random walk (can be < T for speed;
              asymptotic distribution converges fast)
    min_sl  : minimum window (must match the value used in get_bsadf)
    lags    : ADF lags (must match)
    reps    : number of Monte Carlo replications
    seed    : random seed for reproducibility

    Returns
    -------
    dict with keys '90%', '95%', '99%'.
    """
    rng = np.random.default_rng(seed)
    sadf_maxima = []

    for _ in range(reps):
        rw = np.cumsum(rng.standard_normal(n))
        rw_s = pd.Series(rw)
        path = get_bsadf(rw_s, min_sl=min_sl, lags=lags)
        mx = float(path.max())
        if not np.isnan(mx):
            sadf_maxima.append(mx)

    arr = np.array(sadf_maxima)
    return {
        '90%': float(np.percentile(arr, 90)),
        '95%': float(np.percentile(arr, 95)),
        '99%': float(np.percentile(arr, 99)),
    }
