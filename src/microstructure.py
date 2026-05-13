"""
Microstructure features — AFML Chapter 19.

Daily-OHLCV approximations of intraday market-microstructure quantities.
Standalone versions of the estimators in src/features.py, made importable
for analysis notebooks.

Implements:
  corwin_schultz_spread   : bid-ask spread from H/L  [AFML §19.3]
  bekker_parkinson_vol    : range-based volatility   [AFML §19.2]
  amihud_illiquidity      : price-impact proxy       [AFML §19.4]
  roll_spread             : Roll (1984) spread        [AFML §19.5]

All functions return pd.Series (or pd.DataFrame for multi-output) indexed
by the input DataFrame's DatetimeIndex.

Limitations acknowledged in the AFML plan:
- Corwin-Schultz: designed for daily H/L data but can produce negative α
  when no-arbitrage bounds are violated. We floor the result at 0.
- Kyle's Lambda and VPIN require tick-level data — not implemented.
- Roll spread from daily data is noisy.
"""
import numpy as np
import pandas as pd


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Corwin-Schultz bid-ask spread estimator.  AFML Snippet 19.1.

    Uses the high-low range over two consecutive days:
        β = [ln(H_t/L_t)]² + [ln(H_{t-1}/L_{t-1})]²
        γ = [ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))]²
        α = (√(2β) - √β) / (3 - 2√2) − √(γ / (3 - 2√2))
        S = 2(eᵅ - 1) / (1 + eᵅ)

    Floored at 0 when α < 0 (no-arbitrage assumption violated).

    Parameters
    ----------
    high, low : pd.Series of daily High/Low prices

    Returns
    -------
    pd.Series of estimated spread (fractional units, e.g. 0.01 = 1%)
    """
    H_L = np.log(high / low) ** 2
    H_L_prev = H_L.shift(1)

    max_H = np.maximum(high, high.shift(1))
    min_L = np.minimum(low, low.shift(1))
    gamma = np.log(max_H / min_L) ** 2

    beta = H_L + H_L_prev
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rename('corwin_schultz_spread')


def bekker_parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Bekker-Parkinson (range-based) daily volatility.  AFML §19.2.

        σ² = (1 / (4 ln 2)) · [ln(H_t / L_t)]²

    Returns
    -------
    pd.Series of daily variance estimates (non-negative by construction)
    """
    return ((1.0 / (4.0 * np.log(2.0))) * np.log(high / low) ** 2
            ).rename('bekker_parkinson_vol')


def amihud_illiquidity(close: pd.Series, volume: pd.Series,
                       window: int = 20) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio.  AFML §19.4.

        λ_t = |r_t| / DollarVolume_t

    Reported as the rolling mean over `window` days.  Tiny in absolute
    terms for large-cap stocks (dollar volume in billions).

    Parameters
    ----------
    close  : pd.Series of Adj Close prices
    volume : pd.Series of share volume
    window : rolling mean window

    Returns
    -------
    pd.Series of rolling-average illiquidity
    """
    ret = np.abs(np.log(close / close.shift(1)))
    dollar_vol = (close * volume).replace(0, np.nan)
    daily_lambda = ret / dollar_vol
    return daily_lambda.rolling(window=window).mean().rename('amihud_illiquidity')


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Roll (1984) bid-ask spread estimate from return serial covariance.

        S = 2 · √(max(0, −Cov(Δp_t, Δp_{t-1})))

    When the rolling covariance is positive (returns exhibit positive
    autocorrelation, violating the model's assumptions) we floor the
    argument of the square root at 0 — equivalent to reporting zero
    estimable spread for that window.

    Parameters
    ----------
    close  : pd.Series of Adj Close prices
    window : rolling window for covariance

    Returns
    -------
    pd.Series of Roll spread estimates
    """
    dp = close.diff()
    dp_prev = dp.shift(1)
    cov = dp.rolling(window=window).cov(dp_prev)
    return (2.0 * np.sqrt(np.maximum(0.0, -cov))).rename('roll_spread')


def all_microstructure_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute all daily-OHLCV microstructure features in one call.

    Parameters
    ----------
    df     : DataFrame with columns [High, Low, Adj Close, Volume]
    window : rolling window for Amihud and Roll

    Returns
    -------
    pd.DataFrame with columns:
        corwin_schultz_spread, bekker_parkinson_vol,
        amihud_illiquidity, roll_spread
    """
    return pd.concat([
        corwin_schultz_spread(df['High'], df['Low']),
        bekker_parkinson_vol(df['High'], df['Low']),
        amihud_illiquidity(df['Adj Close'], df['Volume'], window=window),
        roll_spread(df['Adj Close'], window=window),
    ], axis=1)
