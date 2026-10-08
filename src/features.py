"""
Stage 3 — feature engineering.

All microstructure measures here are *daily-OHLCV approximations* of the
intraday quantities described in AFML Ch 19. In particular:

- Corwin-Schultz: derived from daily H/L/L_prev/H_prev. The estimator can
  produce alpha < 0 (and therefore a negative spread) when its no-arbitrage
  assumptions are violated by the data; with daily bars these violations are
  more frequent than with intraday data. We follow Corwin & Schultz (2012)
  and AFML Ch 19 by flooring the spread at 0.
- Bekker-Parkinson: uses (1/(4 ln 2)) * [ln(H/L)]^2, which is non-negative
  by construction.
- Amihud: |log return| / dollar volume, smoothed by a 20-day mean. Tiny
  in absolute terms when dollar volume is in billions — that's expected.
- Roll: 2 * sqrt(-cov(dp_t, dp_{t-1})). When the rolling covariance is
  positive (the model's mean-reverting-noise assumption fails) we floor
  the input at 0 so the square root is real, equivalent to reporting
  "no estimable spread" for that window.
"""
import pandas as pd
import numpy as np
from scipy import stats

def compute_momentum_features(close: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    
    # Rolling returns
    for d in [5, 10, 20, 60]:
        features[f'ret_{d}d'] = np.log(close / close.shift(d))
        
    # Momentum 12-1 (252-day minus 21-day)
    ret_252 = np.log(close / close.shift(252))
    ret_21 = np.log(close / close.shift(21))
    features['momentum_12_1'] = ret_252 - ret_21
    
    # 14-day RSI (Wilder's Smoothing)
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = down.abs()
    
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    features['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    return features

def compute_volatility_features(close: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    log_ret = np.log(close / close.shift(1))
    
    for d in [20, 50]:
        features[f'vol_{d}d'] = log_ret.rolling(window=d).std() * np.sqrt(252)
        
    return features

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    features['log_dollar_volume'] = np.log((df['Adj Close'] * df['Volume']).replace(0, np.nan))
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    return features

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    
    # Corwin-Schultz spread
    H_L = np.log(high / low) ** 2
    H_L_prev = H_L.shift(1)
    
    max_H = np.maximum(high, high.shift(1))
    min_L = np.minimum(low, low.shift(1))
    gamma = np.log(max_H / min_L) ** 2
    
    beta = H_L + H_L_prev
    alpha_num = np.sqrt(2 * beta) - np.sqrt(beta)
    alpha_den = 3 - 2 * np.sqrt(2)
    alpha = alpha_num / alpha_den - np.sqrt(gamma / alpha_den)
    
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    # Floor at 0: alpha<0 implies the estimator's no-arbitrage assumption
    # is violated for that pair of days. The accepted convention (Corwin &
    # Schultz 2012; AFML Ch 19) is to clip the resulting spread to 0 rather
    # than report a negative bid-ask spread, which is meaningless.
    spread = spread.clip(lower=0.0)
    features['corwin_schultz_spread'] = spread
    
    # Bekker-Parkinson vol
    features['bekker_parkinson_vol'] = (1.0 / (4 * np.log(2))) * H_L
    
    # Amihud illiquidity
    ret = np.abs(np.log(close / close.shift(1)))
    dollar_vol = close * df['Volume']
    amihud = ret / dollar_vol.replace(0, np.nan)
    features['amihud_illiquidity'] = amihud.rolling(window=20).mean()
    
    # Roll spread
    dp = close.diff()
    dp_prev = dp.shift(1)
    
    cov_dp = dp.rolling(window=20).cov(dp_prev)
    features['roll_spread'] = 2 * np.sqrt(np.maximum(0, -cov_dp))
    
    return features

def lempel_ziv_complexity(binary_sequence, normalize: bool = False) -> float:
    """
    Lempel-Ziv (LZ-76) complexity of a binary sequence.

    Implements Kaspar & Schuster's (1987) production-rule algorithm:
    we walk a pointer along S; for each position we look for the longest
    word starting there that has already appeared anywhere to its left.
    Each time the prefix-matching pointer reaches the start of the
    current (yet-unparsed) tail, we have just identified one new
    production and increment the complexity counter.

    Parameters
    ----------
    binary_sequence : 1-D iterable of ints/floats coercible to {0, 1}.
    normalize : bool
        If True, divide by n / log2(n) (the asymptotic upper bound for a
        uniformly random binary string of length n). Normalised values
        sit in [0, ≈1] regardless of window length, which is what we want
        when this function is called inside a rolling window of fixed
        size.

    Returns
    -------
    float (or int when normalize=False).

    Notes
    -----
    The previous in-repo implementation had off-by-one index pairs in the
    inner comparison and never advanced the production counter for any
    realistic input — every rolling window collapsed to the initial value
    of 1. This implementation is the textbook one and matches the
    Kaspar-Schuster reference: `'0010'` → c = 3, an all-zero string of
    length 32 → c = 2, and an alternating 32-bit string → c = 3.
    """
    s = "".join(str(int(x)) for x in binary_sequence)
    n = len(s)
    if n == 0:
        return 0.0 if normalize else 0

    i = 0          # prefix-search pointer
    c = 1          # number of productions
    u = 1          # current word length
    v = 1          # boundary between "history" and "yet-unparsed tail"
    vmax = v       # longest word matched so far for this production
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
                i = 0
                u = 1
                vmax = 1
            else:
                u = 1

    if normalize:
        if n <= 1:
            return float(c)
        upper = n / np.log2(n)
        return float(c) / upper
    return c

def compute_entropy_features(returns: pd.Series, window: int = 50) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    
    # Rolling Shannon entropy
    def shannon_entropy(x):
        if len(x) == 0: return np.nan
        hist, _ = np.histogram(x, bins=10)
        p = hist / np.sum(hist)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
        
    features['shannon_entropy'] = returns.rolling(window=window).apply(shannon_entropy, raw=True)
    
    # Lempel-Ziv complexity
    # Binary encode sign of returns
    signs = np.where(returns > 0, 1, 0)
    signs_series = pd.Series(signs, index=returns.index)
    
    features['lempel_ziv_complexity'] = signs_series.rolling(window=window).apply(
        lambda x: lempel_ziv_complexity(x.astype(int), normalize=True),
        raw=True,
    )
    
    return features

def build_feature_matrix(df: pd.DataFrame, fracdiff: pd.Series, events: pd.DataFrame, labels: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    close = df['Adj Close']
    ret = np.log(close / close.shift(1))
    
    # Compute feature groups
    mom_features = compute_momentum_features(close)
    vol_features = compute_volatility_features(close)
    volume_features = compute_volume_features(df)
    micro_features = compute_microstructure_features(df)
    ent_features = compute_entropy_features(ret)
    
    # Combine features
    all_features = pd.concat([
        mom_features, vol_features, volume_features, micro_features, ent_features
    ], axis=1)
    
    # fracdiff should be passed as a DataFrame or Series. Assume fracdiff is a Series or a DataFrame with one column
    if isinstance(fracdiff, pd.Series):
        all_features['fracdiff'] = fracdiff
    else:
        all_features['fracdiff'] = fracdiff.iloc[:, 0]
    
    # Align to event timestamps from labels
    dataset = all_features.loc[labels.index].copy()
    
    # Merge with labels and weights
    dataset['label'] = labels['bin']
    dataset['weight'] = weights['weight'] if 'weight' in weights else weights.iloc[:, 0]
    
    # Also include the t1 exit time (used later for purged CV)
    if 't1' in events.columns:
        dataset['t1'] = events['t1']
    elif 't1' in labels.columns:
        dataset['t1'] = labels['t1']
        
    # Drop rows with any NaN
    dataset.dropna(inplace=True)
    
    return dataset
