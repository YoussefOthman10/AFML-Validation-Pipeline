"""
Backtesting engine and performance statistics - AFML Chapters 11-14.

Implements:
  - backtest_strategy     : PnL from positions + prices  (Ch 11)
  - sharpe_ratio          : annualised SR               (Ch 14.7.1)
  - prob_sharpe_ratio     : PSR                         (Ch 14.7.2)
  - deflated_sharpe_ratio : DSR                         (Ch 14.7.3)
  - compute_dd_tuw        : drawdown & time under water (Snippet 14.4)
  - calmar_ratio, hit_ratio, profit_factor
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_strategy(positions: pd.Series, prices: pd.Series,
                      cost_bps: float = 5.0) -> pd.DataFrame:
    """
    Compute daily PnL from a position series and price series.

    Daily gross return  = position_{t-1} * (price_t / price_{t-1} - 1)
    Transaction cost    = |position_t - position_{t-1}| * cost_bps / 10_000
    Net return          = gross - cost

    positions.shift(1) ensures the position determined at end of day t-1
    drives returns from t-1 to t (no look-ahead).

    Parameters
    ----------
    positions : daily position series in [-1, 1], indexed by date
    prices    : daily Adj Close prices, indexed by date
    cost_bps  : transaction cost in basis points per unit of turnover

    Returns
    -------
    DataFrame with columns:
        position, price_return, gross_return, cost, net_return, cumulative,
        hwm, drawdown
    """
    pos = positions.reindex(prices.index).ffill().fillna(0.0)
    price_ret = prices.pct_change().fillna(0.0)

    # One-day lag: position from yesterday drives today's return
    gross = pos.shift(1).fillna(0.0) * price_ret
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * cost_bps / 10_000.0
    net = gross - cost

    cum = (1 + net).cumprod()
    hwm = cum.expanding().max()
    dd  = 1 - cum / hwm

    return pd.DataFrame({
        "position":    pos,
        "price_return": price_ret,
        "gross_return": gross,
        "cost":         cost,
        "net_return":   net,
        "cumulative":   cum,
        "hwm":          hwm,
        "drawdown":     dd,
    })


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe Ratio. SR = (mu / sigma) * sqrt(T)."""
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    return float((mu / sigma) * np.sqrt(periods_per_year))


def prob_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (AFML Ch 14.7.2).

    PSR[SR*] = Phi( (SR_hat - SR*) * sqrt(T-1) /
                    sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )

    Uses non-annualised SR internally; SR* must be passed in the same units.
    """
    sr   = returns.mean() / returns.std() if returns.std() > 0 else 0.0
    T    = len(returns)
    skew = float(returns.skew())
    # scipy kurtosis is excess; PSR formula needs regular (excess + 3)
    kurt = float(returns.kurtosis()) + 3.0

    num = (sr - sr_benchmark) * np.sqrt(T - 1)
    den = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(norm.cdf(num / den))


def deflated_sharpe_ratio(returns: pd.Series, num_trials: int,
                          var_sr_trials: float = None) -> float:
    """
    Deflated Sharpe Ratio (AFML Ch 14.7.3).

    Adjusts SR* for multiple testing across num_trials strategy evaluations.

    SR* = sqrt(V[SR]) * ((1-gamma)*Phi^{-1}(1 - 1/N)
                        + gamma * Phi^{-1}(1 - 1/(N*e)))
    where gamma = Euler-Mascheroni constant ~ 0.5772.

    Then DSR = PSR(SR*).
    """
    euler_gamma = 0.5772156649015329

    sr    = returns.mean() / returns.std() if returns.std() > 0 else 0.0
    T     = len(returns)
    skew  = float(returns.skew())
    kurt  = float(returns.kurtosis()) + 3.0

    if var_sr_trials is None:
        var_sr_trials = max(
            1e-12,
            (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(T - 1, 1)
        )

    N = max(num_trials, 2)
    sr_star = np.sqrt(var_sr_trials) * (
        (1 - euler_gamma) * norm.ppf(1 - 1 / N) +
        euler_gamma       * norm.ppf(1 - 1 / (N * np.e))
    )

    # Convert SR* from non-annualised to match PSR input convention
    return prob_sharpe_ratio(returns, sr_benchmark=sr_star)


def compute_dd_tuw(returns: pd.Series) -> tuple:
    """
    Compute drawdown series and time-under-water duration (AFML Snippet 14.4).

    Returns
    -------
    dd         : pd.Series of drawdown fractions (0 = at HWM)
    max_dd     : float, maximum drawdown fraction
    max_tuw    : int, longest time-under-water in calendar days
    peak_date  : date of HWM before largest drawdown
    trough_date: date of trough of largest drawdown
    """
    cum  = (1 + returns).cumprod()
    hwm  = cum.expanding().max()
    dd   = 1 - cum / hwm

    max_dd = float(dd.max())
    peak_date   = hwm.idxmax()
    trough_date = dd.idxmax()

    # Time under water: duration of each contiguous drawdown episode
    in_dd = dd > 1e-10
    edges_start = in_dd & ~in_dd.shift(1, fill_value=False)
    edges_end   = ~in_dd & in_dd.shift(1, fill_value=False)

    starts = edges_start[edges_start].index.tolist()
    ends   = edges_end[edges_end].index.tolist()

    tuw_days = []
    for s in starts:
        later_ends = [e for e in ends if e > s]
        if later_ends:
            tuw_days.append((later_ends[0] - s).days)

    max_tuw = int(max(tuw_days)) if tuw_days else 0
    return dd, max_dd, max_tuw, peak_date, trough_date


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised return / |max drawdown|."""
    ann_ret = float((1 + returns.mean()) ** periods_per_year - 1)
    _, max_dd, *_ = compute_dd_tuw(returns)
    if max_dd == 0:
        return np.inf
    return ann_ret / max_dd


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of trading days with positive return."""
    active = returns[returns != 0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross profits / |gross losses|."""
    gains  = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf")
    return float(abs(gains / losses))


def summary_table(returns: pd.Series, num_trials: int,
                  periods_per_year: int = 252) -> pd.Series:
    """
    Compile T11: all backtest statistics into a single Series.

    Parameters
    ----------
    returns    : daily net returns (non-zero only where position exists)
    num_trials : total number of model evaluations tried (for DSR)
    """
    sr   = sharpe_ratio(returns, periods_per_year)
    psr  = prob_sharpe_ratio(returns, sr_benchmark=0.0)
    dsr  = deflated_sharpe_ratio(returns, num_trials=num_trials)
    dd, max_dd, max_tuw, peak, trough = compute_dd_tuw(returns)
    calmar = calmar_ratio(returns, periods_per_year)
    ann_ret = float((1 + returns.mean()) ** periods_per_year - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))

    return pd.Series({
        "Start Date":            str(returns.index.min().date()),
        "End Date":              str(returns.index.max().date()),
        "Total Days":            len(returns),
        "Annualised Return":     f"{ann_ret:.4%}",
        "Annualised Volatility": f"{ann_vol:.4%}",
        "Sharpe Ratio":          f"{sr:.4f}",
        "PSR (SR*=0)":           f"{psr:.4f}",
        f"DSR (N={num_trials})": f"{dsr:.4f}",
        "Max Drawdown":          f"{max_dd:.4%}",
        "Time Under Water (d)":  max_tuw,
        "Calmar Ratio":          f"{calmar:.4f}",
        "Hit Ratio":             f"{hit_ratio(returns):.4%}",
        "Profit Factor":         f"{profit_factor(returns):.4f}",
        "Peak Date":             str(peak.date()) if hasattr(peak, "date") else str(peak),
        "Trough Date":           str(trough.date()) if hasattr(trough, "date") else str(trough),
    }, name="Backtest Statistics")
