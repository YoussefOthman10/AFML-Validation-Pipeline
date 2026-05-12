"""
Synthetic data generation for pipeline validation - AFML Chapter 13.

generate_trending       : GBM with drift (planted trend signal)
generate_mean_reverting : Ornstein-Uhlenbeck process (planted MR signal)
"""

import numpy as np
import pandas as pd


def generate_trending(n: int = 5000, drift: float = 0.0005,
                      vol: float = 0.02, seed: int = 42) -> pd.Series:
    """
    Geometric Brownian Motion with positive drift (trending series).

    r_t = drift + vol * eps_t,  eps_t ~ N(0,1)
    p_t = p_0 * exp(cumsum(r))

    A momentum / trend-following model should detect the planted signal.
    """
    rng = np.random.default_rng(seed)
    r   = drift + vol * rng.standard_normal(n)
    p   = 100.0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(end="2025-01-01", periods=n)
    return pd.Series(p, index=idx, name="synthetic_trending")


def generate_mean_reverting(n: int = 5000, theta: float = 0.1,
                             mu: float = 100.0, vol: float = 2.0,
                             seed: int = 42) -> pd.Series:
    """
    Ornstein-Uhlenbeck (mean-reverting) process.

    dp = theta * (mu - p) * dt + vol * dW

    A model with short-horizon momentum signals should detect reversals
    relative to the mean.
    """
    rng    = np.random.default_rng(seed)
    prices = [mu]
    for _ in range(1, n):
        dp = theta * (mu - prices[-1]) + vol * rng.standard_normal()
        prices.append(prices[-1] + dp)
    idx = pd.bdate_range(end="2025-01-01", periods=n)
    return pd.Series(prices, index=idx, name="synthetic_mean_reverting")
