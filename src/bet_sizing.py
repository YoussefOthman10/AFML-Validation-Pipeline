"""
Bet sizing module - AFML Chapter 10.

Implements:
  - get_signal         : probability to signal in [-1, 1]  (Snippet 10.1)
  - avg_active_signals : average signal across overlapping events (Snippet 10.2)
  - discrete_signal    : round signal to step_size  (Snippet 10.3)
  - build_daily_positions : expand event signals to full daily series
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_signal(side, meta_prob, num_classes=2, step_size=0.0):
    """
    Convert meta-model probability to a directional signal.

    AFML Snippet 10.1 (binary case, num_classes=2):
        z = (p - 1/K) / sqrt(p*(1-p))     where K = num_classes
        size = 2*Phi(z) - 1                in [0, 1]
        signal = side * size               in [-1, 1]

    p=0.5 -> z=0 -> size=0 -> no trade.
    p->1  -> z->inf -> size->1 -> full position in predicted direction.
    """
    p = meta_prob.clip(1e-6, 1 - 1e-6)
    z = (p - 1.0 / num_classes) / np.sqrt(p * (1 - p))
    size = 2 * norm.cdf(z) - 1
    signal = side * size

    if step_size > 0:
        signal = discrete_signal(signal, step_size)

    return signal


def avg_active_signals(signals, t1):
    """
    At each event time t, compute mean of all currently active signals.

    A signal is active at t if event_start <= t <= barrier_end (t1).
    AFML Snippet 10.2.
    """
    t_starts = signals.index.tolist()
    t_ends   = t1.dropna().tolist()
    t_points = sorted(set(t_starts + t_ends))

    pos = {}
    for t in t_points:
        mask_started = signals.index <= t
        mask_active  = t1 >= t
        active = signals[mask_started & mask_active]
        pos[t] = active.mean() if len(active) > 0 else 0.0

    return pd.Series(pos)


def discrete_signal(signal, step_size=0.1):
    """Round signal to nearest step_size, then clip to [-1, 1]. AFML Snippet 10.3."""
    return ((signal / step_size).round() * step_size).clip(-1, 1)


def build_daily_positions(avg_positions, daily_index):
    """
    Expand event-level average positions to full daily frequency via forward-fill.

    Before first event: 0. After all signals expire: 0.
    """
    daily_pos = avg_positions.reindex(daily_index, method="ffill").fillna(0.0)
    return daily_pos
