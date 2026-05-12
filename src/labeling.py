import pandas as pd
import numpy as np

def get_daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """
    Compute daily volatility: EWMA std of log returns.
    σ_t = EWMA_std(log_returns, span=50)
    """
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.ewm(span=span).std()
    return vol

def add_vertical_barrier(close: pd.Series, events: pd.DatetimeIndex, num_days: int) -> pd.Series:
    """
    For each event timestamp, find the date num_days ahead in the index (or last date).
    t₁ = t₀ + max_holding_days
    """
    t1 = close.index.searchsorted(events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=events[:t1.shape[0]])
    return t1

def apply_triple_barrier(close: pd.Series, events: pd.DataFrame, pt_sl: list = [1.0, 1.0], molecule: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    events DataFrame has columns: t1 (vertical barrier), trgt (daily vol at event).
    pt_sl = [profit_take_multiplier, stop_loss_multiplier].
    Return DataFrame with columns [t1 (actual exit), sl (stop-loss time or NaT), pt (profit-take time or NaT)].
    """
    if molecule is None:
        molecule = events.index
        
    out = events[['t1']].copy(deep=True)
    out['pt'] = pd.NaT
    out['sl'] = pd.NaT
    
    pt, sl = pt_sl[0], pt_sl[1]
    
    for t0 in molecule:
        trgt = events.loc[t0, 'trgt']
        t1 = events.loc[t0, 't1']
        
        if pd.isna(t1):
            path = close.loc[t0:]
        else:
            path = close.loc[t0:t1]
            
        p0 = close.loc[t0]
        
        # Upper barrier: Close_t >= Close_{t0} * (1 + pt * σ_{t0})
        if pt > 0:
            upper_barrier = p0 * (1 + pt * trgt)
            pt_touches = path[path >= upper_barrier].index
            if not pt_touches.empty:
                out.loc[t0, 'pt'] = pt_touches[0]
                
        # Lower barrier: Close_t <= Close_{t0} * (1 - sl * σ_{t0})
        if sl > 0:
            lower_barrier = p0 * (1 - sl * trgt)
            sl_touches = path[path <= lower_barrier].index
            if not sl_touches.empty:
                out.loc[t0, 'sl'] = sl_touches[0]
                
    return out

def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Compute return = (close[t1] / close[t0]) - 1.
    label = sign(return), but 0 if return exactly 0.
    """
    out = pd.DataFrame(index=events.index)
    
    # The actual exit time is the earliest of pt, sl, t1
    events_ = events.dropna(subset=['t1'])
    earliest_exit = events_[['t1', 'pt', 'sl']].min(axis=1)
    
    out['t1'] = earliest_exit
    
    # Calculate returns
    p0 = close.loc[events_.index]
    p1 = close.loc[earliest_exit.values].values
    
    ret = (p1 / p0) - 1
    
    out['ret'] = ret
    out['bin'] = np.sign(ret)
    
    # label = sign(return), but 0 if return exactly 0
    out.loc[out['ret'] == 0, 'bin'] = 0
    
    return out

def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    Remove events where any label class has fewer than min_pct of samples.
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('Dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events
