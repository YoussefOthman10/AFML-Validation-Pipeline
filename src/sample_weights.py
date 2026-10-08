import pandas as pd
import numpy as np

def num_co_events(close_idx: pd.DatetimeIndex, t1: pd.Series, molecule: pd.DatetimeIndex) -> pd.Series:
    """
    Compute number of concurrent events per bar.
    c_t = Σ_i 1_{t ∈ [t₀ⁱ, t₁ⁱ]}
    """
    # Find events that span over each bar in close_idx
    t1 = t1.fillna(close_idx[-1]) # Unclosed events must still be active
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:molecule[-1]]
    
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1] + 1])
    
    for t0, t1_i in t1.items():
        count.loc[t0:t1_i] += 1
        
    return count.loc[molecule[0]:t1.max()]

def sample_tw(t1: pd.Series, num_co_events: pd.Series, molecule: pd.DatetimeIndex) -> pd.Series:
    """
    Compute average uniqueness for each event.
    ū_i = (1/(t₁ⁱ - t₀ⁱ + 1)) × Σ_{t=t₀}^{t₁} (1/c_t)
    """
    tw = pd.Series(index=molecule, dtype=float)
    for t0 in molecule:
        t1_i = t1.loc[t0]
        if pd.isna(t1_i):
            continue
        tw.loc[t0] = (1.0 / num_co_events.loc[t0:t1_i]).mean()
    return tw

def get_ind_matrix(bar_idx: pd.DatetimeIndex, t1: pd.Series) -> pd.DataFrame:
    """
    Binary indicator matrix: rows = bars, columns = events.
    ind[t, i] = 1 if t ∈ [t0_i, t1_i].
    """
    ind_matrix = pd.DataFrame(0, index=bar_idx, columns=range(t1.shape[0]))
    for i, (t0, t1_i) in enumerate(t1.items()):
        if pd.isna(t1_i):
            continue
        ind_matrix.loc[t0:t1_i, i] = 1
    return ind_matrix

def seq_bootstrap(ind_matrix: pd.DataFrame, s_length: int = None) -> list:
    """
    Sequential bootstrap: draw proportional to uniqueness given already-selected samples.
    Optimized with NumPy vectorization.
    """
    if s_length is None:
        s_length = ind_matrix.shape[1]
        
    ind_mat = ind_matrix.values
    ind_mat_drawn = np.zeros(ind_mat.shape[0])
    active_mask = ind_mat > 0
    active_counts = active_mask.sum(axis=0)
    
    phi = []
    
    for _ in range(s_length):
        # C_t_all is the count of drawn events per bar PLUS the candidate event
        C_t_all = ind_mat_drawn[:, None] + ind_mat
        
        # Calculate 1 / C_t only where the candidate event is active
        inv_C_t = np.zeros_like(C_t_all, dtype=float)
        inv_C_t[active_mask] = 1.0 / C_t_all[active_mask]
        
        # U_i is the mean of inv_C_t over active bars
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = inv_C_t.sum(axis=0) / active_counts
            prob[active_counts == 0] = 0
            
        # Normalize to probability distribution
        prob = prob / prob.sum()
        
        # Draw sample
        chosen = np.random.choice(ind_mat.shape[1], p=prob)
        phi.append(chosen)
        
        # Update drawn counts
        ind_mat_drawn += ind_mat[:, chosen]
        
    return phi

def get_return_attribution(events: pd.DataFrame) -> pd.Series:
    """
    w_i = |ret_i| / sum(|ret|), then multiply by n.
    """
    ret = events['ret'].abs()
    w = ret / ret.sum()
    w = w * len(ret)
    return w

def get_time_decay(tw: pd.Series, c_lf: float = 0.5) -> pd.Series:
    """
    Time decay: d_i = c^{x_i} where x_i = (i - 0) / (N - 1).
    Piecewise linear: oldest gets c_lf, newest gets 1. Normalise so mean = 1.
    """
    clf = pd.Series(tw.values).sort_index()
    clf.loc[:] = 1
    clf.iloc[0] = c_lf
    clf = clf.interpolate()
    
    # Alternatively: x_i = i / (N - 1), d_i = c_lf + (1 - c_lf) * x_i
    N = len(tw)
    x = np.arange(N) / max((N - 1), 1)
    d = c_lf + (1 - c_lf) * x
    
    decay = pd.Series(d, index=tw.index)
    decay = decay / decay.mean()
    return decay

def get_sample_weight(events: pd.DataFrame, close: pd.Series, num_threads: int = 1) -> pd.Series:
    """
    Combine: uniqueness × return_attribution × time_decay. Normalise.
    """
    # 1. Uniqueness
    num_co = num_co_events(close.index, events['t1'], events.index)
    tw = sample_tw(events['t1'], num_co, events.index)
    
    # 2. Return Attribution
    # Note: Using |ret| directly without multiplying by tw because the prompt says:
    # "w_i = |ret_i| / sum(|ret|), then multiply by n." (This replaces tw if we strictly follow that,
    # but the prompt says combine: uniqueness × return_attribution × time_decay. Let's multiply them).
    ret_attr = get_return_attribution(events)
    
    # 3. Time Decay
    decay = get_time_decay(tw)
    
    # Combine
    w = tw * ret_attr * decay
    w = w / w.mean()
    
    return w
