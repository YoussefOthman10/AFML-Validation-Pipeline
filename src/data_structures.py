import pandas as pd
import numpy as np

def cusum_filter(close: pd.Series, h: float) -> pd.DatetimeIndex:
    """
    Implement symmetric CUSUM on log returns.
    y_t = log(close_t / close_{t-1}), E[y] = expanding mean.
    Trigger event when S⁺ > h or S⁻ < -h, then reset.
    """
    events = []
    
    # Calculate log returns
    y = np.log(close / close.shift(1)).dropna()
    
    # Compute expanding mean
    E_y = y.expanding().mean()
    
    S_plus, S_minus = 0.0, 0.0
    
    for t in y.index:
        y_t = y.loc[t]
        E_y_t = E_y.loc[t]
        
        # CUSUM equations
        # S⁺_t = max(0, S⁺_{t-1} + y_t - E[y] - h)
        # S⁻_t = min(0, S⁻_{t-1} + y_t - E[y] + h)
        
        S_plus = max(0.0, S_plus + y_t - E_y_t - h)
        S_minus = min(0.0, S_minus + y_t - E_y_t + h)
        
        if S_plus > h:
            events.append(t)
            S_plus, S_minus = 0.0, 0.0
        elif S_minus < -h:
            events.append(t)
            S_plus, S_minus = 0.0, 0.0
            
    return pd.DatetimeIndex(events)

def get_dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    df has columns [Open, High, Low, Close, Adj Close, Volume].
    Compute dollar_volume = Adj Close × Volume each day.
    Accumulate dollar volume; when cumsum ≥ threshold, emit one bar.
    Bar OHLCV: Open = first day's Open, High = max of constituent Highs, 
    Low = min of Lows, Close = last day's Close, Volume = sum.
    Return DataFrame indexed by bar end date.
    """
    bars = []
    
    # Ensure dollar volume is calculated
    dollar_volume = df['Adj Close'] * df['Volume']
    
    cum_dollar_volume = 0.0
    
    # Variables to track current bar
    current_open = None
    current_high = -np.inf
    current_low = np.inf
    current_volume = 0.0
    
    for t, row in df.iterrows():
        if current_open is None:
            current_open = row['Open']
            
        current_high = max(current_high, row['High'])
        current_low = min(current_low, row['Low'])
        current_volume += row['Volume']
        
        cum_dollar_volume += dollar_volume.loc[t]
        
        if cum_dollar_volume >= threshold:
            bars.append({
                'Date': t,
                'Open': current_open,
                'High': current_high,
                'Low': current_low,
                'Close': row['Close'],
                'Adj Close': row['Adj Close'],
                'Volume': current_volume
            })
            
            # Reset
            cum_dollar_volume = 0.0
            current_open = None
            current_high = -np.inf
            current_low = np.inf
            current_volume = 0.0
            
    # Convert to DataFrame
    bars_df = pd.DataFrame(bars)
    if not bars_df.empty:
        bars_df.set_index('Date', inplace=True)
    return bars_df

def calibrate_cusum_h(close: pd.Series, target_events: int = 400) -> float:
    """
    Binary search over h to find value producing ~target_events.
    """
    # Define bounds for h
    low = 0.0001
    high = 0.5
    
    best_h = high
    min_diff = float('inf')
    
    # Binary search
    for _ in range(20): # max 20 iterations
        mid = (low + high) / 2
        events = cusum_filter(close, mid)
        n_events = len(events)
        
        diff = abs(n_events - target_events)
        if diff < min_diff:
            min_diff = diff
            best_h = mid
            
        if n_events > target_events:
            # Too many events, h is too small
            low = mid
        else:
            # Too few events, h is too big
            high = mid
            
        if diff == 0:
            break
            
    return best_h

def calibrate_dollar_bar_threshold(df: pd.DataFrame, target_bars_per_year: int = 252) -> float:
    """
    Calibrate the dollar bar threshold to target a specific number of bars per year.
    """
    dollar_volume = df['Adj Close'] * df['Volume']
    total_dollar_volume = dollar_volume.sum()
    
    # Calculate total number of years in the dataset
    if len(df) == 0:
        return 0.0
    days = (df.index.max() - df.index.min()).days
    if days == 0:
        years = len(df) / 252.0
    else:
        years = days / 365.25
        
    target_total_bars = target_bars_per_year * years
    
    if target_total_bars == 0:
        return 0.0
        
    threshold = total_dollar_volume / target_total_bars
    
    return threshold
