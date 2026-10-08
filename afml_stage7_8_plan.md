# AFML NVDA Pipeline — Stage 7–8 Implementation Plan

**Document Type:** Production-Grade Implementation Blueprint
**Author:** Senior Quantitative Research Architect
**Date:** May 12, 2026
**Scope:** Stage 7 (Meta-Labeling & Bet Sizing), Stage 8 (Backtesting & Statistics)
**Authoritative Reference:** López de Prado, M. *Advances in Financial Machine Learning*, Wiley (2018)
**Prerequisite:** Audit Report confirming Stages 0–6 CONDITIONALLY VALID

---

## 1. Executive Summary

This document provides a complete, implementation-ready blueprint for Stages 7 and 8 of the AFML NVDA pipeline. Stage 7 implements meta-labeling (AFML Chapter 3 §3.6–3.8) and bet sizing (Chapter 10). Stage 8 implements backtesting with full performance statistics (Chapters 11–14), CPCV-based robustness analysis (Chapter 12), and synthetic-data validation (Chapter 13).

Every design decision is driven by three constraints: (1) AFML fidelity to the book's snippets and methodology, (2) leakage prevention at every step, and (3) statistical realism given 195 samples of daily OHLCV data.

The plan is structured for sequential execution in Claude Code, with explicit checkpoints, artifact definitions, and validation gates.

---

## 2. Current Pipeline State — Verified Inputs

### 2.1 Available Artifacts (Safe to Reuse)

| Artifact | Contents | Usage in Stage 7–8 |
|----------|----------|-------------------|
| `nvda_clean.parquet` | Daily OHLCV, 5114 rows, 2005–2025 | Price series for backtesting PnL |
| `nvda_cusum_events.parquet` | CUSUM-filtered event timestamps | Reference only |
| `nvda_labels.parquet` | Triple-barrier labels: t0, t1, ret, label | Meta-label generation (t1 series, returns) |
| `nvda_sample_weights.parquet` | Uniqueness-based sample weights | Propagate into meta-model |
| `nvda_modelling_dataset.parquet` | 195 × 20: 15 features + label + weight + t1 + 2 aux | Primary model training substrate |
| `models/best_params.json` | Tuned RF & XGB hyperparameters | Primary model configuration |
| `data/processed/tuning_log.parquet` | 50 tuning trials | num_trials for DSR computation |

### 2.2 Artifacts That MUST NOT Be Reused Directly

| Artifact | Reason | Correct Alternative |
|----------|--------|-------------------|
| `models/model_final.pkl` | Fit on ALL 195 samples — in-sample | Retrain via PurgedKFold for OOS predictions |
| `models/model_rf.pkl` | Full-data fit — in-sample | Same |
| `models/model_xgb.pkl` | Full-data fit — in-sample | Same |

### 2.3 Pipeline Parameters (Inferred)

```
n_samples        = 195
n_features       = 15 (after Stage 6 pruning)
label_classes    = {-1, +1} (binary, no label=0)
majority_baseline = 0.5846 (+1 class)
purged_cv_splits = 5
embargo_pct      = 0.01
best_model       = Tuned RF (DSR=0.935 > XGB DSR=0.659)
best_rf_params   = {n_estimators: 100, max_depth: 7, min_samples_leaf: 20, max_features: 'sqrt'}
vertical_barrier = 10 trading days
pt_sl            = [1.0, 1.0]
vol_span         = 50
```

---

## 3. Stage 7 Architecture — Meta-Labeling & Bet Sizing

### 3.1 Purpose of Meta-Labeling (AFML Ch. 3.6–3.8)

Meta-labeling decouples two decisions that standard ML conflates:

1. **Side** — should we go long or short? (primary model)
2. **Size** — how much should we bet? Including zero. (meta-model)

The primary model determines the direction of each trade. The meta-model determines whether to take the trade and how large to make it. The meta-label is binary: 1 if the trade (taken in the predicted direction) would have been profitable, 0 otherwise.

This separation has four critical advantages per AFML §3.7: (a) allows white-box primary models with ML overlay, (b) limits overfitting since ML only learns size not side, (c) enables asymmetric long/short strategies, (d) focuses ML on the sizing decision which is where most money is lost.

### 3.2 The Three Models — Precise Definitions

**Primary Model:** The tuned Random Forest from Stage 5/6. It predicts label ∈ {-1, +1} for each event. Its predicted class determines `side`.

**Side Prediction:** `side_i = sign(primary_model_OOS_prediction_i)`. This is the direction in which we would trade event i. For probabilistic models: `side_i = sign(2 * P(y=+1) - 1)`.

**Meta-Label:** For each event i with side prediction `side_i` and realized return `ret_i` (from the triple-barrier outcome):
```
meta_label_i = 1   if ret_i × side_i > 0   (trade was profitable)
meta_label_i = 0   if ret_i × side_i ≤ 0   (trade was unprofitable or break-even)
```

This matches AFML Snippet 3.7's `getBins` with side: `out['ret'] *= events_['side']` then `out.loc[out['ret']<=0,'bin'] = 0`.

**Meta-Model:** A classifier trained on (features, side) → meta_label ∈ {0, 1}. Its predicted probability P(meta_label=1) becomes the bet size.

### 3.3 Why OOS Predictions Are Mandatory

If the primary model's predictions are generated in-sample (i.e., the model predicts on data it was trained on), two fatal problems occur:

1. **Inflated side accuracy.** The primary model will appear to predict sides correctly far more often than it actually can, because it has memorized the training data. The meta-labels will be dominated by 1s (profitable trades), making the meta-model's task trivially easy but meaningless.

2. **Leakage into meta-model.** If the primary model's overfit predictions feed into the meta-model's training, the meta-model learns to trust signals that are artifacts of overfitting, not genuine patterns. All downstream positions and backtest results are contaminated.

**Correct approach:** Generate OOS predictions via PurgedKFold. For each of K folds, train the primary model on the training set (with purging and embargo), predict on the test set. Concatenate all test-set predictions. Every sample receives exactly one prediction from a model that never saw it.

---

## 4. OOS Prediction Generation — Detailed Design

### 4.1 Algorithm

```
Input:  X (195 × 15), y (195 × 1), w (195 × 1), t1 (195 × 1), rf_params
Output: oos_side (195 × 1), oos_prob (195 × 1)

1. Instantiate PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
2. Initialize empty arrays: oos_pred_class, oos_pred_prob, indexed by event time
3. For each (train_idx, test_idx) in PurgedKFold.split(X):
   a. X_train, y_train, w_train = X[train_idx], y[train_idx], w[train_idx]
   b. X_test = X[test_idx]
   c. clf = RandomForestClassifier(**rf_params, random_state=42)
   d. clf.fit(X_train, y_train, sample_weight=w_train)
   e. pred_class = clf.predict(X_test)
   f. pred_prob = clf.predict_proba(X_test)  # shape: (n_test, 2)
   g. Store pred_class and pred_prob[:, 1] (P(y=+1)) for test indices
4. oos_side = sign(2 * oos_prob_positive - 1)   # or equivalently, the predicted class
5. Return oos_side, oos_prob_positive
```

### 4.2 Why Not cross_val_predict

sklearn's `cross_val_predict` does NOT support PurgedKFold natively. It would use standard KFold internally, breaking purging/embargo. We must implement the loop manually using our custom PurgedKFold splitter.

### 4.3 Validation Checks for OOS Predictions

1. **Coverage:** Every sample in the modelling dataset must have exactly one OOS prediction. Assert `len(oos_pred) == 195`.
2. **No train-test overlap:** For each fold, verify that no training index appears in the test set and that purging/embargo are applied.
3. **OOS accuracy vs in-sample accuracy:** The OOS accuracy should be ≤ the in-sample accuracy. If OOS > IS, something is wrong.
4. **Side distribution:** Report the distribution of predicted sides (+1 vs -1). If overwhelmingly one-sided (>90%), the primary model may have collapsed to always predicting the majority class.
5. **Comparison to Stage 4 CV:** The OOS accuracy from this procedure should approximately match the purged CV accuracy from Stage 4 (0.628 for tuned RF).

### 4.4 Probability Calibration Decision

**Decision: Do NOT calibrate primary model probabilities at this stage.**

Rationale: (1) Calibration (isotonic or Platt) requires yet another held-out set, which we cannot afford with 195 samples. (2) The meta-model will learn the mapping from primary features to profitability directly, so uncalibrated probabilities are acceptable as input features (we actually won't pass probabilities as features — see §5.3). (3) Calibration introduces additional overfitting risk on small samples.

Exception: If the meta-model uses probabilities for bet sizing (which it does via P(meta=1)), calibration of the meta-model's probabilities is more important. We address this in §6.

---

## 5. Meta-Model Design

### 5.1 Meta-Label Construction

For each sample i:

```python
# From nvda_labels.parquet
ret_i = labels.loc[i, 'return']  # realized return from triple-barrier

# From OOS predictions
side_i = oos_side[i]  # +1 or -1

# Meta-label
if ret_i * side_i > 0:
    meta_label_i = 1  # profitable trade
else:
    meta_label_i = 0  # unprofitable trade (includes break-even)
```

**Edge case — side_i disagrees with original label:** This is expected and correct. The primary model gets some predictions wrong. When the model predicts +1 but the true label is -1, the trade would lose money, and meta_label = 0. The meta-model's job is to learn when the primary model is likely wrong and filter those trades out.

### 5.2 Expected Meta-Label Distribution

Given the primary model's OOS accuracy of ~0.628:
- ~62.8% of trades will be correctly sided → meta_label = 1
- ~37.2% of trades will be incorrectly sided → meta_label = 0

The meta-labeling problem is less imbalanced than the original labeling problem, which is favorable. The meta-model's baseline is ~0.628 (always predict meta=1).

### 5.3 Meta-Model Feature Set

The meta-model receives the following features:

```
Feature group 1: All 15 original features from Stage 6
Feature group 2: side (the primary model's predicted direction, +1 or -1)
```

**Total meta-model features: 16**

**What NOT to include as a feature:**
- Primary model probability P(y=+1): This would leak the primary model's confidence. While not technically label leakage, it creates a tight coupling that makes the meta-model a simple threshold on the primary probability, undermining the purpose of having a separate model. AFML does not include primary probability as a meta-feature.
- The original label (y): This is the target of the primary model, not a feature.
- The return (ret): This is future information used to construct the meta-label, not available at prediction time.

### 5.4 Meta-Model Classifier Choice

**Primary choice: Random Forest (small configuration)**

Rationale: (1) Consistent with AFML's preference for ensemble methods. (2) Handles non-linear interactions. (3) Provides predicted probabilities for bet sizing. (4) With 195 samples and 16 features, we need a constrained model.

**Configuration:**
```python
meta_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,       # SHALLOW — to prevent overfitting on 195 samples
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced',  # Handle meta-label imbalance
    random_state=42
)
```

**Why max_depth=3:** With 195 samples and a meta-label distribution of ~63/37, a deep tree can easily memorize the training data. Depth 3 allows at most 8 leaf nodes, which is appropriate for this sample size.

**Alternative: Logistic Regression**

If RF meta-model overfits (OOS accuracy < baseline), fall back to logistic regression:
```python
from sklearn.linear_model import LogisticRegression
meta_clf = LogisticRegression(
    C=0.1,  # Strong regularization
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
```

### 5.5 Meta-Model Training Protocol

```
1. Construct meta-feature matrix:
   X_meta = original_features ∪ {side}   (195 × 16)
   y_meta = meta_labels                   (195 × 1, values in {0, 1})
   w_meta = sample_weights                (195 × 1, from Stage 2)
   t1_meta = barrier_end_times            (195 × 1, from Stage 2)

2. Evaluate with PurgedKFold:
   - Use F1-score as primary metric (per AFML Snippet 9.1: "if set(lbl.values)=={0,1}: scoring='f1'")
   - Also report: accuracy, precision, recall, log-loss
   - Use weighted scoring (consistent with Stage 4–6 protocol)

3. Report metrics:
   - Mean F1 ± std across 5 folds
   - Precision (ability to avoid false positives — betting when we shouldn't)
   - Recall (ability to capture true positives — not missing good trades)
   - Compare to meta-label baseline (~0.628 always predict 1)

4. Generate OOS meta-probabilities:
   - Same PurgedKFold loop as §4.1
   - Store P(meta_label=1) for each sample OOS
   - These probabilities become the bet sizes
```

### 5.6 Scoring Metric — Why F1

AFML Snippet 9.1 explicitly specifies: for meta-labeling where labels ∈ {0,1}, use `scoring='f1'`. This is because the meta-model's purpose is to filter false positives from the primary model. F1 captures the trade-off between precision (not taking bad trades) and recall (not missing good trades). Accuracy alone would be misleading because always predicting 1 gives ~0.628 accuracy.

---

## 6. Bet Sizing Architecture

### 6.1 Probability to Signal (AFML Snippet 10.1)

For binary classification with meta-labeling:

```python
# p = P(meta_label=1) from OOS meta-model predictions
# For binary case (num_classes=2):
z = (p - 1/num_classes) / (p * (1 - p))**0.5    # t-statistic
signal0 = side * (2 * norm.cdf(z) - 1)           # signal = side × size
```

Where:
- `p` is the meta-model's predicted probability of a profitable trade
- `z` is the test statistic testing H0: p = 0.5 (no skill)
- `norm.cdf(z)` maps z to [0.5, 1] for positive z, giving bet size in [0, 1]
- `side` gives direction, `(2*Φ(z)-1)` gives magnitude
- Final signal ∈ [-1, 1]

**When p = 0.5:** z = 0, size = 0 → no trade (correct: no edge)
**When p = 1.0:** z → ∞, size → 1 → full position (correct: maximum confidence)
**When p = 0.0:** z → -∞, size → -1 → position against predicted side (degenerate, should not happen if meta-model is reasonable)

### 6.2 Averaging Active Signals (AFML Snippet 10.2)

Multiple events may be active simultaneously (overlapping holding periods). At each point in time t, the portfolio position is the average of all signals still active:

```python
def avg_active_signals(signals, t1):
    """
    signals: pd.Series indexed by event start time, values = signal
    t1: pd.Series indexed by event start time, values = event end time
    
    At time t, position = mean of all signals where:
      - signal was issued at or before t (signal.index <= t)
      - signal has not expired (t1[signal.index] >= t or t1 is NaT)
    """
    # Collect all time points where signals change
    t_points = sorted(set(signals.index.tolist() + t1.dropna().tolist()))
    
    positions = pd.Series(index=t_points, dtype=float)
    for t in t_points:
        # Active signals at time t
        active = signals[(signals.index <= t) & 
                        ((t1 >= t) | t1.isna())]
        if len(active) > 0:
            positions[t] = active.mean()
        else:
            positions[t] = 0.0
    return positions
```

### 6.3 Signal Discretization (AFML Snippet 10.3)

```python
def discrete_signal(signal, step_size=0.1):
    """Discretize signal to prevent overtrading."""
    signal = (signal / step_size).round() * step_size
    signal = signal.clip(-1, 1)
    return signal
```

Step size recommendation: `step_size = 0.2` for 195 samples (5 possible position sizes per side: 0, ±0.2, ±0.4, ±0.6, ±0.8, ±1.0). Finer discretization would cause excessive turnover relative to the sparse event-driven signal.

### 6.4 Position Clipping and Constraints

```
- Maximum position: signal ∈ [-1.0, +1.0]
- No leverage: |position| ≤ 1.0
- Long-short allowed: NVDA is a single equity — both long and short are valid
- No-trade zone: if |signal| < step_size/2, position = 0
```

### 6.5 Complete Bet Sizing Pipeline

```
Input: oos_meta_prob (195 × 1), oos_side (195 × 1), t1 (195 × 1)
Output: nvda_positions.parquet

1. Compute signal = side × size from meta-probabilities (§6.1)
2. Average active signals across overlapping events (§6.2)
3. Discretize to step_size=0.2 (§6.3)
4. Clip to [-1, 1] (§6.4)
5. Expand to daily position series (forward-fill from event to next event or t1)
6. Save as nvda_positions.parquet with columns [date, position, raw_signal, meta_prob, side]
```

### 6.6 Daily Position Series Construction

Events are sparse (195 over ~5100 trading days). Between events, the position is held constant (the average of active signals). When no signals are active, position = 0.

```python
# Expand event-level positions to daily frequency
daily_idx = clean_data.index  # full daily index
daily_pos = avg_positions.reindex(daily_idx).ffill().fillna(0)
```

---

## 7. Stage 7 Validation Framework

### 7.1 Required Diagnostics

| Check | Expected Range | Failure Condition |
|-------|---------------|-------------------|
| OOS prediction coverage | exactly 195 | ≠ 195 |
| OOS primary accuracy | 0.55–0.70 | < 0.50 (worse than random) |
| Meta-label distribution | 55–70% class=1 | > 90% or < 40% one class |
| Meta-model OOS F1 | > 0.50 | < 0.40 |
| Meta-model OOS precision | > 0.55 | < 0.50 |
| Signal distribution | mean near 0, spread ≥ 0.2 | all same sign, or all zero |
| Position time series | non-trivial, changes over time | constant position |
| Active signal count | 1–5 at typical time | > 10 (excessive overlap) |

### 7.2 Required Plots (Stage 7)

| Plot | Description |
|------|-------------|
| P_meta_1 | OOS primary model: predicted side vs actual label (confusion matrix) |
| P_meta_2 | Meta-label distribution (bar chart: 0 vs 1) |
| P_meta_3 | Meta-model OOS F1/precision/recall per fold (bar chart) |
| P_meta_4 | Meta-model predicted probability distribution (histogram) |
| P_meta_5 | Signal distribution before and after discretization |
| P18 | Position time series overlaid on NVDA price chart |
| P_meta_6 | Bet size distribution histogram |
| P_meta_7 | Number of active signals over time |

### 7.3 Leakage Checks

1. **Temporal ordering:** For each OOS prediction, verify that the training data for that fold does NOT include any sample whose label period overlaps the test sample's feature observation time.
2. **No future returns in features:** Verify that `side` is computed from OOS predictions only, not from realized returns.
3. **Weight propagation:** Verify that sample weights from Stage 2 are passed to both fit() and score() in the meta-model.
4. **t1 propagation:** Verify that t1 (barrier end times) from Stage 2 are used in PurgedKFold for the meta-model.

### 7.4 What Would Invalidate Stage 7

- Meta-model OOS F1 < 0.40 (model has no skill at filtering)
- All OOS predictions are the same class (primary model collapsed)
- Position series is constant (no signal variation)
- OOS primary accuracy < 0.50 (primary model worse than random)
- Any detected leakage (see §7.3)

If invalidated: document honestly, proceed to Stage 8 backtesting anyway to quantify performance (it may still be instructive), but flag the result as "meta-labeling did not improve over naive strategy."

---

## 8. Stage 8 — Backtesting Architecture

### 8.1 Why Naive Backtesting Is Dangerous (AFML Ch. 11)

AFML's "Second Law of Backtesting" (Snippet 11.1): backtesting is not a research tool. The pipeline was designed via feature importance (Chapter 8), not backtesting. Stage 8 evaluates a fully specified strategy; no parameter changes are permitted based on backtest results.

Critical pitfalls to avoid:
- **Look-ahead bias:** Positions must be derived from OOS predictions that use only past data.
- **Survivorship bias:** NVDA is a single stock that survived to present (inherent limitation — acknowledge in report).
- **Selection bias:** The backtest is run on the one strategy that was developed; DSR corrects for the number of trials.
- **Unrealistic execution:** Include transaction costs; do not assume execution at close.

### 8.2 Backtest Engine Design

#### 8.2.1 Return Computation

```python
def backtest_strategy(positions, prices, cost_bps=5):
    """
    positions: pd.Series of daily positions in [-1, 1]
    prices: pd.Series of daily Adj Close prices
    cost_bps: transaction cost in basis points per trade
    
    Returns: pd.DataFrame with daily strategy returns
    """
    # Price returns
    price_ret = prices.pct_change()
    
    # Strategy returns (position from previous day × today's return)
    strategy_ret = positions.shift(1) * price_ret
    
    # Transaction costs: proportional to |change in position|
    turnover = positions.diff().abs()
    costs = turnover * cost_bps / 10000
    
    # Net return
    net_ret = strategy_ret - costs
    
    # Build output
    result = pd.DataFrame({
        'position': positions,
        'price_return': price_ret,
        'gross_return': strategy_ret,
        'cost': costs,
        'net_return': net_ret,
        'cumulative': (1 + net_ret).cumprod()
    })
    return result
```

**Key design decisions:**
- `positions.shift(1)`: We use the position at end of day t-1 to compute the return from day t-1 to day t. This prevents look-ahead.
- Transaction costs are deducted from the gross return on the day the position changes.
- No slippage model beyond the fixed bps cost (reasonable for daily OHLCV of a liquid large-cap stock like NVDA).

#### 8.2.2 Chronological Execution Guarantee

The position on day t is determined by:
1. Whether any event was triggered on or before day t
2. The OOS meta-model prediction for that event (generated from data available at event time)
3. Average of all active signals at day t

Because OOS predictions are generated via PurgedKFold (each sample predicted by a model trained only on data that does not overlap), and positions are applied with a one-day lag (`shift(1)`), there is no look-ahead.

### 8.3 Transaction Cost Model

#### 8.3.1 Fixed BPS Model (Primary)

```
cost_bps = 5   (5 basis points per unit of turnover)
```

Justification: NVDA is a highly liquid large-cap stock. Bid-ask spreads are typically 1-2 bps, and commission + market impact adds 2-3 bps. 5 bps total is conservative.

#### 8.3.2 Feature-Informed Cost Model (Optional Enhancement)

Use the Corwin-Schultz spread from the feature set as a time-varying cost estimate:

```python
spread_cost = cs_spread_daily.reindex(positions.index).ffill()
variable_cost = turnover * spread_cost / 2  # half-spread per side
```

This is informational only — the primary backtest should use fixed 5 bps for comparability.

### 8.4 Backtest Metrics — Complete Implementation

#### 8.4.1 Sharpe Ratio (AFML §14.7.1)

```python
def sharpe_ratio(returns, periods_per_year=252):
    """Annualized Sharpe ratio."""
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    return (mu / sigma) * np.sqrt(periods_per_year)
```

#### 8.4.2 Probabilistic Sharpe Ratio (AFML §14.7.2)

```python
def prob_sharpe_ratio(returns, sr_benchmark=0.0):
    """
    PSR: probability that true SR exceeds sr_benchmark.
    
    PSR[SR*] = Z[ (SR_hat - SR*) * sqrt(T-1) / 
                   sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) ]
    """
    sr = returns.mean() / returns.std()  # non-annualized
    T = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3  # scipy kurtosis is excess; AFML uses regular
    
    numerator = (sr - sr_benchmark) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    
    if denominator <= 0:
        return np.nan
    
    z = numerator / denominator
    return norm.cdf(z)
```

#### 8.4.3 Deflated Sharpe Ratio (AFML §14.7.3)

```python
def deflated_sharpe_ratio(returns, num_trials, var_sr_trials=None):
    """
    DSR: PSR where SR* is adjusted for multiple testing.
    
    SR* = sqrt(V[SR]) * ((1 - gamma) * Z^{-1}(1 - 1/N) + gamma * Z^{-1}(1 - 1/(N*e)))
    where gamma ≈ 0.5772 (Euler-Mascheroni constant)
    """
    euler_mascheroni = 0.5772156649
    
    if var_sr_trials is None:
        # Estimate from returns' higher moments
        sr = returns.mean() / returns.std()
        T = len(returns)
        var_sr = (1 - returns.skew() * sr + 
                  (returns.kurtosis() + 3 - 1) / 4 * sr**2) / (T - 1)
        var_sr_trials = var_sr
    
    N = max(num_trials, 2)
    
    sr_star = np.sqrt(var_sr_trials) * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1/N) +
        euler_mascheroni * norm.ppf(1 - 1/(N * np.e))
    )
    
    return prob_sharpe_ratio(returns, sr_benchmark=sr_star)
```

**Critical distinction from Stage 5:**
- Stage 5 "DSR" was computed from CV fold accuracy scores, not strategy returns. That was a CV-trial multiple-testing correction.
- Stage 8 DSR is the true Deflated Sharpe Ratio, computed from realized strategy returns, with num_trials = total number of model configurations tried (≥50 from tuning log, plus any additional Stage 7 configurations).

#### 8.4.4 Drawdown and Time Under Water (AFML Snippet 14.4)

```python
def compute_dd_tuw(returns):
    """
    Compute drawdown series and time-under-water series.
    Returns (dd_series, tuw_series, max_dd, max_tuw)
    """
    cum_returns = (1 + returns).cumprod()
    hwm = cum_returns.expanding().max()
    dd = 1 - cum_returns / hwm  # drawdown as fraction of HWM
    
    max_dd = dd.max()
    
    # Time under water: duration of each drawdown episode
    in_dd = dd > 0
    dd_start = in_dd & ~in_dd.shift(1, fill_value=False)
    dd_end = ~in_dd & in_dd.shift(1, fill_value=False)
    
    starts = dd_start[dd_start].index
    ends = dd_end[dd_end].index
    
    tuw_days = []
    for s in starts:
        matching_ends = ends[ends > s]
        if len(matching_ends) > 0:
            tuw_days.append((matching_ends[0] - s).days)
    
    max_tuw = max(tuw_days) if tuw_days else 0
    
    return dd, max_dd, max_tuw
```

#### 8.4.5 Additional Metrics

```python
def calmar_ratio(returns, periods_per_year=252):
    """Annualized return / max drawdown."""
    ann_return = (1 + returns.mean())**periods_per_year - 1
    _, max_dd, _ = compute_dd_tuw(returns)
    if max_dd == 0:
        return np.inf
    return ann_return / max_dd

def hit_ratio(returns):
    """Fraction of positive returns."""
    return (returns > 0).mean()

def profit_factor(returns):
    """Gross profits / gross losses."""
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return np.inf
    return abs(gains / losses)
```

### 8.5 Summary Statistics Table (T11)

```
T11: Backtest Statistics
─────────────────────────────────────────
  Metric                        Value
─────────────────────────────────────────
  Start Date                    yyyy-mm-dd
  End Date                      yyyy-mm-dd
  Total Trading Days            N
  Events (Bets)                 195
  Annualized Return             X.XX%
  Annualized Volatility         X.XX%
  Sharpe Ratio                  X.XX
  PSR (SR* = 0)                 X.XX
  DSR (N_trials = XX)           X.XX
  Max Drawdown                  X.XX%
  Time Under Water (days)       N
  Calmar Ratio                  X.XX
  Hit Ratio                     X.XX%
  Profit Factor                 X.XX
  Avg Turnover (daily)          X.XX
  Avg Exposure                  X.XX
  Transaction Costs (total)     X.XX%
  Correlation to Underlying     X.XX
─────────────────────────────────────────
```

---

## 9. CPCV / Robustness Framework

### 9.1 Combinatorial Purged Cross-Validation (AFML Ch. 12)

CPCV generates multiple backtest paths from a single dataset by choosing test groups of size p from K groups, producing C(K,p) unique train-test combinations. Each combination produces an OOS path segment; these segments are concatenated to form complete backtest paths.

**Configuration for 195 samples:**
```
K = 6 groups (32-33 samples each)
p = 2 test groups per split
C(6,2) = 15 unique splits
Number of complete backtest paths = C(K-1, p-1) = C(5,1) = 5 paths
```

**Implementation:**
```python
class CombinatorialPurgedKFold:
    def __init__(self, n_splits=6, n_test_splits=2, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
    
    def split(self, X, y=None):
        """Generate C(n_splits, n_test_splits) train-test combinations."""
        from itertools import combinations
        
        indices = np.arange(len(X))
        groups = np.array_split(indices, self.n_splits)
        
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups[i] for i in test_combo])
            train_idx = np.setdiff1d(indices, test_idx)
            
            # Apply purging and embargo
            train_idx = self._purge_embargo(train_idx, test_idx, X)
            
            yield train_idx, test_idx
```

Each of the 15 splits produces OOS predictions for its 2 test groups (~65 samples). The 5 backtest paths are assembled by selecting the appropriate test predictions for each chronological segment.

### 9.2 CPCV Backtest Path Distribution

For each backtest path:
1. Concatenate OOS predictions in chronological order
2. Derive positions via bet sizing pipeline
3. Compute Sharpe ratio

Report: distribution of Sharpe ratios across the 5 paths (mean, std, min, max). If the mean Sharpe is positive and the majority of paths are positive, the strategy shows robustness.

### 9.3 Synthetic Data Validation (AFML Ch. 13)

Generate two synthetic price series and run the full pipeline on each:

**Trending series:**
```python
def generate_trending(n=5000, drift=0.0005, vol=0.02, seed=42):
    np.random.seed(seed)
    returns = drift + vol * np.random.randn(n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.bdate_range(end='2025-01-01', periods=n))
```

**Mean-reverting series (O-U process):**
```python
def generate_mean_reverting(n=5000, theta=0.1, mu=100, vol=2, seed=42):
    np.random.seed(seed)
    prices = [mu]
    for i in range(1, n):
        dp = theta * (mu - prices[-1]) + vol * np.random.randn()
        prices.append(prices[-1] + dp)
    return pd.Series(prices, index=pd.bdate_range(end='2025-01-01', periods=n))
```

**Expectation:** The pipeline should detect the planted signal in the trending series (positive Sharpe) and the mean-reverting series (positive Sharpe if barriers are calibrated correctly). If it fails on both, the pipeline lacks statistical power.

---

## 10. Leakage & Statistical Risk Prevention

### 10.1 Complete Risk Registry

| Risk ID | Description | Severity | Prevention |
|---------|------------|----------|------------|
| L1 | model_final.pkl used for OOS predictions | CRITICAL | Retrain via PurgedKFold loop |
| L2 | Meta-model trained on in-sample meta-labels | CRITICAL | Generate meta-labels from OOS primary predictions |
| L3 | Backtest uses in-sample positions | CRITICAL | All positions from OOS predictions only |
| L4 | t1 not propagated to meta-model PurgedKFold | HIGH | Pass t1 explicitly to all CV splitters |
| L5 | Sample weights dropped in meta-model | HIGH | Pass w to fit() and score() |
| L6 | Position applied on same day as signal (no lag) | HIGH | Use positions.shift(1) in backtest |
| L7 | Transaction costs omitted | MEDIUM | Include 5 bps per unit turnover |
| L8 | DSR computed with wrong num_trials | MEDIUM | Count all tuning trials + meta trials |
| L9 | CPCV paths constructed with leakage | MEDIUM | Apply purging/embargo within each CPCV split |
| L10 | Backtesting used to modify strategy parameters | MEDIUM | No parameter changes after Stage 7 |
| L11 | Returns computed on wrong price series | LOW | Use Adj Close from nvda_clean.parquet |
| L12 | Survivorship bias in NVDA selection | LOW | Acknowledge in report (cannot fix with single stock) |

### 10.2 Automated Leakage Verification

The implementation must include a `verify_no_leakage()` function that:

1. For each OOS prediction, confirms the training set does not contain overlapping samples.
2. For each position, confirms it is derived from predictions available at or before the position date.
3. Confirms that the backtest return computation uses `shift(1)` on positions.
4. Confirms transaction costs are non-zero.

---

## 11. File / Notebook / Artifact Structure

### 11.1 New Source Modules

```
src/
├── meta_labeling.py          # Meta-label generation, OOS prediction generation
├── bet_sizing.py             # Signal generation, averaging, discretization (update existing)
├── backtesting.py            # Backtest engine, return computation (update existing)
├── backtest_stats.py         # SR, PSR, DSR, DD, TuW, Calmar, etc.
├── cpcv.py                   # CombinatorialPurgedKFold implementation
├── synthetic.py              # Synthetic data generation (update existing)
└── validation.py             # Leakage checks, diagnostic functions
```

### 11.2 New Notebooks

```
notebooks/
├── 10_meta_labeling_bet_sizing.ipynb    # Stage 7: complete meta-labeling + bet sizing
└── 11_backtesting.ipynb                 # Stage 8: backtest + statistics + CPCV + synthetic
```

### 11.3 New Data Artifacts

```
data/processed/
├── nvda_oos_predictions.parquet    # OOS primary model predictions (side, prob)
├── nvda_meta_labels.parquet        # Meta-labels + meta-features
├── nvda_meta_predictions.parquet   # OOS meta-model predictions (meta_prob)
├── nvda_positions.parquet          # Daily position series
├── backtest_results.parquet        # Daily backtest returns and metrics
└── cpcv_results.parquet            # CPCV path statistics

models/
├── model_meta.pkl                  # Meta-model (for reference only)
└── cpcv_models/                    # CPCV fold models (optional)

reports/figures/
├── P18_position_series.png
├── P19_equity_curve.png
├── P20_cpcv_sharpe_distribution.png
├── P21_synthetic_equity.png
├── P_meta_*.png                    # Stage 7 diagnostic plots
```

---

## 12. Claude Code Execution Roadmap

### Phase A — Pre-Flight Checks (30 min)

```
A1. Load nvda_modelling_dataset.parquet
A2. Verify: 195 rows, 20 columns, no NaN
A3. Verify: t1 column exists and all values are valid timestamps
A4. Verify: weight column exists, all positive, sums to ~195
A5. Load best_params.json, verify RF parameters
A6. Load tuning_log.parquet, count total trials → num_trials_tuning
A7. Rerun PurgedKFold on current dataset to confirm fold sizes
```

### Phase B — OOS Primary Predictions (1 hour)

```
B1. Implement generate_oos_predictions() in src/meta_labeling.py
B2. Run PurgedKFold OOS prediction loop with tuned RF
B3. Assert: 195 OOS predictions, coverage = 100%
B4. Compute OOS accuracy, compare to Stage 4 CV accuracy
B5. Save nvda_oos_predictions.parquet
B6. Print side distribution (+1 vs -1)
```

### Phase C — Meta-Label Generation (30 min)

```
C1. Load OOS predictions + nvda_labels.parquet
C2. Compute meta_label = 1 if ret × side > 0 else 0
C3. Print meta-label distribution
C4. Construct meta-feature matrix (15 features + side)
C5. Save nvda_meta_labels.parquet
```

### Phase D — Meta-Model Training & Evaluation (1 hour)

```
D1. Train meta-model RF with PurgedKFold
D2. Compute F1, precision, recall, accuracy per fold
D3. Generate OOS meta-probabilities
D4. Save nvda_meta_predictions.parquet
D5. Generate diagnostic plots (P_meta_1 through P_meta_4)
```

### Phase E — Bet Sizing (30 min)

```
E1. Compute signal from meta-probabilities (§6.1)
E2. Average active signals (§6.2)
E3. Discretize signals (§6.3)
E4. Expand to daily position series
E5. Save nvda_positions.parquet
E6. Generate P18 (position on price chart), P_meta_5-7
```

### Phase F — Backtesting (1 hour)

```
F1. Load positions + nvda_clean.parquet
F2. Run backtest with cost_bps=5
F3. Compute all metrics: SR, PSR, DSR, MaxDD, TuW, Calmar, etc.
F4. Generate T11 (statistics table)
F5. Generate P19 (equity curve with drawdown)
F6. Save backtest_results.parquet
```

### Phase G — CPCV Robustness (1 hour)

```
G1. Implement CombinatorialPurgedKFold
G2. Run CPCV with K=6, p=2
G3. For each backtest path: compute Sharpe
G4. Generate P20 (Sharpe distribution across paths)
G5. Save cpcv_results.parquet
```

### Phase H — Synthetic Validation (30 min)

```
H1. Generate trending and mean-reverting series
H2. Run full pipeline on each (CUSUM → label → model → meta → backtest)
H3. Generate P21 (synthetic equity curves)
H4. Report Sharpe on synthetic data
```

### Phase I — Final Validation & Report (30 min)

```
I1. Run verify_no_leakage()
I2. Generate all remaining report figures
I3. Compile summary table T11
I4. Write Stage 7-8 section of the report
I5. Git commit all artifacts
```

---

## 13. Final Recommended Workflow

### 13.1 Execution Order

```
1. Phase A (pre-flight)           — must pass before proceeding
2. Phase B (OOS predictions)      — gate: OOS accuracy > 0.50
3. Phase C (meta-labels)          — gate: meta-label dist. not degenerate
4. Phase D (meta-model)           — gate: F1 > 0.40
5. Phase E (bet sizing)           — gate: positions non-trivial
6. Phase F (backtesting)          — informational, no gate
7. Phase G (CPCV)                 — informational, no gate
8. Phase H (synthetic)            — informational, no gate
9. Phase I (validation + report)  — final quality check
```

### 13.2 Regeneration Rules

If any Stage 7 parameter changes (e.g., meta-model hyperparameters), ALL of Phases D–I must be rerun. If OOS predictions change (Phase B), ALL of Phases C–I must be rerun.

### 13.3 Git Workflow

```
git checkout -b stage-7-8
# After Phase B:
git add -A && git commit -m "Stage 7: generate OOS primary predictions"
# After Phase E:
git add -A && git commit -m "Stage 7: meta-labeling and bet sizing"
# After Phase F:
git add -A && git commit -m "Stage 8: backtesting with full statistics"
# After Phase H:
git add -A && git commit -m "Stage 8: CPCV robustness and synthetic validation"
# After Phase I:
git add -A && git commit -m "Stages 7-8: final validation and report artifacts"
```

### 13.4 What Success Looks Like

A successful Stage 7-8 implementation produces:
1. Honest, leakage-free backtest results — even if the Sharpe ratio is negative
2. Properly computed DSR that accounts for all trials
3. CPCV robustness analysis showing the distribution of outcomes
4. Synthetic data validation confirming the pipeline can detect signal when present
5. Clear separation between "what the model predicts" and "what we wish it predicted"

The pipeline is designed to produce truthful results, not impressive ones. If NVDA daily OHLCV with 195 samples yields a negative Sharpe after transaction costs, that is itself a valuable finding aligned with AFML's philosophy that most strategies fail under proper validation.

---

*End of Implementation Plan*
