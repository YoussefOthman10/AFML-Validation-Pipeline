# AFML Implementation Plan — NVDA Dataset

## Complete Pipeline from *Advances in Financial Machine Learning* (López de Prado)

---

## 0 — Dataset Summary

| Property | Value |
|---|---|
| Asset | NVIDIA (NVDA) |
| Columns | Date, Open, High, Low, Close, Adj Close, Volume |
| Rows | 5 114 |
| Frequency | Daily (trading days only) |
| Date range | 2005-01-03 → 2025-04-30 |
| Missing values | 0 in every column |
| Price range (Adj Close) | $0.135 → $149.38 |
| Volume range | 45.6 M → 3.69 B shares |
| Splits reflected | Yes — Adj Close is split-adjusted |

**Key limitation:** This is **daily OHLCV** data, not tick or order-book data. Many AFML methods are designed for tick-level data. Each chapter section below explicitly states which methods work directly, which require approximation, and which are theoretical only with this dataset.

---

## Assumptions to State in the Report

1. Adj Close is used for all return calculations (accounts for dividends and splits).
2. Volume is in shares and is split-adjusted.
3. Dollar volume is approximated as `Adj Close × Volume` on each day.
4. We use NVDA as a single-asset study; portfolio-level methods are applied to NVDA alone.
5. We do not have tick data, bid-ask spreads, or order-book depth.
6. Transaction costs are not modelled unless explicitly stated.
7. All timestamps are end-of-day; intraday bar methods are approximations.
8. Python 3.10+, scikit-learn, NumPy, pandas, matplotlib, statsmodels, scipy are assumed available.

---

## AFML Methods Not Faithfully Implementable with Daily OHLCV

| Method | Reason | Possible Approximation |
|---|---|---|
| Tick bars, volume bars, dollar bars (Ch 2) | Require tick-level timestamps and prices | Approximate dollar bars by cumulating daily dollar volume and sampling when a threshold is crossed |
| Tick imbalance / run bars (Ch 2) | Need signed tick sequences | Not implementable; explain theory only |
| VPIN (Ch 19) | Requires tick-level volume classification (BVC) | Approximate volume classification from daily close vs open, but accuracy is poor |
| Kyle's Lambda, Amihud's Lambda (Ch 19) | Lambda needs trade-by-trade price impact | Amihud's illiquidity ratio works with daily data; Kyle's Lambda does not |
| Roll Model spread estimate (Ch 19) | Needs serial covariance of price changes at high frequency | Can compute from daily returns but estimate is noisy |
| Hasbrouck information share (Ch 19) | Needs multivariate tick data | Not implementable |
| Exact CUSUM on tick returns (Ch 2) | Designed for intraday returns | Apply to daily returns as a reasonable approximation |
| True information-driven bars (Ch 2) | Need real-time signed order flow | Not implementable |

---

# PART 0 — PREAMBLE

## Chapter 1: Financial Machine Learning as a Distinct Subject

**Concept:** Financial ML differs from general ML because financial data is not IID, labels must be engineered, backtesting is prone to overfitting, and the signal-to-noise ratio is extremely low.

**Why it matters:** Sets the philosophical foundation. Every subsequent chapter addresses a specific pitfall (non-stationarity, label leakage, look-ahead bias, overfitting via multiple testing).

**Implementability:** No code — this is the motivation section of the report.

**Report deliverable:** A 1-page introduction summarising the five unique challenges of financial ML per de Prado: (1) non-IID data, (2) labels are not given, (3) sample weights are unequal, (4) CV must be purged, (5) backtests overfit.

---

# PART 1 — DATA ANALYSIS

---

## Chapter 2: Financial Data Structures

### 2.1 Concept

Standard time bars (daily OHLCV) sample at equal time intervals. De Prado argues that information does not arrive at equal intervals — high-activity periods should be sampled more often. Alternative bars include:

- **Tick bars**: sample every N ticks.
- **Volume bars**: sample every V shares traded.
- **Dollar bars**: sample every $D transacted.

Dollar bars are preferred because they normalise for price changes over time.

### 2.2 Key Equations

**Dollar value of a daily bar:**

```
dv_t = Close_t × Volume_t
```

**Dollar-bar threshold:** Choose threshold Θ so that the cumulated dollar volume triggers a new bar:

```
Σ_{i=t₀}^{t} dv_i ≥ Θ  →  emit bar, reset accumulator
```

A common heuristic: set Θ so that the number of bars ≈ number of calendar days or ≈ 1/50 of total dollar volume.

**CUSUM filter for event sampling (Snippet 2.4):**

The symmetric CUSUM filter detects shifts in the mean of a process:

```
S_t⁺ = max(0, S_{t-1}⁺ + (y_t - E[y_t] - h))
S_t⁻ = min(0, S_{t-1}⁻ + (y_t - E[y_t] + h))
```

An event is triggered when `S_t⁺ > h` or `S_t⁻ < -h`.

For daily data: `y_t = log(Close_t / Close_{t-1})` and `E[y_t]` can be an expanding-window mean. `h` is calibrated so that events fire at a reasonable frequency (e.g., 200–500 events over the full history).

### 2.3 Implementability with NVDA_raw.csv

| Method | Status |
|---|---|
| Time bars | Already available (daily rows) |
| Dollar bars | Approximate: cumulate daily dollar volume, emit bar when threshold crossed. Each bar gets OHLCV aggregated from constituent days. |
| Volume bars | Same approach using share volume |
| Tick bars | NOT implementable — need tick data |
| Imbalance / run bars | NOT implementable — need signed tick flow |
| CUSUM event filter | Fully implementable on daily log returns |

### 2.4 Implementation Tasks

1. Compute daily log returns and daily dollar volume.
2. Implement `get_dollar_bars(df, threshold)` → DataFrame of dollar bars with OHLCV.
3. Implement `cusum_filter(returns, h)` → DatetimeIndex of event timestamps.
4. Calibrate `h` to produce ~300–600 events over the 20-year window.
5. Plot: bar count per year for time bars vs dollar bars; event frequency from CUSUM filter.

### 2.5 Inputs / Outputs

- **Input:** `NVDA_raw.csv`
- **Outputs:** `nvda_dollar_bars.parquet`, `nvda_cusum_events.parquet`

### 2.6 Suggested Modules

```
src/data_structures.py
    get_dollar_bars(df, threshold) → pd.DataFrame
    get_volume_bars(df, threshold) → pd.DataFrame
    cusum_filter(close, h) → pd.DatetimeIndex
```

### 2.7 Plots / Tables

- Histogram: number of dollar bars per year vs number of time bars per year.
- Returns distribution: Jarque-Bera normality test for dollar-bar returns vs daily returns.
- CUSUM events plotted on the price chart (scatter overlay).

### 2.8 Common Mistakes

- Using unadjusted Close instead of Adj Close for dollar volume → spurious jumps at split dates.
- Choosing Θ too small → too many bars → overfitting.
- Applying CUSUM to price levels instead of returns.

### 2.9 Connection to Next Chapter

The CUSUM-filtered events become the **timestamps at which we evaluate labels** (Chapter 3). Dollar bars become the price series on which triple-barrier labels are computed.

---

## Chapter 3: Labeling

### 3.1 Concept

We must engineer labels. Three methods:

1. **Fixed-time horizon** — label based on return over a fixed window. Simple but ignores volatility.
2. **Triple-barrier method** — set an upper profit-taking barrier, a lower stop-loss barrier, and a maximum holding period (vertical barrier). The label is determined by which barrier is hit first.
3. **Meta-labeling** — a secondary model that predicts the *size* of the bet given a primary model's directional signal.

### 3.2 Key Equations

**Fixed-time horizon label:**

```
r_{t,t+h} = (Close_{t+h} / Close_t) - 1
y_t = sign(r_{t,t+h})
```

**Dynamic volatility (used for barrier width):**

Daily volatility estimated via exponentially weighted standard deviation of returns:

```
σ_t = EWMA_std(r, span=S)   (typical S = 50–100 days)
```

**Triple-barrier labeling (Snippet 3.1-3.6):**

Given event time t₀ and volatility σ:
- Upper barrier: `Close_t ≥ Close_{t₀} × (1 + pt × σ_{t₀})`   where pt is a multiplier (e.g., 1.0–2.0)
- Lower barrier: `Close_t ≤ Close_{t₀} × (1 - sl × σ_{t₀})`
- Vertical barrier: t₁ = t₀ + max_holding_period

Label:
```
y = +1 if upper barrier hit first
y = -1 if lower barrier hit first
y =  0 if vertical barrier hit first (or sign of return at expiry)
```

The function returns for each event: `{t₀, t₁, ret, label, barrier_hit}`.

**Meta-labeling:**

Primary model produces a side prediction: `side ∈ {+1, -1}`.
Triple barrier is applied only in the direction of `side`.
Meta-label: `y_meta = 1` if the trade was profitable, `0` otherwise.
The meta-model predicts `P(profit | side)`, and this probability becomes the **bet size**.

### 3.3 Implementability

| Method | Status |
|---|---|
| Fixed-time horizon | Fully implementable |
| Dynamic volatility | Fully implementable with EWMA |
| Triple-barrier | Fully implementable on daily data (barriers in terms of daily bars) |
| Meta-labeling | Fully implementable (requires a primary model from Ch 6) |

### 3.4 Implementation Tasks

1. Implement `get_daily_volatility(close, span=50)` → Series of σ.
2. Implement `get_events(close, timestamps, pt_sl, target, min_ret, num_threads, vertical_barrier_days)` → DataFrame with columns `[t1, trgt, side]`.
3. Implement `apply_triple_barrier(close, events, pt_sl)` → DataFrame with `[ret, label, barrier_type]`.
4. Implement fixed-horizon labeling as a baseline.
5. Examine label distribution (class balance).

### 3.5 Inputs / Outputs

- **Input:** `nvda_dollar_bars.parquet` or daily Adj Close + `nvda_cusum_events.parquet`
- **Output:** `nvda_labels.parquet` with columns `[event_time, barrier_time, return, label, barrier_type, target_vol]`

### 3.6 Suggested Modules

```
src/labeling.py
    get_daily_vol(close, span) → pd.Series
    add_vertical_barrier(close, events, num_days) → pd.Series
    apply_triple_barrier(close, events, pt_sl, molecule) → pd.DataFrame
    get_bins(events, close) → pd.DataFrame   # assign +1/−1/0
    drop_labels(events, min_pct) → pd.DataFrame  # drop rare labels
    meta_labeling(primary_side, events, close) → pd.DataFrame
```

### 3.7 Plots / Tables

- Bar chart of label distribution (+1 / −1 / 0).
- Scatter plot of events on price chart coloured by label.
- Sensitivity analysis: how label balance changes with pt/sl multipliers and holding period.

### 3.8 Common Mistakes

- **Look-ahead bias**: Using future volatility to set barriers. Use only information up to t₀.
- Setting symmetric pt = sl when asset has a drift.
- Not removing events whose vertical barrier extends beyond the dataset end.

### 3.9 Connection to Next

Labels feed into **sample weights** (Ch 4) and eventually into **model training** (Ch 6).

---

## Chapter 4: Sample Weights

### 4.1 Concept

Financial labels overlap in time: the triple-barrier holding period for event t₀ extends to t₁, during which other events may start. Concurrent labels share information, making samples non-independent. We must down-weight concurrent samples and account for return attribution.

### 4.2 Key Equations

**Concurrency (Snippet 4.1):**

For a label spanning [t₀, t₁], define indicator `1_{t}` = 1 if t₀ ≤ t ≤ t₁ for that label.

Number of concurrent labels at time t:

```
c_t = Σ_i 1_{t ∈ [t₀ⁱ, t₁ⁱ]}
```

**Average uniqueness of sample i (Snippet 4.2):**

```
ū_i = (1 / (t₁ⁱ - t₀ⁱ + 1)) × Σ_{t=t₀ⁱ}^{t₁ⁱ} (1 / c_t)
```

A sample with ū_i close to 1 is unique; close to 0 means highly overlapping.

**Sequential bootstrap (Snippet 4.5):**

Draw samples one at a time. At each draw, compute average uniqueness of each candidate given the already-selected set, and sample with probability proportional to uniqueness. This produces a bootstrap that respects the temporal overlap structure.

**Return attribution weight (Snippet 4.10):**

```
w_i = |r_i| / Σ_j |r_j|
```

where r_i is the label return for sample i. This up-weights events with large absolute returns (where the model had more to learn).

Alternatively, combine uniqueness and return-attribution:

```
w_i = ū_i × |r_i|
```

Normalise so weights sum to n.

**Time decay (Snippet 4.11):**

Apply an exponential or piecewise-linear decay so that recent samples carry more weight:

```
d_i = c^{x_i}    where x_i = (t_i - t_min) / (t_max - t_min), c is the decay factor
```

With `c = 1`: uniform. With `c → 0`: only the most recent sample matters. A typical choice: `c` such that the oldest 50 % of samples have half the total weight.

### 4.3 Implementability

Fully implementable with daily data. The concurrency is measured in **bar indices** (days).

### 4.4 Implementation Tasks

1. Build concurrency matrix: for each timestamp t, count how many labels are live.
2. Compute average uniqueness for each sample.
3. Implement sequential bootstrap.
4. Compute return-attribution weights.
5. Implement time-decay function.
6. Combine into final sample weight vector.

### 4.5 Inputs / Outputs

- **Input:** `nvda_labels.parquet` (must contain t₀, t₁ for each event)
- **Output:** `nvda_sample_weights.parquet` with columns `[event_time, uniqueness, ret_weight, time_decay, final_weight]`

### 4.6 Suggested Modules

```
src/sample_weights.py
    mp_num_co_events(close_idx, t1, molecule) → pd.Series  # concurrency count
    mp_sample_tw(t1, num_co_events, molecule) → pd.Series   # average uniqueness
    seq_bootstrap(ind_matrix, s_length) → list               # sequential bootstrap indices
    get_return_attribution(events) → pd.Series
    get_time_decay(tw, c_lf) → pd.Series
    get_sample_weight(events, close, num_threads) → pd.Series
```

### 4.7 Plots / Tables

- Time series of concurrency count.
- Histogram of average uniqueness.
- Comparison: standard bootstrap uniqueness vs sequential bootstrap uniqueness.
- Weight distribution (box plot).

### 4.8 Common Mistakes

- Ignoring concurrency → highly correlated samples → inflated model accuracy.
- Using uniform sample weights in `sklearn` estimators.

### 4.9 Connection to Next

Sample weights are passed to all model fitting (Ch 6, 7, 9) via `sample_weight` parameter.

---

## Chapter 5: Fractionally Differentiated Features

### 5.1 Concept

Financial time series are non-stationary (unit root), so modellers often take first differences `Δy_t = y_t - y_{t-1}`. But first differencing destroys long-memory. Fractional differentiation with order `d ∈ (0,1)` removes just enough memory to achieve stationarity while preserving signal.

### 5.2 Key Equations

**Fractional differencing weights (Snippet 5.1-5.3):**

The binomial series gives weights:

```
w_k = -w_{k-1} × (d - k + 1) / k      for k = 1, 2, ...
w_0 = 1
```

The fractionally differentiated series:

```
X̃_t = Σ_{k=0}^{K} w_k × X_{t-k}
```

In practice, truncate when |w_k| < τ (a threshold like 1e-5).

**Fixed-width window fracdiff (FFD, Snippet 5.4):**

Instead of variable-length weights, use a fixed window length `l*` determined by the smallest window where all |w_k| ≥ τ. This avoids losing the first part of the series.

**Finding minimum d for stationarity:**

1. For d ∈ {0.0, 0.1, 0.2, ..., 1.0}, compute FFD series.
2. Run Augmented Dickey-Fuller (ADF) test on each.
3. Choose the smallest d where ADF p-value < 0.05 (reject unit root).

Correlation between original series and fracdiff series should remain high (> 0.9).

### 5.3 Implementability

Fully implementable. Apply to Adj Close (log prices).

### 5.4 Implementation Tasks

1. Implement `get_weights(d, size, threshold)` → array of weights.
2. Implement `get_weights_ffd(d, threshold)` → fixed-window weights.
3. Implement `frac_diff(series, d, threshold)` → fractionally differentiated series.
4. Implement `frac_diff_ffd(series, d, threshold)` → FFD series.
5. Sweep d ∈ [0, 1] at 0.05 increments, compute ADF p-value and correlation with original.
6. Select optimal d*.
7. Add fracdiff feature to feature set.

### 5.5 Inputs / Outputs

- **Input:** Adj Close series from `NVDA_raw.csv` or dollar-bar close.
- **Output:** `nvda_fracdiff.parquet` with column `fracdiff_close` at optimal d*.

### 5.6 Suggested Modules

```
src/fracdiff.py
    get_weights(d, size, threshold=1e-5) → np.array
    get_weights_ffd(d, threshold=1e-5) → np.array
    frac_diff(series, d, threshold=1e-5) → pd.Series
    frac_diff_ffd(series, d, threshold=1e-5) → pd.Series
    find_min_d(series, d_range, threshold=1e-5) → float
    plot_min_ffd(series) → matplotlib.Figure
```

### 5.7 Plots / Tables

- Plot: d vs ADF statistic (with 1 % / 5 % critical values) and d vs correlation with original.
- Overlay: original series, d = 1 series (returns), d* series.
- Table: d, ADF stat, p-value, correlation, number of weights kept.

### 5.8 Common Mistakes

- Applying fracdiff to returns instead of log prices.
- Not using FFD → losing the first few hundred observations.
- Using too large a threshold τ → inaccurate weights.

### 5.9 Connection to Next

The fracdiff series is a **feature** used in model training (Ch 6). Combined with other engineered features (Stage 6).

---

# PART 2 — MODELLING

---

## Chapter 6: Ensemble Methods

### 6.1 Concept

De Prado recommends ensemble methods — Random Forests and boosted trees — for financial ML because:

- They handle non-linear interactions.
- Bagging (RF) reduces variance; boosting reduces bias.
- Feature importance is built in.

Key insight: RF's bootstrap sampling interacts badly with overlapping financial labels. Use **sample weights** from Ch 4 instead of (or in addition to) standard bagging.

### 6.2 Key Points

- Use `sample_weight` in `RandomForestClassifier.fit()`.
- For RF: set `max_samples` to control the bag size; use sequential bootstrap (Ch 4) if possible.
- Consider BaggingClassifier with a base estimator and custom sampling.
- Boosted trees (XGBoost/LightGBM): pass `sample_weight` in training.

### 6.3 Implementability

Fully implementable.

### 6.4 Implementation Tasks

1. Build feature matrix X: fracdiff close, rolling statistics (mean, std, skew of returns), volume features, momentum indicators, and any other features from Part 4.
2. Use labels from Ch 3 as y.
3. Use sample weights from Ch 4.
4. Train RandomForestClassifier and XGBClassifier.
5. Use purged K-fold CV from Ch 7 for evaluation.

### 6.5 Inputs / Outputs

- **Input:** Feature matrix, labels, sample weights.
- **Output:** Trained model objects, CV accuracy scores.

### 6.6 Suggested Modules

```
src/modelling.py
    build_feature_matrix(dollar_bars, fracdiff, events) → pd.DataFrame
    train_rf(X, y, sample_weight, cv) → dict
    train_xgb(X, y, sample_weight, cv) → dict
```

### 6.7 Plots / Tables

- CV accuracy (mean ± std) for RF and XGBoost.
- Feature importance bar chart (MDI from RF).
- Confusion matrix.

### 6.8 Common Mistakes

- Not passing sample weights → overfit on overlapping samples.
- Using standard train/test split instead of purged CV.

### 6.9 Connection to Next

Models are evaluated via purged K-fold (Ch 7) and tuned via Ch 9.

---

## Chapter 7: Cross-Validation in Finance

### 7.1 Concept

Standard K-fold CV leaks information because:
1. Labels overlap in time (concurrency).
2. Train and test sets may share return information.

Solution: **Purged K-fold CV** removes from the training set any sample whose label period overlaps with the test set. An additional **embargo** period after each test fold removes samples whose information might leak forward.

### 7.2 Key Equations

**Purging logic:**

For test fold with samples spanning [t_test_start, t_test_end]:
- Remove from training any sample i where `[t₀ⁱ, t₁ⁱ] ∩ [t_test_start, t_test_end] ≠ ∅`.

**Embargo:**

Additionally remove from training any sample i where:
```
t₀ⁱ ∈ (t_test_end, t_test_end + embargo_period]
```

Typical embargo: 1 % of the total sample size (in time steps).

**Combinatorial Purged CV (CPCV, Snippet 7.4):**

Instead of standard K-fold, choose test groups of size p from K groups. This produces `C(K, p)` train-test combinations, allowing backtest path generation. Each sample appears in exactly `C(K-1, p-1)` test sets.

### 7.3 Implementability

Fully implementable.

### 7.4 Implementation Tasks

1. Implement `PurgedKFold(n_splits, t1, embargo_pct)` as a custom sklearn splitter.
2. Verify: for each fold, check that no training sample's label period overlaps the test period.
3. Implement CPCV as a more advanced variant.
4. Use `cross_val_score` with the custom splitter.

### 7.5 Inputs / Outputs

- **Input:** Feature matrix, labels, t₁ series (barrier times), embargo fraction.
- **Output:** Array of fold scores; custom CV splitter object.

### 7.6 Suggested Modules

```
src/cross_validation.py
    class PurgedKFold(KFold):
        __init__(n_splits, t1, pct_embargo)
        split(X, y, groups) → generator of (train_idx, test_idx)

    class CombinatorialPurgedKFold:
        __init__(n_splits, n_test_splits, t1, pct_embargo)
        split(X, y) → generator

    cv_score(clf, X, y, sample_weight, scoring, cv) → pd.Series
```

### 7.7 Plots / Tables

- Fold composition diagram (which dates in train vs test per fold).
- CV score per fold (bar chart + mean line).
- Comparison: purged CV accuracy vs naive CV accuracy (to show leakage effect).

### 7.8 Common Mistakes

- Not purging → inflated accuracy from label leakage.
- Not applying embargo → subtle forward leakage.
- Using GroupKFold by year — correct direction but does not account for barrier overlap near boundaries.

### 7.9 Connection to Next

Purged CV is used in feature importance (Ch 8) and hyperparameter tuning (Ch 9).

---

## Chapter 8: Feature Importance

### 8.1 Concept

Three methods to assess feature importance:

1. **MDI (Mean Decrease Impurity):** Measures the total reduction in impurity from splits on each feature across all trees in the forest. Biased toward high-cardinality features. In-sample metric.
2. **MDA (Mean Decrease Accuracy):** Permute each feature in the OOS (test) fold and measure the drop in accuracy. Unbiased but noisy. Uses purged CV.
3. **SFI (Single Feature Importance):** Train the model on each feature individually (with purged CV) and record accuracy. Identifies features that carry information on their own.

### 8.2 Key Equations

**MDI for feature j:**

```
MDI_j = (1/T) × Σ_{t=1}^{T} Σ_{node ∈ tree_t where feature = j} (p_node × ΔImpurity_node)
```

Normalised so that `Σ_j MDI_j = 1`.

**MDA for feature j:**

```
MDA_j = (1/K) × Σ_{k=1}^{K} [score_k - score_k^{(j permuted)}]
```

where score_k is the OOS accuracy of fold k. Reported with standard deviation across folds.

**SFI for feature j:**

```
SFI_j = CV_score(model trained on feature j alone)
```

### 8.3 Implementability

Fully implementable.

### 8.4 Implementation Tasks

1. Compute MDI from fitted RF (`clf.feature_importances_`).
2. Implement MDA using purged CV: for each fold, permute each feature in the test set and record accuracy drop.
3. Implement SFI: loop over features, train single-feature model with purged CV.
4. Compare MDI, MDA, SFI rankings.

### 8.5 Suggested Modules

```
src/feature_importance.py
    feat_imp_MDI(clf, feature_names) → pd.DataFrame
    feat_imp_MDA(clf, X, y, cv, sample_weight, scoring) → pd.DataFrame
    feat_imp_SFI(clf_template, X, y, cv, sample_weight, scoring) → pd.DataFrame
    plot_feature_importance(imp_df, method_name) → Figure
```

### 8.6 Plots / Tables

- Horizontal bar chart: MDI importance (with std across trees).
- Horizontal bar chart: MDA importance (with std across folds).
- Horizontal bar chart: SFI scores.
- Correlation heatmap: MDI rank vs MDA rank vs SFI rank.

### 8.7 Common Mistakes

- Reporting MDI as the sole importance metric → biased.
- Not using purged CV in MDA/SFI → leakage.
- Interpreting high SFI as sufficient for inclusion — features may be redundant.

### 8.8 Connection to Next

Feature importance guides **feature selection** (drop uninformative features before hyperparameter tuning in Ch 9) and is a key report deliverable.

---

## Chapter 9: Hyperparameter Tuning with Cross-Validation

### 9.1 Concept

Grid search / random search using purged CV. De Prado emphasises:

- Always use purged CV as the inner CV.
- Prefer randomised search (covers more of the hyperparameter space).
- Log all experiments to detect overfitting through multiple testing.

### 9.2 Implementation Tasks

1. Define hyperparameter grids for RF and XGBoost.
2. Use `RandomizedSearchCV` with `cv=PurgedKFold(...)`.
3. Record all trial results.
4. Apply **Deflated Sharpe Ratio** logic (Ch 14) to test whether the best CV score is statistically significant given the number of trials.

### 9.3 Suggested Modules

```
src/hyperparameter_tuning.py
    purged_grid_search(clf, X, y, param_grid, cv, sample_weight) → dict
    purged_random_search(clf, X, y, param_dist, cv, n_iter, sample_weight) → dict
    log_trials(results) → pd.DataFrame
```

### 9.4 Plots / Tables

- Best hyperparameters table.
- CV score distribution across trials (histogram).
- Heatmap of score vs two key hyperparameters.

---

# PART 3 — BACKTESTING

---

## Chapter 10: Bet Sizing

### 10.1 Concept

A model's predicted probability should be translated into a position size. The further the probability is from 0.5, the larger the bet.

### 10.2 Key Equations

**From probability to bet size (Snippet 10.1-10.2):**

```
m = 2 × P[ŷ = 1] - 1       (centred probability, range [−1, 1])
bet_size = m × (1 + |m|) / 2   (concave, optional)
```

Or use the CDF of the fitted probability distribution to map to position size:

```
z = (P - 0.5) / σ_P
bet_size = 2 × Φ(z) - 1
```

where Φ is the standard normal CDF and σ_P is estimated from the model's predicted probabilities.

**Average active bets:**

At each time t, the position is the sum of bet sizes for all currently active (non-expired) signals.

### 10.3 Implementability

Fully implementable.

### 10.4 Implementation Tasks

1. Get predicted probabilities from the classifier.
2. Implement `get_signal(prob, num_classes)` → signal in [−1, 1].
3. Implement `avg_active_signals(signals, events)` → position time series.
4. Implement `discrete_signal(signal, step_size)` → discretised position.

### 10.5 Suggested Modules

```
src/bet_sizing.py
    get_signal(events, step_size, prob, pred, num_classes, num_threads) → pd.Series
    avg_active_signals(signals, molecule) → pd.Series
    discrete_signal(signal, step_size) → pd.Series
```

---

## Chapters 11–14: Backtesting Framework

### 11–12 Concept

Chapter 11 catalogues backtesting pitfalls: look-ahead bias, survivorship bias, overfitting via selection, unrealistic fills. Chapter 12 proposes backtesting through CPCV — generating multiple backtest paths from combinatorial purged cross-validation (avoiding the single historical path problem).

### Key Equations — Backtest Statistics (Chapter 14)

**Sharpe Ratio (annualised):**

```
SR = (μ̂ / σ̂) × √252
```

where μ̂ is mean daily return and σ̂ is daily standard deviation.

**Probabilistic Sharpe Ratio (PSR):**

```
PSR(SR*) = Φ( (SR̂ - SR*) × √(n-1) / √(1 - γ₃ × SR̂ + (γ₄ - 1)/4 × SR̂²) )
```

where γ₃ = skewness, γ₄ = kurtosis of returns, SR* = benchmark Sharpe (often 0), n = number of observations.

PSR answers: "What is the probability that the true Sharpe exceeds SR*?"

**Deflated Sharpe Ratio (DSR):**

Adjusts for multiple testing. Given N strategy trials:

```
SR* = √(V[SR̂]) × ((1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N×e)))
```

where γ ≈ 0.5772 (Euler-Mascheroni constant), V[SR̂] is the variance of Sharpe estimates across trials.

Then compute PSR with this SR* as the benchmark.

**Maximum Drawdown:**

```
DD_t = (HWM_t - Price_t) / HWM_t
MaxDD = max_t DD_t
```

**Time Under Water:**

Length of the longest period where cumulative return stays below the previous high-water mark.

### 13 — Backtesting on Synthetic Data

Generate synthetic price paths with known properties (trend, mean-reversion) and test the strategy on them to validate that the pipeline can detect a planted signal.

### Implementation Tasks

1. Implement backtest loop: given position series from bet sizing + price series, compute PnL.
2. Compute Sharpe, PSR, DSR.
3. Compute max drawdown, time under water, Calmar ratio.
4. Implement CPCV-based backtest path generation.
5. Generate synthetic data and run the pipeline on it as a sanity check.

### Suggested Modules

```
src/backtesting.py
    backtest_strategy(positions, prices, cost_bps=0) → pd.DataFrame  # daily PnL
    sharpe_ratio(returns) → float
    prob_sharpe_ratio(returns, sr_benchmark=0) → float
    deflated_sharpe_ratio(returns, num_trials) → float
    max_drawdown(returns) → (float, pd.Timestamp, pd.Timestamp)
    time_under_water(returns) → pd.Timedelta

src/synthetic.py
    generate_trending_series(n, drift, vol) → pd.Series
    generate_mean_reverting_series(n, theta, mu, vol) → pd.Series
```

### Plots / Tables

- Equity curve with drawdown shading.
- Table: SR, PSR, DSR, MaxDD, Calmar, Time Under Water.
- CPCV: distribution of Sharpe across backtest paths.
- Synthetic data: equity curves on 5 synthetic datasets.

---

## Chapter 15: Understanding Strategy Risk

### Concept

Examine concentration of returns, tail risk, and how bets are distributed over time. Key metrics:

- Herfindahl-Hirschman Index (HHI) of returns across time buckets.
- Maximum loss from a single bet.
- Percentage of PnL from top N bets.

### Implementation

Compute HHI, tail-risk metrics, and create a bet-level attribution table.

---

## Chapter 16: Machine Learning Asset Allocation

### Concept

Apply hierarchical risk parity (HRP) for portfolio construction. Since we have a single asset, this chapter is **primarily theoretical** for our dataset. However, we can demonstrate HRP on a multi-asset universe by adding a few tickers (e.g., SPY, QQQ, AAPL alongside NVDA) if desired.

### Implementation

Present the HRP algorithm. If single-asset only: explain theory and note that it requires multiple assets.

---

# PART 4 — USEFUL FINANCIAL FEATURES

---

## Chapter 17: Structural Breaks

### 17.1 Concept

Detect regime changes in the price series. Key tests:

1. **CUSUM test:** Detects shifts in the mean.
2. **Chow-type breakpoint test.**
3. **Supremum Augmented Dickey-Fuller (SADF):** Tests for explosive (bubble) behaviour.
4. **Generalized SADF (GSADF):** More powerful variant using a double recursion.

### 17.2 Key Equations

**SADF (Snippet 17.1):**

Run ADF regressions with expanding or rolling windows [r₁, r₂]:

```
ADF_{r1,r2}: Δy_t = α + β × y_{t-1} + Σ γ_j Δy_{t-j} + ε_t
```

```
SADF = sup_{r2 ∈ [r₀, 1]} ADF_{0, r2}
```

Reject H₀ (no bubble) if SADF exceeds critical values from Monte Carlo simulation.

**GSADF extends to:**

```
GSADF = sup_{r2 ∈ [r₀, 1], r1 ∈ [0, r2 - r₀]} ADF_{r1, r2}
```

### 17.3 Implementability

Fully implementable on daily prices.

### 17.4 Implementation Tasks

1. Implement `get_sadf(log_prices, min_window, lags)` → SADF statistic + timestamp series of ADF values.
2. Compute critical values via Monte Carlo (simulate random walks, compute SADF distribution).
3. Plot SADF sequence; highlight periods where it exceeds 95 % critical value (potential bubbles).

### 17.5 Suggested Modules

```
src/structural_breaks.py
    get_bsadf(log_p, min_sl, lags) → pd.Series  # SADF sequence
    get_gsadf(log_p, min_sl, lags) → float        # GSADF statistic
    cv_sadf(n, min_sl, lags, reps=1000) → dict     # Monte Carlo critical values
```

### 17.6 Plots

- SADF time series with 95 % critical value line overlaid on the NVDA price chart.
- Highlight detected bubble regimes.

---

## Chapter 18: Entropy Features

### 18.1 Concept

Measure the information content / predictability of the return series using entropy estimators:

- **Shannon entropy** of discretised returns.
- **Plug-in (maximum likelihood) estimator.**
- **Lempel-Ziv complexity** (related to Kolmogorov complexity).
- **Kontoyiannis entropy estimator** (based on match lengths).

Lower entropy → more predictable → potentially more profitable to trade.

### 18.2 Key Equations

**Shannon entropy:**

```
H(X) = -Σ_x p(x) × log₂(p(x))
```

Discretise daily returns into bins (e.g., quantiles) to estimate p(x).

**Lempel-Ziv complexity (Snippet 18.3):**

Count the number of distinct substrings in a binary encoding of the return series (1 if positive, 0 if negative). Normalise by theoretical maximum.

**Kontoyiannis estimator:**

```
Ĥ = (n / Σ_{i=1}^{n} L_i) × log₂(n)
```

where L_i is the longest match length starting at position i.

### 18.3 Implementability

Fully implementable on daily returns.

### 18.4 Implementation Tasks

1. Discretise returns into bins.
2. Compute rolling Shannon entropy (e.g., 50-day window).
3. Implement Lempel-Ziv estimator on binary-encoded return signs.
4. Implement Kontoyiannis estimator.
5. Add entropy features to feature matrix.

### 18.5 Suggested Modules

```
src/entropy.py
    shannon_entropy(msg, base=2) → float
    lempel_ziv_complexity(binary_string) → float
    kontoyiannis_entropy(msg, window) → float
    rolling_entropy(returns, window, n_bins) → pd.Series
```

---

## Chapter 19: Microstructural Features

### 19.1 Concept

Features derived from market microstructure: bid-ask spread, informed trading probability, price impact.

### 19.2 Key Equations & Applicability

| Feature | Formula | Daily OHLCV? |
|---|---|---|
| **Corwin-Schultz spread** | `S = (2(eᵅ - 1)) / (1 + eᵅ)` where α is estimated from high-low ratios over two consecutive bars | **Yes** — uses High/Low |
| **Bekker-Parkinson volatility** | `σ² = (1/4ln2) × (ln(H/L))²` | **Yes** |
| **Kyle's Lambda** | Regression of ΔP on signed volume | **No** — needs tick-level signed volume |
| **Amihud illiquidity** | `λ = |r_t| / DollarVolume_t` | **Yes** |
| **VPIN** | Volume-synchronised probability of informed trading | **No** — needs tick-level BVC classification |
| **Roll spread** | `Roll = 2√(-Cov(Δp_t, Δp_{t-1}))` | Approximation — use daily returns, but estimate is noisy |

### 19.3 Implementation Tasks

1. Implement Corwin-Schultz estimator from High/Low prices (Snippet 19.1).
2. Compute Bekker-Parkinson volatility.
3. Compute Amihud illiquidity ratio.
4. Compute Roll spread estimate from daily returns.
5. Explain VPIN and Kyle's Lambda theoretically; note limitation.

### 19.4 Suggested Modules

```
src/microstructure.py
    corwin_schultz_spread(high, low, window=1) → pd.Series
    bekker_parkinson_vol(high, low) → pd.Series
    amihud_illiquidity(close, volume, window=20) → pd.Series
    roll_spread(close, window=20) → pd.Series
```

### 19.5 Plots

- Rolling spread estimate (Corwin-Schultz) over time.
- Rolling Amihud illiquidity.
- Correlation table of microstructure features with labels.

---

# PART 5 — HIGH-PERFORMANCE COMPUTING

---

## Chapter 20: Multiprocessing and Vectorization

### Concept

De Prado provides recipes for parallelising computations across multiple cores using Python's `multiprocessing`. Key patterns:

- **`mp_pandas_obj`** (Snippet 20.5): A generic function that distributes pandas computations across cores. Accepts a callback function, a molecule (subset of index), and keyword arguments.
- Vectorise inner loops using NumPy wherever possible.

### Implementability

Fully implementable. Wrap computationally intensive functions (label generation, sample weights, sequential bootstrap) in `mp_pandas_obj`.

### Suggested Module

```
src/multiprocess.py
    mp_pandas_obj(func, pd_obj, num_threads, mp_batches, lin_mols, **kwargs) → pd.DataFrame
    process_jobs(jobs) → list
    expand_call(kargs) → result   # helper for Pool.imap_unordered
```

---

## Chapters 21–22

Theoretical discussion of quantum computing and brute-force combinatorial methods. **Not implementable** — include a brief theoretical summary in the report.

---

# STAGED ROADMAP

## Stage 0: Dataset Inspection and Cleaning

- Load NVDA_raw.csv, parse dates, set index.
- Confirm: 5 114 rows, 7 columns, no nulls.
- Compute basic statistics, check for outliers (> 5σ daily return).
- Compute adjusted dollar volume.
- Export cleaned dataset.
- **Output:** `nvda_clean.parquet`

## Stage 1: Financial Data Structures and Event Sampling

- Compute dollar bars with threshold calibration.
- Apply CUSUM filter on daily log returns.
- Calibrate h to produce ~300–600 events.
- **Output:** `nvda_dollar_bars.parquet`, `nvda_cusum_events.parquet`

## Stage 2: Labels and Sample Weights

- Compute daily volatility (EWMA).
- Apply triple-barrier labeling on CUSUM events.
- Compute concurrency and average uniqueness.
- Compute return-attribution weights.
- Apply time decay.
- Implement sequential bootstrap.
- **Output:** `nvda_labels.parquet`, `nvda_sample_weights.parquet`

## Stage 3: Fractional Differentiation and Feature Engineering

- Compute FFD on log(Adj Close) for d ∈ [0, 1].
- Find optimal d* via ADF test.
- Compute rolling features: returns, volatility, momentum, RSI-analog.
- Compute microstructure features: Corwin-Schultz spread, Amihud illiquidity, Bekker-Parkinson vol.
- Compute entropy features: rolling Shannon entropy, Lempel-Ziv.
- Compute structural-break features: rolling SADF.
- Merge all features aligned to event timestamps.
- **Output:** `nvda_features.parquet`

## Stage 4: Model Training

- Merge features, labels, weights into a single modelling dataset.
- Train RandomForestClassifier with sample weights.
- Train XGBClassifier with sample weights.
- Use purged K-fold CV for evaluation.
- **Output:** `model_rf.pkl`, `model_xgb.pkl`, `cv_results.parquet`

## Stage 5: Purged CV and Hyperparameter Tuning

- Implement PurgedKFold splitter.
- Run RandomizedSearchCV with purged CV.
- Log all trials.
- Compute DSR to test significance of best trial.
- **Output:** `best_params.json`, `tuning_log.parquet`

## Stage 6: Feature Importance and Interpretation

- Compute MDI, MDA, SFI.
- Rank features by all three methods.
- Drop features with consistently low importance.
- Retrain model on reduced feature set.
- **Output:** `feature_importance.parquet`, `model_final.pkl`

## Stage 7: Meta-Labeling and Bet Sizing

- Use final model's directional signal as the primary model.
- Generate meta-labels.
- Train meta-model (predict probability of profit).
- Convert probabilities to bet sizes.
- **Output:** `nvda_meta_labels.parquet`, `nvda_positions.parquet`

## Stage 8: Backtesting and Backtest Statistics

- Compute PnL from position series.
- Compute: Sharpe, PSR, DSR, MaxDD, Time Under Water, Calmar.
- Run CPCV backtest path generation.
- Run pipeline on synthetic data as sanity check.
- **Output:** `backtest_results.parquet`, `equity_curve.png`

## Stage 9: Structural Breaks, Entropy, Microstructure Features

(These features are computed in Stage 3 but their analysis and report sections are here.)

- Run SADF/GSADF and overlay on price chart.
- Analyse rolling entropy features.
- Analyse microstructure features.
- **Output:** Report plots and analysis.

## Stage 10: Performance Optimisation

- Wrap slow functions in `mp_pandas_obj`.
- Benchmark: single-threaded vs multi-threaded runtimes.
- **Output:** `src/multiprocess.py`, benchmark table.

## Stage 11: Final Report and Presentation

- Compile all results into report sections.
- Generate all final plots and tables.
- Write limitations section.
- **Output:** Final report document, slide deck.

---

# EXACT FINAL DATASET SCHEMA BEFORE MODELLING

```
nvda_modelling_dataset.parquet
Columns:
    event_time          : datetime64    (index, from CUSUM events)
    barrier_time        : datetime64    (from triple-barrier)
    label               : int8          (+1, -1, 0)
    return              : float64       (label return)
    barrier_type        : str           ('upper', 'lower', 'vertical')
    target_vol          : float64       (daily vol at event time)
    sample_weight       : float64       (combined weight)
    uniqueness          : float64       (average uniqueness)
    fracdiff_close      : float64       (FFD of log price at optimal d*)
    ret_5d              : float64       (5-day rolling return)
    ret_20d             : float64       (20-day rolling return)
    vol_20d             : float64       (20-day rolling volatility)
    vol_50d             : float64       (50-day rolling volatility)
    momentum_12_1       : float64       (12-month minus 1-month return)
    rsi_14              : float64       (14-day RSI-analog)
    log_dollar_volume   : float64       (log daily dollar volume)
    cs_spread           : float64       (Corwin-Schultz bid-ask spread)
    amihud_illiq        : float64       (Amihud illiquidity)
    bp_vol              : float64       (Bekker-Parkinson volatility)
    roll_spread         : float64       (Roll model spread estimate)
    shannon_entropy_50  : float64       (50-day rolling Shannon entropy)
    lz_complexity_50    : float64       (50-day Lempel-Ziv complexity)
    sadf                : float64       (rolling SADF statistic)
```

---

# FINAL REPORT TABLES AND PLOTS

### Tables

| # | Table | Source Stage |
|---|---|---|
| T1 | Dataset summary statistics | Stage 0 |
| T2 | Dollar-bar calibration: threshold vs bar count | Stage 1 |
| T3 | CUSUM filter: h vs event count | Stage 1 |
| T4 | Label distribution by barrier type | Stage 2 |
| T5 | Sample weight statistics | Stage 2 |
| T6 | Fracdiff: d vs ADF p-value vs correlation | Stage 3 |
| T7 | Feature list with descriptions | Stage 3 |
| T8 | CV accuracy: RF vs XGBoost (purged CV) | Stage 4 |
| T9 | Best hyperparameters | Stage 5 |
| T10 | Feature importance: MDI / MDA / SFI rankings | Stage 6 |
| T11 | Backtest statistics: SR, PSR, DSR, MaxDD, Calmar | Stage 8 |
| T12 | CPCV path statistics | Stage 8 |
| T13 | AFML methods not implementable with daily OHLCV | Report |
| T14 | Multiprocessing speedup benchmarks | Stage 10 |

### Plots

| # | Plot | Source Stage |
|---|---|---|
| P1 | NVDA price chart with stock splits annotated | Stage 0 |
| P2 | Daily returns distribution + QQ plot | Stage 0 |
| P3 | Dollar bars per year vs time bars per year | Stage 1 |
| P4 | CUSUM events overlaid on price chart | Stage 1 |
| P5 | Triple-barrier events coloured by label on price chart | Stage 2 |
| P6 | Label distribution bar chart | Stage 2 |
| P7 | Concurrency count time series | Stage 2 |
| P8 | Uniqueness histogram | Stage 2 |
| P9 | Fracdiff: d vs ADF / correlation dual-axis plot | Stage 3 |
| P10 | Original vs fracdiff vs first-diff series | Stage 3 |
| P11 | Feature correlation heatmap | Stage 3 |
| P12 | RF confusion matrix (best fold) | Stage 4 |
| P13 | Purged CV scores per fold | Stage 4 |
| P14 | Hyperparameter search landscape (2D heatmap) | Stage 5 |
| P15 | MDI importance bar chart (with error bars) | Stage 6 |
| P16 | MDA importance bar chart | Stage 6 |
| P17 | SFI bar chart | Stage 6 |
| P18 | Bet size distribution | Stage 7 |
| P19 | Equity curve with drawdown shading | Stage 8 |
| P20 | CPCV: distribution of Sharpe across paths | Stage 8 |
| P21 | Synthetic-data equity curves | Stage 8 |
| P22 | SADF time series with critical values + price | Stage 9 |
| P23 | Rolling entropy time series | Stage 9 |
| P24 | Corwin-Schultz spread time series | Stage 9 |

---

# RECOMMENDED REPOSITORY STRUCTURE

```
afml-nvda/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── NVDA_raw.csv
│   └── processed/
│       ├── nvda_clean.parquet
│       ├── nvda_dollar_bars.parquet
│       ├── nvda_cusum_events.parquet
│       ├── nvda_labels.parquet
│       ├── nvda_sample_weights.parquet
│       ├── nvda_features.parquet
│       ├── nvda_modelling_dataset.parquet
│       └── nvda_positions.parquet
├── src/
│   ├── __init__.py
│   ├── data_structures.py       # Ch 2: bars, CUSUM
│   ├── labeling.py              # Ch 3: triple barrier, meta-labeling
│   ├── sample_weights.py        # Ch 4: concurrency, uniqueness, weights
│   ├── fracdiff.py              # Ch 5: fractional differentiation
│   ├── modelling.py             # Ch 6: RF, XGBoost wrappers
│   ├── cross_validation.py      # Ch 7: PurgedKFold, CPCV
│   ├── feature_importance.py    # Ch 8: MDI, MDA, SFI
│   ├── hyperparameter_tuning.py # Ch 9: search with purged CV
│   ├── bet_sizing.py            # Ch 10: probability → position size
│   ├── backtesting.py           # Ch 11-14: PnL, SR, PSR, DSR, drawdown
│   ├── synthetic.py             # Ch 13: synthetic data generation
│   ├── structural_breaks.py     # Ch 17: SADF, GSADF
│   ├── entropy.py               # Ch 18: Shannon, Lempel-Ziv
│   ├── microstructure.py        # Ch 19: spread estimators, Amihud
│   ├── multiprocess.py          # Ch 20: mp_pandas_obj
│   └── utils.py                 # shared helpers
├── notebooks/
│   ├── 00_data_inspection.ipynb
│   ├── 01_data_structures.ipynb
│   ├── 02_labeling.ipynb
│   ├── 03_sample_weights.ipynb
│   ├── 04_fracdiff.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_purged_cv.ipynb
│   ├── 08_feature_importance.ipynb
│   ├── 09_hyperparameter_tuning.ipynb
│   ├── 10_meta_labeling_bet_sizing.ipynb
│   ├── 11_backtesting.ipynb
│   ├── 12_structural_breaks.ipynb
│   ├── 13_entropy_microstructure.ipynb
│   └── 14_final_report_plots.ipynb
├── models/
│   ├── model_rf.pkl
│   ├── model_xgb.pkl
│   └── best_params.json
├── reports/
│   ├── figures/
│   └── final_report.pdf
└── tests/
    ├── test_labeling.py
    └── test_cv.py
```

---

# MINIMUM VIABLE VERSION vs ADVANCED VERSION

### Minimum Viable (MVP)

1. Daily time bars (skip dollar bars).
2. CUSUM event filter.
3. Triple-barrier labeling with fixed pt = sl = 1.0.
4. Sample weights (uniqueness + return attribution).
5. Fractional differentiation at optimal d*.
6. Basic feature set: fracdiff, rolling returns, rolling vol.
7. Random Forest with purged K-fold CV.
8. MDI feature importance.
9. Simple backtest: equity curve, Sharpe, MaxDD.
10. Final report with core tables and plots.

### Advanced Version (adds to MVP)

1. Dollar bars with calibrated threshold.
2. Meta-labeling + bet sizing.
3. XGBoost + hyperparameter tuning with DSR significance test.
4. MDA + SFI feature importance.
5. CPCV backtest paths.
6. SADF structural-break features.
7. Entropy features.
8. Microstructure features (Corwin-Schultz, Amihud).
9. Sequential bootstrap.
10. Synthetic-data validation.
11. Multiprocessing wrappers.

---

# REALISTIC TIMELINE (4-person team, 6 weeks)

| Week | Stages | Deliverables |
|---|---|---|
| 1 | Stage 0, 1 | Clean data, dollar bars, CUSUM events |
| 2 | Stage 2, 3 | Labels, weights, fracdiff, feature engineering |
| 3 | Stage 4, 5 | Model training, purged CV, hyperparameter tuning |
| 4 | Stage 6, 7 | Feature importance, meta-labeling, bet sizing |
| 5 | Stage 8, 9 | Backtesting, structural breaks, entropy, microstructure |
| 6 | Stage 10, 11 | Performance optimisation, final report assembly |

---

# IMPLEMENTATION PROMPTS

Below are self-contained prompts to paste into Claude one at a time. Each prompt is designed to produce a working notebook or module.

---

## Prompt 1: Dataset Inspection and Cleaning

**Goal:** Create a Jupyter notebook that loads, inspects, cleans, and profiles the NVDA dataset.

**Inputs:** `NVDA_raw.csv`

**Outputs:** `nvda_clean.parquet`, summary statistics, plots.

**Functions to create:**
- `load_and_clean(path)` → pd.DataFrame with DatetimeIndex
- `detect_outliers(returns, sigma=5)` → boolean mask
- `compute_dollar_volume(df)` → Series

**Validation checks:**
- Assert no NaN values.
- Assert dates are monotonically increasing.
- Assert all prices > 0.
- Assert Volume > 0 on all rows.

**Plots:** P1 (price chart), P2 (returns distribution + QQ plot), dollar volume time series.

**Prompt text:**
```
Create a complete Python notebook (00_data_inspection.ipynb) that:
1. Loads NVDA_raw.csv with columns [Date, Adj Close, Close, High, Low, Open, Volume].
2. Parses Date as datetime, sets as index.
3. Computes: daily log returns from Adj Close, daily dollar volume = Adj Close × Volume.
4. Reports: shape, date range, missing values, basic statistics.
5. Flags outlier returns beyond ±5σ (report dates and values).
6. Produces these plots:
   a. NVDA Adj Close price chart (log scale) over 2005–2025.
   b. Daily log return distribution with Jarque-Bera test result annotated.
   c. QQ plot of daily returns vs normal.
   d. Rolling 60-day volatility (annualised).
   e. Dollar volume time series.
7. Saves cleaned DataFrame as data/processed/nvda_clean.parquet.
Use pandas, numpy, matplotlib, scipy.stats. No external financial libraries needed.
```

---

## Prompt 2: CUSUM Event Detection and Dollar Bars

**Goal:** Implement CUSUM filter and approximate dollar bars from daily data.

**Inputs:** `nvda_clean.parquet`

**Outputs:** `nvda_cusum_events.parquet`, `nvda_dollar_bars.parquet`

**Functions to create:**
- `cusum_filter(close, h)` → DatetimeIndex
- `get_dollar_bars(df, threshold)` → DataFrame
- `calibrate_cusum_h(close, target_events)` → float
- `calibrate_dollar_bar_threshold(df, target_bars_per_year)` → float

**Key equations:**
- CUSUM: `S⁺_t = max(0, S⁺_{t-1} + y_t - E[y] - h)`, `S⁻_t = min(0, S⁻_{t-1} + y_t - E[y] + h)`, event when `|S| > h`.
- Dollar bar: emit bar when cumulative dollar volume ≥ Θ.

**Validation checks:**
- CUSUM events should be a subset of the date index.
- Dollar bar count should be roughly proportional to total dollar volume / threshold.
- Bar OHLCV should satisfy H ≥ max(O, C) and L ≤ min(O, C).

**Plots:** P3 (bars per year comparison), P4 (CUSUM events on price chart), histogram of CUSUM inter-event durations.

**Prompt text:**
```
Create src/data_structures.py and notebook 01_data_structures.ipynb that:

Module (src/data_structures.py):
1. cusum_filter(close: pd.Series, h: float) → pd.DatetimeIndex
   - Implement symmetric CUSUM on log returns.
   - y_t = log(close_t / close_{t-1}), E[y] = expanding mean.
   - Trigger event when S⁺ > h or S⁻ < -h, then reset.
   
2. get_dollar_bars(df: pd.DataFrame, threshold: float) → pd.DataFrame
   - df has columns [Open, High, Low, Close, Adj Close, Volume].
   - Compute dollar_volume = Adj Close × Volume each day.
   - Accumulate dollar volume; when cumsum ≥ threshold, emit one bar.
   - Bar OHLCV: Open = first day's Open, High = max of constituent Highs, Low = min of Lows, Close = last day's Close, Volume = sum.
   - Return DataFrame indexed by bar end date.

3. calibrate_cusum_h(close, target_events=400) → float
   - Binary search over h to find value producing ~target_events.

4. calibrate_dollar_bar_threshold(df, target_bars_per_year=252) → float

Notebook (01_data_structures.ipynb):
1. Load nvda_clean.parquet.
2. Calibrate and apply CUSUM filter (target ~400 events). Print h value and event count.
3. Calibrate and generate dollar bars.
4. Compare: bar count per year for daily vs dollar bars.
5. Plot CUSUM events as red dots on the price chart.
6. Plot histogram of days between consecutive CUSUM events.
7. Test normality of dollar-bar returns vs daily returns (Jarque-Bera).
8. Save outputs as parquet files.

Include the mathematical equations as comments in the code.
```

---

## Prompt 3: Triple-Barrier Labeling

**Goal:** Implement triple-barrier labeling and dynamic volatility threshold.

**Inputs:** `nvda_clean.parquet`, `nvda_cusum_events.parquet`

**Outputs:** `nvda_labels.parquet`

**Functions to create:**
- `get_daily_vol(close, span=50)` → pd.Series
- `add_vertical_barrier(close, events, num_days)` → pd.Series
- `apply_triple_barrier(close, events, pt_sl, molecule)` → pd.DataFrame
- `get_bins(events, close)` → pd.DataFrame
- `drop_labels(events, min_pct=0.05)` → pd.DataFrame

**Key equations:**
- `σ_t = EWMA_std(log_returns, span=50)`
- Upper barrier: `Close_t ≥ Close_{t₀} × (1 + pt × σ_{t₀})`
- Lower barrier: `Close_t ≤ Close_{t₀} × (1 - sl × σ_{t₀})`
- Vertical barrier: `t₁ = t₀ + max_holding_days`
- Label = sign(return at first barrier touched)

**Validation checks:**
- Every event should have a barrier time t₁ ≤ end of dataset.
- Upper-barrier events should have positive returns; lower-barrier events negative.
- Label values ∈ {-1, 0, +1}.

**Plots:** P5, P6, sensitivity of label balance to pt/sl multipliers.

**Prompt text:**
```
Create src/labeling.py and notebook 02_labeling.ipynb that:

Module (src/labeling.py):
1. get_daily_vol(close, span=50) → pd.Series
   - Compute log returns, then EWMA std with given span.
   
2. add_vertical_barrier(close, events, num_days) → pd.Series
   - For each event timestamp, find the date num_days ahead in the index (or last date).
   
3. apply_triple_barrier(close, events, pt_sl=[1.0, 1.0], molecule=None) → pd.DataFrame
   - events DataFrame has columns: t1 (vertical barrier), trgt (daily vol at event).
   - pt_sl = [profit_take_multiplier, stop_loss_multiplier].
   - For each event in molecule (subset of events.index):
     a. Set upper barrier = close[t0] * (1 + pt * trgt[t0])
     b. Set lower barrier = close[t0] * (1 - sl * trgt[t0])
     c. Walk forward from t0 to t1, check if close crosses upper or lower.
     d. Record first barrier touched and the return.
   - Return DataFrame with columns [t1 (actual exit), sl (stop-loss time or NaT), pt (profit-take time or NaT)].

4. get_bins(events, close) → pd.DataFrame
   - Compute return = (close[t1] / close[t0]) - 1.
   - label = sign(return), but 0 if return exactly 0.
   - Return DataFrame [ret, bin].

5. drop_labels(events, min_pct=0.05) → pd.DataFrame
   - Remove events where any label class has fewer than min_pct of samples.

Notebook (02_labeling.ipynb):
1. Load data and CUSUM events.
2. Compute daily volatility (span=50).
3. Set vertical barrier = 10 trading days.
4. Apply triple barrier with pt_sl = [1.0, 1.0].
5. Get labels.
6. Print label distribution and barrier-type distribution.
7. Plot events on price chart coloured by label (+1 green, -1 red, 0 grey).
8. Sensitivity table: vary pt ∈ {0.5, 1.0, 1.5, 2.0} and sl ∈ {0.5, 1.0, 1.5, 2.0}, report label balance for each.
9. Save nvda_labels.parquet.

Show all equations as comments. Use numpy vectorisation where possible.
```

---

## Prompt 4: Sample Uniqueness and Weights

**Goal:** Implement concurrent-label counting, average uniqueness, sequential bootstrap, return-attribution weights, and time decay.

**Inputs:** `nvda_labels.parquet`, `nvda_clean.parquet`

**Outputs:** `nvda_sample_weights.parquet`

**Functions to create:**
- `num_co_events(close_idx, t1, molecule)` → pd.Series
- `sample_tw(t1, num_co_events, molecule)` → pd.Series
- `get_ind_matrix(bar_idx, t1)` → pd.DataFrame (indicator matrix)
- `seq_bootstrap(ind_matrix, s_length=None)` → list
- `get_return_attribution(events)` → pd.Series
- `get_time_decay(tw, c_lf=0.5)` → pd.Series
- `get_sample_weight(events, close, num_threads=1)` → pd.Series

**Key equations:**
- `c_t = Σ_i 1_{t ∈ [t₀ⁱ, t₁ⁱ]}`
- `ū_i = (1/(t₁ⁱ - t₀ⁱ + 1)) × Σ_{t=t₀}^{t₁} (1/c_t)`
- Sequential bootstrap: draw proportional to uniqueness given already-selected samples.
- `w_i = ū_i × |r_i|`, normalised.
- Time decay: `d_i = c^{x_i}` where `x_i = (i - 0) / (N - 1)`.

**Validation checks:**
- All uniqueness values ∈ (0, 1].
- Sum of weights ≈ n (normalised).
- Sequential bootstrap should produce average uniqueness > standard bootstrap.

**Plots:** P7, P8, sequential vs standard bootstrap uniqueness comparison.

**Prompt text:**
```
Create src/sample_weights.py and notebook 03_sample_weights.ipynb that:

Module (src/sample_weights.py):
1. num_co_events(close_idx, t1, molecule) → pd.Series
   - close_idx: full DatetimeIndex of price bars.
   - t1: Series mapping event start → event end.
   - molecule: subset of event indices to process.
   - For each bar in close_idx, count how many events from t1 are active.

2. sample_tw(t1, num_co_events, molecule) → pd.Series
   - For each event in molecule, compute average uniqueness:
     ū_i = mean(1/c_t for t in [t0_i, t1_i]).

3. get_ind_matrix(bar_idx, t1) → pd.DataFrame
   - Binary indicator matrix: rows = bars, columns = events.
   - ind[t, i] = 1 if t ∈ [t0_i, t1_i].

4. seq_bootstrap(ind_matrix, s_length=None) → list of int
   - s_length defaults to number of columns (events).
   - Draw one sample at a time; at each step compute average uniqueness
     of each candidate given already-drawn set; sample proportional to uniqueness.

5. get_return_attribution(events) → pd.Series
   - w_i = |ret_i| / sum(|ret|), then multiply by n.

6. get_time_decay(tw, c_lf=0.5) → pd.Series
   - Piecewise linear: oldest sample gets weight c_lf, newest gets 1.
   - Linearly interpolate. Normalise so mean = 1.

7. get_sample_weight(events, close, num_threads=1) → pd.Series
   - Combine: uniqueness × return_attribution × time_decay. Normalise.

Notebook (03_sample_weights.ipynb):
1. Load labels (with t0, t1, ret).
2. Compute concurrency count for each bar.
3. Compute average uniqueness for each event.
4. Run sequential bootstrap (100 draws) and compare average uniqueness with standard bootstrap.
5. Compute return-attribution weights.
6. Compute time decay with c_lf = 0.5.
7. Compute final combined weights.
8. Plot: concurrency over time, uniqueness histogram, weight distribution boxplot.
9. Save nvda_sample_weights.parquet.

Include all equations as docstrings and inline comments.
```

---

## Prompt 5: Fractional Differentiation

**Goal:** Implement FFD and find the minimum d that achieves stationarity.

**Inputs:** `nvda_clean.parquet`

**Outputs:** `nvda_fracdiff.parquet`

**Functions to create:**
- `get_weights(d, size, threshold=1e-5)` → np.array
- `get_weights_ffd(d, threshold=1e-5)` → np.array
- `frac_diff_ffd(series, d, threshold=1e-5)` → pd.Series
- `find_min_d(series, d_range, threshold=1e-5)` → float
- `plot_min_ffd(series)` → Figure

**Key equations:**
- `w_0 = 1`; `w_k = -w_{k-1} × (d - k + 1) / k`
- `X̃_t = Σ_{k=0}^{K} w_k × X_{t-k}` where K is truncated when `|w_k| < τ`
- ADF test: reject H₀ (unit root) at 5 % significance.

**Validation checks:**
- d = 0 should return the original series.
- d = 1 should approximate first differences.
- Correlation between fracdiff and original should be > 0.9 at optimal d*.
- ADF p-value at optimal d* should be < 0.05.

**Plots:** P9, P10.

**Prompt text:**
```
Create src/fracdiff.py and notebook 04_fracdiff.ipynb that:

Module (src/fracdiff.py):
1. get_weights(d, size, threshold=1e-5) → np.array
   - Compute binomial weights: w_0=1, w_k = -w_{k-1}*(d-k+1)/k.
   - Stop when |w_k| < threshold. Return array of length min(size, k).

2. get_weights_ffd(d, threshold=1e-5) → np.array
   - Fixed-width: compute weights until |w_k| < threshold. Return all.

3. frac_diff_ffd(series, d, threshold=1e-5) → pd.Series
   - Apply FFD weights to the series using a dot product on a rolling window.
   - Width = len(get_weights_ffd(d, threshold)).
   - Drop initial NaN rows (width - 1 lost).

4. find_min_d(series, d_range=np.arange(0, 1.05, 0.05), threshold=1e-5) → float
   - For each d, compute FFD, run ADF test.
   - Return smallest d where ADF p-value < 0.05.

5. plot_min_ffd(series) → Figure
   - Dual y-axis: left = ADF statistic, right = correlation with original.
   - Mark 1% and 5% critical values for ADF.
   - Mark the chosen d* with a vertical line.

Notebook (04_fracdiff.ipynb):
1. Load nvda_clean.parquet, extract log(Adj Close).
2. Sweep d ∈ [0, 1] in steps of 0.05.
3. For each d: compute FFD series, ADF test (statistic, p-value), correlation with original.
4. Print table: d | ADF stat | p-value | correlation | window_size.
5. Identify optimal d* (smallest d with p < 0.05).
6. Plot the dual-axis chart (P9).
7. Plot overlay of original, returns (d=1), fracdiff at d* (P10).
8. Save fracdiff column at d* as nvda_fracdiff.parquet.

Use statsmodels.tsa.stattools.adfuller for the ADF test.
```

---

## Prompt 6: Feature Engineering Pipeline

**Goal:** Build the complete feature matrix for modelling.

**Inputs:** `nvda_clean.parquet`, `nvda_fracdiff.parquet`, `nvda_cusum_events.parquet`, `nvda_labels.parquet`

**Outputs:** `nvda_features.parquet`, `nvda_modelling_dataset.parquet`

**Functions to create:**
- `compute_momentum_features(close)` → DataFrame
- `compute_volatility_features(close)` → DataFrame
- `compute_volume_features(df)` → DataFrame
- `compute_microstructure_features(df)` → DataFrame
- `compute_entropy_features(returns, window)` → DataFrame
- `build_feature_matrix(df, fracdiff, events)` → DataFrame

**Prompt text:**
```
Create src/features.py and notebook 05_feature_engineering.ipynb that:

Module (src/features.py):
1. compute_momentum_features(close) → pd.DataFrame
   - ret_5d, ret_10d, ret_20d, ret_60d: rolling returns.
   - momentum_12_1: 252-day return minus 21-day return.
   - rsi_14: 14-day RSI (use Wilder's smoothing: up/down averages).

2. compute_volatility_features(close) → pd.DataFrame
   - vol_20d, vol_50d: rolling std of log returns, annualised (×√252).

3. compute_volume_features(df) → pd.DataFrame
   - log_dollar_volume: log(Adj Close × Volume).
   - volume_ratio: Volume / 20-day average Volume.

4. compute_microstructure_features(df) → pd.DataFrame
   - Corwin-Schultz spread: from High/Low of consecutive bars.
     β = [ln(H_t/L_t)]² + [ln(H_{t-1}/L_{t-1})]² 
     γ = [ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))]²
     α = (√(2β) - √β) / (3 - 2√2) - √(γ/(3 - 2√2))
     spread = 2(e^α - 1)/(1 + e^α)
   - Bekker-Parkinson vol: σ² = (1/(4 ln 2)) × [ln(H/L)]²
   - Amihud illiquidity: |r_t| / dollar_volume_t, 20-day rolling mean.
   - Roll spread: 2√(max(0, -Cov(Δp_t, Δp_{t-1}))), 20-day rolling.

5. compute_entropy_features(returns, window=50) → pd.DataFrame
   - Rolling Shannon entropy: discretise returns into 10 bins, compute entropy.
   - Lempel-Ziv complexity: binary encode sign of returns, compute on rolling window.

6. build_feature_matrix(df, fracdiff, events, labels, weights) → pd.DataFrame
   - Compute all features on the full daily index.
   - Align to event timestamps from labels.
   - Merge with fracdiff, labels, weights.
   - Drop rows with any NaN.
   - Return the final modelling dataset.

Notebook (05_feature_engineering.ipynb):
1. Load all prior outputs.
2. Compute each feature group.
3. Build final modelling dataset.
4. Print shape, column list, summary statistics.
5. Plot feature correlation heatmap (P11).
6. Check for NaN, infinite values.
7. Save nvda_features.parquet and nvda_modelling_dataset.parquet.
```

---

## Prompt 7: Purged K-Fold Cross-Validation

**Goal:** Implement PurgedKFold and combinatorial purged CV as sklearn-compatible splitters.

**Inputs:** `nvda_modelling_dataset.parquet`

**Outputs:** Custom CV splitter classes.

**Functions to create:**
- `class PurgedKFold(KFold)` with purging and embargo.
- `class CombinatorialPurgedKFold` for CPCV.
- `cv_score(clf, X, y, sample_weight, cv, scoring)` → pd.Series

**Key equations:**
- Purge: remove train samples where `[t0_i, t1_i] ∩ [test_start, test_end] ≠ ∅`.
- Embargo: also remove train samples where `t0_i ∈ (test_end, test_end + embargo]`.

**Prompt text:**
```
Create src/cross_validation.py and notebook 07_purged_cv.ipynb that:

Module (src/cross_validation.py):
1. class PurgedKFold(BaseCrossValidator):
   - __init__(self, n_splits=5, t1=None, pct_embargo=0.01)
   - t1: pd.Series mapping each sample's event time to its barrier end time.
   - split(X, y=None, groups=None):
     a. Sort indices by time.
     b. Split into n_splits contiguous groups.
     c. For each test fold:
        - test_start, test_end = min/max time of test indices.
        - Purge: remove from train any index where t1[idx] > test_start and idx < test_start (label leaks into test).
        - Also purge: remove from train any index in test period.
        - Embargo: remove from train any index within embargo period after test_end.
     d. Yield (train_indices, test_indices).

2. cv_score(clf, X, y, sample_weight, scoring, cv, t1) → pd.Series
   - For each fold: fit clf on train with sample_weight, score on test.
   - Return Series of scores.

Notebook (07_purged_cv.ipynb):
1. Load modelling dataset.
2. Extract X (feature columns), y (label), sample_weight, t1 (barrier times).
3. Instantiate PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01).
4. Run cv_score with RandomForestClassifier(n_estimators=100).
5. Print mean and std of CV accuracy.
6. Compare with naive KFold (no purging) — show inflation.
7. Visualise fold composition: for each fold, show train/test/purged/embargo date ranges (P13).
8. Save CV results.
```

---

## Prompt 8: Model Training (RF + XGBoost)

**Goal:** Train ensemble models with sample weights and purged CV.

**Inputs:** `nvda_modelling_dataset.parquet`

**Outputs:** `model_rf.pkl`, `model_xgb.pkl`, `cv_results.parquet`

**Prompt text:**
```
Create src/modelling.py and notebook 06_model_training.ipynb that:

Module (src/modelling.py):
1. train_and_evaluate(clf, X, y, sample_weight, cv, scoring='accuracy') → dict
   - Run purged CV.
   - Return {mean_score, std_score, fold_scores, fitted_clf (on full data)}.

Notebook (06_model_training.ipynb):
1. Load modelling dataset; split into X, y, w, t1.
2. Define models:
   - RF: RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=10,
         class_weight='balanced_subsample', random_state=42)
   - XGB: XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, random_state=42)
3. For each model:
   a. Run purged 5-fold CV with sample weights.
   b. Print accuracy ± std.
   c. Plot confusion matrix for one representative fold.
4. Compare RF vs XGBoost in a table.
5. Fit final models on full data (for downstream feature importance and bet sizing).
6. Save models as pickle files and CV results.

pip install xgboost if needed.
```

---

## Prompt 9: Feature Importance (MDI, MDA, SFI)

**Goal:** Compute and compare three feature importance methods.

**Inputs:** `nvda_modelling_dataset.parquet`, trained RF model

**Outputs:** `feature_importance.parquet`

**Prompt text:**
```
Create src/feature_importance.py and notebook 08_feature_importance.ipynb that:

Module (src/feature_importance.py):
1. feat_imp_MDI(clf, feature_names) → pd.DataFrame
   - Extract clf.feature_importances_ (mean impurity decrease).
   - Also compute std across trees: for each tree, get tree.feature_importances_.
   - Return DataFrame with columns [mean, std], index = feature names.

2. feat_imp_MDA(clf, X, y, cv, sample_weight, scoring='neg_log_loss') → pd.DataFrame
   - For each fold in purged CV:
     a. Fit model on train.
     b. Score on test → baseline.
     c. For each feature j: permute column j in test, score → permuted_score.
     d. MDA_j_fold = baseline - permuted_score.
   - Average across folds. Return DataFrame [mean, std].

3. feat_imp_SFI(clf_template, X, y, cv, sample_weight, scoring) → pd.DataFrame
   - For each feature j:
     a. X_j = X[[j]] (single column).
     b. Run purged CV with clf_template.
     c. SFI_j = mean CV score.
   - Return DataFrame [mean, std].

4. plot_feature_importance(imp, title) → Figure
   - Sorted horizontal bar chart with error bars.

Notebook (08_feature_importance.ipynb):
1. Load modelling dataset and fitted RF.
2. Compute MDI, MDA, SFI.
3. Plot P15, P16, P17.
4. Create rank comparison table.
5. Identify features that rank in bottom 25% across all three methods — candidates for removal.
6. Save feature_importance.parquet.
```

---

## Prompt 10: Meta-Labeling

**Goal:** Implement meta-labeling pipeline: use the primary model's side as direction, train a secondary model to predict probability of profit.

**Inputs:** `nvda_modelling_dataset.parquet`, trained primary model

**Outputs:** `nvda_meta_labels.parquet`, meta-model

**Prompt text:**
```
Create notebook 10_meta_labeling_bet_sizing.ipynb (Part A: meta-labeling) that:

1. Load modelling dataset.
2. Use the final RF model to predict side (sign of prediction) for each event using OOS predictions from purged CV.
3. For each event with a side prediction:
   a. Apply triple-barrier in the direction of the predicted side only:
      - If side = +1: set sl = 1.0, pt = 1.0 (both barriers active).
      - The meta-label y_meta = 1 if the trade was profitable (return × side > 0), else 0.
4. Build a meta-label dataset: same features, y = y_meta, side = primary prediction.
5. Train a meta-model (RF or XGB) to predict P(profit | side, features).
6. Evaluate with purged CV.
7. Print: accuracy, precision, recall, F1.
8. Save nvda_meta_labels.parquet with columns [event_time, side, meta_label, meta_prob].
```

---

## Prompt 11: Bet Sizing

**Goal:** Convert meta-model probabilities to position sizes.

**Inputs:** `nvda_meta_labels.parquet`

**Outputs:** `nvda_positions.parquet`

**Prompt text:**
```
Continue notebook 10_meta_labeling_bet_sizing.ipynb (Part B: bet sizing) that:

Module (src/bet_sizing.py):
1. get_signal(prob, num_classes=2, step_size=0.0) → pd.Series
   - Compute: m = 2*prob - 1 (for binary classification).
   - signal = side × |m| (direction × confidence).
   - If step_size > 0: discretise to nearest step_size.

2. avg_active_signals(signals, t1) → pd.Series
   - At each bar, compute the average signal of all currently active events.
   - Active = events where t0 ≤ bar ≤ t1.

3. discrete_signal(signal, step_size=0.1) → pd.Series
   - Round signal to nearest step_size.

Notebook:
1. Compute signal from meta-model probabilities.
2. Apply avg_active_signals to get position time series.
3. Discretise positions.
4. Plot: position time series overlaid on price chart (P18).
5. Plot: bet size distribution histogram.
6. Save nvda_positions.parquet with columns [date, position, signal].
```

---

## Prompt 12: Backtesting and Metrics

**Goal:** Implement the full backtest loop and compute all performance statistics.

**Inputs:** `nvda_positions.parquet`, `nvda_clean.parquet`

**Outputs:** `backtest_results.parquet`, plots

**Prompt text:**
```
Create src/backtesting.py and notebook 11_backtesting.ipynb that:

Module (src/backtesting.py):
1. backtest_strategy(positions, prices, cost_bps=5) → pd.DataFrame
   - Daily PnL = position_{t-1} × (price_t / price_{t-1} - 1) - |Δposition_t| × cost_bps/10000.
   - Return DataFrame [date, position, daily_return, cumulative_return, hwm, drawdown].

2. sharpe_ratio(returns, periods_per_year=252) → float
   - SR = mean(r) / std(r) × √252.

3. prob_sharpe_ratio(returns, sr_benchmark=0) → float
   - PSR = Φ( (SR - SR*) × √(n-1) / √(1 - skew×SR + (kurt-1)/4 × SR²) )

4. deflated_sharpe_ratio(returns, num_trials, sr_benchmark=0) → float
   - Compute SR* from the distribution of trial Sharpes:
     SR* = √(V[SR]) × ((1-γ)×Φ⁻¹(1-1/N) + γ×Φ⁻¹(1-1/(N×e)))
   - Then compute PSR with that SR*.

5. max_drawdown(returns) → (value, peak_date, trough_date)

6. time_under_water(returns) → pd.Timedelta (longest period below HWM)

7. calmar_ratio(returns) → float (annualised_return / |max_drawdown|)

Notebook (11_backtesting.ipynb):
1. Load positions and clean price data.
2. Run backtest with cost = 5 bps per trade.
3. Compute and print table T11: SR, PSR, DSR (using num_trials from hyperparameter tuning), MaxDD, Time Under Water, Calmar.
4. Plot equity curve with drawdown shading (P19).
5. Plot monthly returns heatmap.
6. Run backtest on synthetic trending series and synthetic mean-reverting series as sanity checks (P21).
7. Save backtest_results.parquet.

Include all equations as docstrings.
```

---

## Prompt 13: Structural Breaks and Advanced Features

**Goal:** Implement SADF for structural-break detection and finalise entropy/microstructure feature analysis.

**Inputs:** `nvda_clean.parquet`

**Outputs:** SADF analysis, entropy analysis, microstructure analysis — plots for the report.

**Prompt text:**
```
Create src/structural_breaks.py and notebook 12_structural_breaks.ipynb that:

Module (src/structural_breaks.py):
1. get_bsadf(log_p, min_sl, lags=1) → pd.Series
   - For each end point r2 from min_sl to len(log_p):
     a. Run ADF regression on log_p[0:r2].
     b. Record ADF statistic.
   - Return Series of ADF statistics (SADF path).

2. cv_sadf(n, min_sl, lags=1, reps=1000) → dict
   - Simulate random walks of length n.
   - For each, compute SADF (= max of the ADF path).
   - Return {90%, 95%, 99%} critical values.

Notebook (12_structural_breaks.ipynb):
1. Load nvda_clean, compute log prices.
2. Run get_bsadf with min_sl = 63 (≈1 quarter).
3. Compute Monte Carlo critical values.
4. Plot SADF path with 95% critical value line; overlay NVDA price on twin axis (P22).
5. Identify periods where SADF exceeds 95% CV — annotate as potential bubbles.

Also create notebook 13_entropy_microstructure.ipynb:
1. Load features.
2. Plot rolling Shannon entropy (P23), Lempel-Ziv complexity.
3. Plot Corwin-Schultz spread (P24), Amihud illiquidity over time.
4. Correlate these features with forward returns and labels.
5. Discuss which features are most informative (table + commentary).
```

---

## Prompt 14: Final Report Tables, Plots, and Assembly

**Goal:** Generate all publication-quality figures and summary tables for the report.

**Inputs:** All prior outputs

**Outputs:** All figures in `reports/figures/`, summary tables as CSVs.

**Prompt text:**
```
Create notebook 14_final_report_plots.ipynb that:

1. Load all intermediate results: labels, weights, features, CV scores, feature importance, backtest results, SADF.

2. Generate publication-quality figures (save as PNG 300 DPI):
   - P1–P24 as defined in the plan (use consistent style: white background, 12pt labels, grid on).
   - Use a consistent colour palette throughout.

3. Generate summary tables (save as CSV):
   - T1: Dataset summary.
   - T2: Dollar bar calibration.
   - T3: CUSUM calibration.
   - T4: Label distribution.
   - T5: Sample weight stats.
   - T6: Fracdiff sweep results.
   - T7: Feature list with descriptions.
   - T8: CV accuracy comparison.
   - T9: Best hyperparameters.
   - T10: Feature importance rankings.
   - T11: Backtest statistics.
   - T12: CPCV path statistics (if CPCV implemented).
   - T13: Methods not implementable with daily OHLCV.
   - T14: Multiprocessing benchmarks.

4. Print suggested report outline:
   - 1. Introduction (Ch 1 motivation)
   - 2. Data (Ch 2)
   - 3. Labeling and Weights (Ch 3–4)
   - 4. Feature Engineering (Ch 5, 17–19)
   - 5. Model Training and CV (Ch 6–7, 9)
   - 6. Feature Importance (Ch 8)
   - 7. Meta-Labeling and Bet Sizing (Ch 10)
   - 8. Backtesting (Ch 11–15)
   - 9. Limitations and Future Work
   - 10. Conclusion

5. Save all outputs to reports/ directory.
```

---

*End of Implementation Plan*
