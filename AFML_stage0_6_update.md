# AFML NVDA Project — Stage 0–6 Implementation Update

## Purpose of this document

This document summarizes what was implemented in the AFML NVDA project so far, what results we obtained, what issues we found during validation, how we fixed them, and what should happen next before moving to Stage 7.

The project follows the implementation plan based on *Advances in Financial Machine Learning* by Marcos López de Prado. The dataset is daily OHLCV data for NVDA, so some AFML methods that require tick/order-book data are approximated using daily data.

---

# 1. Current project status

## Completed stages

| Stage | Name | Status |
|---|---|---|
| Stage 0 | Dataset inspection and cleaning | Completed by teammate |
| Stage 1 | Financial data structures and CUSUM events | Completed by teammate |
| Stage 2 | Labeling and sample weights | Completed by teammate |
| Stage 3 | Fractional differentiation and feature engineering | Completed, then fixed/validated |
| Stage 4 | Model training | Implemented and regenerated |
| Stage 5 | Hyperparameter tuning | Implemented and regenerated |
| Stage 6 | Feature importance and interpretation | Implemented and regenerated |

## Not implemented yet

| Stage | Name |
|---|---|
| Stage 7 | Meta-labeling and bet sizing |
| Stage 8 | Backtesting and backtest statistics |
| Stage 9 | Structural breaks, entropy, and microstructure analysis/reporting |
| Stage 10 | Performance optimization |
| Stage 11 | Final report and presentation |

---

# 2. Files implemented or regenerated

## Source code files

| File | Purpose |
|---|---|
| `src/modelling.py` | Stage 4 model training utilities for RF/XGBoost with purged CV |
| `src/hyperparameter_tuning.py` | Stage 5 manual randomized search using PurgedKFold |
| `src/feature_importance.py` | Stage 6 MDI, MDA, and SFI feature importance |
| `src/cross_validation.py` | Updated to use weighted scoring during CV evaluation |
| `src/fracdiff.py` | Fixed fractional differentiation implementation |
| `src/features.py` | Fixed microstructure and entropy feature engineering |

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/04_fracdiff.ipynb` | Fixed fractional differentiation sweep and plots |
| `notebooks/05_feature_engineering.ipynb` | Regenerated feature matrix after feature fixes |
| `notebooks/06_model_training.ipynb` | Stage 4 baseline RF/XGB model training |
| `notebooks/09_hyperparameter_tuning.ipynb` | Stage 5 weighted purged-CV randomized tuning |
| `notebooks/08_feature_importance.ipynb` | Stage 6 MDI/MDA/SFI feature importance and reduced model |

## Data/model artifacts

| Artifact | Purpose |
|---|---|
| `data/processed/nvda_fracdiff.parquet` | Corrected fractionally differentiated feature |
| `data/processed/nvda_features.parquet` | Corrected full feature matrix |
| `data/processed/nvda_modelling_dataset.parquet` | Final modelling dataset before Stage 4 |
| `data/processed/cv_results.parquet` | Stage 4 RF/XGB CV results |
| `data/processed/tuning_log.parquet` | Stage 5 tuning trials |
| `data/processed/feature_importance.parquet` | Stage 6 MDI/MDA/SFI results and ranks |
| `models/model_rf.pkl` | Baseline RF model |
| `models/model_xgb.pkl` | Baseline XGBoost model |
| `models/best_params.json` | Best tuned parameters and CV-trial DSR-style results |
| `models/model_final.pkl` | Final tuned RF model after Stage 6 feature pruning |

## Figures generated

| Figure | Purpose |
|---|---|
| `reports/figures/P9_fracdiff_adf_correlation.png` | ADF/correlation sweep for fractional differentiation |
| `reports/figures/P10_fracdiff_overlay.png` | Original vs fracdiff overlay |
| `reports/figures/P11_feature_correlation_heatmap.png` | Feature correlation heatmap |
| `reports/figures/P15_mdi_importance.png` | MDI feature importance |
| `reports/figures/P16_mda_importance.png` | MDA feature importance |
| `reports/figures/P17_sfi_importance.png` | SFI feature importance |

---

# 3. Final modelling dataset after fixes

The corrected final modelling dataset is:

```text
data/processed/nvda_modelling_dataset.parquet
```

Shape:

```text
195 rows × 20 columns
```

It contains:

- 17 feature columns
- `label`
- `weight`
- `t1`

Label distribution:

```text
+1: 114
-1: 81
```

Majority-class baseline:

```text
114 / 195 = 0.5846
```

Final feature list:

```text
ret_5d
ret_10d
ret_20d
ret_60d
momentum_12_1
rsi_14
vol_20d
vol_50d
log_dollar_volume
volume_ratio
corwin_schultz_spread
bekker_parkinson_vol
amihud_illiquidity
roll_spread
shannon_entropy
lempel_ziv_complexity
fracdiff
```

---

# 4. Stage 4 — Model training

## What was implemented

Stage 4 trains baseline ensemble models using the prepared modelling dataset:

- Random Forest classifier
- XGBoost classifier
- Purged K-Fold cross-validation
- sample weights from the AFML sample-weighting stage

The models use:

```text
X = all feature columns
y = label
sample_weight = weight
t1 = barrier end time for PurgedKFold
```

## Results after corrected weighted scoring

| Model | Mean weighted purged-CV accuracy | Std | Beats majority baseline? |
|---|---:|---:|---|
| Majority baseline | 0.5846 | — | — |
| Stage 4 RF, untuned | 0.5240 | 0.081 | No |
| Stage 4 XGB, untuned | 0.5283 | 0.102 | No |

## Interpretation

The untuned models do **not** beat the majority-class baseline. This is important and should be stated honestly in the report.

This does not mean the pipeline failed. Financial ML signals are noisy, and AFML emphasizes that naive models often overstate performance unless validation is done carefully.

---

# 5. Stage 5 — Hyperparameter tuning

## What was implemented

Stage 5 runs manual randomized hyperparameter search using:

- Purged K-Fold CV
- weighted fitting
- weighted test-fold scoring
- 25 trials for RF
- 25 trials for XGB
- tuning log saved to `data/processed/tuning_log.parquet`

## Tuned results

| Model | Mean weighted purged-CV accuracy | Std | Beats majority baseline? |
|---|---:|---:|---|
| Majority baseline | 0.5846 | — | — |
| Tuned RF | 0.6280 | 0.072 | Yes |
| Tuned XGB | 0.6456 | 0.080 | Yes |

## Best RF hyperparameters

```text
n_estimators = 100
max_depth = 7
min_samples_leaf = 20
max_features = sqrt
class_weight = None
```

## Best XGBoost hyperparameters

```text
n_estimators = 200
max_depth = 7
learning_rate = 0.01
subsample = 0.8
colsample_bytree = 0.5
gamma = 0.0
reg_lambda = 10.0
```

## DSR note

The values in `models/best_params.json` use a **CV-trial DSR-style correction**, based on the Sharpe-like ratio of CV fold scores.

This is **not** a true trading-strategy Deflated Sharpe Ratio. A true DSR must be computed later from realized strategy returns during Stage 8 backtesting.

Summary:

```text
RF best CV-trial Sharpe = 8.77
RF expected max SR under null = 3.18
RF DSR(CV trials) = 0.935

XGB best CV-trial Sharpe = 8.05
XGB expected max SR under null = 6.84
XGB DSR(CV trials) = 0.659
```

Interpretation:

- RF looks more robust after multiple-testing correction.
- XGB has higher mean accuracy, but its CV-trial DSR is less convincing.

---

# 6. Stage 6 — Feature importance

## What was implemented

Stage 6 implements AFML Chapter 8 feature importance methods:

1. **MDI — Mean Decrease Impurity**
2. **MDA — Mean Decrease Accuracy**, using weighted test-fold scoring
3. **SFI — Single Feature Importance**, using weighted Purged CV

Output:

```text
data/processed/feature_importance.parquet
```

## Top features by average rank

| Rank | Feature | Interpretation |
|---:|---|---|
| 1 | `amihud_illiquidity` | liquidity / price impact proxy |
| 2 | `log_dollar_volume` | trading activity / scale |
| 3 | `fracdiff` | stationary memory-preserving price feature |
| 4 | `ret_20d` | medium-horizon return signal |
| 5 | `volume_ratio` | abnormal volume activity |

## Features pruned

Two features were consistently weak across MDI, MDA, and SFI:

```text
momentum_12_1
bekker_parkinson_vol
```

They were removed from the reduced model.

## Final Stage 6 model

```text
models/model_final.pkl
```

This is a tuned Random Forest refit on the reduced 15-feature set.

Result:

| Model | Mean weighted purged-CV accuracy | Std | Beats baseline? |
|---|---:|---:|---|
| Stage 6 reduced RF | 0.6411 | 0.093 | Yes |

Interpretation:

The reduced RF is close to the tuned RF and above the majority baseline. This suggests that pruning did not materially hurt performance.

---

# 7. Issues we found and how we fixed them

## Issue 1 — Fractional differentiation was wrong

### Problem

The old fractional differentiation output had:

```text
d* = 0.20
ADF p-value passed
correlation with log price ≈ -0.008
```

This violated the AFML requirement: fractional differentiation should make the series more stationary while preserving memory.

### Root cause

The FFD weights were computed correctly, but applied in the wrong order during convolution. This caused the present-day weight `w_0 = 1` to be applied to the oldest observation in the window.

### Fix

We fixed the weight direction in `src/fracdiff.py`, hardened index alignment, and changed the selection rule to require both:

```text
ADF p-value < 0.05
correlation with aligned log price >= 0.9
```

### New result

```text
d* = 0.25
ADF p-value = 0.0119
correlation with log price = 0.916
n_obs retained = 2310
window length = 2804
```

Plan requirement satisfied.

---

## Issue 2 — Corwin-Schultz spread had negative values

### Problem

The original `corwin_schultz_spread` feature had many negative values:

```text
121 / 171 values were negative
mean ≈ -0.0506
```

A spread estimate should not be negative.

### Root cause

The Corwin-Schultz formula can produce negative values when its assumptions are violated. This happens more often with daily OHLCV approximations.

### Fix

We clipped the spread at zero:

```python
spread = spread.clip(lower=0)
```

and documented that this is a daily-OHLCV approximation.

### New result

```text
min = 0.0000
max = 0.1022
mean = 0.0081
std = 0.0180
negative count = 0
zero count = 140
```

---

## Issue 3 — Lempel-Ziv complexity was constant

### Problem

The original Lempel-Ziv feature was constant:

```text
min = 1.0
max = 1.0
std = 0.0
unique values = 1
```

A constant feature is useless for modelling.

### Root cause

The previous LZ logic had an indexing/boundary bug and did not implement the standard LZ-76 complexity algorithm properly.

### Fix

We replaced it with a proper Kaspar-Schuster-style LZ-76 implementation and normalized it by `n / log2(n)`.

### New result

```text
min = 0.9030
max = 1.4674
mean = 1.2301
std = 0.0992
unique values = 6
```

The feature now varies and was kept in the modelling dataset.

---

## Issue 4 — CV scoring ignored sample weights

### Problem

The code originally passed sample weights into model fitting, but not into test-fold scoring.

That meant:

```text
fit = weighted
score = unweighted
```

This was inconsistent with the AFML idea that overlapping samples should have unequal weights.

### Root cause

`cv_score()` and `feat_imp_MDA()` used sklearn scorer callables from `get_scorer()`, which did not forward `sample_weight` to the scoring function.

### Fix

We added a custom `weighted_score()` helper in `src/cross_validation.py` supporting:

- weighted accuracy
- weighted negative log-loss
- weighted F1

Then we updated:

- `cv_score()`
- `feat_imp_MDA()`

Now the default path is:

```text
fit = weighted
score = weighted
```

### Validation

Four-way validation showed that weighted scoring changes the fold scores:

| Mode | Fit weights | Score weights | Meaning |
|---|---|---|---|
| A | yes | yes | new correct default |
| B | yes | no | old behavior |
| C | no | yes | scoring-only weight sensitivity |
| D | no | no | no weights |

This confirmed test-fold sample weights are now actually being used.

---

## Issue 5 — Stage 4–6 outputs became stale after fixes

### Problem

After fixing Stage 3 features and weighted scoring, old model/tuning/importance outputs were stale.

### Fix

We reran:

```text
06_model_training.ipynb
09_hyperparameter_tuning.ipynb
08_feature_importance.ipynb
```

and regenerated:

```text
models/model_rf.pkl
models/model_xgb.pkl
models/best_params.json
models/model_final.pkl
data/processed/cv_results.parquet
data/processed/tuning_log.parquet
data/processed/feature_importance.parquet
reports/figures/P15_mdi_importance.png
reports/figures/P16_mda_importance.png
reports/figures/P17_sfi_importance.png
```

---

# 8. Git commits made

Current local commit history:

```text
647561d Regenerate stages 4-6 after validation fixes
76576a8 Fix fractional differentiation memory preservation
5e81626 Use weighted scoring in purged CV and feature importance
024369d Fix microstructure and entropy feature engineering
a6de5c7 Implement stage 6 feature importance analysis
c9d3d7e Implement stages 4-5 model training and tuning
```

A separate cleanup commit was also made to remove tracked Python cache files from Git tracking.

---

# 9. Important limitations to mention in the report

## Dataset limitation

The dataset is daily OHLCV only. It does not contain:

- tick data
- bid/ask quotes
- order-book depth
- signed trade volume

Therefore, some AFML methods are only approximated.

## Microstructure limitation

Corwin-Schultz, Amihud, Roll spread, and Bekker-Parkinson features are computed from daily OHLCV and should be treated as approximations.

## DSR limitation

The Stage 5 DSR-style values are computed from CV trial scores, not strategy returns.

True Deflated Sharpe Ratio should be computed in Stage 8 after backtesting.

## Model-performance limitation

Untuned models do not beat the majority baseline. Tuned models do, but this still does not prove trading profitability.

Profitability must be evaluated in Stage 8 using:

- positions
- transaction costs
- PnL
- Sharpe
- PSR
- DSR
- max drawdown
- time under water

---

# 10. Things still to clean up

Before final submission, we should still do the following:

## Repository cleanup

- Fill `README.md`
- Fill `requirements.txt`
- Ensure `.gitignore` excludes:

```text
__pycache__/
*.pyc
.claude/
.ipynb_checkpoints/
```

## Notebook cleanup

- Rerun `notebooks/02_labeling.ipynb` so it shows executed outputs.
- Rerun `notebooks/07_purged_cv.ipynb` because it may still reflect the old 171-row dataset instead of the current 195-row dataset.

## Missing figures for final report

The final report plan expects Stage 4–5 figures:

```text
P12_rf_confusion_matrix.png
P13_purged_cv_scores.png
P14_hyperparameter_search.png
```

These should be saved before the final report.

## Sequential bootstrap validation

The sample-weight notebook previously showed sequential bootstrap uniqueness slightly below standard bootstrap uniqueness. Since we are not using sequential bootstrap directly in training yet, this is not blocking Stages 4–6, but we should avoid claiming it improved uniqueness unless we retest/fix it.

---

# 11. Recommended next steps

## Before Stage 7

1. Push the current branch to GitHub.
2. Open a pull request.
3. Have teammates review Stages 4–6 and the validation fixes.
4. Optionally perform the cleanup tasks listed above.

## Stage 7 — Meta-labeling and bet sizing

Next stage should:

1. Use the final model's directional predictions as the primary model.
2. Create meta-labels:

```text
meta_label = 1 if trade was profitable given predicted side
meta_label = 0 otherwise
```

3. Train a meta-model to predict probability of profit.
4. Convert probabilities into bet sizes.
5. Save:

```text
data/processed/nvda_meta_labels.parquet
data/processed/nvda_positions.parquet
models/model_meta.pkl
```

## Stage 8 — Backtesting

After Stage 7, backtest the generated positions and compute:

- cumulative returns
- Sharpe ratio
- probabilistic Sharpe ratio
- deflated Sharpe ratio
- max drawdown
- time under water
- Calmar ratio
- monthly returns

---

# 12. Bottom line

Stages 0–6 are now structurally aligned with the AFML implementation plan.

The most important fixes were:

1. correcting fractional differentiation so it preserves memory,
2. fixing invalid/constant features,
3. applying sample weights during both fitting and scoring,
4. rerunning Stages 4–6 after those fixes.

The project is now ready to push for teammate review and then continue to Stage 7.
