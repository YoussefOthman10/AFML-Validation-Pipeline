# AFML NVDA Pipeline — Independent Technical Audit Report

**Auditor Role:** Senior Quantitative Researcher & AFML Implementation Auditor
**Date:** May 11, 2026
**Scope:** Stages 0–6, pipeline integrity, AFML fidelity, readiness for Stages 7–8
**Authoritative Reference:** López de Prado, M. *Advances in Financial Machine Learning*, Wiley (2018)

---

## 1. Executive Summary

This audit evaluates a multi-stage financial machine learning pipeline built on daily OHLCV data for NVIDIA (NVDA), implementing core methodologies from AFML Chapters 2–9. The pipeline spans dataset cleaning (Stage 0) through feature importance analysis (Stage 6), with Stages 7–11 yet to be implemented.

**Overall Assessment: CONDITIONALLY VALID — Proceed with Caution**

The pipeline demonstrates genuine AFML alignment in its structural design and has undergone meaningful self-correction. Five substantive bugs were identified and fixed by the team: a critical fracdiff convolution-order error, a broken Lempel-Ziv feature, negative Corwin-Schultz values, unweighted test-fold scoring, and stale downstream artifacts. These fixes show intellectual honesty and methodological seriousness.

However, several issues remain that require attention before Stage 7:

- **195 samples** is at the lower boundary of viability for 17-feature ML. Variance is high, and results are fragile.
- **Sequential bootstrap anomaly** — reported uniqueness below standard bootstrap — indicates a likely implementation bug.
- **Two notebooks (02, 07) not rerun** after dataset changes — possible stale outputs in the pipeline.
- **Dollar bars are generated but not clearly used** as the primary price series for labeling — the pipeline may be using daily close instead.
- **No CPCV implementation** yet, limiting backtest-path diversity in Stage 8.

The pipeline is safe to proceed to Stage 7 after the blockers identified in Section 9 are resolved.

---

## 2. Comparison Between Original and Updated Plans

### 2.1 Changes Identified

| Component | Original Plan | Updated Implementation | Assessment |
|-----------|--------------|----------------------|------------|
| Fracdiff d* | Not specified (placeholder) | d*=0.25, ADF p=0.0119, corr=0.916 | **FIXED** — was d*=0.20, corr=-0.008 |
| Corwin-Schultz | Raw formula output | Clipped at zero | **REASONABLE** approximation for daily data |
| Lempel-Ziv | Standard implementation | Replaced with LZ-76 + normalization | **FIXED** — was constant=1.0 |
| CV scoring | Weighted fit, unweighted score | Weighted fit, weighted score | **FIXED** — critical consistency correction |
| Dataset size | Not specified | 195 rows × 20 columns | **CONCERN** — small for 17 features |
| Label distribution | Not specified | +1: 114, -1: 81 (no label=0) | **NOTE** — imbalanced, no vertical-barrier-only labels |
| Stages 4–6 artifacts | Initial run | Regenerated after fixes | **CORRECT** — necessary cascade |
| DSR computation | Plan: from strategy returns | Implemented: from CV trial scores | **NOTED** — correctly flagged as not true DSR |

### 2.2 What Was Added

The update introduced several elements not explicit in the original plan:

1. **Four-way validation matrix** for weighted scoring (fit-weight × score-weight combinations) — excellent diagnostic practice.
2. **Explicit majority-class baseline** (0.5846) as the performance bar — a rigorous standard.
3. **Honest reporting** that untuned models fail to beat baseline — aligns with AFML's emphasis on skepticism.
4. **Feature pruning** based on tri-method consensus (MDI+MDA+SFI) — methodologically sound.
5. **Git commit history** documenting the fix sequence — good reproducibility practice.

### 2.3 What Was Removed or Deferred

1. **SADF as a feature** — listed in original plan's feature schema but not in the final 17-feature set. Likely dropped due to sample loss or computational cost. Acceptable if documented.
2. **CPCV** — mentioned in the plan but not implemented. Needed for Stage 8 backtest-path generation.
3. **Sequential bootstrap for training** — noted as not yet used in training despite being implemented.

### 2.4 Were Changes Necessary?

All five fixes were necessary and improve AFML alignment:

- **Fracdiff fix:** Without it, the fracdiff feature was anti-correlated with log price, destroying the core premise of Chapter 5 (stationarity with memory preservation). This was a **pipeline-invalidating** bug.
- **LZ fix:** A constant feature adds noise to the model and wastes a feature slot. Removing it or fixing it was required.
- **CS clipping:** Daily-OHLCV Corwin-Schultz frequently violates the spread-positivity assumption. Clipping is the standard workaround.
- **Weighted scoring:** AFML's sample-weighting philosophy (Chapter 4) extends to evaluation, not just fitting. Unweighted test-fold scoring contradicts the rationale for computing weights in the first place.
- **Artifact regeneration:** Any fix to features or scoring invalidates all downstream artifacts. Not regenerating would mean Stages 4–6 were evaluated on stale data.

---

## 3. Stage-by-Stage Validation (Stages 0–6)

### Stage 0 — Dataset Inspection and Cleaning

**AFML Reference:** Chapter 1 (philosophical foundation), Chapter 2 (data quality)

**Expected Behavior:**
- Load NVDA daily OHLCV (5,114 rows, 2005–2025)
- Validate no nulls, monotonic dates, positive prices/volumes
- Compute Adj Close-based dollar volume
- Flag outlier returns
- Export clean parquet

**Assessment:**

The update document confirms Stage 0 was "completed by teammate" and is not the subject of the validation fixes. The output `nvda_clean.parquet` is the foundation for all subsequent stages.

**Concerns:**
- No evidence of outlier treatment beyond detection. If 5σ returns were left in, they may skew volatility estimates and triple-barrier calibration.
- Adj Close is used for dollar volume — correct per the plan's stated assumptions.

**Verdict: VALID** — Standard data-cleaning stage with appropriate OHLCV limitations acknowledged.

---

### Stage 1 — Financial Data Structures and CUSUM Events

**AFML Reference:** Chapter 2, Snippets 2.4 (CUSUM filter), dollar-bar construction

**Expected Behavior:**
- Construct approximate dollar bars from daily data
- Implement symmetric CUSUM filter on daily log returns
- Calibrate h to produce ~300–600 events
- Output: `nvda_dollar_bars.parquet`, `nvda_cusum_events.parquet`

**Assessment:**

The CUSUM filter on daily log returns is a faithful approximation of Snippet 2.4, adapted for daily frequency. The book uses `E_{t-1}[y_t] = y_{t-1}` in the snippet; the plan uses an expanding-window mean, which is a reasonable alternative.

Dollar bars from daily data are acknowledged as an approximation — each "bar" comprises one or more complete trading days, so intraday granularity is lost. This is acceptable given the data limitation.

**Concerns:**

1. **Dollar bar usage downstream is unclear.** The labeling stage (Stage 2) takes `nvda_clean.parquet` + `nvda_cusum_events.parquet` as input. The plan says labels can be computed on "dollar-bar close OR daily Adj Close + CUSUM events." It is not clear which was actually used. If daily close was used (which appears likely from the data flow), then dollar bars were generated but not consumed — they become decorative. This is acceptable for a daily-data project but should be documented.

2. **CUSUM event count** — the final dataset has 195 labeled samples. Given that CUSUM was calibrated for ~300–600 events, some events were dropped (likely due to NaN from feature lookback windows or label truncation at dataset end). This is expected but the attrition rate should be tracked.

**Verdict: VALID** — Faithful daily-data approximation of Chapter 2 methods. Dollar bars may be underutilized.

---

### Stage 2 — Labeling and Sample Weights

**AFML Reference:** Chapter 3 (Snippets 3.1–3.8), Chapter 4 (Snippets 4.1–4.11)

**Expected Behavior:**
- Compute daily volatility via EWMA (Snippet 3.1)
- Apply triple-barrier method (Snippets 3.2–3.6)
- Compute concurrency and average uniqueness (Snippets 4.1–4.2)
- Implement sequential bootstrap (Snippet 4.5)
- Compute return-attribution weights (Snippet 4.10)
- Apply time decay (Snippet 4.11)
- Output: `nvda_labels.parquet`, `nvda_sample_weights.parquet`

**Assessment:**

The update confirms Stage 2 was "completed by teammate" and not directly modified during the fix cycle. The final label distribution is +1: 114, -1: 81 with no label=0 events.

**Concerns:**

1. **No label=0 events.** The AFML triple-barrier method assigns label=0 when the vertical barrier is hit first (or the sign of return at expiry, depending on implementation). The absence of label=0 events suggests one of:
   - The implementation uses `label = sign(return at first barrier touch)`, assigning +1 or -1 even when the vertical barrier is hit. This matches Snippet 3.5's `getBins` which returns `sign(ret)`.
   - OR the vertical barrier was set too wide relative to pt/sl, so horizontal barriers always fire first.
   Either way, this means the model is a binary classifier (+1/-1), not a ternary one. This is acceptable and common, but means there is no explicit "no-trade" signal from the primary model — that role will fall to meta-labeling in Stage 7.

2. **Sequential bootstrap anomaly.** The update states: "The sample-weight notebook previously showed sequential bootstrap uniqueness slightly below standard bootstrap uniqueness." This **contradicts** the core theoretical result of Chapter 4 (Snippets 4.5–4.8): sequential bootstrap should always yield higher average uniqueness than standard bootstrap. Possible causes:
   - Implementation bug in the indicator matrix or uniqueness computation
   - Edge case with very small sample size
   - Incorrect comparison methodology
   This is a **medium-severity concern** because sequential bootstrap is not yet used in training, but it casts doubt on the correctness of the underlying concurrency/uniqueness calculations that DO feed into sample weights.

3. **Sample weight normalization.** Snippet 4.10 normalizes weights so they sum to n: `out['w'] *= out.shape[0] / out['w'].sum()`. This should be verified in the implementation.

4. **Notebook 02 not rerun.** The update explicitly notes: "Rerun notebooks/02_labeling.ipynb so it shows executed outputs." This means the notebook's visible outputs may be stale, though the saved parquet files should be from the original run (before the Stage 3+ fixes). Since Stage 2 outputs were not directly modified by the fixes, the parquet files should be valid, but the notebook should be rerun for consistency.

**Verdict: PARTIALLY VALID** — Core labeling appears correct. Sequential bootstrap anomaly is a red flag that warrants investigation. Notebook needs rerunning.

---

### Stage 3 — Fractional Differentiation and Feature Engineering

**AFML Reference:** Chapter 5 (Snippets 5.1–5.4), Chapters 17–19 (structural breaks, entropy, microstructure)

**Expected Behavior:**
- Compute FFD on log(Adj Close) with d sweep
- Find minimum d* for ADF stationarity while preserving correlation > 0.9
- Engineer momentum, volatility, volume, microstructure, and entropy features
- Merge all features aligned to event timestamps
- Output: `nvda_fracdiff.parquet`, `nvda_features.parquet`, `nvda_modelling_dataset.parquet`

**Assessment — Fracdiff:**

This was the stage with the most critical bug. The original implementation had d*=0.20 with correlation ≈ -0.008, violating the fundamental premise of Chapter 5.

**Root Cause Analysis:** The FFD weights were applied in the wrong order during convolution. In AFML's Snippet 5.3 (`fracDiff_FFD`), the `getWeights_FFD` function produces weights ordered such that when dotted with the price series `seriesF.loc[loc0:loc1]` (ordered oldest to newest), the present-day weight w₀=1 multiplies the newest observation. If the implementation reversed this convention — applying w₀=1 to the oldest observation — the result would be a nearly uncorrelated (or anti-correlated) series, exactly as observed.

**Post-Fix Result:** d*=0.25, ADF p-value=0.0119, correlation=0.916, n_obs=2310, window=2804. This is consistent with AFML's expectation (Chapter 5, Figure 5.5 shows d*≈0.35 for E-mini S&P 500 futures with correlation ~0.995; a slightly lower d* for a more volatile single stock is plausible). The correlation of 0.916 exceeds the 0.9 threshold and indicates memory is substantially preserved.

**Assessment — Feature Engineering:**

Three feature bugs were found and fixed:

1. **Corwin-Schultz spread (clipped at 0):** The CS formula (AFML Chapter 19) can produce negative α when the two-bar high-low range is smaller than predicted by the model. This is a known limitation with daily data. Clipping at zero is standard practice. However, note that 140/195 values are now exactly zero — the feature has very low variance and may have limited predictive power. This is acceptable but should be noted.

2. **Lempel-Ziv complexity (fixed algorithm):** The original was constant=1.0, meaning the LZ-76 algorithm was not implemented correctly. The fix produces a range of [0.903, 1.467] with 6 unique values. Six unique values across 195 samples is still quite low — this feature has limited granularity. However, it now varies, so it can contribute to the model.

3. **Feature alignment to events:** Features computed on the full daily index are aligned to event timestamps. This is correct as long as features are computed using only past data (no look-ahead). Rolling features inherently use past windows, so this should be safe.

**Final Feature Set (17 features):** ret_5d, ret_10d, ret_20d, ret_60d, momentum_12_1, rsi_14, vol_20d, vol_50d, log_dollar_volume, volume_ratio, corwin_schultz_spread, bekker_parkinson_vol, amihud_illiquidity, roll_spread, shannon_entropy, lempel_ziv_complexity, fracdiff.

**Concerns:**

1. **17 features with 195 samples = ratio of ~11.5 samples per feature.** This is dangerously low. A common rule of thumb for tree-based models is ≥30–50 samples per feature. The risk of overfitting is substantial. The feature pruning in Stage 6 (removing 2 features to get 15) helps marginally but does not resolve the fundamental issue.

2. **Missing features from original plan:** The original modelling dataset schema specified `sadf` (rolling SADF) as a feature. It is absent from the final 17-feature set. This is acceptable if SADF was too computationally expensive or caused excessive sample loss, but should be documented.

3. **NaN-induced sample loss:** The original CUSUM filter likely produced more events than 195. Feature lookback windows (60-day returns, 50-day volatility, 252-day momentum, rolling entropy) cause the first ~252 observations to have NaN values. When these are dropped, the effective sample count falls to 195. This attrition is methodologically correct but painful for sample size.

**Verdict: VALID after fixes** — The fracdiff fix was essential and correctly applied. Feature engineering follows AFML methodology with appropriate daily-data approximations. Sample size is the primary concern.

---

### Stage 4 — Model Training

**AFML Reference:** Chapter 6 (Snippet 6.2), Chapter 7 (Snippet 7.3–7.4)

**Expected Behavior:**
- Train RF and XGBoost classifiers with sample weights
- Evaluate with PurgedKFold CV
- Report accuracy against majority-class baseline

**Assessment:**

The implementation trains both models with purged CV and sample weights, which is the core AFML requirement. Results after the weighted-scoring fix:

| Model | CV Accuracy | Baseline |
|-------|-----------|----------|
| Untuned RF | 0.524 ± 0.081 | 0.585 |
| Untuned XGB | 0.528 ± 0.102 | 0.585 |

Neither untuned model beats baseline. The team reports this honestly, which is itself a positive signal — it shows they are not cherry-picking or inflating results.

**Concerns:**

1. **Weighted accuracy metric.** The update says scoring now uses weighted accuracy. This is unusual — most AFML examples use `neg_log_loss` for scoring (see Snippet 7.4's `cvScore` which defaults to `neg_log_loss`, and Snippet 9.1 which uses `neg_log_loss` for non-binary labels). Weighted accuracy conflates the weighting scheme with the evaluation metric. **Recommendation:** Use `neg_log_loss` as the primary CV scoring metric (as AFML recommends) and report weighted accuracy as a secondary metric.

2. **Full-data refit.** The plan mentions fitting final models on the full data for downstream use. If this model is used for OOS predictions in Stage 7 (meta-labeling), it creates leakage — the model has seen all the data it is now predicting on. Stage 7 must use OOS predictions from purged CV, not from a full-data-fit model.

3. **Missing figures.** P12 (confusion matrix) and P13 (purged CV scores per fold) were noted as missing in the update document. These should be generated for completeness.

**Verdict: VALID** — Correct methodology with honest performance reporting. Scoring metric choice could be improved.

---

### Stage 5 — Hyperparameter Tuning

**AFML Reference:** Chapter 9 (Snippets 9.1–9.3)

**Expected Behavior:**
- Randomized search with PurgedKFold as inner CV
- Log all trials
- Apply DSR-style correction for multiple testing
- Output: `best_params.json`, `tuning_log.parquet`

**Assessment:**

The implementation runs 25 trials each for RF and XGB using purged CV with weighted scoring. Tuned results:

| Model | CV Accuracy | Baseline |
|-------|-----------|----------|
| Tuned RF | 0.628 ± 0.072 | 0.585 |
| Tuned XGB | 0.646 ± 0.080 | 0.585 |

Both now beat baseline, but the margins are thin (4–6 percentage points) and within 1σ of baseline. This is a realistic result for financial ML on daily data.

**DSR-style validation:** The team correctly notes that their "DSR" is computed from CV trial scores (treating each trial's CV accuracy as a "Sharpe-like" measure), not from actual strategy returns. The RF's DSR(CV) = 0.935 and XGB's = 0.659. This is a creative but non-standard application of the DSR concept. The true DSR (Chapter 14) must be computed from strategy returns in Stage 8.

**Concerns:**

1. **25 trials per model** = 50 total trials. With this many trials and only 195 samples, there is a real risk of trial-level overfitting (selection bias). The DSR correction attempts to address this, but 195 samples provide very limited statistical power.

2. **Best XGB max_depth=7** is potentially too deep for 195 samples. With 15–17 features and max_depth=7, individual trees can create highly specific decision boundaries. The regularization (reg_lambda=10.0) helps, but this configuration warrants scrutiny.

3. **No nested CV.** The plan and implementation use a single PurgedKFold for tuning. Ideally, an outer loop would provide a truly uncontaminated estimate of generalization performance. With only 195 samples, nested CV may not be practical, but this limitation should be acknowledged.

**Verdict: PARTIALLY VALID** — Methodology follows AFML Chapter 9, but the small sample size makes tuning results fragile. DSR-style correction is creative but non-standard.

---

### Stage 6 — Feature Importance and Interpretation

**AFML Reference:** Chapter 8 (Snippets 8.2–8.4, 8.8)

**Expected Behavior:**
- Compute MDI, MDA, SFI
- Rank features by all three methods
- Prune consistently weak features
- Retrain on reduced feature set
- Output: `feature_importance.parquet`, `model_final.pkl`

**Assessment:**

All three importance methods were implemented with weighted scoring:

- **MDI** (Snippet 8.2): In-sample metric from RF. Biased toward high-cardinality features but useful for screening.
- **MDA** (Snippet 8.3): OOS permutation importance using purged CV with weighted scoring. This is the most reliable method per AFML.
- **SFI** (Snippet 8.4): Single-feature purged CV scores. Identifies standalone predictive features.

Top features by average rank: amihud_illiquidity, log_dollar_volume, fracdiff, ret_20d, volume_ratio. This is plausible — microstructure features (Amihud) and volume features tend to be informative for daily-frequency models.

Pruned features: momentum_12_1, bekker_parkinson_vol. Both were consistently bottom-ranked across all three methods. Removing them is justified.

Reduced model (15 features): CV accuracy = 0.641 ± 0.093, above baseline. The slight drop from tuned RF (0.628) to reduced RF (0.641) — actually an improvement — is likely within noise given the high variance (0.093 std).

**Concerns:**

1. **MDA with weighted scoring.** The book's Snippet 8.3 uses `log_loss` or `accuracy` without sample weights in the scoring call. The update adds weighted scoring, which is a deliberate improvement over the book's implementation. This is methodologically defensible but deviates from the book.

2. **SFI with 195 samples and 5-fold CV.** Each SFI evaluation trains a model on ~156 samples with a single feature. The signal-to-noise ratio per single feature is extremely low at this sample size. SFI results should be interpreted with caution.

3. **Pruning based on rank consensus.** This is a reasonable heuristic but not a formal statistical test. With only 17 features and 195 samples, the marginal value of removing 2 features is small.

**Verdict: VALID** — Faithful implementation of AFML Chapter 8 with appropriate extensions. Feature importance rankings are plausible.

---

## 4. AFML Fidelity Assessment

### 4.1 Concept-by-Concept Evaluation

| AFML Concept | Chapter | Implementation Status | Fidelity |
|-------------|---------|---------------------|----------|
| Dollar bars | Ch. 2 | Generated but likely not used for labeling | Partial |
| CUSUM filter | Ch. 2, Snippet 2.4 | Implemented on daily log returns | Faithful approximation |
| Triple-barrier labeling | Ch. 3, Snippets 3.2–3.6 | Implemented with pt_sl=[1,1], 10-day vertical | Faithful |
| Daily volatility (EWMA) | Ch. 3, Snippet 3.1 | Implemented with span=50 | Faithful |
| Concurrency counting | Ch. 4, Snippet 4.1 | Implemented | Faithful |
| Average uniqueness | Ch. 4, Snippet 4.2 | Implemented but sequential bootstrap anomaly | Possibly buggy |
| Sequential bootstrap | Ch. 4, Snippet 4.5 | Implemented but not used in training; anomalous results | Not validated |
| Return-attribution weights | Ch. 4, Snippet 4.10 | Implemented | Faithful |
| Time decay | Ch. 4, Snippet 4.11 | Implemented | Faithful |
| FFD fracdiff | Ch. 5, Snippet 5.3 | Fixed — now correct | Faithful |
| Minimum d* finding | Ch. 5, Snippet 5.4 | Fixed — d*=0.25, corr=0.916 | Faithful |
| RF with sample weights | Ch. 6, Snippet 6.2 | Implemented | Faithful |
| PurgedKFold | Ch. 7, Snippet 7.3 | Implemented with embargo | Faithful |
| cvScore with weights | Ch. 7, Snippet 7.4 | Extended to include weighted scoring | Enhanced |
| MDI | Ch. 8, Snippet 8.2 | Implemented | Faithful |
| MDA (weighted) | Ch. 8, Snippet 8.3 | Implemented with weighted scoring | Enhanced |
| SFI | Ch. 8, Snippet 8.4 | Implemented | Faithful |
| Purged hyperparameter search | Ch. 9, Snippet 9.1/9.3 | Implemented as manual randomized search | Faithful |
| CPCV | Ch. 7 | **Not implemented** | Missing |

### 4.2 Daily OHLCV Limitations — Where Approximations Are Acceptable

The following approximations are acceptable given daily data:

- **Dollar bars from daily dollar volume:** Loses intraday granularity but preserves the core idea of information-driven sampling.
- **CUSUM on daily log returns:** The book applies CUSUM to any time series of interest. Daily log returns are a valid input.
- **Triple-barrier on daily closes:** Barriers may be hit intraday but are only evaluated at daily resolution. This introduces noise but does not introduce bias.
- **Corwin-Schultz from daily H/L:** This is the intended use case in the microstructure literature (Corwin and Schultz, 2012). The formula was designed for daily data.
- **Amihud illiquidity from daily data:** This is the standard application (Amihud, 2002).
- **Bekker-Parkinson from daily H/L:** This is a standard daily-frequency volatility estimator.

The following approximations are marginal:

- **Roll spread from daily returns:** Very noisy; the serial covariance of daily returns is dominated by other effects. The feature may not capture spread information reliably.
- **Entropy features from daily returns:** Shannon entropy and LZ complexity have low granularity with daily data. With only 50 observations per rolling window and 10 bins, the entropy estimates are coarse. LZ complexity with only 6 unique values across 195 samples is borderline useful.

---

## 5. Statistical Validity Assessment

### 5.1 Sample Size Analysis

The most significant statistical concern is sample size. With 195 samples and 17 features (15 after pruning):

- **Samples per feature ratio:** 11.5 (pre-pruning) to 13.0 (post-pruning). Standard recommendations for tree-based models suggest ≥30–50.
- **Per-fold test size:** With 5-fold purged CV plus purging and embargo, each test fold has ~35–40 samples. Performance estimates on individual folds have very high variance.
- **Effective degrees of freedom:** After accounting for overlapping labels (concurrency), the effective sample size is further reduced. Average uniqueness < 1.0 means samples share information.

### 5.2 Performance Significance

The tuned RF achieves 0.628 ± 0.072 against a baseline of 0.585. The excess accuracy is 0.043, which is less than 1σ (0.072). A one-sample t-test against baseline would not reject the null at conventional significance levels. This does not mean the model is worthless — it means we cannot statistically distinguish it from random at this sample size.

The tuned XGB at 0.646 ± 0.080 has a slightly larger excess (0.061), still within 1σ.

**Implication for downstream stages:** Meta-labeling (Stage 7) will train on the primary model's OOS predictions. If the primary model has near-random accuracy, the meta-model may struggle to find meaningful signal.

### 5.3 Multiple Testing

With 50 total tuning trials (25 RF + 25 XGB), the DSR-style correction is appropriate. The RF's DSR(CV)=0.935 is reassuring; the XGB's DSR(CV)=0.659 is less so. However, these are computed on CV accuracy, not strategy returns, so their practical meaning is limited.

---

## 6. Leakage and Overfitting Assessment

### 6.1 Identified Leakage Vectors

| Potential Leakage Source | Status | Severity |
|--------------------------|--------|----------|
| PurgedKFold implementation | Appears correct (purge + embargo) | Low risk |
| Feature look-ahead | Rolling features use past windows — no look-ahead | Low risk |
| Full-data model refit | model_final.pkl is fit on all data — MUST NOT be used for OOS predictions in Stage 7 | **HIGH RISK if misused** |
| Fracdiff computed on full history | FFD uses fixed window, not expanding — acceptable | Low risk |
| Volatility target (EWMA) computed on full series | EWMA is backward-looking by construction — acceptable | Low risk |
| Feature standardization | No evidence of train/test leakage via scaling — not mentioned | Check required |

### 6.2 Critical Leakage Risk for Stage 7

The most dangerous leakage vector going forward is the use of `model_final.pkl` for generating primary-model predictions in Stage 7. If the final model (fit on all 195 samples) is used to predict `side` for meta-labeling, those predictions are in-sample and will overstate the primary model's accuracy, contaminating the meta-label.

**Required approach for Stage 7:** Generate OOS predictions via purged CV (as the plan states in Prompt 10, Step 2: "use OOS predictions from purged CV"). Each sample must receive a prediction from a model that was NOT trained on that sample. This is the only leakage-safe approach.

### 6.3 Overfitting Indicators

1. **Untuned models below baseline, tuned models above:** This pattern is consistent with tuning extracting marginal signal, but it could also indicate overfitting to the CV folds.
2. **High CV variance (0.072–0.093):** Individual folds range widely, suggesting the model is sensitive to which samples are in the test set.
3. **max_depth=7 for XGB on 195 samples:** High tree depth relative to sample count is an overfitting risk, though reg_lambda=10.0 provides regularization.

---

## 7. Pipeline Consistency Assessment

### 7.1 Data Flow Verification

| Stage | Input | Output | Consistency |
|-------|-------|--------|-------------|
| 0 → 1 | nvda_clean.parquet | nvda_dollar_bars.parquet, nvda_cusum_events.parquet | ✓ |
| 1 → 2 | nvda_clean.parquet + nvda_cusum_events.parquet | nvda_labels.parquet, nvda_sample_weights.parquet | ✓ |
| 2 → 3 | nvda_clean.parquet + nvda_labels.parquet + nvda_sample_weights.parquet | nvda_fracdiff.parquet, nvda_features.parquet, nvda_modelling_dataset.parquet | ✓ (after fix) |
| 3 → 4 | nvda_modelling_dataset.parquet | model_rf.pkl, model_xgb.pkl, cv_results.parquet | ✓ (regenerated) |
| 4 → 5 | nvda_modelling_dataset.parquet | best_params.json, tuning_log.parquet | ✓ (regenerated) |
| 5 → 6 | nvda_modelling_dataset.parquet + tuned params | feature_importance.parquet, model_final.pkl | ✓ (regenerated) |

### 7.2 Critical Propagation Checks

| Artifact | Propagation | Status |
|----------|-------------|--------|
| t1 (barrier end times) | labels → modelling dataset → PurgedKFold | Must be verified — t1 must be in the modelling dataset and passed to CV splitter |
| sample_weight | weights → modelling dataset → fit() + score() | ✓ (confirmed by weighted scoring fix) |
| Label consistency | labels → modelling dataset → y | ✓ (195 rows, binary +1/-1) |
| Feature alignment | features computed on daily index → aligned to event timestamps | ✓ (if point-in-time alignment was used) |

### 7.3 Stale Artifact Check

| Artifact | Status |
|----------|--------|
| nvda_clean.parquet | Original — unchanged by fixes |
| nvda_cusum_events.parquet | Original — unchanged by fixes |
| nvda_labels.parquet | Original — unchanged by fixes |
| nvda_sample_weights.parquet | Original — unchanged by fixes |
| nvda_fracdiff.parquet | **Regenerated** after fracdiff fix |
| nvda_features.parquet | **Regenerated** after feature fixes |
| nvda_modelling_dataset.parquet | **Regenerated** after feature fixes |
| model_rf.pkl, model_xgb.pkl | **Regenerated** after all fixes |
| cv_results.parquet | **Regenerated** |
| tuning_log.parquet | **Regenerated** |
| feature_importance.parquet | **Regenerated** |
| model_final.pkl | **Regenerated** |
| Notebook 02 (labeling) | **STALE** — needs rerun for display consistency |
| Notebook 07 (purged CV) | **STALE** — may reflect old 171-row dataset |

**The stale notebook outputs (02 and 07) are a concern.** If Notebook 07 reflects a 171-row dataset (mentioned in the update), then the PurgedKFold demonstration in that notebook is inconsistent with the current 195-row dataset. This does not affect the model artifacts (which were regenerated from the correct dataset), but it means the notebook-level narrative is misleading.

---

## 8. Remaining Risks

### 8.1 High-Severity

1. **model_final.pkl leakage in Stage 7.** If used directly for meta-label prediction instead of OOS CV predictions, all downstream results are contaminated.
2. **Sequential bootstrap anomaly.** If the uniqueness/concurrency calculations are buggy, the sample weights fed to all models may be incorrect.
3. **195-sample fragility.** All results are subject to high variance. Small perturbations in the data or methodology could flip conclusions.

### 8.2 Medium-Severity

4. **Notebook 07 reflects old dataset.** The PurgedKFold demonstration may be inconsistent.
5. **No CPCV implementation.** Stage 8 backtest-path generation requires CPCV, which is not yet implemented.
6. **Feature standardization unclear.** If features are not standardized before model training, tree-based models are unaffected, but any distance-based or gradient-based methods downstream could be impacted.
7. **Scoring metric choice.** Weighted accuracy is used instead of AFML's preferred neg_log_loss. This may affect hyperparameter selection.

### 8.3 Low-Severity

8. **Missing SADF feature.** Mentioned in original schema but absent from final features.
9. **Low-granularity features.** CS spread (140/195 zero) and LZ complexity (6 unique values) provide limited information.
10. **Missing figures (P12, P13, P14).** Needed for final report but not blocking.

---

## 9. Required Fixes Before Stage 7

### Blocker — Must Fix

1. **Verify OOS prediction generation mechanism.** Before Stage 7 begins, confirm that the code generates OOS predictions via purged CV (each sample predicted by a model not trained on it). The plan's Prompt 10 specifies this, but it must be implemented correctly. DO NOT use model_final.pkl predictions as the primary model's "side" input.

2. **Investigate and fix sequential bootstrap.** Run the Monte Carlo test from Snippet 4.8 to verify that sequential bootstrap uniqueness > standard bootstrap uniqueness. If it does not, the uniqueness and concurrency calculations need debugging. Since these feed into sample weights, a bug here affects all model results.

3. **Rerun Notebook 07 (purged CV).** Ensure it reflects the current 195-row dataset and that the PurgedKFold splitter is demonstrably correct on the current data.

### Recommended — Should Fix

4. **Rerun Notebook 02 (labeling).** Ensure displayed outputs match the current pipeline state.

5. **Add neg_log_loss as a secondary scoring metric.** AFML's cvScore (Snippet 7.4) defaults to neg_log_loss. Running both weighted accuracy and neg_log_loss provides complementary information.

6. **Document dollar bar usage.** Clarify whether dollar bars were used for any downstream computation or are purely for analysis. If not used, note this in the report as a simplification.

---

## 10. Recommended Architecture for Stage 7–8

### 10.1 Stage 7 — Meta-Labeling and Bet Sizing

**Critical requirements:**

1. **Generate OOS predictions** from the tuned RF (or final model) using PurgedKFold. For each fold, train on train samples, predict on test samples. Concatenate all test predictions to get a full set of OOS predictions. Each sample must have exactly one OOS prediction.

2. **Derive side** from OOS predictions: `side = sign(prediction)` or, for probabilistic models, `side = sign(2*P(y=1) - 1)`.

3. **Apply triple-barrier in the direction of side.** As per Snippet 3.6, set the stop-loss barrier on the side opposite to the prediction. The meta-label is:
   - `meta_label = 1` if the trade was profitable (return × side > 0)
   - `meta_label = 0` otherwise

4. **Train meta-model** (RF or logistic regression) on the same features plus `side` as an additional feature, with `y = meta_label`. Use PurgedKFold for evaluation. The meta-model predicts P(profit | side, features).

5. **Bet sizing:** Convert meta-model probability to position size via `m = 2P - 1`, then `signal = side × |m|`. Use avg_active_signals to handle overlapping events.

**Architecture caution:** The meta-model must NOT use the primary model's probability as a feature — it already gets `side`, which encodes the primary model's direction. Including the raw probability would leak the primary model's confidence into the meta-model in a potentially leaky way.

### 10.2 Stage 8 — Backtesting

**Critical requirements:**

1. **Use OOS positions only.** The position series must be derived entirely from OOS predictions (both primary and meta). No in-sample positions.

2. **Include transaction costs.** The plan specifies 5 bps per trade. This is reasonable for a liquid large-cap stock.

3. **Compute true DSR.** Use the actual strategy returns (not CV accuracy) with `num_trials` = total number of tuning trials + any other model variations tried.

4. **CPCV backtest paths.** Implement CombinatorialPurgedKFold to generate multiple backtest paths and compute the distribution of Sharpe ratios across paths. This provides a more robust performance estimate than a single historical simulation.

5. **Synthetic data validation.** Run the pipeline on synthetic trending and mean-reverting series to verify that the model can detect planted signals. If it cannot, the pipeline has insufficient power.

---

## 11. Final Verdict

### Pipeline AFML Compliance: **SUBSTANTIALLY COMPLIANT**

The pipeline implements the core AFML methodology with fidelity: triple-barrier labeling, sample weighting via concurrency/uniqueness/return-attribution, FFD fractional differentiation, PurgedKFold cross-validation, ensemble models with sample weights, and three-method feature importance. The daily-OHLCV limitations are acknowledged and handled with appropriate approximations.

### Stages 0–6 Trustworthiness: **CONDITIONALLY TRUSTWORTHY**

After the five bug fixes and artifact regeneration, Stages 0–6 produce internally consistent results. The fracdiff fix was critical and correctly applied. The weighted-scoring fix aligned evaluation with training. However, the sequential bootstrap anomaly casts uncertainty on the sample-weight calculations, and the very small sample size (195) means all numerical results should be interpreted with wide confidence intervals.

### Downstream Safety: **PROCEED WITH CAUTION**

Stage 7 can proceed safely IF:
- OOS predictions are generated via purged CV (not from model_final.pkl)
- Sequential bootstrap is investigated and fixed if buggy
- Notebook 07 is rerun on current data

Stage 8 can proceed after Stage 7 IF:
- CPCV is implemented for backtest-path generation
- True DSR is computed from strategy returns
- Synthetic data validation is performed

### What Must Still Be Corrected

| Priority | Item | Effort |
|----------|------|--------|
| BLOCKER | Verify/implement OOS prediction generation for Stage 7 | 2–4 hours |
| BLOCKER | Investigate sequential bootstrap anomaly | 2–4 hours |
| BLOCKER | Rerun Notebook 07 on current dataset | 30 minutes |
| RECOMMENDED | Rerun Notebook 02 | 30 minutes |
| RECOMMENDED | Add neg_log_loss scoring | 1 hour |
| RECOMMENDED | Document dollar bar usage | 30 minutes |

### Confidence in Current Implementation

**Confidence Level: 70%**

The pipeline architecture is sound. The self-correction process demonstrates methodological rigor. The main risks are (a) the sequential bootstrap anomaly, which could invalidate sample weights, (b) the tiny sample size, which limits statistical power, and (c) potential misuse of the full-data-fit model for Stage 7 predictions. If the three blockers are resolved, confidence rises to approximately 85%, bounded above by the inherent limitations of 195-sample daily-OHLCV financial ML.

---

*End of Audit Report*
