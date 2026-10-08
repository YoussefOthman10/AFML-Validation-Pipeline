"""
Meta-labeling module — AFML Chapter 3.6–3.8.

Implements:
  - generate_oos_predictions  : PurgedKFold OOS loop for primary model
  - make_meta_labels          : meta_label = 1 if ret × side > 0 else 0
  - build_meta_feature_matrix : original features ∪ {side}
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from src.cross_validation import PurgedKFold


FEATURE_COLS_15 = [
    "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "rsi_14", "vol_20d", "vol_50d",
    "log_dollar_volume", "volume_ratio",
    "corwin_schultz_spread", "amihud_illiquidity",
    "roll_spread", "shannon_entropy", "lempel_ziv_complexity",
    "fracdiff",
]


def generate_oos_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    clf_params: dict,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate out-of-sample predictions via PurgedKFold loop.

    Every sample receives exactly one prediction from a model that was NOT
    trained on that sample (or any overlapping sample).

    Parameters
    ----------
    X              : feature matrix (n_samples × n_features)
    y              : labels (-1 or +1)
    sample_weight  : per-sample weights from Stage 2
    t1             : barrier end times (index = event times)
    clf_params     : RandomForestClassifier keyword args
    n_splits       : number of PurgedKFold folds
    pct_embargo    : fraction of data to embargo after each test fold
    random_state   : RNG seed for reproducibility

    Returns
    -------
    DataFrame indexed by event time with columns:
        pred_class  : predicted label (-1 or +1)
        pred_prob   : P(y = +1) from predict_proba
        side        : sign(2 * pred_prob - 1) — same as pred_class for RF
        fold        : which fold produced this prediction
    """
    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

    out_class = pd.Series(index=X.index, dtype=float)
    out_prob  = pd.Series(index=X.index, dtype=float)
    out_fold  = pd.Series(index=X.index, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        sw_tr = sample_weight.iloc[train_idx].values

        X_te = X.iloc[test_idx]

        params = {**clf_params, "random_state": random_state}
        clf = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)

        pred_class = clf.predict(X_te)
        pred_proba = clf.predict_proba(X_te)

        # P(y = +1): find column index for class +1
        classes = list(clf.classes_)
        pos_col = classes.index(1) if 1 in classes else classes.index(1.0)

        out_class.iloc[test_idx] = pred_class
        out_prob.iloc[test_idx]  = pred_proba[:, pos_col]
        out_fold.iloc[test_idx]  = fold_idx

    # side = sign of (2p - 1); equivalent to pred_class for any threshold=0.5 RF
    out_side = np.sign(2 * out_prob - 1).replace(0, 1)  # break ties to long

    result = pd.DataFrame({
        "pred_class": out_class,
        "pred_prob":  out_prob,
        "side":       out_side,
        "fold":       out_fold,
    }, index=X.index)

    _validate_oos_predictions(result, y)
    return result


def _validate_oos_predictions(oos: pd.DataFrame, y: pd.Series) -> None:
    n = len(y)
    assert len(oos) == n, f"OOS coverage {len(oos)} ≠ {n}"
    assert oos["pred_prob"].isna().sum() == 0, "NaN in OOS probabilities"
    assert oos["side"].isin([-1.0, 1.0]).all(), "side must be -1 or +1"

    oos_acc = (oos["pred_class"] == y).mean()
    print(f"[OOS validation] n={n}, accuracy={oos_acc:.4f}")
    print(f"  side distribution: {oos['side'].value_counts().to_dict()}")
    print(f"  fold coverage: {oos['fold'].value_counts().sort_index().to_dict()}")


def make_meta_labels(
    oos_preds: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct meta-labels from OOS primary predictions and realized returns.

    meta_label_i = 1  if  ret_i × side_i > 0   (trade profitable)
    meta_label_i = 0  if  ret_i × side_i <= 0  (trade unprofitable / break-even)

    Matches AFML Snippet 3.7: getBins with side applied.

    Parameters
    ----------
    oos_preds : DataFrame with columns [pred_class, pred_prob, side, fold]
    labels    : DataFrame with columns [t1, pt, sl, ret, bin]
                indexed by event time

    Returns
    -------
    DataFrame indexed by event time with columns:
        side, pred_prob, fold, ret, original_label, meta_label
    """
    # Align labels to OOS predictions (both indexed by event time)
    common_idx = oos_preds.index.intersection(labels.index)
    if len(common_idx) < len(oos_preds):
        print(f"[meta-label] Warning: {len(oos_preds) - len(common_idx)} events "
              "not found in labels parquet; dropping them.")

    ret = labels.loc[common_idx, "ret"]
    side = oos_preds.loc[common_idx, "side"]

    # AFML §3.7 — meta-label is 1 if trade in predicted direction was profitable
    aligned_product = ret * side
    meta_label = (aligned_product > 0).astype(int)

    result = pd.DataFrame({
        "side":           side,
        "pred_prob":      oos_preds.loc[common_idx, "pred_prob"],
        "fold":           oos_preds.loc[common_idx, "fold"],
        "ret":            ret,
        "original_label": labels.loc[common_idx, "bin"],
        "meta_label":     meta_label,
    }, index=common_idx)

    dist = result["meta_label"].value_counts().to_dict()
    n = len(result)
    print(f"[meta-label] n={n}, dist={dist}, "
          f"class-1 rate={dist.get(1,0)/n:.4f}")
    return result


def build_meta_feature_matrix(
    modelling_dataset: pd.DataFrame,
    meta_labels: pd.DataFrame,
    feature_cols: list = None,
) -> tuple:
    """
    Build X_meta, y_meta, w_meta, t1_meta for meta-model training.

    Feature set: original 15 features + side (16 features total).
    AFML §3.7 note: do NOT include primary model probability as a feature —
    side already encodes direction; probability would leak model confidence.

    Returns
    -------
    X_meta : (n, 16) DataFrame
    y_meta : (n,) Series, values in {0, 1}
    w_meta : (n,) Series of sample weights
    t1_meta: (n,) Series of barrier end times
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_15

    common_idx = modelling_dataset.index.intersection(meta_labels.index)

    X_base = modelling_dataset.loc[common_idx, feature_cols].copy()
    X_meta = X_base.copy()
    X_meta["side"] = meta_labels.loc[common_idx, "side"].values

    y_meta  = meta_labels.loc[common_idx, "meta_label"].astype(int)
    w_meta  = modelling_dataset.loc[common_idx, "weight"]
    t1_meta = modelling_dataset.loc[common_idx, "t1"]

    print(f"[meta features] X_meta shape: {X_meta.shape}")
    print(f"  feature cols: {list(X_meta.columns)}")
    print(f"  NaN count: {X_meta.isna().sum().sum()}")

    return X_meta, y_meta, w_meta, t1_meta


def generate_oos_meta_predictions(
    X_meta: pd.DataFrame,
    y_meta: pd.Series,
    w_meta: pd.Series,
    t1_meta: pd.Series,
    meta_clf_params: dict = None,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate OOS meta-model predictions via PurgedKFold.

    Returns
    -------
    DataFrame indexed by event time with columns:
        meta_pred_class : 0 or 1
        meta_pred_prob  : P(meta_label = 1)
        meta_fold       : fold index
    """
    if meta_clf_params is None:
        # AFML recommendation: shallow RF to prevent overfitting on small sample
        meta_clf_params = {
            "n_estimators":    100,
            "max_depth":       3,
            "min_samples_leaf": 20,
            "max_features":    "sqrt",
            "class_weight":    "balanced",
        }

    cv = PurgedKFold(n_splits=n_splits, t1=t1_meta, pct_embargo=pct_embargo)

    out_class = pd.Series(index=X_meta.index, dtype=float)
    out_prob  = pd.Series(index=X_meta.index, dtype=float)
    out_fold  = pd.Series(index=X_meta.index, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_meta, y_meta)):
        X_tr = X_meta.iloc[train_idx]
        y_tr = y_meta.iloc[train_idx]
        sw_tr = w_meta.iloc[train_idx].values

        X_te = X_meta.iloc[test_idx]

        params = {**meta_clf_params, "random_state": random_state}
        clf = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)

        pred_class = clf.predict(X_te)
        pred_proba = clf.predict_proba(X_te)

        classes = list(clf.classes_)
        pos_col = classes.index(1)

        out_class.iloc[test_idx] = pred_class
        out_prob.iloc[test_idx]  = pred_proba[:, pos_col]
        out_fold.iloc[test_idx]  = fold_idx

    result = pd.DataFrame({
        "meta_pred_class": out_class,
        "meta_pred_prob":  out_prob,
        "meta_fold":       out_fold,
    }, index=X_meta.index)

    print(f"[OOS meta] n={len(result)}, NaN prob: {result['meta_pred_prob'].isna().sum()}")
    print(f"  meta pred dist: {result['meta_pred_class'].value_counts().to_dict()}")
    return result
