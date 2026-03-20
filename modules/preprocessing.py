"""
preprocessing.py — Automotive Decision Intelligence Platform

Provides:
  load_dataset()         — CSV / Excel ingestion
  build_pipeline()       — sklearn Pipeline (impute → scale) for a given model
  detect_outliers()      — IQR-based outlier flags + Winsorisation
  add_missing_indicators() — binary flags for missing values before imputation

Pipeline design
---------------
Using sklearn.pipeline.Pipeline guarantees the scaler is ALWAYS fit only on
the training fold — never on validation or test data. This eliminates the
data-leakage bug that exists when fit_transform() is called on the full
dataset before the train/test split.

The Pipeline is passed directly into RandomizedSearchCV, so inside every CV
fold the sequence is:
    fit:       imputer.fit(X_train_fold) → scaler.fit(X_train_fold_imputed)
    transform: applied to X_val_fold using training-fold statistics only

Scaling choice
--------------
RobustScaler (median + IQR) instead of StandardScaler (mean + std).
Factory sensor data regularly contains spike readings that inflate std and
shift the mean, distorting the StandardScaler transformation for all other
samples. RobustScaler is unaffected by values outside the IQR.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(uploaded_file):
    """Load CSV or Excel file. Raises ValueError for unsupported formats."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "unknown"
        raise ValueError(
            f"Unsupported file format: .{ext}\n\n"
            "Only CSV (.csv) and Excel (.xlsx, .xls) files are accepted.\n"
            "If your data is in another format, export it to CSV first."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def add_missing_indicators(df):
    """
    For every numeric column that has at least one missing value, add a binary
    indicator column (col_name + '_was_missing') before imputation.

    A missing sensor reading in factory data often means the sensor failed —
    itself a useful signal that the model should see. Imputing the value fills
    in a plausible number, but without the indicator the model never knows the
    original value was absent.

    Only adds columns where missingness actually exists (avoids bloat on clean
    datasets where no values are missing).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER HANDLING — PURE WINSORISATION
# ─────────────────────────────────────────────────────────────────────────────
def detect_and_winsorise_outliers(df, target_col=None):
    """
    IQR-based Winsorisation — caps outlier values at the fence, no extra columns.

    Why pure Winsorisation (no flag columns, no row removal):
    ──────────────────────────────────────────────────────────
    Flag columns:  one binary column per outlier-containing feature adds
                   significant bloat for minimal gain. On clean data every flag
                   is all-zeros. Even on dirty data, adding a full column for
                   a handful of outlier rows is disproportionate.

    Row removal:   discards the entire row because one sensor reading was extreme.
                   The other 7–10 features on that row are perfectly valid and
                   throwing them away loses training data unnecessarily.

    Winsorisation: keeps every row, keeps every column. Simply replaces the
                   extreme value with the fence value (Q1 − 1.5×IQR or
                   Q3 + 1.5×IQR). The model sees a capped, realistic value
                   rather than a spike that would distort scaling. No feature
                   count increase, no data loss.

    Algorithm:
      1. Compute Q1, Q3, IQR per numeric feature column (target excluded)
      2. Lower fence = Q1 − 1.5 × IQR
      3. Upper fence = Q3 + 1.5 × IQR
      4. Clip values to [lower_fence, upper_fence]  — that's it

    Returns
    -------
    df_out         : DataFrame with outlier values capped (same shape as input)
    outlier_report : dict  col → {"n_capped": int, "lower": float, "upper": float}
                     Only contains entries for columns where capping occurred.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)

    outlier_report = {}

    for col in numeric_cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue   # constant column — skip

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col]  = df[col].clip(lower=lower, upper=upper)

        if n_capped > 0:
            outlier_report[col] = {
                "n_capped":   n_capped,
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
            }

    return df, outlier_report


# ─────────────────────────────────────────────────────────────────────────────
# ORDINAL ENCODING FOR CLASS TARGETS
# ─────────────────────────────────────────────────────────────────────────────
def encode_target(y):
    """
    Encode ordinal string labels to integers preserving the natural order.

    Low=0, Medium=1, High=2

    The ordering matters: 'Low' and 'High' are further apart than 'Low' and
    'Medium'. Using OrdinalEncoder instead of LabelEncoder makes this explicit.

    Returns encoded array and the fitted encoder (for inverse_transform later).
    """
    enc = OrdinalEncoder(
        categories=[["Low", "Medium", "High"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    y_encoded = enc.fit_transform(y.values.reshape(-1, 1)).ravel().astype(int)
    return y_encoded, enc


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessing_pipeline(model):
    """
    Build a sklearn Pipeline that:
      1. Imputes missing values with the column median  (SimpleImputer)
      2. Removes near-zero-variance features  (VarianceThreshold, threshold=0.01)
      3. Scales with RobustScaler  (median + IQR, robust to sensor spikes)
      4. Applies the model

    The Pipeline is designed to be passed directly into RandomizedSearchCV so
    that steps 1–3 are always fit only on the training fold — never on
    validation or test data.

    Parameters
    ----------
    model : any sklearn-compatible estimator

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("imputer",   SimpleImputer(strategy="median")),
        ("var_thresh",VarianceThreshold(threshold=0.01)),
        ("scaler",    RobustScaler()),
        ("model",     model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY COMPAT — keep old preprocess() signature so nothing else breaks
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    """
    Legacy function retained for backward compatibility with Dashboard.py
    and other callers that don't go through the training pipeline.

    For training, use build_preprocessing_pipeline() instead — it prevents
    data leakage by fitting the scaler inside each CV fold.
    """
    df = df.copy()
    non_numeric  = df.select_dtypes(exclude=["float64", "int64"]).copy()
    numeric_df   = df.select_dtypes(include=["float64", "int64"]).copy()

    imputer = SimpleImputer(strategy="median")
    cleaned = pd.DataFrame(
        imputer.fit_transform(numeric_df), columns=numeric_df.columns
    )

    scaler = RobustScaler()   # upgraded from StandardScaler
    scaled = pd.DataFrame(
        scaler.fit_transform(cleaned), columns=numeric_df.columns
    )

    cleaned_full = pd.concat([cleaned, non_numeric.reset_index(drop=True)], axis=1)
    return cleaned_full, scaled, imputer, scaler