"""
Train one XGBoost model per target variable with STRICT time-based splitting.

APPROACH: Direct Multi-Horizon Training
========================================
Instead of training on single-step (t -> t+1) data and then recursing at
prediction time (which causes predictions to collapse to the mean), we
expand the training set so that for every origin time t and every horizon
h in [1..MAX_HORIZON] there is a training row.  The model learns to predict
directly at any horizon using only real past observations.

Leakage safeguards
-------------------
1. Train/test split is purely chronological — no shuffling.
2. StandardScaler is fit ONLY on training data; test data is transformed
   with the same (frozen) scaler.
3. Validation set = last 14 days of training data (also chronological).
4. Every lag/rolling feature in the expanded dataset references only real
   observations strictly before the origin time — no predicted values ever
   appear in the feature set.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import (
    RAW_CSV,
    FEATURES_CSV,
    TRAIN_CUTOFF,
    TARGET_VARIABLES,
    XGB_PARAMS,
    SCALER_PATH,
    model_path,
)
from feature_engineering import get_feature_columns, build_direct_training_data


def train_models(df: pd.DataFrame | None = None):
    """Train and persist one XGBoost model per target variable.

    Parameters
    ----------
    df : DataFrame, optional
        The base feature DataFrame produced by ``engineer_features()``.
        If *None*, the raw CSV is loaded and direct training data is built
        from scratch.

    Returns
    -------
    (train_df_base, test_df_base) : tuple of DataFrames
        The original (unexpanded) train/test splits, for use by the
        prediction step.
    """
    # ── Load raw data for direct expansion ───────────────────────────────
    if df is None:
        df = pd.read_csv(FEATURES_CSV, parse_dates=["time"])

    # We need the raw data (with time + target columns) for the direct
    # multi-horizon expansion.  Read it from RAW_CSV if available,
    # otherwise fall back to the features CSV which also has the columns.
    raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
    raw = raw.sort_values("time").reset_index(drop=True)

    # ── Split raw data chronologically BEFORE expansion ──────────────────
    train_raw = raw[raw["time"] <= TRAIN_CUTOFF].copy()
    test_raw = raw[raw["time"] > TRAIN_CUTOFF].copy()
    print(f"Raw train: {len(train_raw)} rows  "
          f"({train_raw['time'].min()} -> {train_raw['time'].max()})")
    print(f"Raw test:  {len(test_raw)} rows  "
          f"({test_raw['time'].min()} -> {test_raw['time'].max()})")

    # ── Build direct multi-horizon training data from TRAIN split only ───
    print("\nExpanding training data for direct multi-horizon forecasting ...")
    direct_train = build_direct_training_data(train_raw)

    feature_cols = get_feature_columns()

    # ── Strict time-based split of expanded data ─────────────────────────
    # Validation = last 14 days of origin times in the training data.
    val_cutoff = direct_train["time"].max() - pd.Timedelta(days=14)
    fit_df = direct_train[direct_train["time"] <= val_cutoff].copy()
    val_df = direct_train[direct_train["time"] > val_cutoff].copy()

    print(f"\nExpanded fit set:  {len(fit_df):,} rows")
    print(f"Expanded val set:  {len(val_df):,} rows")

    # ── Scale features ──────────────────────────────────────────────────
    scaler = StandardScaler()
    fit_df[feature_cols] = scaler.fit_transform(fit_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved -> {SCALER_PATH}")

    # ── Train one model per target ──────────────────────────────────────
    for target in TARGET_VARIABLES:
        print(f"\n{'='*60}")
        print(f"Training model for: {target}")
        print(f"{'='*60}")

        model = XGBRegressor(**XGB_PARAMS)

        X_fit = fit_df[feature_cols]
        y_fit = fit_df[target]
        X_val = val_df[feature_cols]
        y_val = val_df[target]

        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validation metrics
        preds_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        r2 = r2_score(y_val, preds_val)
        print(f"  Validation MAE:  {mae:.4f}")
        print(f"  Validation RMSE: {rmse:.4f}")
        print(f"  Validation R2:   {r2:.4f}")

        path = model_path(target)
        joblib.dump(model, path)
        print(f"  Model saved -> {path}")

    print("\nAll models trained and saved.")

    # Return the original base splits for use by prediction.py
    train_base = df[df["time"] <= TRAIN_CUTOFF].copy()
    test_base = df[df["time"] > TRAIN_CUTOFF].copy()
    return train_base, test_base


if __name__ == "__main__":
    train_models()
