"""
Training Pipeline — Train LightGBM multi-output weather forecast model.

Steps:
  1. Load & merge historical datasets
  2. Create features (lag, rolling, time, Open-Meteo)
  3. Split into train / validation (last N days for validation)
  4. Train one LightGBM regressor per target variable
  5. Evaluate on validation set (MAE, RMSE, R²)
  6. Feature importance analysis
  7. Save trained model + evaluation report
"""

import json
import logging
import sys
import time
from datetime import timedelta

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    TARGET_COLUMNS,
    LIGHTGBM_PARAMS,
    MODEL_FILE,
    EVAL_REPORT_FILE,
    VALIDATION_DAYS,
    FORECAST_STEPS,
    MODEL_DIR,
)
from data_loader import merge_historical_datasets
from feature_engineering import create_features, get_feature_columns

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def train_model():
    """Full training pipeline."""
    start_time = time.time()

    # ──────────────────────────────────────────────
    # Step 1: Load data
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading historical datasets")
    logger.info("=" * 60)

    df_raw = merge_historical_datasets()
    logger.info("Raw merged data: %d rows", len(df_raw))

    # ──────────────────────────────────────────────
    # Step 2: Feature engineering
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering")
    logger.info("=" * 60)

    df = create_features(df_raw, om_df=None, drop_na=True)
    feature_cols = get_feature_columns(df)
    logger.info("Features: %d columns", len(feature_cols))

    # ──────────────────────────────────────────────
    # Step 3: Train / Validation split
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Train/Validation split (last %d days)", VALIDATION_DAYS)
    logger.info("=" * 60)

    split_date = df["timestamp"].max() - timedelta(days=VALIDATION_DAYS)
    train_mask = df["timestamp"] <= split_date
    val_mask = df["timestamp"] > split_date

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    logger.info("Train: %d rows (%s → %s)",
                len(df_train), df_train["timestamp"].min(), df_train["timestamp"].max())
    logger.info("Val:   %d rows (%s → %s)",
                len(df_val), df_val["timestamp"].min(), df_val["timestamp"].max())

    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values

    # ──────────────────────────────────────────────
    # Step 4: Train models
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Training LightGBM models")
    logger.info("=" * 60)

    models = {}
    eval_results = {}

    for target in TARGET_COLUMNS:
        logger.info("─── Training model for: %s ───", target)

        y_train = df_train[target].values
        y_val = df_val[target].values

        params = LIGHTGBM_PARAMS.copy()
        early_stop = params.pop("early_stopping_rounds", 50)
        n_est = params.pop("n_estimators", 1000)

        model = lgb.LGBMRegressor(
            n_estimators=n_est,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop),
                lgb.log_evaluation(period=100),
            ],
        )

        models[target] = model

        # ──────────────────────────────────────────────
        # Step 5: Evaluate
        # ──────────────────────────────────────────────
        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        eval_results[target] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "best_iteration": model.best_iteration_,
        }

        logger.info("  %s → MAE: %.4f | RMSE: %.4f | R²: %.4f",
                     target, mae, rmse, r2)

    # ──────────────────────────────────────────────
    # Step 6: Feature importance
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Feature importance analysis")
    logger.info("=" * 60)

    importance_report = {}

    for target in TARGET_COLUMNS:
        model = models[target]
        importances = model.feature_importances_
        feat_imp = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        top_10 = feat_imp[:10]
        importance_report[target] = [
            {"feature": name, "importance": int(imp)}
            for name, imp in top_10
        ]
        logger.info("  Top 5 for %s: %s",
                     target,
                     ", ".join(f"{n}={i}" for n, i in top_10[:5]))

    # ──────────────────────────────────────────────
    # Step 7: Save model + report
    # ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Saving model and evaluation report")
    logger.info("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "models": models,
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
        "forecast_steps": FORECAST_STEPS,
        "training_rows": len(df_train),
        "validation_rows": len(df_val),
        "train_date_range": [
            str(df_train["timestamp"].min()),
            str(df_train["timestamp"].max()),
        ],
    }
    joblib.dump(model_bundle, MODEL_FILE)
    logger.info("Model saved → %s", MODEL_FILE)

    report = {
        "evaluation": eval_results,
        "feature_importance": importance_report,
        "dataset_info": {
            "total_rows": len(df),
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "feature_count": len(feature_cols),
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        },
        "training_time_seconds": round(time.time() - start_time, 1),
    }

    with open(EVAL_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved → %s", EVAL_REPORT_FILE)

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)
    for target in TARGET_COLUMNS:
        r = eval_results[target]
        logger.info("  %-15s MAE=%.4f  RMSE=%.4f  R²=%.4f",
                     target, r["mae"], r["rmse"], r["r2"])
    logger.info("Model: %s", MODEL_FILE)
    logger.info("Report: %s", EVAL_REPORT_FILE)

    return models, eval_results


if __name__ == "__main__":
    train_model()
