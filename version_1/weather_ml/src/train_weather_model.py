"""
Training Pipeline V2 — Fine-tuned LightGBM with VEDAS integration.

Enhancements over V1:
  - Higher n_estimators (2000), lower learning rate (0.03), more leaves (255)
  - Extended lag features (t-192, t-288)
  - Interaction features (GHI/DHI ratio, temp×wind)
  - VEDAS + Open-Meteo features
  - Early stopping with patience=100
"""

import json
import logging
import sys
import time
from datetime import timedelta

import joblib
import lightgbm as lgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor

from config import (
    TARGET_COLUMNS, LIGHTGBM_PARAMS, XGBOOST_PARAMS, MODEL_FILE, EVAL_REPORT_FILE,
    VALIDATION_DAYS, FORECAST_STEPS, MODEL_DIR,
)
from data_loader import merge_historical_datasets
from feature_engineering import create_features, get_feature_columns
import mlflow

mlflow.set_tracking_uri(
    "http://localhost:5001"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_v2")


def train_model():
    start_time = time.time()

    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading historical datasets")
    logger.info("=" * 60)
    df_raw = merge_historical_datasets()
    logger.info("Raw merged data: %d rows", len(df_raw))

    # Step 2: Features
    logger.info("=" * 60)
    logger.info("STEP 2: V2 Feature engineering")
    logger.info("=" * 60)
    df = create_features(df_raw, om_df=None, vedas_df=None, drop_na=True)
    feature_cols = get_feature_columns(df)
    logger.info("Features: %d columns", len(feature_cols))

    # Step 3: Split
    logger.info("=" * 60)
    logger.info("STEP 3: Train/Validation split (last %d days)", VALIDATION_DAYS)
    logger.info("=" * 60)
    split_date = df["timestamp"].max() - timedelta(days=VALIDATION_DAYS)
    df_train = df[df["timestamp"] <= split_date].copy()
    df_val = df[df["timestamp"] > split_date].copy()

    logger.info("Train: %d rows (%s → %s)",
                len(df_train), df_train["timestamp"].min(), df_train["timestamp"].max())
    logger.info("Val:   %d rows (%s → %s)",
                len(df_val), df_val["timestamp"].min(), df_val["timestamp"].max())

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]

    # Step 4: Train
    logger.info("=" * 60)
    logger.info("STEP 4: Training fine-tuned LightGBM V2 models")
    logger.info("=" * 60)

    models = {}
    eval_results = {}

    for target in TARGET_COLUMNS:
        logger.info("─── Training: %s ───", target)
        y_train = df_train[target].values
        y_val = df_val[target].values

        params_lgb = LIGHTGBM_PARAMS.copy()
        early_stop = params_lgb.pop("early_stopping_rounds", 100)
        n_est = params_lgb.pop("n_estimators", 2000)

        model_lgb = lgb.LGBMRegressor(n_estimators=n_est, **params_lgb)
        
        params_xgb = XGBOOST_PARAMS.copy()
        early_stop_xgb = params_xgb.pop("early_stopping_rounds", 100)
        n_est_xgb = params_xgb.pop("n_estimators", 2000)
        
        # XGBoost handles early stopping via fit params
        model_xgb = XGBRegressor(n_estimators=n_est_xgb, **params_xgb)

        # We fit individually first so we can use early stopping on validation sets
        logger.info("    -> Fitting LightGBM...")
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        
        logger.info("    -> Fitting XGBoost...")
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        ensemble = VotingRegressor(
            estimators=[('lgb', model_lgb), ('xgb', model_xgb)]
        )
        # We already fit the estimators individually, so we don't need VotingRegressor to refit them.
        # To bypass VotingRegressor.fit(), we just manually set the fitted attributes.
        ensemble.estimators_ = [model_lgb, model_xgb]
        
        models[target] = ensemble

        y_pred = ensemble.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        eval_results[target] = {
            "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "best_iteration_lgb": model_lgb.best_iteration_,
        }
        logger.info("  %s → MAE=%.4f | RMSE=%.4f | R²=%.4f", target, mae, rmse, r2)

    # Step 5: Feature importance
    logger.info("=" * 60)
    logger.info("STEP 5: Feature importance analysis")
    logger.info("=" * 60)

    importance_report = {}
    for target in TARGET_COLUMNS:
        # Get importances from the LightGBM model inside the ensemble
        lgb_model = models[target].estimators_[0]
        importances = lgb_model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        importance_report[target] = [
            {"feature": n, "importance": int(i)} for n, i in feat_imp[:15]
        ]
        logger.info("  Top 5 for %s (LGBM): %s", target,
                     ", ".join(f"{n}={i}" for n, i in feat_imp[:5]))

    # Step 6: Save
    logger.info("=" * 60)
    logger.info("STEP 6: Saving model and report")
    logger.info("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "models": models,
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
        "forecast_steps": FORECAST_STEPS,
        "training_rows": len(df_train),
        "validation_rows": len(df_val),
        "version": "v2",
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
        "model_version": "v2",
    }
    with open(EVAL_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved → %s", EVAL_REPORT_FILE)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("V2 TRAINING COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)
    for target in TARGET_COLUMNS:
        r = eval_results[target]
        logger.info("  %-15s MAE=%.4f  RMSE=%.4f  R²=%.4f", target, r["mae"], r["rmse"], r["r2"])

    return models, eval_results


if __name__ == "__main__":
    train_model()
