"""
Prediction Module for Weather Forecasting Pipeline
=====================================================
Loads trained models and generates 3-day weather forecasts.

Usage:
    python predict.py                                # Full 3-day forecast
    python predict.py --targets temperature_2m       # Single target
    python predict.py --output my_forecast.csv       # Custom output
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_RESOLUTION,
    ENSEMBLE_WEIGHTS,
    FORECAST_HORIZONS,
    FORECAST_OUTPUT_PATH,
    LATITUDE,
    LONGITUDE,
    MODELS_DIR,
    OPENMETEO_FORECAST_URL,
    OPENMETEO_PARAMETERS,
    STEPS_PER_HOUR,
    TARGET_COLUMNS,
    TARGET_DISPLAY_NAMES,
)
from feature_engineering import build_features
from utils import api_request_with_retry, setup_logger

logger = setup_logger("predict")


# ═══════════════════════════════════════════════
# Fetch Recent Data
# ═══════════════════════════════════════════════

def fetch_recent_data(past_days: int = 10) -> pd.DataFrame:
    """
    Fetch recent weather data from Open-Meteo Forecast API.

    We need enough past data for lag/rolling features (up to 72h).
    Fetching 10 days of past data provides a safe margin.

    Parameters
    ----------
    past_days : int
        Number of past days to fetch.

    Returns
    -------
    pd.DataFrame
        Recent weather data with DatetimeIndex (hourly).
    """
    if past_days < 10:
        past_days = 10  # Ensure we have enough for lag features

    logger.info(f"Fetching recent {past_days}-day data from Open-Meteo API ...")

    params_str = ",".join(OPENMETEO_PARAMETERS)

    url = (
        f"{OPENMETEO_FORECAST_URL}?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly={params_str}"
        f"&past_days={past_days}"
        f"&forecast_days=1"
        f"&timezone=auto"
    )

    data = api_request_with_retry(url, logger=logger)

    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo returned no hourly data: {data}")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Resample to 15-min intervals
    df = df.resample(DATA_RESOLUTION).interpolate(method="linear")

    # Filter out future timestamps so we only operate on strict actuals up to now
    # Handle timezone awareness matching
    now_local = pd.Timestamp.now().tz_localize(df.index.tz) if df.index.tz else pd.Timestamp.now()
    df = df[df.index <= now_local]

    logger.info(f"Fetched & resampled to 15-min: {len(df):,} rows: {df.index.min()} → {df.index.max()}")

    return df


# ═══════════════════════════════════════════════
# Load Models
# ═══════════════════════════════════════════════

def load_models(target: str, horizon: str) -> tuple:
    """
    Load LightGBM and XGBoost models for a target × horizon.

    Returns
    -------
    (lgb_model, xgb_model, feature_names) : tuple
    """
    prefix = f"{target}_{horizon}"
    lgb_path = os.path.join(MODELS_DIR, f"{prefix}_lgb.joblib")
    xgb_path = os.path.join(MODELS_DIR, f"{prefix}_xgb.joblib")
    feat_path = os.path.join(MODELS_DIR, f"{prefix}_features.json")

    if not os.path.exists(lgb_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Model files not found for {prefix}. Run train_model.py first.")

    lgb_model = joblib.load(lgb_path)
    xgb_model = joblib.load(xgb_path)

    with open(feat_path, "r") as f:
        feature_names = json.load(f)

    return lgb_model, xgb_model, feature_names


# ═══════════════════════════════════════════════
# Generate Forecast
# ═══════════════════════════════════════════════

def generate_forecast(
    targets: list = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate a 3-day weather forecast.

    For each target × horizon, loads the trained model, builds features
    from recent data, and makes predictions.

    Parameters
    ----------
    targets : list, optional
        Subset of TARGET_COLUMNS. Default: all.
    output_path : str, optional
        Output CSV path. Default: from config.

    Returns
    -------
    pd.DataFrame
        Forecast table with columns: target, horizon, forecast_time, predicted_value
    """
    targets = targets or TARGET_COLUMNS
    output_path = output_path or FORECAST_OUTPUT_PATH

    # Fetch recent data
    recent_df = fetch_recent_data(past_days=10)

    now = datetime.now()
    forecasts = []

    for target in targets:
        if target not in recent_df.columns:
            logger.warning(f"Target {target} not in recent data, skipping.")
            continue

        for horizon_name, horizon_steps in FORECAST_HORIZONS.items():
            logger.info(f"Predicting: {TARGET_DISPLAY_NAMES.get(target, target)} @ {horizon_name}")

            try:
                lgb_model, xgb_model, feature_names = load_models(target, horizon_name)
            except FileNotFoundError as e:
                logger.warning(f"  {e}")
                continue

            # Build features (we only need the last available row for prediction)
            try:
                X, _ = build_features(recent_df.copy(), target, horizon_steps, is_training=False)
            except Exception as e:
                logger.error(f"  Feature engineering failed: {e}")
                continue

            if len(X) == 0:
                logger.warning(f"  No valid samples after feature engineering.")
                continue

            # Align features with training feature names
            missing_feats = [f for f in feature_names if f not in X.columns]
            for mf in missing_feats:
                X[mf] = 0  # Fill missing features with 0
            X = X[feature_names]

            # Use the most recent rows for prediction at 15-min intervals
            last_rows = X.tail(96 * 3)  # 3 days of 15-min predictions

            lgb_pred = lgb_model.predict(last_rows)
            xgb_pred = xgb_model.predict(last_rows)
            ens_pred = (
                ENSEMBLE_WEIGHTS["lgbm"] * lgb_pred
                + ENSEMBLE_WEIGHTS["xgb"] * xgb_pred
            )

            for i, (idx, pred) in enumerate(zip(last_rows.index, ens_pred)):
                # horizon_steps is in 15-min steps, convert to timedelta
                forecast_time = idx + timedelta(minutes=horizon_steps * 15)

                # Stitch them together into a continuous 3-day forecast:
                # 24h model provides from: now to now+24h
                # 48h model provides from: now+24h to now+48h
                # 72h model provides from: now+48h to now+72h
                
                # 'now' is considered the last timestamp in our recent_df (latest actual)
                last_actual_time = recent_df.index.max()
                
                is_valid = False
                if horizon_steps == 96: # 24h
                    is_valid = last_actual_time < forecast_time <= last_actual_time + timedelta(hours=24)
                elif horizon_steps == 192: # 48h
                    is_valid = last_actual_time + timedelta(hours=24) < forecast_time <= last_actual_time + timedelta(hours=48)
                elif horizon_steps == 288: # 72h
                    is_valid = last_actual_time + timedelta(hours=48) < forecast_time <= last_actual_time + timedelta(hours=72)
                
                if is_valid:
                    forecasts.append({
                        "base_time": idx,
                        "forecast_time": forecast_time,
                        "target": target,
                        "target_display": TARGET_DISPLAY_NAMES.get(target, target),
                        "horizon": "continuous_3d",
                        "horizon_steps": horizon_steps,
                        "predicted_value": round(pred, 2),
                    })

    forecast_df = pd.DataFrame(forecasts)

    if len(forecast_df) > 0:
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Forecast saved: {output_path}")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("  3-DAY WEATHER FORECAST SUMMARY")
        logger.info("=" * 70)

        for target in targets:
            tdf = forecast_df[forecast_df["target"] == target]
            if tdf.empty:
                continue
            display = TARGET_DISPLAY_NAMES.get(target, target)
            logger.info(f"\n  {display}:")
            
            # Print a few key points from the continuous 3d forecast
            for hours_ahead in [24, 48, 72]:
                # Find closest point
                last_actual = recent_df.index.max()
                target_time = last_actual + timedelta(hours=hours_ahead)
                
                # Get closest forecast
                if not tdf.empty:
                    tdf_sorted = tdf.copy()
                    tdf_sorted['time_diff'] = abs(tdf_sorted['forecast_time'] - target_time)
                    closest = tdf_sorted.loc[tdf_sorted['time_diff'].idxmin()]
                    
                    logger.info(
                        f"    ~{hours_ahead}h ahead (horizon={closest['horizon_steps']/4:.0f}h model) → {closest['predicted_value']:.1f} "
                        f"(forecast for {closest['forecast_time']})"
                    )

        logger.info("\n" + "=" * 70)
    else:
        logger.warning("No forecasts generated. Are models trained?")

    return forecast_df


# ═══════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate weather forecasts")
    parser.add_argument("--targets", nargs="+", default=None, help="List of targets to forecast")
    parser.add_argument("--output", type=str, default=FORECAST_OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    generate_forecast(targets=args.targets, output_path=args.output)


if __name__ == "__main__":
    main()
