"""
Forecast Service — Real-time weather forecast generation.

Workflow:
  1. Fetch latest observed weather (Open-Meteo archive, last 2–3 days)
  2. Fetch Open-Meteo forecast (next 7 days, used as input features)
  3. Load trained LightGBM model
  4. Recursive multi-step prediction for 384 steps (4 days ahead)
  5. Export to forecast_output.csv (ML + Open-Meteo reference side by side)
  6. Insert into PostgreSQL weather_forecast table
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from config import (
    TARGET_COLUMNS,
    FORECAST_STEPS,
    FORECAST_INTERVAL_MIN,
    FORECAST_OUTPUT_CSV,
    MODEL_FILE,
    LAG_STEPS,
    ROLLING_WINDOWS,
    PHYSICAL_BOUNDS,
)
from data_loader import fetch_openmeteo_forecast, fetch_openmeteo_recent
from feature_engineering import (
    add_time_features,
    get_feature_columns,
)
from postgres_writer import insert_forecast

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forecast")


# ──────────────────────────────────────────────────────────────
# Recursive multi-step forecast
# ──────────────────────────────────────────────────────────────


def _build_lag_features(history: pd.DataFrame, col: str) -> dict:
    """Build lag features for a single column from history buffer."""
    features = {}
    vals = history[col].values
    for lag in LAG_STEPS:
        if lag <= len(vals):
            features[f"{col}_lag{lag}"] = vals[-lag]
        else:
            features[f"{col}_lag{lag}"] = vals[0]  # fallback: oldest value
    return features


def _build_rolling_features(history: pd.DataFrame, col: str) -> dict:
    """Build rolling mean/std features for a single column."""
    features = {}
    vals = history[col].values
    for window_name, window_size in ROLLING_WINDOWS.items():
        window = vals[-window_size:] if len(vals) >= window_size else vals
        features[f"{col}_rmean_{window_name}"] = np.mean(window)
        features[f"{col}_rstd_{window_name}"] = np.std(window) if len(window) > 1 else 0.0
    return features


def _build_time_features(ts: pd.Timestamp) -> dict:
    """Build time features for a single timestamp."""
    hour_frac = ts.hour + ts.minute / 60.0
    doy = ts.timetuple().tm_yday
    return {
        "hour_sin": np.sin(2 * np.pi * hour_frac / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_frac / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "month": ts.month,
        "is_daytime": int(6 <= ts.hour < 18),
        "weekday": ts.weekday(),
    }


def _build_om_features(om_df: pd.DataFrame, ts: pd.Timestamp) -> dict:
    """Look up Open-Meteo forecast values for a given timestamp."""
    features = {}
    om_cols_map = {
        "om_ghi": "om_ghi",
        "om_dhi": "om_dhi",
        "om_temperature": "om_temperature",
        "om_windspeed": "om_windspeed",
        "om_cloud_cover": "om_cloud_cover",
    }

    if om_df is not None and not om_df.empty:
        # Find closest timestamp
        mask = om_df["timestamp"] == ts
        if mask.any():
            row = om_df[mask].iloc[0]
            for feat_name, col_name in om_cols_map.items():
                features[feat_name] = row.get(col_name, 0.0)
        else:
            # Nearest neighbor fallback
            idx = (om_df["timestamp"] - ts).abs().idxmin()
            row = om_df.loc[idx]
            for feat_name, col_name in om_cols_map.items():
                features[feat_name] = row.get(col_name, 0.0)
    else:
        for feat_name in om_cols_map:
            features[feat_name] = 0.0

    return features


def recursive_forecast(
    model_bundle: dict,
    seed_data: pd.DataFrame,
    om_df: pd.DataFrame,
    n_steps: int = FORECAST_STEPS,
) -> pd.DataFrame:
    """
    Generate n_steps of recursive weather forecast.

    Args:
        model_bundle: Dict with 'models', 'feature_columns', etc.
        seed_data: Recent observed data with target columns + timestamp.
        om_df: Open-Meteo forecast DataFrame.
        n_steps: Number of steps to forecast.

    Returns:
        DataFrame with timestamp + predicted target columns.
    """
    models = model_bundle["models"]
    feature_cols = model_bundle["feature_columns"]

    # Build history buffer from seed data
    history = seed_data[["timestamp"] + TARGET_COLUMNS].copy()
    history = history.sort_values("timestamp").reset_index(drop=True)

    # Start forecasting from the last known timestamp
    last_ts = history["timestamp"].iloc[-1]
    delta = timedelta(minutes=FORECAST_INTERVAL_MIN)

    predictions = []

    logger.info("Starting recursive forecast: %d steps from %s", n_steps, last_ts)

    for step in range(1, n_steps + 1):
        current_ts = last_ts + delta * step

        # Build feature vector
        feat_dict = {}

        # Lag + rolling features from history
        for col in TARGET_COLUMNS:
            feat_dict.update(_build_lag_features(history, col))
            feat_dict.update(_build_rolling_features(history, col))

        # Time features
        feat_dict.update(_build_time_features(current_ts))

        # Open-Meteo features
        feat_dict.update(_build_om_features(om_df, current_ts))

        # Create feature vector in correct order
        X = np.array([[feat_dict.get(f, 0.0) for f in feature_cols]])

        # Predict each target
        pred_row = {"timestamp": current_ts}
        for target in TARGET_COLUMNS:
            pred = models[target].predict(X)[0]

            # Clip to physical bounds
            lo, hi = PHYSICAL_BOUNDS.get(target, (-np.inf, np.inf))
            pred = np.clip(pred, lo, hi)

            pred_row[target] = pred

        predictions.append(pred_row)

        # Update history buffer (append prediction as new "observed" row)
        new_row = pd.DataFrame([pred_row])
        history = pd.concat([history, new_row], ignore_index=True)

        # Keep buffer manageable (last 200 rows)
        if len(history) > 200:
            history = history.iloc[-200:].reset_index(drop=True)

        if step % 96 == 0:
            logger.info("  Step %d/%d (day %d)", step, n_steps, step // 96)

    df_pred = pd.DataFrame(predictions)
    logger.info("Forecast complete: %d rows", len(df_pred))
    return df_pred


# ──────────────────────────────────────────────────────────────
# Main forecast routine
# ──────────────────────────────────────────────────────────────


def run_forecast():
    """Execute one full forecast cycle."""
    cycle_start = time.time()
    logger.info("=" * 60)
    logger.info("FORECAST CYCLE START — %s", datetime.utcnow().isoformat())
    logger.info("=" * 60)

    # 1. Load model
    logger.info("Loading model from %s …", MODEL_FILE)
    if not MODEL_FILE.exists():
        logger.error("Model file not found! Run train_weather_model.py first.")
        return False

    model_bundle = joblib.load(MODEL_FILE)
    logger.info("Model loaded (%d features, %d targets)",
                len(model_bundle["feature_columns"]),
                len(model_bundle["target_columns"]))

    # 2. Fetch seed data (recent observations)
    seed_data = fetch_openmeteo_recent(days_back=3)
    if seed_data.empty or len(seed_data) < 10:
        logger.error("Insufficient seed data (%d rows) — aborting", len(seed_data))
        return False

    # 3. Fetch Open-Meteo forecast (input features)
    om_df = fetch_openmeteo_forecast()

    # 4. Generate forecast
    df_pred = recursive_forecast(model_bundle, seed_data, om_df, n_steps=FORECAST_STEPS)

    # 5. Build output CSV (ML predictions + Open-Meteo reference)
    df_ml = df_pred.copy()
    df_ml = df_ml.rename(columns={
        "ghi": "GHI_pred",
        "dhi": "DHI_pred",
        "temperature": "temperature_pred",
        "windspeed": "windspeed_pred",
        "cloud_cover": "cloud_cover_pred",
    })
    df_ml["source"] = "ml_model"

    # Add Open-Meteo reference rows
    if om_df is not None and not om_df.empty:
        df_om_ref = om_df.copy()
        # Filter to same time range as predictions
        min_ts = df_pred["timestamp"].min()
        max_ts = df_pred["timestamp"].max()
        df_om_ref = df_om_ref[
            (df_om_ref["timestamp"] >= min_ts) & (df_om_ref["timestamp"] <= max_ts)
        ]
        df_om_ref = df_om_ref.rename(columns={
            "om_ghi": "GHI_pred",
            "om_dhi": "DHI_pred",
            "om_temperature": "temperature_pred",
            "om_windspeed": "windspeed_pred",
            "om_cloud_cover": "cloud_cover_pred",
        })
        df_om_ref["source"] = "openmeteo_reference"
        df_om_ref = df_om_ref[df_ml.columns]  # align columns

        df_output = pd.concat([df_ml, df_om_ref], ignore_index=True)
    else:
        df_output = df_ml

    df_output = df_output.sort_values(["timestamp", "source"]).reset_index(drop=True)

    # 6. Export CSV
    df_output.to_csv(FORECAST_OUTPUT_CSV, index=False)
    logger.info("Forecast CSV → %s (%d rows)", FORECAST_OUTPUT_CSV, len(df_output))

    # 7. Insert into PostgreSQL
    db_df = df_pred.copy()  # Use raw predictions for DB
    success = insert_forecast(db_df, model_version="lgbm_v1")
    if success:
        logger.info("PostgreSQL insert successful")
    else:
        logger.warning("PostgreSQL insert skipped (see logs above)")

    elapsed = time.time() - cycle_start
    logger.info("=" * 60)
    logger.info("FORECAST CYCLE COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)

    return True


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecast Service")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single forecast cycle and exit",
    )
    args = parser.parse_args()

    run_forecast()
