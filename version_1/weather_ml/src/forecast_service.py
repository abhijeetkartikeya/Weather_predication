"""
Forecast Service V2 — 3-day forecast with VEDAS + Open-Meteo + InfluxDB.

Changes from V1:
  - 288 steps (3 days) instead of 384
  - VEDAS forecast as additional input features
  - Writes predictions to InfluxDB for Grafana
  - Interaction features in recursive loop
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
    TARGET_COLUMNS, FORECAST_STEPS, FORECAST_INTERVAL_MIN,
    FORECAST_OUTPUT_CSV, MODEL_FILE,
    LAG_STEPS, ROLLING_WINDOWS, PHYSICAL_BOUNDS,
)
from data_loader import fetch_openmeteo_forecast, fetch_openmeteo_recent, fetch_vedas_forecast
from feature_engineering import get_feature_columns
from postgres_writer import insert_forecast
from influxdb_writer import write_forecast_to_influxdb, write_vedas_to_influxdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forecast_v2")


def _build_lag_features(history, col):
    features = {}
    vals = history[col].values
    for lag in LAG_STEPS:
        features[f"{col}_lag{lag}"] = vals[-lag] if lag <= len(vals) else vals[0]
    return features


def _build_rolling_features(history, col):
    features = {}
    vals = history[col].values
    for wname, wsize in ROLLING_WINDOWS.items():
        window = vals[-wsize:] if len(vals) >= wsize else vals
        features[f"{col}_rmean_{wname}"] = np.mean(window)
        features[f"{col}_rstd_{wname}"] = np.std(window) if len(window) > 1 else 0.0
    return features


def _build_time_features(ts):
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


def _build_interaction_features(pred_row):
    ghi = pred_row.get("ghi", 0)
    dhi = pred_row.get("dhi", 0)
    temp = pred_row.get("temperature", 0)
    wind = pred_row.get("windspeed", 0)
    cloud = pred_row.get("cloud_cover", 0)
    return {
        "ghi_dhi_ratio": ghi / dhi if dhi > 1 else 0.0,
        "temp_wind_interaction": temp * wind / 100.0,
        "cloud_ghi_interaction": cloud * ghi / 1000.0,
    }


def _build_om_features(om_df, ts):
    cols = {"om_ghi": "om_ghi", "om_dhi": "om_dhi", "om_temperature": "om_temperature",
            "om_windspeed": "om_windspeed", "om_cloud_cover": "om_cloud_cover"}
    features = {k: 0.0 for k in cols}
    if om_df is not None and not om_df.empty:
        mask = om_df["timestamp"] == ts
        if mask.any():
            row = om_df[mask].iloc[0]
            for k, c in cols.items():
                features[k] = row.get(c, 0.0)
        else:
            idx = (om_df["timestamp"] - ts).abs().idxmin()
            row = om_df.loc[idx]
            for k, c in cols.items():
                features[k] = row.get(c, 0.0)
    return features


def _build_vedas_features(vedas_df, ts):
    features = {"vedas_temperature": 0.0, "vedas_solar_potential": 0.0}
    if vedas_df is not None and not vedas_df.empty:
        mask = vedas_df["timestamp"] == ts
        if mask.any():
            row = vedas_df[mask].iloc[0]
            features["vedas_temperature"] = row.get("vedas_temperature", 0.0)
            features["vedas_solar_potential"] = row.get("vedas_solar_potential", 0.0)
        else:
            idx = (vedas_df["timestamp"] - ts).abs().idxmin()
            row = vedas_df.loc[idx]
            features["vedas_temperature"] = row.get("vedas_temperature", 0.0)
            features["vedas_solar_potential"] = row.get("vedas_solar_potential", 0.0)
    return features


def recursive_forecast(model_bundle, seed_data, om_df, vedas_df, n_steps=FORECAST_STEPS):
    models = model_bundle["models"]
    feature_cols = model_bundle["feature_columns"]

    history = seed_data[["timestamp"] + TARGET_COLUMNS].copy()
    history = history.sort_values("timestamp").reset_index(drop=True)
    last_ts = history["timestamp"].iloc[-1]
    delta = timedelta(minutes=FORECAST_INTERVAL_MIN)

    predictions = []
    logger.info("Recursive forecast: %d steps from %s", n_steps, last_ts)

    for step in range(1, n_steps + 1):
        current_ts = last_ts + delta * step
        feat_dict = {}

        for col in TARGET_COLUMNS:
            feat_dict.update(_build_lag_features(history, col))
            feat_dict.update(_build_rolling_features(history, col))

        feat_dict.update(_build_time_features(current_ts))
        feat_dict.update(_build_om_features(om_df, current_ts))
        feat_dict.update(_build_vedas_features(vedas_df, current_ts))

        # Predict each target first to build interaction features
        pred_row = {"timestamp": current_ts}
        X = np.array([[feat_dict.get(f, 0.0) for f in feature_cols]])

        for target in TARGET_COLUMNS:
            pred = models[target].predict(X)[0]
            lo, hi = PHYSICAL_BOUNDS.get(target, (-np.inf, np.inf))
            pred_row[target] = np.clip(pred, lo, hi)

        predictions.append(pred_row)

        new_row = pd.DataFrame([pred_row])
        history = pd.concat([history, new_row], ignore_index=True)
        if len(history) > 400:
            history = history.iloc[-400:].reset_index(drop=True)

        if step % 96 == 0:
            logger.info("  Step %d/%d (day %d)", step, n_steps, step // 96)

    return pd.DataFrame(predictions)


def run_forecast():
    cycle_start = time.time()
    logger.info("=" * 60)
    logger.info("V2 FORECAST CYCLE — %s", datetime.utcnow().isoformat())
    logger.info("=" * 60)

    if not MODEL_FILE.exists():
        logger.error("Model not found! Run train_weather_model.py first.")
        return False

    model_bundle = joblib.load(MODEL_FILE)
    logger.info("Model loaded (v=%s, %d features)",
                model_bundle.get("version", "?"), len(model_bundle["feature_columns"]))

    seed_data = fetch_openmeteo_recent(days_back=3)
    if seed_data.empty or len(seed_data) < 10:
        logger.error("Insufficient seed data — aborting")
        return False

    om_df = fetch_openmeteo_forecast()
    vedas_df = fetch_vedas_forecast()

    # Write VEDAS data to InfluxDB for Grafana comparison
    if not vedas_df.empty:
        write_vedas_to_influxdb(vedas_df)

    df_pred = recursive_forecast(model_bundle, seed_data, om_df, vedas_df, n_steps=FORECAST_STEPS)

    # Build CSV output
    df_ml = df_pred.rename(columns={
        "ghi": "GHI_pred", "dhi": "DHI_pred", "temperature": "temperature_pred",
        "windspeed": "windspeed_pred", "cloud_cover": "cloud_cover_pred",
    })
    df_ml["source"] = "ml_model"

    if om_df is not None and not om_df.empty:
        min_ts, max_ts = df_pred["timestamp"].min(), df_pred["timestamp"].max()
        df_om_ref = om_df[(om_df["timestamp"] >= min_ts) & (om_df["timestamp"] <= max_ts)].copy()
        df_om_ref = df_om_ref.rename(columns={
            "om_ghi": "GHI_pred", "om_dhi": "DHI_pred", "om_temperature": "temperature_pred",
            "om_windspeed": "windspeed_pred", "om_cloud_cover": "cloud_cover_pred",
        })
        df_om_ref["source"] = "openmeteo_reference"
        df_om_ref = df_om_ref[df_ml.columns]
        df_output = pd.concat([df_ml, df_om_ref], ignore_index=True)
    else:
        df_output = df_ml

    df_output = df_output.sort_values(["timestamp", "source"]).reset_index(drop=True)
    df_output.to_csv(FORECAST_OUTPUT_CSV, index=False)
    logger.info("CSV → %s (%d rows)", FORECAST_OUTPUT_CSV, len(df_output))

    # PostgreSQL
    insert_forecast(df_pred, model_version="lgbm_v2")

    # InfluxDB
    write_forecast_to_influxdb(df_pred, source="ml_model", model_version="lgbm_v2")

    # Also write Open-Meteo reference to InfluxDB
    if om_df is not None and not om_df.empty:
        om_for_influx = om_df.rename(columns={
            "om_ghi": "ghi", "om_dhi": "dhi", "om_temperature": "temperature",
            "om_windspeed": "windspeed", "om_cloud_cover": "cloud_cover",
        })
        write_forecast_to_influxdb(om_for_influx, source="openmeteo", model_version="api")

    elapsed = time.time() - cycle_start
    logger.info("V2 FORECAST CYCLE COMPLETE in %.1f seconds", elapsed)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecast Service V2")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    args = parser.parse_args()
    run_forecast()
