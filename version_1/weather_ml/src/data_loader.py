"""
Data Loader V2 — Load, merge, clean all data sources + VEDAS integration.

Sources:
  1. historical_15min_data.csv  (Open-Meteo archive, 15-min, 2015–2026)
  2. nasa_historical_data.csv   (NASA POWER, 15-min, 2015–2026)
  3. live_forecast_data.csv     (Open-Meteo realtime, 15-min)
  4. Open-Meteo Forecast API    (fetched at inference time)
  5. VEDAS Forecast API         (ISRO SAC, 3-day temp + solar potential)
"""

import logging
import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    HISTORICAL_CSV, NASA_CSV, LIVE_FORECAST_CSV,
    LATITUDE, LONGITUDE, TARGET_COLUMNS, PHYSICAL_BOUNDS,
    OPENMETEO_FORECAST_URL, OPENMETEO_FORECAST_DAYS,
    VEDAS_FORECAST_URL,
)

logger = logging.getLogger(__name__)

CANONICAL_COLS = ["timestamp"] + TARGET_COLUMNS


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"cloudcover": "cloud_cover", "CLOUD_AMT": "cloud_cover"}
    df = df.rename(columns=rename_map)
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    keep = [c for c in CANONICAL_COLS if c in df.columns]
    return df[keep].copy()


def _clip_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# Individual loaders
# ──────────────────────────────────────────────────────────────

def load_historical_csv() -> pd.DataFrame:
    logger.info("Loading %s …", HISTORICAL_CSV)
    df = pd.read_csv(HISTORICAL_CSV)
    df = _standardize_columns(df)
    df = _clean_dataframe(df)
    logger.info("  → %d rows", len(df))
    return df


def load_nasa_csv() -> pd.DataFrame:
    logger.info("Loading %s …", NASA_CSV)
    df = pd.read_csv(NASA_CSV)
    df = _standardize_columns(df)
    df = _clean_dataframe(df)
    logger.info("  → %d rows", len(df))
    return df


def load_live_csv() -> pd.DataFrame:
    logger.info("Loading %s …", LIVE_FORECAST_CSV)
    if not LIVE_FORECAST_CSV.exists():
        logger.warning("Live forecast CSV not found — skipping")
        return pd.DataFrame(columns=CANONICAL_COLS)
    df = pd.read_csv(LIVE_FORECAST_CSV)
    df = _standardize_columns(df)
    df = _clean_dataframe(df)
    logger.info("  → %d rows", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# Open-Meteo Forecast API
# ──────────────────────────────────────────────────────────────

def fetch_openmeteo_forecast() -> pd.DataFrame:
    logger.info("Fetching Open-Meteo forecast (next %d days) …", OPENMETEO_FORECAST_DAYS)
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "minutely_15": "temperature_2m,windspeed_10m,cloudcover,shortwave_radiation,diffuse_radiation,relative_humidity_2m,surface_pressure",
        "forecast_days": OPENMETEO_FORECAST_DAYS, "timezone": "auto",
    }
    try:
        resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Open-Meteo API failed: %s", e)
        return pd.DataFrame()

    if "minutely_15" not in data:
        logger.error("Unexpected API response: %s", list(data.keys()))
        return pd.DataFrame()

    m15 = data["minutely_15"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(m15["time"]),
        "om_ghi": m15["shortwave_radiation"],
        "om_dhi": m15["diffuse_radiation"],
        "om_temperature": m15["temperature_2m"],
        "om_windspeed": m15["windspeed_10m"],
        "om_cloud_cover": m15["cloudcover"],
        "om_relative_humidity": m15["relative_humidity_2m"],
        "om_surface_pressure": m15["surface_pressure"],
    })
    logger.info("  → %d forecast rows from Open-Meteo", len(df))
    return df


def fetch_openmeteo_recent(days_back: int = 3) -> pd.DataFrame:
    logger.info("Fetching recent %d days from Open-Meteo archive …", days_back)
    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,windspeed_10m,cloudcover,shortwave_radiation,diffuse_radiation,relative_humidity_2m,surface_pressure"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Open-Meteo archive API failed: %s", e)
        return pd.DataFrame(columns=CANONICAL_COLS)

    if "hourly" not in data:
        return pd.DataFrame(columns=CANONICAL_COLS)

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "ghi": hourly["shortwave_radiation"],
        "dhi": hourly["diffuse_radiation"],
        "temperature": hourly["temperature_2m"],
        "windspeed": hourly["windspeed_10m"],
        "cloud_cover": hourly["cloudcover"],
        "relative_humidity": hourly["relative_humidity_2m"],
        "surface_pressure": hourly["surface_pressure"],
    })
    df = df.set_index("timestamp").resample("15min").interpolate(method="time").reset_index()
    logger.info("  → %d rows (upsampled to 15-min)", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# VEDAS Forecast API (ISRO SAC)
# ──────────────────────────────────────────────────────────────

def _parse_vedas_time(time_str: str, year: int = None) -> pd.Timestamp:
    """Parse VEDAS time format like '10Mar:05.30' into a proper datetime."""
    if year is None:
        year = datetime.utcnow().year
    match = re.match(r"(\d{1,2})(\w{3}):(\d{2})\.(\d{2})", time_str)
    if not match:
        return pd.NaT
    day, month_str, hour, minute = match.groups()
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    month = month_map.get(month_str, 1)
    try:
        return pd.Timestamp(year=year, month=month, day=int(day),
                            hour=int(hour), minute=int(minute))
    except ValueError:
        return pd.NaT


def fetch_vedas_forecast() -> pd.DataFrame:
    """
    Fetch 3-day forecast from VEDAS (ISRO SAC).
    Returns DataFrame with: timestamp, vedas_temperature, vedas_solar_potential
    """
    logger.info("Fetching VEDAS forecast …")
    try:
        resp = requests.get(VEDAS_FORECAST_URL, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("VEDAS API failed: %s", e)
        return pd.DataFrame()

    if isinstance(data, list) and len(data) > 0:
        record = data[0]
    elif isinstance(data, dict):
        record = data
    else:
        logger.error("Unexpected VEDAS response format")
        return pd.DataFrame()

    time_list = record.get("timeList", [])
    temp_list = record.get("temprList", [])
    potential_list = record.get("potentialList", [])

    if not time_list:
        logger.warning("VEDAS returned empty timeList")
        return pd.DataFrame()

    # Parse timestamps
    timestamps = [_parse_vedas_time(t) for t in time_list]

    # Build DataFrame
    df = pd.DataFrame({"timestamp": timestamps})
    if temp_list and len(temp_list) == len(timestamps):
        df["vedas_temperature"] = [float(v) if v is not None else np.nan for v in temp_list]
    if potential_list and len(potential_list) == len(timestamps):
        df["vedas_solar_potential"] = [float(v) if v is not None else np.nan for v in potential_list]

    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    logger.info("  → %d rows from VEDAS (temp + solar potential)", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# Merge pipeline
# ──────────────────────────────────────────────────────────────

def merge_historical_datasets() -> pd.DataFrame:
    """Load & merge all historical data sources into a single clean DataFrame."""
    df_hist = load_historical_csv()
    df_nasa = load_nasa_csv()
    df_live = load_live_csv()

    logger.info("Merging historical + NASA datasets …")
    df = pd.merge(df_hist, df_nasa, on="timestamp", how="outer", suffixes=("", "_nasa"))
    for col in TARGET_COLUMNS:
        nasa_col = f"{col}_nasa"
        if nasa_col in df.columns:
            df[col] = df[col].fillna(df[nasa_col])
            df = df.drop(columns=[nasa_col])

    if len(df_live) > 0:
        max_hist_ts = df["timestamp"].max()
        df_live_new = df_live[df_live["timestamp"] > max_hist_ts]
        if len(df_live_new) > 0:
            logger.info("Appending %d new live rows", len(df_live_new))
            df = pd.concat([df, df_live_new], ignore_index=True)

    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp").resample("15min").mean()
    df = df.interpolate(method="time", limit=8)
    df = df.reset_index()
    df = _clip_physical_bounds(df)

    for col in TARGET_COLUMNS:
        if df[col].isna().any():
            median_val = df[col].median()
            n_fill = df[col].isna().sum()
            logger.warning("Filling %d NaN in '%s' with median %.2f", n_fill, col, median_val)
            df[col] = df[col].fillna(median_val)

    logger.info("Merged dataset: %d rows, %s → %s",
                len(df), df["timestamp"].min(), df["timestamp"].max())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = merge_historical_datasets()
    print(f"\nShape: {df.shape}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\n{df[TARGET_COLUMNS].describe().round(2)}")
