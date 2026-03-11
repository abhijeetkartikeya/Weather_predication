"""
Data Loader — Load, merge, clean, and normalize all weather data sources.

Sources:
  1. historical_15min_data.csv  (Open-Meteo archive, 15-min, 2015–2026)
  2. nasa_historical_data.csv   (NASA POWER, 15-min, 2015–2026)
  3. live_forecast_data.csv     (Open-Meteo realtime, 15-min)
  4. Open-Meteo Forecast API    (fetched at inference time)
"""

import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    HISTORICAL_CSV,
    NASA_CSV,
    LIVE_FORECAST_CSV,
    LATITUDE,
    LONGITUDE,
    TARGET_COLUMNS,
    PHYSICAL_BOUNDS,
    OPENMETEO_FORECAST_URL,
    OPENMETEO_FORECAST_DAYS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Canonical column set
# ──────────────────────────────────────────────────────────────
CANONICAL_COLS = ["timestamp"] + TARGET_COLUMNS


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to the canonical names."""
    rename_map = {
        "cloudcover": "cloud_cover",
        "cloud_cover": "cloud_cover",
        "CLOUD_AMT": "cloud_cover",
    }
    df = df.rename(columns=rename_map)

    # Ensure all target columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            logger.warning("Column '%s' missing — filling with NaN", col)
            df[col] = np.nan

    # Keep only canonical columns
    keep = [c for c in CANONICAL_COLS if c in df.columns]
    return df[keep].copy()


def _clip_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values to physically plausible ranges."""
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            before_clip = df[col].notna().sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            logger.debug("Clipped '%s' to [%s, %s]", col, lo, hi)
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a single DataFrame: parse timestamps, drop dupes, sort."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# Individual loaders
# ──────────────────────────────────────────────────────────────


def load_historical_csv() -> pd.DataFrame:
    """Load the Open-Meteo archive CSV (primary dataset)."""
    logger.info("Loading %s …", HISTORICAL_CSV)
    df = pd.read_csv(HISTORICAL_CSV)
    df = _standardize_columns(df)
    df = _clean_dataframe(df)
    logger.info("  → %d rows", len(df))
    return df


def load_nasa_csv() -> pd.DataFrame:
    """Load the NASA POWER historical CSV."""
    logger.info("Loading %s …", NASA_CSV)
    df = pd.read_csv(NASA_CSV)
    df = _standardize_columns(df)
    df = _clean_dataframe(df)
    logger.info("  → %d rows", len(df))
    return df


def load_live_csv() -> pd.DataFrame:
    """Load the latest live / forecast CSV from the realtime updater."""
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
# Open-Meteo Forecast API (used as input features)
# ──────────────────────────────────────────────────────────────


def fetch_openmeteo_forecast() -> pd.DataFrame:
    """
    Fetch the latest Open-Meteo forecast for the configured location.
    Returns a DataFrame with columns prefixed 'om_' for use as input features.
    """
    logger.info("Fetching Open-Meteo forecast (next %d days) …", OPENMETEO_FORECAST_DAYS)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "minutely_15": (
            "temperature_2m,windspeed_10m,cloudcover,"
            "shortwave_radiation,diffuse_radiation"
        ),
        "forecast_days": OPENMETEO_FORECAST_DAYS,
        "timezone": "auto",
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
    })

    logger.info("  → %d forecast rows from Open-Meteo", len(df))
    return df


def fetch_openmeteo_recent(days_back: int = 3) -> pd.DataFrame:
    """
    Fetch recent observed weather from Open-Meteo archive API.
    Used as seed data for the forecast service.
    """
    logger.info("Fetching recent %d days from Open-Meteo archive …", days_back)

    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&hourly=temperature_2m,windspeed_10m,cloudcover,"
        f"shortwave_radiation,diffuse_radiation"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Open-Meteo archive API failed: %s", e)
        return pd.DataFrame(columns=CANONICAL_COLS)

    if "hourly" not in data:
        logger.error("Unexpected response: %s", list(data.keys()))
        return pd.DataFrame(columns=CANONICAL_COLS)

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "ghi": hourly["shortwave_radiation"],
        "dhi": hourly["diffuse_radiation"],
        "temperature": hourly["temperature_2m"],
        "windspeed": hourly["windspeed_10m"],
        "cloud_cover": hourly["cloudcover"],
    })

    # Upsample hourly → 15-min
    df = df.set_index("timestamp")
    df = df.resample("15min").interpolate(method="time")
    df = df.reset_index()

    logger.info("  → %d rows (upsampled to 15-min)", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# Merge pipeline
# ──────────────────────────────────────────────────────────────


def merge_historical_datasets() -> pd.DataFrame:
    """
    Load & merge all historical data sources into a single clean DataFrame.
    Priority: historical CSV > NASA CSV (fill gaps) > live CSV (append recent).
    """
    df_hist = load_historical_csv()
    df_nasa = load_nasa_csv()
    df_live = load_live_csv()

    # ----- Merge historical + NASA on timestamp -----
    logger.info("Merging historical + NASA datasets …")
    df = pd.merge(
        df_hist, df_nasa,
        on="timestamp", how="outer",
        suffixes=("", "_nasa"),
    )

    # For each target, fill gaps from NASA
    for col in TARGET_COLUMNS:
        nasa_col = f"{col}_nasa"
        if nasa_col in df.columns:
            df[col] = df[col].fillna(df[nasa_col])
            df = df.drop(columns=[nasa_col])

    # ----- Append live data (recent timestamps) -----
    if len(df_live) > 0:
        max_hist_ts = df["timestamp"].max()
        df_live_new = df_live[df_live["timestamp"] > max_hist_ts]
        if len(df_live_new) > 0:
            logger.info("Appending %d new live rows", len(df_live_new))
            df = pd.concat([df, df_live_new], ignore_index=True)

    # ----- Final cleanup -----
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Resample to exact 15-min grid
    df = df.set_index("timestamp")
    df = df.resample("15min").mean()  # handles any slight misalignment
    df = df.interpolate(method="time", limit=8)  # fill gaps up to 2 hours
    df = df.reset_index()

    # Clip to physical bounds
    df = _clip_physical_bounds(df)

    # Replace remaining NaN with column median (rare edge case)
    for col in TARGET_COLUMNS:
        if df[col].isna().any():
            median_val = df[col].median()
            n_fill = df[col].isna().sum()
            logger.warning("Filling %d remaining NaN in '%s' with median %.2f", n_fill, col, median_val)
            df[col] = df[col].fillna(median_val)

    logger.info("Merged dataset: %d rows, %s → %s",
                len(df), df["timestamp"].min(), df["timestamp"].max())
    return df


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = merge_historical_datasets()
    print("\n=== Merged Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nColumn stats:\n{df[TARGET_COLUMNS].describe().round(2)}")
    print(f"\nNaN counts:\n{df[TARGET_COLUMNS].isna().sum()}")
