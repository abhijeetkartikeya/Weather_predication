"""
Feature Engineering V2 — Enhanced features with VEDAS integration.

Features:
  1. Lag features (extended: t-192, t-288 for multi-day patterns)
  2. Rolling statistics (added 12-hour window)
  3. Cyclical time features
  4. Open-Meteo forecast alignment
  5. VEDAS forecast alignment (temperature + solar potential)
  6. Interaction features (GHI/DHI ratio, temp-wind)
"""

import logging
import numpy as np
import pandas as pd

from config import TARGET_COLUMNS, LAG_STEPS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding lag features (steps: %s) …", LAG_STEPS)
    for col in TARGET_COLUMNS:
        for lag in LAG_STEPS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding rolling features …")
    for col in TARGET_COLUMNS:
        for window_name, window_size in ROLLING_WINDOWS.items():
            df[f"{col}_rmean_{window_name}"] = (
                df[col].rolling(window=window_size, min_periods=1).mean()
            )
            df[f"{col}_rstd_{window_name}"] = (
                df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
            )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding time features …")
    ts = pd.to_datetime(df["timestamp"])
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    doy = ts.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = ts.dt.month
    df["is_daytime"] = ((ts.dt.hour >= 6) & (ts.dt.hour < 18)).astype(int)
    df["weekday"] = ts.dt.weekday
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived interaction features."""
    logger.info("Adding interaction features …")
    # GHI/DHI ratio (solar beam fraction)
    df["ghi_dhi_ratio"] = np.where(
        df["dhi"] > 1, df["ghi"] / df["dhi"], 0.0
    )
    # Temperature × windspeed interaction (wind chill proxy)
    df["temp_wind_interaction"] = df["temperature"] * df["windspeed"] / 100.0
    # Cloud × GHI interaction
    df["cloud_ghi_interaction"] = df["cloud_cover"] * df["ghi"] / 1000.0
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features like EMA and differences."""
    logger.info("Adding advanced features (EMA, Differences) …")
    for col in TARGET_COLUMNS:
        # Exponential Moving Averages (EMA)
        for span in [4, 12, 24, 48]: # 1h, 3h, 6h, 12h
            df[f"{col}_ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()
        
        # Difference features (Current vs Lagged)
        if f"{col}_lag4" in df.columns:
            df[f"{col}_diff_1h"] = df[col] - df[f"{col}_lag4"]
        if f"{col}_lag96" in df.columns:
            df[f"{col}_diff_24h"] = df[col] - df[f"{col}_lag96"]
            
    return df


def add_openmeteo_features(df: pd.DataFrame, om_df: pd.DataFrame) -> pd.DataFrame:
    if om_df is None or om_df.empty:
        logger.warning("No Open-Meteo data — filling with zeros")
        for col in TARGET_COLUMNS:
            df[f"om_{col}"] = 0.0
        return df

    logger.info("Merging %d Open-Meteo forecast rows …", len(om_df))
    om_df = om_df.copy()
    om_df["timestamp"] = pd.to_datetime(om_df["timestamp"])
    om_cols = ["timestamp"] + [c for c in om_df.columns if c.startswith("om_")]
    om_df = om_df[om_cols]
    df = df.merge(om_df, on="timestamp", how="left")

    for col in [c for c in df.columns if c.startswith("om_")]:
        df[col] = df[col].ffill(limit=4).fillna(0)
    return df


def add_vedas_features(df: pd.DataFrame, vedas_df: pd.DataFrame) -> pd.DataFrame:
    """Merge VEDAS forecast features (temperature + solar potential)."""
    if vedas_df is None or vedas_df.empty:
        logger.warning("No VEDAS data — filling with zeros")
        df["vedas_temperature"] = 0.0
        df["vedas_solar_potential"] = 0.0
        return df

    logger.info("Merging %d VEDAS forecast rows …", len(vedas_df))
    vedas_df = vedas_df.copy()
    vedas_df["timestamp"] = pd.to_datetime(vedas_df["timestamp"])
    vedas_cols = ["timestamp"] + [c for c in vedas_df.columns if c.startswith("vedas_")]
    vedas_df = vedas_df[[c for c in vedas_cols if c in vedas_df.columns]]
    df = df.merge(vedas_df, on="timestamp", how="left")

    for col in [c for c in df.columns if c.startswith("vedas_")]:
        df[col] = df[col].ffill(limit=4).fillna(0)
    return df


def create_features(
    df: pd.DataFrame,
    om_df: pd.DataFrame = None,
    vedas_df: pd.DataFrame = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Full feature pipeline: lags → rolling → time → interactions → Open-Meteo → VEDAS."""
    logger.info("Running V2 feature engineering pipeline …")
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_interaction_features(df)
    df = add_advanced_features(df)
    df = add_openmeteo_features(df, om_df)
    df = add_vedas_features(df, vedas_df)

    if drop_na:
        lag_cols = [c for c in df.columns if "_lag" in c]
        if lag_cols:
            before = len(df)
            df = df.dropna(subset=lag_cols).reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                logger.info("Dropped %d rows with NaN from lag features", dropped)
        feature_cols = get_feature_columns(df)
        df[feature_cols] = df[feature_cols].fillna(0)

    logger.info("Feature engineering complete → %d rows, %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = set(["timestamp"] + TARGET_COLUMNS)
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    dates = pd.date_range("2024-01-01", periods=400, freq="15min")
    test_df = pd.DataFrame({
        "timestamp": dates,
        "ghi": np.random.uniform(0, 800, 400),
        "dhi": np.random.uniform(0, 300, 400),
        "temperature": np.random.uniform(15, 40, 400),
        "windspeed": np.random.uniform(0, 30, 400),
        "cloud_cover": np.random.uniform(0, 100, 400),
    })
    result = create_features(test_df)
    feats = get_feature_columns(result)
    print(f"\nFeature columns ({len(feats)}):")
    for col in feats:
        print(f"  {col}")
