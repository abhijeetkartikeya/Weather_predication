"""
Feature Engineering — Create time-series features for the weather forecast model.

Features created:
  1. Lag features for each target variable
  2. Rolling statistics (mean, std)
  3. Cyclical time features (hour, day-of-year)
  4. Open-Meteo forecast alignment features
"""

import logging
import numpy as np
import pandas as pd

from config import TARGET_COLUMNS, LAG_STEPS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged values for each target variable."""
    logger.info("Adding lag features (steps: %s) …", LAG_STEPS)
    for col in TARGET_COLUMNS:
        for lag in LAG_STEPS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and std for each target variable."""
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
    """Add cyclical time-of-day and day-of-year features."""
    logger.info("Adding time features …")
    ts = pd.to_datetime(df["timestamp"])

    # Hour of day (0–23.75 at 15-min resolution) — encoded as sin/cos
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)

    # Day of year (1–366) — encoded as sin/cos
    doy = ts.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Month (useful for seasonal patterns)
    df["month"] = ts.dt.month

    # Is daytime flag (approximate: 6 AM – 6 PM)
    df["is_daytime"] = ((ts.dt.hour >= 6) & (ts.dt.hour < 18)).astype(int)

    # Weekday (0=Mon, 6=Sun) — minor relevance but cheap
    df["weekday"] = ts.dt.weekday

    return df


def add_openmeteo_features(df: pd.DataFrame, om_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Open-Meteo forecast variables as input features.
    om_df must have: timestamp, om_ghi, om_dhi, om_temperature, om_windspeed, om_cloud_cover
    """
    if om_df is None or om_df.empty:
        logger.warning("No Open-Meteo data — filling with zeros")
        for col in TARGET_COLUMNS:
            df[f"om_{col}"] = 0.0
        return df

    logger.info("Merging %d Open-Meteo forecast rows as features …", len(om_df))

    # Ensure timestamp types match
    om_df = om_df.copy()
    om_df["timestamp"] = pd.to_datetime(om_df["timestamp"])

    # Keep only om_ columns + timestamp
    om_cols = ["timestamp"] + [c for c in om_df.columns if c.startswith("om_")]
    om_df = om_df[om_cols]

    df = df.merge(om_df, on="timestamp", how="left")

    # Forward-fill Open-Meteo features for historical rows where no forecast existed
    for col in [c for c in df.columns if c.startswith("om_")]:
        df[col] = df[col].fillna(method="ffill", limit=4)  # 1 hour max
        df[col] = df[col].fillna(0)  # remaining → 0

    return df


def create_features(
    df: pd.DataFrame,
    om_df: pd.DataFrame = None,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Full feature pipeline: lags → rolling → time → Open-Meteo.

    Args:
        df: DataFrame with timestamp + target columns.
        om_df: Optional Open-Meteo forecast DataFrame.
        drop_na: If True, drop rows where lag features created NaN (start of series).

    Returns:
        DataFrame ready for model training / inference.
    """
    logger.info("Running full feature engineering pipeline …")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_openmeteo_features(df, om_df)

    if drop_na:
        # Only drop rows where LAG features have NaN (start of series)
        # Don't drop based on Open-Meteo or other optional columns
        lag_cols = [c for c in df.columns if "_lag" in c]
        if lag_cols:
            before = len(df)
            df = df.dropna(subset=lag_cols).reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                logger.info("Dropped %d rows with NaN from lag features", dropped)
        # Fill any remaining NaN in non-target columns with 0
        feature_cols = get_feature_columns(df)
        df[feature_cols] = df[feature_cols].fillna(0)

    logger.info("Feature engineering complete → %d rows, %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (everything except timestamp + targets)."""
    exclude = set(["timestamp"] + TARGET_COLUMNS)
    return [c for c in df.columns if c not in exclude]


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Create a small test DataFrame
    dates = pd.date_range("2024-01-01", periods=200, freq="15min")
    test_df = pd.DataFrame({
        "timestamp": dates,
        "ghi": np.random.uniform(0, 800, 200),
        "dhi": np.random.uniform(0, 300, 200),
        "temperature": np.random.uniform(15, 40, 200),
        "windspeed": np.random.uniform(0, 30, 200),
        "cloud_cover": np.random.uniform(0, 100, 200),
    })

    result = create_features(test_df)
    print(f"\nFeature columns ({len(get_feature_columns(result))}):")
    for col in get_feature_columns(result):
        print(f"  {col}")
