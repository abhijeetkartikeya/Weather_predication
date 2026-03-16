"""
Feature Engineering Module for Weather Forecasting Pipeline
=============================================================
Transforms raw hourly weather data into ML-ready features with
temporal encoding, lag features, rolling statistics, and solar position.

Usage:
    from feature_engineering import build_features
    X, y = build_features(df, target="temperature_2m", horizon=24)
"""

import numpy as np
import pandas as pd

from config import (
    DIFF_PERIODS,
    FORECAST_HORIZONS,
    LAG_STEPS,
    LATITUDE,
    LONGITUDE,
    ROLLING_WINDOWS,
    STEPS_PER_HOUR,
    TARGET_COLUMNS,
)
from utils import compute_solar_features, setup_logger

logger = setup_logger("features")


# ═══════════════════════════════════════════════
# Temporal Features
# ═══════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical temporal features from the DatetimeIndex.

    Features added:
        hour_sin, hour_cos, month_sin, month_cos,
        day_of_year_sin, day_of_year_cos, week_sin, week_cos,
        is_weekend
    """
    df = df.copy()

    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    week = df.index.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)

    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    return df


# ═══════════════════════════════════════════════
# Lag Features
# ═══════════════════════════════════════════════

def add_lag_features(df: pd.DataFrame, columns: list, lags: list = None) -> pd.DataFrame:
    """
    Add lagged values for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 15-min frequency.
    columns : list
        Columns to create lags for.
    lags : list
        Lag periods in number of 15-min steps.
    """
    df = df.copy()
    lags = lags or LAG_STEPS

    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            # Convert steps to readable label
            minutes = lag * 15
            if minutes < 60:
                label = f"{minutes}m"
            elif minutes < 1440:
                label = f"{minutes // 60}h"
            else:
                label = f"{minutes // 1440}d"
            df[f"{col}_lag_{label}"] = df[col].shift(lag)

    return df


# ═══════════════════════════════════════════════
# Rolling Statistics
# ═══════════════════════════════════════════════

def add_rolling_features(
    df: pd.DataFrame,
    columns: list,
    windows: list = None,
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 15-min frequency.
    columns : list
        Columns to compute rolling stats for.
    windows : list
        Window sizes in number of 15-min steps.
    """
    df = df.copy()
    windows = windows or ROLLING_WINDOWS

    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            # Convert steps to readable label
            minutes = w * 15
            if minutes < 60:
                label = f"{minutes}m"
            elif minutes < 1440:
                label = f"{minutes // 60}h"
            else:
                label = f"{minutes // 1440}d"
            roll = df[col].rolling(window=w, min_periods=max(1, w // 4))
            df[f"{col}_rmean_{label}"] = roll.mean()
            df[f"{col}_rstd_{label}"] = roll.std()
            df[f"{col}_rmin_{label}"] = roll.min()
            df[f"{col}_rmax_{label}"] = roll.max()

    return df


# ═══════════════════════════════════════════════
# Difference Features
# ═══════════════════════════════════════════════

def add_diff_features(df: pd.DataFrame, columns: list, periods: list = None) -> pd.DataFrame:
    """
    Add hour-over-hour and day-over-day difference features.
    """
    df = df.copy()
    periods = periods or DIFF_PERIODS

    for col in columns:
        if col not in df.columns:
            continue
        for p in periods:
            minutes = p * 15
            if minutes < 60:
                label = f"{minutes}m"
            elif minutes < 1440:
                label = f"{minutes // 60}h"
            else:
                label = f"{minutes // 1440}d"
            df[f"{col}_diff_{label}"] = df[col].diff(p)

    return df


# ═══════════════════════════════════════════════
# Interaction Features
# ═══════════════════════════════════════════════

def add_wind_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose wind direction (degrees) into U/V vector components.

    Raw wind direction in degrees is CIRCULAR (359° ≈ 1°), which tree-based
    models cannot interpret. Converting to U (east-west) and V (north-south)
    components makes wind direction a proper continuous feature.

    Also creates wind-cloud advection features: these capture how wind
    transports clouds, which is critical for cloud cover and radiation
    forecasting at 24-72h horizons.

    Features added:
        wind_u, wind_v                   — Wind vector components
        wind_u_x_cloud, wind_v_x_cloud   — Cloud advection by wind direction
        wind_dir_sin, wind_dir_cos       — Cyclical encoding of direction
        cloud_advection_rate             — Speed at which clouds move
        nasa_wind_u, nasa_wind_v         — Satellite-derived wind components
    """
    df = df.copy()

    # ── Open-Meteo wind decomposition ──
    if "wind_direction_10m" in df.columns and "wind_speed_10m" in df.columns:
        wd_rad = np.radians(df["wind_direction_10m"])

        # U = east-west component (+ = westerly, wind blowing FROM west)
        # V = north-south component (+ = southerly, wind blowing FROM south)
        df["wind_u"] = -df["wind_speed_10m"] * np.sin(wd_rad)
        df["wind_v"] = -df["wind_speed_10m"] * np.cos(wd_rad)

        # Cyclical encoding of raw direction
        df["wind_dir_sin"] = np.sin(wd_rad)
        df["wind_dir_cos"] = np.cos(wd_rad)

        # Cloud advection features — how wind pushes clouds
        if "cloud_cover" in df.columns:
            cloud_frac = df["cloud_cover"] / 100.0
            df["wind_u_x_cloud"] = df["wind_u"] * cloud_frac
            df["wind_v_x_cloud"] = df["wind_v"] * cloud_frac
            df["cloud_advection_rate"] = df["wind_speed_10m"] * cloud_frac

        # Wind gust ratio (turbulence indicator → affects cloud breakup)
        if "wind_gusts_10m" in df.columns:
            ws_safe = df["wind_speed_10m"].replace(0, np.nan)
            df["gust_factor"] = df["wind_gusts_10m"] / ws_safe
            df["gust_factor"] = df["gust_factor"].fillna(1.0)

    # ── NASA POWER satellite-derived wind decomposition ──
    if "nasa_wind_direction" in df.columns and "nasa_wind_speed" in df.columns:
        wd_rad = np.radians(df["nasa_wind_direction"])
        df["nasa_wind_u"] = -df["nasa_wind_speed"] * np.sin(wd_rad)
        df["nasa_wind_v"] = -df["nasa_wind_speed"] * np.cos(wd_rad)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically meaningful feature interactions.
    """
    df = df.copy()

    # Temperature × Humidity → heat index proxy
    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["temp_x_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100.0

    # Wind speed × cloud cover → convective proxy
    if "wind_speed_10m" in df.columns and "cloud_cover" in df.columns:
        df["wind_x_cloud"] = df["wind_speed_10m"] * df["cloud_cover"] / 100.0

    # GHI × (1 - cloud_cover) → clear sky GHI proxy
    if "shortwave_radiation" in df.columns and "cloud_cover" in df.columns:
        df["ghi_x_clearsky"] = df["shortwave_radiation"] * (1 - df["cloud_cover"] / 100.0)

    # Pressure tendency (differenced) already comes from diff features
    # Temperature – dew point → dryness indicator
    if "temperature_2m" in df.columns and "dew_point_2m" in df.columns:
        df["temp_dewpoint_spread"] = df["temperature_2m"] - df["dew_point_2m"]

    # DHI / GHI ratio → diffuse fraction (important for solar)
    if "diffuse_radiation" in df.columns and "shortwave_radiation" in df.columns:
        ghi_safe = df["shortwave_radiation"].replace(0, np.nan)
        df["diffuse_fraction"] = df["diffuse_radiation"] / ghi_safe
        df["diffuse_fraction"] = df["diffuse_fraction"].clip(0, 1).fillna(0)

    # Cloud layer interaction — high clouds affect DHI more than GHI
    if "cloud_cover_high" in df.columns and "cloud_cover_low" in df.columns:
        df["cloud_layer_ratio"] = (
            df["cloud_cover_high"] / df["cloud_cover_low"].replace(0, np.nan)
        ).fillna(0).clip(0, 10)

    # Pressure × wind speed → synoptic forcing indicator
    if "pressure_msl" in df.columns and "wind_speed_10m" in df.columns:
        df["pressure_x_wind"] = df["pressure_msl"] * df["wind_speed_10m"] / 1000.0

    return df


# ═══════════════════════════════════════════════
# Target Creation (shifted)
# ═══════════════════════════════════════════════

def create_target(df: pd.DataFrame, target_col: str, horizon_hours: int) -> pd.Series:
    """
    Create a target variable shifted forward by `horizon_hours`.
    The model predicts future values, so the target is the value
    `horizon_hours` steps ahead.

    Parameters
    ----------
    target_col : str
        Column to predict.
    horizon_hours : int
        Number of hours ahead to forecast.

    Returns
    -------
    pd.Series
        Shifted target.
    """
    return df[target_col].shift(-horizon_hours)


# ═══════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    target_col: str,
    horizon_hours: int,
    feature_columns: list = None,
    is_training: bool = True,
) -> tuple:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged dataset with DatetimeIndex.
    target_col : str
        Which target to predict.
    horizon_hours : int
        Forecast horizon in hours.
    feature_columns : list, optional
        Columns to generate lag/rolling features for. Defaults to TARGET_COLUMNS.

    Returns
    -------
    (X, y) : tuple of pd.DataFrame, pd.Series
        Feature matrix and target vector (NaN rows dropped).
    """
    logger.info(f"Building features for {target_col} @ {horizon_hours}h horizon ...")

    feat_cols = feature_columns or [c for c in TARGET_COLUMNS if c in df.columns]

    # Also include auxiliary columns for lag/rolling
    aux_cols = [c for c in df.columns if c.startswith("nasa_") or c in [
        "relative_humidity_2m", "dew_point_2m", "pressure_msl",
        "wind_direction_10m", "wind_gusts_10m",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "direct_radiation", "direct_normal_irradiance",
    ]]
    all_source_cols = list(set(feat_cols + aux_cols))
    all_source_cols = [c for c in all_source_cols if c in df.columns]

    # Step 1: Temporal
    featured = add_temporal_features(df)

    # Step 2: Solar position
    featured = compute_solar_features(featured)

    # Step 3: Wind decomposition (U/V components + cloud advection)
    featured = add_wind_decomposition(featured)

    # Step 4: Add minute-of-hour feature for 15-min resolution
    featured["minute_of_hour"] = featured.index.minute / 60.0
    featured["step_of_day"] = (featured.index.hour * 4 + featured.index.minute // 15) / 96.0

    # Step 4: Lag features (only on main targets + key aux)
    lag_cols = [c for c in feat_cols if c in featured.columns]
    # Also lag wind components and cloud advection (critical for cloud cover prediction)
    wind_lag_cols = [c for c in ["wind_u", "wind_v", "cloud_advection_rate"] if c in featured.columns]
    lag_cols = list(set(lag_cols + wind_lag_cols))
    featured = add_lag_features(featured, lag_cols)

    # Step 5: Rolling statistics
    featured = add_rolling_features(featured, lag_cols)

    # Step 6: Difference features
    featured = add_diff_features(featured, lag_cols)

    # Step 6: Interaction features
    featured = add_interaction_features(featured)

    # Step 7: Create target
    y = create_target(featured, target_col, horizon_hours)

    # Step 8: Drop target columns and raw string/object columns
    drop_cols = [target_col]  # Don't leak current target value
    # Keep the target lags though — the model uses past values to predict future
    X = featured.drop(columns=[c for c in drop_cols if c in featured.columns], errors="ignore")

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Step 9: Drop rows with NaN in target or features
    if is_training:
        mask = y.notna() & X.notna().all(axis=1)
    else:
        mask = X.notna().all(axis=1)
        
    X = X.loc[mask]
    y = y.loc[mask]

    logger.info(f"  Features: {X.shape[1]}, Samples: {len(X):,}")

    return X, y


def build_all_feature_sets(df: pd.DataFrame) -> dict:
    """
    Build feature sets for all target × horizon combinations.

    Returns
    -------
    dict
        Nested dict: {target: {horizon_name: (X, y)}}
    """
    result = {}

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            logger.warning(f"Target column {target} not found in dataset, skipping.")
            continue
        result[target] = {}
        for horizon_name, horizon_hours in FORECAST_HORIZONS.items():
            X, y = build_features(df, target, horizon_hours)
            result[target][horizon_name] = (X, y)
            logger.info(f"  {target} @ {horizon_name}: X={X.shape}, y={len(y)}")

    return result
