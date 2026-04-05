"""
Feature engineering with STRICT leakage prevention.

Every feature is derived from PAST values only.  We never peek at the current
or future target value when constructing inputs.

APPROACH: Direct Multi-Horizon Forecasting
==========================================
Instead of recursive/autoregressive prediction (which causes predictions to
collapse to the mean after a few steps), we train the model to predict
directly at any forecast horizon h = 1..MAX_HORIZON.

For each observation at time t and each horizon h:
  - Lag features use real data from time t (shifted by h+lag, so they always
    reference REAL past observations, never predictions)
  - Rolling features use real data from time t
  - Cyclical time features are computed for the TARGET timestamp (t+h)
  - forecast_horizon (normalized) + sin/cos encoding is added as a feature
  - Target = actual value at time t+h

This creates ~24x more training rows but every single feature comes from
real past observations.  At prediction time there is NO recursion — each
hour is predicted independently using only real past data.
"""

import pandas as pd
import numpy as np

from config import (
    RAW_CSV,
    FEATURES_CSV,
    TARGET_VARIABLES,
    LAG_HOURS,
    ROLLING_WINDOWS,
    MAX_HORIZON,
)

# Short aliases used in column names
_ALIAS = {
    "temperature_2m": "temp",
    "cloud_cover": "cloud",
    "wind_speed_10m": "wind",
    "shortwave_radiation": "ghi",
}


def engineer_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the base feature matrix from raw weather data.

    This produces the traditional single-step feature set (lag, rolling,
    cyclical time) which is saved to FEATURES_CSV.  The direct multi-horizon
    expansion is done separately by ``build_direct_training_data()``.

    Returns a DataFrame with:
      - time column (datetime)
      - lag features for every target variable (PAST values only)
      - rolling mean/std features (PAST values only, via shift(1))
      - cyclical time features (sin/cos hour, day-of-year)
      - original target columns (for training / evaluation)
    """
    if df is None:
        df = pd.read_csv(RAW_CSV, parse_dates=["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # ── Lag features ────────────────────────────────────────────────────
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        for lag in LAG_HOURS:
            col_name = f"{alias}_lag_{lag}"
            df[col_name] = df[var].shift(lag)

    # ── Rolling features ──────────────────────────────────────────────
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        shifted = df[var].shift(1)  # exclude current value
        for window in ROLLING_WINDOWS:
            df[f"{alias}_roll_mean_{window}"] = shifted.rolling(
                window=window, min_periods=max(1, window // 2)
            ).mean()
            df[f"{alias}_roll_std_{window}"] = shifted.rolling(
                window=window, min_periods=max(1, window // 2)
            ).std()

    # ── Cyclical time features ────────────────────────────────────────
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    doy = df["time"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["is_daytime"] = ((hour >= 6) & (hour <= 18)).astype(int)

    # ── Drop rows that have NaN from lagging/rolling ──────────────────
    max_lag = max(LAG_HOURS)
    rows_before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {rows_before - len(df)} NaN rows (from lag window of "
          f"{max_lag} hours). Remaining: {len(df)} rows.")

    # ── Save ──────────────────────────────────────────────────────────
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Features saved -> {FEATURES_CSV}")
    return df


def build_direct_training_data(df: pd.DataFrame,
                               max_horizon: int = MAX_HORIZON,
                               max_years: int = 5) -> pd.DataFrame:
    """
    Expand the feature DataFrame into direct multi-horizon training data.

    Uses fully VECTORIZED numpy operations for speed (no Python loops over
    rows).  Limits training data to the most recent *max_years* to keep
    memory and compute manageable while still capturing seasonal patterns.

    For each origin time *t* and each horizon *h* in [1 .. max_horizon]:
      - Lag features reference real data at t - lag
      - Rolling mean/std are anchored at time t (shift(1) equivalent)
      - Cyclical time features are for the TARGET timestamp t + h
      - forecast_horizon + sin/cos encoding added as features
      - Target = actual value at time t + h

    Parameters
    ----------
    df : DataFrame
        Raw weather data with ``time`` + TARGET_VARIABLES columns.
    max_horizon : int
        Maximum forecast horizon (hours).
    max_years : int
        Limit to the most recent N years of data for training speed.

    Returns
    -------
    DataFrame ready for model training (features + target columns + time).
    """
    df = df.sort_values("time").reset_index(drop=True)

    # ── Limit to recent years for speed ────────────────────────────────
    cutoff_date = df["time"].max() - pd.Timedelta(days=max_years * 365)
    df = df[df["time"] >= cutoff_date].reset_index(drop=True)
    print(f"Using last {max_years} years: {len(df):,} rows "
          f"({df['time'].min()} -> {df['time'].max()})")

    n = len(df)
    time_arr = df["time"].values  # datetime64[ns]
    var_arrays = {var: df[var].values.astype(np.float64) for var in TARGET_VARIABLES}

    min_origin = max(LAG_HOURS)
    max_origin = n - max_horizon  # exclusive

    if max_origin <= min_origin:
        raise ValueError("Not enough data for the requested horizon / lag combination.")

    # Origin indices (vectorised)
    origins = np.arange(min_origin, max_origin)
    n_origins = len(origins)
    n_horizons = max_horizon
    total_rows = n_origins * n_horizons

    print(f"Building direct training data: {n_origins:,} origins × "
          f"{n_horizons} horizons = {total_rows:,} rows (vectorised) ...")

    # ── Pre-allocate output dict of arrays ─────────────────────────────
    data = {}

    # Tile origins and horizons
    # origins_rep[i] = origin index for row i
    # horizons_rep[i] = horizon (1-based) for row i
    origins_rep = np.repeat(origins, n_horizons)          # [o0,o0,..,o0,o1,o1,..]
    horizons_rep = np.tile(np.arange(1, n_horizons + 1), n_origins)  # [1,2,..,24,1,2,..]
    target_indices = origins_rep + horizons_rep

    # ── Lag features (vectorised) ──────────────────────────────────────
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        arr = var_arrays[var]
        for lag in LAG_HOURS:
            lag_indices = origins_rep - lag
            data[f"{alias}_lag_{lag}"] = arr[lag_indices]

    # ── Rolling features (vectorised via cumsum) ───────────────────────
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        arr = var_arrays[var]
        # Prefix sums for fast window mean/std
        cumsum = np.concatenate([[0.0], np.cumsum(arr)])
        cumsum2 = np.concatenate([[0.0], np.cumsum(arr ** 2)])

        for window in ROLLING_WINDOWS:
            # Window = [origin - window, origin) (exclusive of origin = shift(1))
            win_end = origins_rep        # exclusive
            win_start = np.maximum(0, origins_rep - window)
            win_len = win_end - win_start
            win_len_safe = np.maximum(win_len, 1)

            s = cumsum[win_end] - cumsum[win_start]
            s2 = cumsum2[win_end] - cumsum2[win_start]

            roll_mean = s / win_len_safe
            # Variance = E[X²] - (E[X])² with Bessel correction
            variance = (s2 / win_len_safe) - (roll_mean ** 2)
            variance = np.maximum(variance, 0.0)  # numerical safety
            # Bessel correction: multiply by n/(n-1)
            bessel = np.where(win_len > 1, win_len_safe / (win_len_safe - 1), 1.0)
            roll_std = np.sqrt(variance * bessel)
            roll_std = np.where(win_len > 1, roll_std, 0.0)

            # Repeat for all horizons (same origin → same rolling features)
            data[f"{alias}_roll_mean_{window}"] = roll_mean
            data[f"{alias}_roll_std_{window}"] = roll_std

    # ── Cyclical time features (for TARGET timestamp) ──────────────────
    target_times = pd.DatetimeIndex(time_arr[target_indices])
    target_hour = target_times.hour + target_times.minute / 60.0
    target_doy = target_times.dayofyear

    data["hour_sin"] = np.sin(2 * np.pi * target_hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * target_hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * target_doy / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * target_doy / 365.25)
    data["dow_sin"] = np.sin(2 * np.pi * target_times.dayofweek / 7)
    data["dow_cos"] = np.cos(2 * np.pi * target_times.dayofweek / 7)
    data["is_daytime"] = ((target_hour >= 6) & (target_hour <= 18)).astype(np.float64)

    # ── Horizon features ───────────────────────────────────────────────
    data["forecast_horizon"] = horizons_rep / max_horizon
    data["forecast_horizon_sin"] = np.sin(2 * np.pi * horizons_rep / 24)
    data["forecast_horizon_cos"] = np.cos(2 * np.pi * horizons_rep / 24)

    # ── Target values at t+h ──────────────────────────────────────────
    for var in TARGET_VARIABLES:
        data[var] = var_arrays[var][target_indices]

    # ── Origin time for chronological splitting ────────────────────────
    data["time"] = time_arr[origins_rep]

    result = pd.DataFrame(data)
    print(f"  Built {len(result):,} direct training rows.")
    return result


def get_base_feature_columns() -> list[str]:
    """Return feature columns for the base (single-step) feature set.

    This is used by leakage tests which operate on the base FEATURES_CSV
    that does not contain horizon columns.
    """
    cols = []
    # Lag features
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        for lag in LAG_HOURS:
            cols.append(f"{alias}_lag_{lag}")
    # Rolling features
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        for window in ROLLING_WINDOWS:
            cols.append(f"{alias}_roll_mean_{window}")
            cols.append(f"{alias}_roll_std_{window}")
    # Cyclical time features
    cols += ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
             "dow_sin", "dow_cos", "is_daytime"]
    return cols


def get_feature_columns() -> list[str]:
    """Return the full list of feature column names for the direct
    multi-horizon model (base features + horizon features)."""
    cols = get_base_feature_columns()
    # Horizon features (direct multi-horizon)
    cols += ["forecast_horizon", "forecast_horizon_sin", "forecast_horizon_cos"]
    return cols


if __name__ == "__main__":
    engineer_features()
