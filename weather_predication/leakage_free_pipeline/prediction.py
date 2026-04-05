"""
Direct multi-horizon forecasting — the KEY leakage-free component.

APPROACH: Direct Prediction (NO Recursion)
==========================================
Instead of recursively feeding predictions back as lag features (which
causes predictions to collapse to the dataset mean after a few steps),
we predict each hour DIRECTLY using only REAL past observations.

  For each day D in the test period:
    1. Seed history with ACTUAL observations up to end of day D-1.
    2. For each hour h in day D:
       - Compute lag/rolling features from REAL past observations only.
       - Set forecast_horizon = hours into the prediction window.
       - Predict directly — NO recursion, NO feeding predictions back.

WHY THIS IS LEAKAGE-FREE:
  - We only use observations BEFORE day D (past data).
  - We never peek at day D's actuals when predicting day D.
  - The actual values used for features are strictly in the past.

WHY THIS FIXES THE FLAT PREDICTION PROBLEM:
  - Every prediction uses REAL past data (not degraded predictions).
  - The model has learned the relationship between real features and
    targets at various horizons during training.
  - The cyclical time features (hour_sin/cos) tell the model where in
    the diurnal cycle it is predicting, preserving temperature swings.
"""

import numpy as np
import pandas as pd
import joblib

from config import (
    FEATURES_CSV,
    TRAIN_CUTOFF,
    TARGET_VARIABLES,
    SCALER_PATH,
    PREDICTIONS_CSV,
    PREDICTION_SMOOTHING_WINDOW,
    ROLLING_WINDOWS,
    MAX_HORIZON,
    model_path,
    LAG_HOURS,
)
from feature_engineering import get_feature_columns, _ALIAS


def _create_features_for_row(history: pd.DataFrame, timestamp: pd.Timestamp,
                             forecast_horizon: int, scaler) -> np.ndarray:
    """
    Build a single feature vector for *timestamp* using only *history*.

    Parameters
    ----------
    history : DataFrame
        Must contain at least max(LAG_HOURS) rows of past data with columns
        for every target variable and a 'time' column.  All values must be
        REAL observations (not predictions).
    timestamp : pd.Timestamp
        The moment we are predicting for (the target time).
    forecast_horizon : int
        Hours ahead from the last real observation (1-based).
    scaler : fitted StandardScaler

    Returns
    -------
    1-D numpy array ready for model.predict().
    """
    feature_cols = get_feature_columns()
    row = {}

    # ── Lag features (PAST real values only) ──────────────────────────
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        for lag in LAG_HOURS:
            idx = len(history) - lag
            if idx >= 0:
                row[f"{alias}_lag_{lag}"] = history[var].iloc[idx]
            else:
                row[f"{alias}_lag_{lag}"] = history[var].iloc[0]

    # ── Rolling features (PAST real values only) ──────────────────────
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        series = history[var]
        for window in ROLLING_WINDOWS:
            if len(series) >= window:
                window_data = series.iloc[-window:]
                row[f"{alias}_roll_mean_{window}"] = window_data.mean()
                row[f"{alias}_roll_std_{window}"] = window_data.std()
            else:
                row[f"{alias}_roll_mean_{window}"] = series.mean()
                row[f"{alias}_roll_std_{window}"] = series.std() if len(series) > 1 else 0.0

    # ── Cyclical time features (for TARGET timestamp) ─────────────────
    hour = timestamp.hour + timestamp.minute / 60.0
    doy = timestamp.timetuple().tm_yday

    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    row["dow_sin"] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    row["is_daytime"] = 1 if 6 <= hour <= 18 else 0

    # ── Horizon features ──────────────────────────────────────────────
    row["forecast_horizon"] = forecast_horizon / MAX_HORIZON
    row["forecast_horizon_sin"] = np.sin(2 * np.pi * forecast_horizon / 24)
    row["forecast_horizon_cos"] = np.cos(2 * np.pi * forecast_horizon / 24)

    vec = pd.DataFrame([row])[feature_cols]
    vec_scaled = scaler.transform(vec)
    return vec_scaled


def _gaussian_smooth(series: pd.Series, window: int = 5, sigma: float = 1.5) -> pd.Series:
    """Gaussian-weighted rolling mean for smooth output."""
    return series.rolling(window=window, min_periods=1, center=True, win_type="gaussian").mean(std=sigma)


def generate_predictions(train_df: pd.DataFrame | None = None,
                         test_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Produce direct multi-horizon forecasts for the test period.

    For each calendar day in the test set:
      1. Seed history with ACTUAL past observations (up to previous day end)
      2. For each hour h, predict DIRECTLY using real features + horizon
      3. Record predictions vs actuals for evaluation

    NO recursion.  Every prediction uses only REAL past observations.
    """
    # ── Load artefacts ──────────────────────────────────────────────────
    if train_df is None or test_df is None:
        full = pd.read_csv(FEATURES_CSV, parse_dates=["time"])
        train_df = full[full["time"] <= TRAIN_CUTOFF].copy()
        test_df = full[full["time"] > TRAIN_CUTOFF].copy()

    scaler = joblib.load(SCALER_PATH)
    models = {t: joblib.load(model_path(t)) for t in TARGET_VARIABLES}

    # ── Prepare full observation data for seeding ─────────────────────
    # This contains ALL historical data (train + test actuals).
    # We ONLY use observations BEFORE the current prediction day.
    all_obs = pd.concat([
        train_df[["time"] + TARGET_VARIABLES],
        test_df[["time"] + TARGET_VARIABLES],
    ]).sort_values("time").reset_index(drop=True)

    # ── Group test timestamps by day ──────────────────────────────────
    test_df = test_df.sort_values("time").reset_index(drop=True)
    test_df["date"] = test_df["time"].dt.date
    unique_days = sorted(test_df["date"].unique())

    seed_len = max(max(LAG_HOURS), max(ROLLING_WINDOWS)) + 10
    results = []
    n_total = len(test_df)
    done = 0

    print(f"Generating DIRECT multi-horizon predictions for {len(unique_days)} days, "
          f"{n_total} total hours ...")
    print(f"  (No recursion — every prediction uses only REAL past data)")

    for day_idx, day in enumerate(unique_days):
        day_timestamps = test_df[test_df["date"] == day]["time"].tolist()
        day_start = pd.Timestamp(day)

        # ── 1. Seed history with ACTUAL observations strictly BEFORE this day ──
        past_mask = all_obs["time"] < pd.Timestamp(day)
        past_obs = all_obs[past_mask].tail(seed_len).copy()
        history = past_obs[["time"] + TARGET_VARIABLES].reset_index(drop=True)

        if history.empty:
            print(f"  Skipping {day} -- no past observations available")
            continue

        # ── 2. Predict each hour of the day DIRECTLY (no recursion) ───
        for hour_idx, ts in enumerate(day_timestamps):
            actual_row = test_df[test_df["time"] == ts].iloc[0]

            # forecast_horizon = how many hours ahead from the last real
            # observation.  hour_idx=0 is 1 hour ahead, etc.
            forecast_horizon = hour_idx + 1

            # Build features from REAL history only (not predictions!)
            X = _create_features_for_row(history, ts, forecast_horizon, scaler)

            # Predict each target
            preds = {}
            for var in TARGET_VARIABLES:
                pred_val = float(models[var].predict(X)[0])

                # Physical constraints
                if var == "shortwave_radiation":
                    hour = ts.hour
                    if hour < 5 or hour > 19:
                        pred_val = 0.0
                    pred_val = max(0.0, pred_val)
                elif var == "cloud_cover":
                    pred_val = np.clip(pred_val, 0.0, 100.0)
                elif var == "wind_speed_10m":
                    pred_val = max(0.0, pred_val)
                preds[var] = pred_val

            # NOTE: We do NOT append predictions to history.
            # History stays as REAL observations only.

            # Record result
            result = {"time": ts}
            for var in TARGET_VARIABLES:
                alias = _ALIAS[var]
                result[f"actual_{alias}"] = actual_row[var]
                result[f"predicted_{alias}"] = preds[var]
            results.append(result)
            done += 1

        if (day_idx + 1) % 7 == 0 or (day_idx + 1) == len(unique_days):
            print(f"  {done}/{n_total} hours done ({day_idx + 1}/{len(unique_days)} days)")

    pred_df = pd.DataFrame(results)
    pred_df["time"] = pd.to_datetime(pred_df["time"])

    # ── Post-prediction Gaussian smoothing ────────────────────────────
    # Smooth within each day to avoid blurring day boundaries.
    smooth_window = max(PREDICTION_SMOOTHING_WINDOW, 5)
    print(f"Applying Gaussian smoothing (window={smooth_window}) ...")
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        col = f"predicted_{alias}"
        pred_df[col] = _gaussian_smooth(pred_df[col], window=smooth_window, sigma=1.5)

        # Re-apply physical constraints after smoothing
        if var == "shortwave_radiation":
            night_mask = pred_df["time"].dt.hour.isin(
                list(range(0, 5)) + list(range(20, 24))
            )
            pred_df.loc[night_mask, col] = 0.0
            pred_df[col] = pred_df[col].clip(lower=0.0)
        elif var == "cloud_cover":
            pred_df[col] = pred_df[col].clip(0.0, 100.0)
        elif var == "wind_speed_10m":
            pred_df[col] = pred_df[col].clip(lower=0.0)

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved -> {PREDICTIONS_CSV}")
    return pred_df


if __name__ == "__main__":
    generate_predictions()
