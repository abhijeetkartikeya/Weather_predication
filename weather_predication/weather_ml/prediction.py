from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd

from .db import ensure_schema, fetch_dataframe, upsert_dataframe
from .features import build_single_feature_row, load_observations
from .logging_utils import get_logger
from .settings import Settings, load_settings, location_token

logger = get_logger("weather_ml.prediction")

# Max rate-of-change per 15-min step (prevents spikes)
_MAX_DELTA_PER_STEP: dict[str, float] = {
    "temperature_2m": 0.8,
    "cloud_cover": 8.0,
    "wind_speed_10m": 1.5,
    "shortwave_radiation": 80.0,
    "diffuse_radiation": 40.0,
}


def _ml_weight(step: int, total_steps: int) -> float:
    """Adaptive blending weight for the ML *correction* (delta) applied on top
    of the provider forecast.

    Near-term (step 0): 70% ML correction weight
    Far-term (last step): ~15% ML correction weight
    Uses exponential decay so trust in ML drops quickly as the horizon grows.
    """
    progress = step / max(total_steps - 1, 1)
    return 0.70 * math.exp(-3.0 * progress)


def _gaussian_smooth(series: pd.Series, window: int = 11, sigma: float = 3.0) -> pd.Series:
    """Gaussian-weighted rolling mean — much smoother than simple rolling."""
    return series.rolling(window=window, min_periods=1, center=True, win_type="gaussian").mean(std=sigma)


def _provider_columns(settings: Settings) -> list[str]:
    return [column for column in settings.observation_columns if not column.startswith("nasa_")]


def _load_provider_forecast(lat: float, lon: float, settings: Settings) -> pd.DataFrame:
    provider_columns = _provider_columns(settings)
    query = f"""
        SELECT issue_time, forecast_time, {", ".join(provider_columns)}
        FROM {settings.postgres_schema}.weather_forecast
        WHERE forecast_type = 'provider'
          AND lat = :lat
          AND lon = :lon
          AND issue_time = (
              SELECT MAX(issue_time)
              FROM {settings.postgres_schema}.weather_forecast
              WHERE forecast_type = 'provider'
                AND lat = :lat
                AND lon = :lon
          )
        ORDER BY forecast_time
    """
    frame = fetch_dataframe(query, {"lat": lat, "lon": lon}, settings=settings, parse_dates=["issue_time", "forecast_time"])
    if frame.empty:
        return frame
    return frame.set_index("forecast_time").sort_index()


def _load_model_bundle(target: str, lat: float, lon: float, settings: Settings):
    location_key = location_token(lat, lon)
    model_path = settings.models_dir / f"{target}_{location_key}_xgb.joblib"
    metadata_path = settings.models_dir / f"{target}_{location_key}_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts for target '{target}' at lat={lat}, lon={lon}. Run train_model.py first."
        )
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def generate_predictions(
    settings: Settings | None = None,
    output_path: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat = float(settings.latitude if lat is None else lat)
    lon = float(settings.longitude if lon is None else lon)
    ensure_schema(settings)

    history = load_observations(settings=settings, lat=lat, lon=lon)
    if history.empty:
        raise RuntimeError("No observations found in PostgreSQL. Run collect_data.py first.")

    provider_forecast = _load_provider_forecast(lat, lon, settings)
    if provider_forecast.empty:
        raise RuntimeError("No provider forecast snapshot found. Run collect_data.py first.")

    provider_forecast = provider_forecast.reindex(columns=["issue_time", *_provider_columns(settings)])
    provider_issue_time = pd.Timestamp(provider_forecast["issue_time"].dropna().iloc[0])
    provider_values = provider_forecast.drop(columns=["issue_time"])
    provider_values = provider_values.reindex(columns=settings.observation_columns)

    full_future_index = pd.date_range(
        start=history.index.max() + pd.Timedelta(minutes=settings.schedule_minutes),
        periods=settings.model_horizon_steps,
        freq=settings.interval,
        tz="UTC",
    )
    provider_values = provider_values.reindex(full_future_index).interpolate(method="time").ffill().bfill()

    bundles = {target: _load_model_bundle(target, lat, lon, settings) for target in settings.target_columns}
    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    total_steps = len(full_future_index)

    # Track previous predicted value per target for rate-of-change clamping
    prev_predicted: dict[str, float] = {}

    for step_idx, forecast_time in enumerate(full_future_index):
        seed_row = provider_values.loc[forecast_time].copy() if forecast_time in provider_values.index else pd.Series(dtype=float)
        history.loc[forecast_time, settings.observation_columns] = seed_row.reindex(settings.observation_columns).values

        feature_row = build_single_feature_row(history, forecast_time, settings)
        forecast_row: dict[str, object] = {
            "issue_time": provider_issue_time.to_pydatetime(),
            "forecast_time": forecast_time.to_pydatetime(),
            "lat": lat,
            "lon": lon,
            "forecast_type": "ml",
            "source": settings.model_algorithm,
            "model_version": "",
        }

        w_ml = _ml_weight(step_idx, total_steps)

        for target, (model, metadata) in bundles.items():
            feature_names = metadata["feature_names"]
            row = feature_row.reindex(columns=feature_names, fill_value=0.0)
            raw_ml = float(model.predict(row)[0])

            # Get provider value as the anchor/base
            provider_val = seed_row.get(target, np.nan) if not seed_row.empty else np.nan

            # ── Physical constraints on raw ML prediction ────
            hour = forecast_time.hour if hasattr(forecast_time, 'hour') else pd.Timestamp(forecast_time).hour
            if target in ("shortwave_radiation", "diffuse_radiation"):
                if hour < 5 or hour > 19:
                    raw_ml = 0.0
                raw_ml = max(0.0, raw_ml)
            elif target == "cloud_cover":
                raw_ml = max(0.0, min(100.0, raw_ml))
            elif target == "wind_speed_10m":
                raw_ml = max(0.0, raw_ml)

            # ── Delta-based ML correction ────
            # ML predicts a correction (delta) on top of provider forecast,
            # preventing the model from drifting far from physical reality.
            if np.isfinite(provider_val):
                delta = raw_ml - provider_val
                blended = provider_val + w_ml * delta
            else:
                # No provider value available — fall back to raw ML
                blended = raw_ml

            # ── Max rate-of-change constraint ────
            if target in _MAX_DELTA_PER_STEP and target in prev_predicted:
                max_change = _MAX_DELTA_PER_STEP[target]
                prev_val = prev_predicted[target]
                blended = max(prev_val - max_change, min(prev_val + max_change, blended))

            # ── Re-apply physical constraints after blending & clamping ────
            if target in ("shortwave_radiation", "diffuse_radiation"):
                if hour < 5 or hour > 19:
                    blended = 0.0
                blended = max(0.0, blended)
            elif target == "cloud_cover":
                blended = max(0.0, min(100.0, blended))
            elif target == "wind_speed_10m":
                blended = max(0.0, blended)

            # Update previous predicted value tracker
            prev_predicted[target] = blended

            # ── Inertia-based feedback ────
            # When feeding back to history for next step's lag features, blend
            # with the previous history value to dampen autoregressive oscillation.
            prev_history_val = history.at[forecast_time, target] if forecast_time in history.index and pd.notna(history.at[forecast_time, target]) else np.nan
            if np.isfinite(prev_history_val):
                feedback_value = 0.7 * blended + 0.3 * prev_history_val
            else:
                feedback_value = blended
            history.at[forecast_time, target] = feedback_value

            forecast_row[target] = blended
            forecast_row["model_version"] = metadata["model_version"]
            long_rows.append(
                {
                    "issue_time": provider_issue_time.to_pydatetime(),
                    "forecast_time": forecast_time.to_pydatetime(),
                    "target": target,
                    "predicted_value": blended,
                    "model_version": metadata["model_version"],
                }
            )

        previous_row = history.shift(1).loc[forecast_time]
        history.loc[forecast_time] = history.loc[forecast_time].fillna(previous_row)
        wide_rows.append(forecast_row)

    forecast_frame = pd.DataFrame(wide_rows)

    # ── Post-prediction Gaussian smoothing ─────────────────────────────
    # Gaussian-weighted rolling mean (window=11, sigma=3.0) produces much
    # smoother curves than a simple box filter while preserving trends.
    smooth_window = 11
    smooth_sigma = 3.0
    for target in settings.target_columns:
        if target in forecast_frame.columns:
            forecast_frame[target] = _gaussian_smooth(
                forecast_frame[target], window=smooth_window, sigma=smooth_sigma
            )
            # Re-apply physical constraints after smoothing
            if target in ("shortwave_radiation", "diffuse_radiation"):
                ft_col = pd.to_datetime(forecast_frame["forecast_time"])
                night_mask = ft_col.dt.hour.isin(list(range(0, 5)) + list(range(20, 24)))
                forecast_frame.loc[night_mask, target] = 0.0
                forecast_frame[target] = forecast_frame[target].clip(lower=0.0)
            elif target == "cloud_cover":
                forecast_frame[target] = forecast_frame[target].clip(0.0, 100.0)
            elif target == "wind_speed_10m":
                forecast_frame[target] = forecast_frame[target].clip(lower=0.0)
    logger.info("Applied Gaussian smoothing (window=%d, sigma=%.1f) to %d forecast rows", smooth_window, smooth_sigma, len(forecast_frame))

    # Update long_rows to match smoothed values
    long_rows = []
    for _, row_data in forecast_frame.iterrows():
        for target in settings.target_columns:
            if target in row_data:
                long_rows.append({
                    "issue_time": row_data["issue_time"],
                    "forecast_time": row_data["forecast_time"],
                    "target": target,
                    "predicted_value": row_data[target],
                    "model_version": row_data.get("model_version", ""),
                })

    upsert_dataframe(
        "weather_forecast",
        forecast_frame,
        conflict_columns=["issue_time", "forecast_time", "lat", "lon", "forecast_type", "source", "model_version"],
        settings=settings,
    )

    prediction_frame = pd.DataFrame(long_rows)
    upsert_dataframe(
        "weather_ml_predictions",
        prediction_frame,
        conflict_columns=["issue_time", "forecast_time", "target", "model_version"],
        settings=settings,
    )

    # Write to simplified weather_predictions table
    simplified_rows = []
    for _, row in forecast_frame.iterrows():
        simplified_rows.append({
            "latitude": lat,
            "longitude": lon,
            "timestamp": row["forecast_time"],
            "predicted_temperature": row.get("temperature_2m"),
            "predicted_cloud_cover": row.get("cloud_cover"),
            "predicted_wind_speed": row.get("wind_speed_10m"),
            "predicted_ghi": row.get("shortwave_radiation"),
            "predicted_dhi": row.get("diffuse_radiation"),
            "model_version": row.get("model_version", ""),
            "issue_time": row["issue_time"],
        })
    simplified_frame = pd.DataFrame(simplified_rows)
    upsert_dataframe(
        "weather_predictions",
        simplified_frame,
        conflict_columns=["latitude", "longitude", "timestamp", "issue_time"],
        settings=settings,
    )

    location_key = location_token(lat, lon)
    output_file = settings.reports_dir / f"latest_predictions_{location_key}.csv"
    if output_path:
        output_file = settings.project_root / output_path
    forecast_frame.to_csv(output_file, index=False)
    logger.info("Saved %s forecast rows to %s", len(forecast_frame), output_file)
    return {
        "predictions_written": int(len(forecast_frame)),
        "issue_time": provider_issue_time.isoformat(),
        "lat": lat,
        "lon": lon,
        "output_path": str(output_file),
    }
