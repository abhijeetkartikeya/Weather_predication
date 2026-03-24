from __future__ import annotations

import json

import joblib
import pandas as pd

from .db import ensure_schema, fetch_dataframe, upsert_dataframe
from .features import build_single_feature_row, load_observations
from .logging_utils import get_logger
from .settings import Settings, load_settings, location_token

logger = get_logger("weather_ml.prediction")


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

    for forecast_time in full_future_index:
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

        for target, (model, metadata) in bundles.items():
            feature_names = metadata["feature_names"]
            row = feature_row.reindex(columns=feature_names, fill_value=0.0)
            prediction = float(model.predict(row)[0])
            history.at[forecast_time, target] = prediction
            forecast_row[target] = prediction
            forecast_row["model_version"] = metadata["model_version"]
            long_rows.append(
                {
                    "issue_time": provider_issue_time.to_pydatetime(),
                    "forecast_time": forecast_time.to_pydatetime(),
                    "target": target,
                    "predicted_value": prediction,
                    "model_version": metadata["model_version"],
                }
            )

        previous_row = history.shift(1).loc[forecast_time]
        history.loc[forecast_time] = history.loc[forecast_time].fillna(previous_row)
        wide_rows.append(forecast_row)

    forecast_frame = pd.DataFrame(wide_rows)
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
    for row in wide_rows:
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
