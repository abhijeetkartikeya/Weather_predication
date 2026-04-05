from __future__ import annotations

import time
from datetime import timezone

import pandas as pd

from .collection import run_collection_cycle
from .db import ensure_schema, fetch_dataframe
from .features import materialize_feature_store
from .logging_utils import get_logger
from .prediction import generate_predictions
from .settings import Settings, load_settings
from .training import models_are_stale, train_models

logger = get_logger("weather_ml.orchestration")

# India bounding box for input validation
INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 37.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 98.0


def validate_india_location(lat: float, lon: float) -> None:
    if not (INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX and INDIA_LON_MIN <= lon <= INDIA_LON_MAX):
        raise ValueError(
            f"Location ({lat}, {lon}) is outside India. "
            f"Valid range: lat [{INDIA_LAT_MIN}-{INDIA_LAT_MAX}], lon [{INDIA_LON_MIN}-{INDIA_LON_MAX}]"
        )


def run_pipeline_cycle(
    settings: Settings | None = None,
    full_backfill: bool = False,
    force_retrain: bool = False,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat = float(lat if lat is not None else settings.latitude)
    lon = float(lon if lon is not None else settings.longitude)
    validate_india_location(lat, lon)

    summary: dict[str, object] = {}
    summary["collection"] = run_collection_cycle(settings=settings, full_backfill=full_backfill, lat=lat, lon=lon)
    summary["features"] = materialize_feature_store(settings=settings, lat=lat, lon=lon)

    if force_retrain or models_are_stale(settings=settings, lat=lat, lon=lon):
        summary["training"] = train_models(settings=settings, force=force_retrain, lat=lat, lon=lon)
    else:
        summary["training"] = {"skipped": True}

    summary["prediction"] = generate_predictions(settings=settings, lat=lat, lon=lon)
    return summary


def run_ondemand_prediction(
    lat: float,
    lon: float,
    settings: Settings | None = None,
) -> list[dict]:
    """Run the full pipeline for an arbitrary Indian location and return forecast rows."""
    settings = settings or load_settings()
    validate_india_location(lat, lon)
    ensure_schema(settings)

    run_pipeline_cycle(settings=settings, lat=lat, lon=lon, force_retrain=False)

    query = f"""
        SELECT
            wf.forecast_time AS timestamp,
            wf.lat AS latitude,
            wf.lon AS longitude,
            wf.temperature_2m AS predicted_temperature,
            wf.cloud_cover AS predicted_cloud_cover,
            wf.wind_speed_10m AS predicted_wind_speed,
            wf.shortwave_radiation AS predicted_ghi,
            wf.diffuse_radiation AS predicted_dhi,
            wf.issue_time,
            wf.model_version
        FROM {settings.postgres_schema}.weather_forecast wf
        WHERE wf.forecast_type = 'ml'
          AND wf.lat = :lat
          AND wf.lon = :lon
          AND wf.issue_time = (
              SELECT MAX(issue_time)
              FROM {settings.postgres_schema}.weather_forecast
              WHERE forecast_type = 'ml'
                AND lat = :lat
                AND lon = :lon
          )
        ORDER BY wf.forecast_time
    """
    frame = fetch_dataframe(query, {"lat": lat, "lon": lon}, settings=settings, parse_dates=["timestamp", "issue_time"])
    if frame.empty:
        return []
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["issue_time"] = frame["issue_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame.to_dict(orient="records")


def fetch_stored_predictions(
    lat: float | None = None,
    lon: float | None = None,
    limit: int = 288,
    settings: Settings | None = None,
) -> list[dict]:
    """Fetch the latest stored ML predictions from the database."""
    settings = settings or load_settings()
    ensure_schema(settings)

    where_clauses = ["wf.forecast_type = 'ml'"]
    params: dict = {}

    if lat is not None and lon is not None:
        where_clauses.append("wf.lat = :lat")
        where_clauses.append("wf.lon = :lon")
        params["lat"] = float(lat)
        params["lon"] = float(lon)

    where_sql = " AND ".join(where_clauses)
    query = f"""
        SELECT
            wf.forecast_time AS timestamp,
            wf.lat AS latitude,
            wf.lon AS longitude,
            wf.temperature_2m AS predicted_temperature,
            wf.cloud_cover AS predicted_cloud_cover,
            wf.wind_speed_10m AS predicted_wind_speed,
            wf.shortwave_radiation AS predicted_ghi,
            wf.diffuse_radiation AS predicted_dhi,
            wf.issue_time,
            wf.model_version
        FROM {settings.postgres_schema}.weather_forecast wf
        WHERE {where_sql}
          AND wf.issue_time = (
              SELECT MAX(issue_time)
              FROM {settings.postgres_schema}.weather_forecast
              WHERE {where_sql}
          )
        ORDER BY wf.forecast_time
        LIMIT :limit
    """
    params["limit"] = limit
    frame = fetch_dataframe(query, params, settings=settings, parse_dates=["timestamp", "issue_time"])
    if frame.empty:
        return []
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["issue_time"] = frame["issue_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame.to_dict(orient="records")


def get_active_locations(settings: Settings) -> list[tuple[float, float]]:
    """Query all distinct locations that have predictions stored."""
    query = f"""
        SELECT DISTINCT latitude, longitude
        FROM {settings.postgres_schema}.weather_predictions
        ORDER BY latitude, longitude
    """
    frame = fetch_dataframe(query, settings=settings)
    if frame.empty:
        return []
    return [(float(row["latitude"]), float(row["longitude"])) for _, row in frame.iterrows()]


def _seconds_until_next_run(settings: Settings) -> float:
    now = pd.Timestamp.now(tz=timezone.utc)
    next_run = now.ceil(settings.interval)
    if next_run == now:
        next_run = now + pd.Timedelta(minutes=settings.schedule_minutes)
    return max((next_run - now).total_seconds(), 1.0)


def run_scheduler(
    settings: Settings | None = None,
    once: bool = False,
    full_backfill: bool = False,
    force_retrain: bool = False,
) -> None:
    settings = settings or load_settings()
    while True:
        started_at = pd.Timestamp.now(tz=timezone.utc)

        # Run default location first
        default_lat = settings.latitude
        default_lon = settings.longitude
        try:
            logger.info("Starting pipeline cycle for default location (%s, %s)", default_lat, default_lon)
            summary = run_pipeline_cycle(
                settings=settings,
                full_backfill=full_backfill,
                force_retrain=force_retrain,
                lat=default_lat,
                lon=default_lon,
            )
            logger.info("Default location cycle completed: %s", summary)
        except Exception as exc:
            logger.exception("Default location pipeline failed: %s", exc)

        # Refresh all other previously-requested locations
        try:
            active_locations = get_active_locations(settings)
            for loc_lat, loc_lon in active_locations:
                if abs(loc_lat - default_lat) < 0.001 and abs(loc_lon - default_lon) < 0.001:
                    continue  # Already processed
                try:
                    logger.info("Refreshing predictions for active location (%s, %s)", loc_lat, loc_lon)
                    loc_summary = run_pipeline_cycle(
                        settings=settings,
                        lat=loc_lat,
                        lon=loc_lon,
                    )
                    logger.info("Location (%s, %s) cycle completed: %s", loc_lat, loc_lon, loc_summary)
                except Exception as exc:
                    logger.exception("Pipeline failed for location (%s, %s): %s", loc_lat, loc_lon, exc)
        except Exception as exc:
            logger.exception("Failed to fetch active locations: %s", exc)

        if once:
            return

        full_backfill = False
        force_retrain = False
        sleep_seconds = _seconds_until_next_run(settings)
        logger.info("Sleeping %.0f seconds until the next scheduled cycle", sleep_seconds)
        time.sleep(sleep_seconds)
