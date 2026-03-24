from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .settings import Settings, load_settings


@lru_cache(maxsize=1)
def get_engine(database_url: str | None = None):
    settings = load_settings()
    return create_engine(database_url or settings.database_url, pool_pre_ping=True, future=True)


def ensure_schema(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    sql = Path(settings.schema_path).read_text(encoding="utf-8")
    raw_connection = get_engine(settings.database_url).raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            cursor.execute(sql)
        raw_connection.commit()
    finally:
        raw_connection.close()
    _apply_runtime_migrations(settings)


def execute_sql(statement: str, params: dict | None = None, settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    with get_engine(settings.database_url).begin() as connection:
        connection.execute(text(statement), params or {})


def table_exists(table_name: str, settings: Settings | None = None) -> bool:
    settings = settings or load_settings()
    return inspect(get_engine(settings.database_url)).has_table(table_name, schema=settings.postgres_schema)


def fetch_dataframe(
    query: str,
    params: dict | None = None,
    settings: Settings | None = None,
    parse_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    with get_engine(settings.database_url).connect() as connection:
        return pd.read_sql(text(query), connection, params=params or {}, parse_dates=list(parse_dates or []))


def upsert_dataframe(
    table_name: str,
    dataframe: pd.DataFrame,
    conflict_columns: list[str],
    settings: Settings | None = None,
) -> int:
    settings = settings or load_settings()
    if dataframe.empty:
        return 0

    engine = get_engine(settings.database_url)
    metadata = MetaData(schema=settings.postgres_schema)
    table = Table(table_name, metadata, autoload_with=engine)

    records = dataframe.replace({pd.NaT: None}).where(pd.notnull(dataframe), None).to_dict(orient="records")
    chunk_size = 1000
    rows_written = 0

    with engine.begin() as connection:
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            insert_stmt = pg_insert(table).values(chunk)
            update_map = {
                column.name: getattr(insert_stmt.excluded, column.name)
                for column in table.columns
                if column.name not in conflict_columns and not column.primary_key
            }
            if "updated_at" in table.columns:
                update_map["updated_at"] = text("CURRENT_TIMESTAMP")

            statement = insert_stmt.on_conflict_do_update(
                index_elements=conflict_columns,
                set_=update_map,
            )
            connection.execute(statement)
            rows_written += len(chunk)

    return rows_written


def insert_raw_snapshot(
    source: str,
    dataset_type: str,
    payload: dict,
    interval_start=None,
    interval_end=None,
    lat: float | None = None,
    lon: float | None = None,
    settings: Settings | None = None,
) -> None:
    settings = settings or load_settings()
    query = """
        INSERT INTO weather_raw_snapshots (source, dataset_type, lat, lon, interval_start, interval_end, payload)
        VALUES (:source, :dataset_type, :lat, :lon, :interval_start, :interval_end, CAST(:payload AS jsonb))
    """
    with get_engine(settings.database_url).begin() as connection:
        connection.execute(
            text(query),
            {
                "source": source,
                "dataset_type": dataset_type,
                "lat": lat,
                "lon": lon,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "payload": json.dumps(payload),
            },
        )


def latest_timestamp(table_name: str, time_column: str = "time", settings: Settings | None = None):
    settings = settings or load_settings()
    query = f"SELECT MAX({time_column}) AS latest_value FROM {settings.postgres_schema}.{table_name}"
    df = fetch_dataframe(query, settings=settings, parse_dates=["latest_value"])
    if df.empty or df["latest_value"].isna().all():
        return None
    return pd.Timestamp(df.iloc[0]["latest_value"])


def _apply_runtime_migrations(settings: Settings) -> None:
    engine = get_engine(settings.database_url)
    inspector = inspect(engine)

    with engine.begin() as connection:
        if inspector.has_table("weather_observations", schema=settings.postgres_schema):
            connection.execute(
                text(
                    f"""
                    UPDATE {settings.postgres_schema}.weather_observations
                    SET lat = COALESCE(lat, :lat),
                        lon = COALESCE(lon, :lon)
                    WHERE lat IS NULL OR lon IS NULL
                    """
                ),
                {"lat": settings.latitude, "lon": settings.longitude},
            )

        if inspector.has_table("weather_feature_store", schema=settings.postgres_schema):
            connection.execute(
                text(
                    f"""
                    UPDATE {settings.postgres_schema}.weather_feature_store
                    SET lat = COALESCE(lat, :lat),
                        lon = COALESCE(lon, :lon)
                    WHERE lat IS NULL OR lon IS NULL
                    """
                ),
                {"lat": settings.latitude, "lon": settings.longitude},
            )

        if (
            inspector.has_table("weather_external_forecasts", schema=settings.postgres_schema)
            and inspector.has_table("weather_forecast", schema=settings.postgres_schema)
        ):
            connection.execute(
                text(
                    f"""
                    INSERT INTO {settings.postgres_schema}.weather_forecast (
                        issue_time,
                        forecast_time,
                        lat,
                        lon,
                        forecast_type,
                        source,
                        model_version,
                        temperature_2m,
                        relative_humidity_2m,
                        dew_point_2m,
                        pressure_msl,
                        cloud_cover,
                        cloud_cover_low,
                        cloud_cover_mid,
                        cloud_cover_high,
                        wind_speed_10m,
                        wind_direction_10m,
                        wind_gusts_10m,
                        shortwave_radiation,
                        direct_radiation,
                        diffuse_radiation,
                        direct_normal_irradiance
                    )
                    SELECT
                        issue_time,
                        forecast_time,
                        :lat,
                        :lon,
                        'provider',
                        source,
                        '',
                        temperature_2m,
                        relative_humidity_2m,
                        dew_point_2m,
                        pressure_msl,
                        cloud_cover,
                        cloud_cover_low,
                        cloud_cover_mid,
                        cloud_cover_high,
                        wind_speed_10m,
                        wind_direction_10m,
                        wind_gusts_10m,
                        shortwave_radiation,
                        direct_radiation,
                        diffuse_radiation,
                        direct_normal_irradiance
                    FROM {settings.postgres_schema}.weather_external_forecasts
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"lat": settings.latitude, "lon": settings.longitude},
            )

        if (
            inspector.has_table("weather_ml_predictions", schema=settings.postgres_schema)
            and inspector.has_table("weather_forecast", schema=settings.postgres_schema)
        ):
            connection.execute(
                text(
                    f"""
                    INSERT INTO {settings.postgres_schema}.weather_forecast (
                        issue_time,
                        forecast_time,
                        lat,
                        lon,
                        forecast_type,
                        source,
                        model_version,
                        temperature_2m,
                        cloud_cover,
                        wind_speed_10m,
                        shortwave_radiation,
                        diffuse_radiation
                    )
                    SELECT
                        issue_time,
                        forecast_time,
                        :lat,
                        :lon,
                        'ml',
                        'xgboost',
                        model_version,
                        MAX(CASE WHEN target = 'temperature_2m' THEN predicted_value END) AS temperature_2m,
                        MAX(CASE WHEN target = 'cloud_cover' THEN predicted_value END) AS cloud_cover,
                        MAX(CASE WHEN target = 'wind_speed_10m' THEN predicted_value END) AS wind_speed_10m,
                        MAX(CASE WHEN target = 'shortwave_radiation' THEN predicted_value END) AS shortwave_radiation,
                        MAX(CASE WHEN target = 'diffuse_radiation' THEN predicted_value END) AS diffuse_radiation
                    FROM {settings.postgres_schema}.weather_ml_predictions
                    GROUP BY issue_time, forecast_time, model_version
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"lat": settings.latitude, "lon": settings.longitude},
            )
