from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .db import ensure_schema, fetch_dataframe, insert_raw_snapshot, table_exists, upsert_dataframe
from .logging_utils import get_logger
from .settings import Settings, load_settings

logger = get_logger("weather_ml.collection")

NASA_PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": "nasa_ghi",
    "ALLSKY_SFC_SW_DIFF": "nasa_dhi",
    "T2M": "nasa_temperature",
    "WS10M": "nasa_wind_speed",
    "WD10M": "nasa_wind_direction",
    "RH2M": "nasa_humidity",
    "CLOUD_AMT": "nasa_cloud_cover",
    "PS": "nasa_pressure",
}

OPENMETEO_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]


def _time_step(settings: Settings) -> pd.Timedelta:
    return pd.Timedelta(minutes=settings.schedule_minutes)


def _session(settings: Settings) -> requests.Session:
    retry = Retry(
        total=settings.api_max_retries,
        backoff_factor=settings.api_retry_backoff_seconds,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_json(session: requests.Session, url: str, timeout: int) -> dict:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.warning("API request failed for %s: %s", url, exc)
        raise
    except ValueError as exc:
        logger.warning("API returned a non-JSON response for %s: %s", url, exc)
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type from {url}: {type(payload)!r}")
    return payload


def _to_utc_timestamp(value: Any, settings: Settings | None = None, round_to_interval: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if round_to_interval and settings is not None:
        timestamp = timestamp.floor(settings.interval)
    return timestamp


def _to_window_end(value: Any, settings: Settings) -> pd.Timestamp:
    timestamp = _to_utc_timestamp(value)
    if isinstance(value, str) and len(value) == 10:
        timestamp = timestamp + pd.Timedelta(days=1) - _time_step(settings)
    return timestamp.floor(settings.interval)


def _ensure_full_interval_index(
    frame: pd.DataFrame,
    settings: Settings,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    start = _to_utc_timestamp(start_time if start_time is not None else frame.index.min(), settings, round_to_interval=True)
    end = _to_utc_timestamp(end_time if end_time is not None else frame.index.max(), settings, round_to_interval=True)
    if end < start:
        return frame.iloc[0:0]

    full_index = pd.date_range(start=start, end=end, freq=settings.interval, tz="UTC")
    return frame.reindex(full_index).interpolate(method="time").ffill().bfill()


def _build_hourly_frame(payload: dict, columns: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time")
    return frame.reindex(columns=columns)


def _ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _resample_15min(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if frame.empty:
        return frame
    resampled = (
        frame.sort_index()
        .resample(settings.interval)
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    return _ensure_full_interval_index(resampled, settings=settings)


def _resolve_location(lat: float | None, lon: float | None, settings: Settings) -> tuple[float, float]:
    return float(lat if lat is not None else settings.latitude), float(lon if lon is not None else settings.longitude)


def fetch_nasa_history(
    start_date: str,
    end_date: str,
    lat: float | None = None,
    lon: float | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    lat, lon = _resolve_location(lat, lon, settings)
    session = _session(settings)
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        year_start = max(pd.Timestamp(start_date), pd.Timestamp(f"{year}-01-01"))
        year_end = min(pd.Timestamp(end_date), pd.Timestamp(f"{year}-12-31"))
        if year_start > year_end:
            continue

        url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            f"?parameters={','.join(NASA_PARAMETERS)}"
            f"&community=RE&longitude={lon}&latitude={lat}"
            f"&start={year_start.strftime('%Y%m%d')}&end={year_end.strftime('%Y%m%d')}&format=JSON"
        )
        payload = _get_json(session, url, settings.api_timeout_seconds)
        insert_raw_snapshot(
            source="nasa_power",
            dataset_type="historical",
            payload=payload,
            interval_start=year_start.to_pydatetime(),
            interval_end=year_end.to_pydatetime(),
            lat=lat,
            lon=lon,
            settings=settings,
        )

        parameter_data = payload.get("properties", {}).get("parameter", {})
        if not parameter_data:
            continue
        frame = pd.DataFrame(parameter_data)
        frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H", utc=True)
        frame = frame.rename(columns=NASA_PARAMETERS)
        frame = frame.replace([-999, -999.0], np.nan)
        frame = frame.reindex(columns=NASA_PARAMETERS.values())
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=NASA_PARAMETERS.values())

    merged = _ensure_utc_index(pd.concat(frames))
    return _resample_15min(merged, settings)


def fetch_openmeteo_archive(
    start_date: str,
    end_date: str,
    lat: float | None = None,
    lon: float | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    lat, lon = _resolve_location(lat, lon, settings)
    session = _session(settings)
    frames: list[pd.DataFrame] = []

    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current <= end:
        chunk_end = min(current + timedelta(days=settings.history_chunk_days - 1), end)
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={current.strftime('%Y-%m-%d')}&end_date={chunk_end.strftime('%Y-%m-%d')}"
            f"&hourly={','.join(OPENMETEO_COLUMNS)}&timezone=UTC"
        )
        payload = _get_json(session, url, settings.api_timeout_seconds)
        insert_raw_snapshot(
            source="open_meteo",
            dataset_type="archive",
            payload=payload,
            interval_start=current.to_pydatetime(),
            interval_end=chunk_end.to_pydatetime(),
            lat=lat,
            lon=lon,
            settings=settings,
        )
        frame = _build_hourly_frame(payload, OPENMETEO_COLUMNS)
        if not frame.empty:
            frames.append(frame)
        current = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OPENMETEO_COLUMNS)

    merged = _ensure_utc_index(pd.concat(frames))
    return _resample_15min(merged, settings)


def fetch_weather(
    lat: float,
    lon: float,
    settings: Settings | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    settings = settings or load_settings()
    session = _session(settings)
    issue_time = pd.Timestamp.now(tz=timezone.utc).floor(settings.interval)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={','.join(OPENMETEO_COLUMNS)}"
        f"&past_days={settings.openmeteo_past_days}&forecast_days={settings.forecast_days}"
        "&timezone=UTC"
    )
    payload = _get_json(session, url, settings.api_timeout_seconds)
    insert_raw_snapshot(
        source="open_meteo",
        dataset_type="live_forecast",
        payload=payload,
        interval_start=issue_time.to_pydatetime(),
        interval_end=(issue_time + timedelta(days=settings.forecast_days)).to_pydatetime(),
        lat=lat,
        lon=lon,
        settings=settings,
    )

    frame = _build_hourly_frame(payload, OPENMETEO_COLUMNS)
    if frame.empty:
        raise RuntimeError("Open-Meteo forecast payload did not contain hourly weather data.")
    frame = _resample_15min(frame, settings)

    actuals = frame[frame.index <= issue_time]
    forecasts = frame[frame.index > issue_time]
    if forecasts.empty:
        raise RuntimeError("Open-Meteo forecast payload did not contain any future forecast rows.")
    return issue_time, actuals, forecasts


def fetch_openmeteo_live(settings: Settings | None = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    settings = settings or load_settings()
    issue_time, actuals, forecasts = fetch_weather(settings.latitude, settings.longitude, settings)
    return issue_time, pd.concat([actuals, forecasts]).sort_index()


def merge_historical_sources(openmeteo_df: pd.DataFrame, nasa_df: pd.DataFrame) -> pd.DataFrame:
    if openmeteo_df.empty and nasa_df.empty:
        return pd.DataFrame()
    if nasa_df.empty:
        return openmeteo_df
    if openmeteo_df.empty:
        return nasa_df
    return openmeteo_df.join(nasa_df, how="outer").sort_index()


def _prepare_observations(frame: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output.index.name = "time"
    output = output.reset_index()
    output.insert(1, "lat", lat)
    output.insert(2, "lon", lon)
    return output


def _prepare_forecast(
    issue_time: pd.Timestamp,
    frame: pd.DataFrame,
    lat: float,
    lon: float,
    forecast_type: str,
    source: str,
    model_version: str = "",
) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output.index.name = "forecast_time"
    output = output.reset_index()
    output.insert(0, "issue_time", issue_time.to_pydatetime())
    output.insert(2, "lat", lat)
    output.insert(3, "lon", lon)
    output.insert(4, "forecast_type", forecast_type)
    output.insert(5, "source", source)
    output.insert(6, "model_version", model_version)
    return output


def _latest_location_timestamp(
    table_name: str,
    time_column: str,
    lat: float,
    lon: float,
    settings: Settings,
    extra_filters: str = "",
    params: dict[str, Any] | None = None,
) -> pd.Timestamp | None:
    query = f"""
        SELECT MAX({time_column}) AS latest_value
        FROM {settings.postgres_schema}.{table_name}
        WHERE lat = :lat
          AND lon = :lon
          {extra_filters}
    """
    frame = fetch_dataframe(
        query,
        {"lat": lat, "lon": lon, **(params or {})},
        settings=settings,
        parse_dates=["latest_value"],
    )
    if frame.empty or frame["latest_value"].isna().all():
        return None
    return _to_utc_timestamp(frame.iloc[0]["latest_value"], settings=settings, round_to_interval=True)


def _missing_observation_ranges(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    lat: float,
    lon: float,
    settings: Settings,
) -> list[dict[str, pd.Timestamp]]:
    if end_time < start_time:
        return []

    query = f"""
        WITH expected AS (
            SELECT generate_series(
                :start_time,
                :end_time,
                CAST(:step_interval AS interval)
            ) AS bucket_time
        ),
        missing AS (
            SELECT e.bucket_time
            FROM expected e
            LEFT JOIN {settings.postgres_schema}.weather_observations o
                ON o.time = e.bucket_time
               AND o.lat = :lat
               AND o.lon = :lon
            WHERE o.time IS NULL
        ),
        grouped AS (
            SELECT
                bucket_time,
                bucket_time - (
                    ROW_NUMBER() OVER (ORDER BY bucket_time) * CAST(:step_interval AS interval)
                ) AS gap_group
            FROM missing
        )
        SELECT
            MIN(bucket_time) AS gap_start,
            MAX(bucket_time) AS gap_end
        FROM grouped
        GROUP BY gap_group
        ORDER BY gap_start
    """
    frame = fetch_dataframe(
        query,
        {
            "start_time": start_time.to_pydatetime(),
            "end_time": end_time.to_pydatetime(),
            "lat": lat,
            "lon": lon,
            "step_interval": f"{settings.schedule_minutes} minutes",
        },
        settings=settings,
        parse_dates=["gap_start", "gap_end"],
    )
    if frame.empty:
        return []
    return [
        {
            "start": _to_utc_timestamp(row["gap_start"], settings=settings, round_to_interval=True),
            "end": _to_utc_timestamp(row["gap_end"], settings=settings, round_to_interval=True),
        }
        for _, row in frame.iterrows()
    ]


def _missing_forecast_ranges(
    issue_time: pd.Timestamp,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    lat: float,
    lon: float,
    settings: Settings,
    forecast_type: str = "provider",
    source: str = "open_meteo",
) -> list[dict[str, pd.Timestamp]]:
    if end_time < start_time:
        return []

    query = f"""
        WITH expected AS (
            SELECT generate_series(
                :start_time,
                :end_time,
                CAST(:step_interval AS interval)
            ) AS bucket_time
        ),
        missing AS (
            SELECT e.bucket_time
            FROM expected e
            LEFT JOIN {settings.postgres_schema}.weather_forecast wf
                ON wf.issue_time = :issue_time
               AND wf.forecast_time = e.bucket_time
               AND wf.lat = :lat
               AND wf.lon = :lon
               AND wf.forecast_type = :forecast_type
               AND wf.source = :source
               AND wf.model_version = :model_version
            WHERE wf.forecast_time IS NULL
        ),
        grouped AS (
            SELECT
                bucket_time,
                bucket_time - (
                    ROW_NUMBER() OVER (ORDER BY bucket_time) * CAST(:step_interval AS interval)
                ) AS gap_group
            FROM missing
        )
        SELECT
            MIN(bucket_time) AS gap_start,
            MAX(bucket_time) AS gap_end
        FROM grouped
        GROUP BY gap_group
        ORDER BY gap_start
    """
    frame = fetch_dataframe(
        query,
        {
            "issue_time": issue_time.to_pydatetime(),
            "start_time": start_time.to_pydatetime(),
            "end_time": end_time.to_pydatetime(),
            "lat": lat,
            "lon": lon,
            "forecast_type": forecast_type,
            "source": source,
            "model_version": "",
            "step_interval": f"{settings.schedule_minutes} minutes",
        },
        settings=settings,
        parse_dates=["gap_start", "gap_end"],
    )
    if frame.empty:
        return []
    return [
        {
            "start": _to_utc_timestamp(row["gap_start"], settings=settings, round_to_interval=True),
            "end": _to_utc_timestamp(row["gap_end"], settings=settings, round_to_interval=True),
        }
        for _, row in frame.iterrows()
    ]


def _format_gap_ranges(ranges: list[dict[str, pd.Timestamp]]) -> list[dict[str, str]]:
    return [{"start": item["start"].isoformat(), "end": item["end"].isoformat()} for item in ranges]


def _write_observation_backfill(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    lat: float,
    lon: float,
    settings: Settings,
) -> dict[str, object]:
    if end_time < start_time:
        return {"start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "observations_written": 0}

    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    openmeteo_df = fetch_openmeteo_archive(start_date, end_date, lat=lat, lon=lon, settings=settings)
    nasa_df = fetch_nasa_history(start_date, end_date, lat=lat, lon=lon, settings=settings)
    merged = merge_historical_sources(openmeteo_df, nasa_df)
    if not merged.empty:
        merged = merged[(merged.index >= start_time) & (merged.index <= end_time)]
        merged = _ensure_full_interval_index(merged, settings=settings, start_time=start_time, end_time=end_time)

    rows = upsert_dataframe(
        "weather_observations",
        _prepare_observations(merged, lat, lon),
        conflict_columns=["time", "lat", "lon"],
        settings=settings,
    )
    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "observations_written": rows,
    }


def recover_missing_data(
    settings: Settings | None = None,
    full_backfill: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, object]:
    settings = settings or load_settings()
    lat, lon = _resolve_location(lat, lon, settings)
    ensure_schema(settings)

    now = pd.Timestamp.now(tz=timezone.utc).floor(settings.interval)
    step = _time_step(settings)
    latest_observation_time = _latest_location_timestamp("weather_observations", "time", lat, lon, settings)

    summary: dict[str, object] = {
        "lat": lat,
        "lon": lon,
        "current_time": now.isoformat(),
        "latest_observation_time": latest_observation_time.isoformat() if latest_observation_time is not None else None,
    }

    if full_backfill or latest_observation_time is None:
        historical_start = _to_utc_timestamp(start_date or settings.history_start_date)
        historical_end = _to_window_end(end_date, settings) if end_date else now
        summary["observation_gaps"] = (
            [{"start": historical_start.isoformat(), "end": historical_end.isoformat()}] if historical_end >= historical_start else []
        )
        summary["historical"] = _write_observation_backfill(
            start_time=historical_start,
            end_time=historical_end,
            lat=lat,
            lon=lon,
            settings=settings,
        )
    else:
        lookback_start = max(
            _to_utc_timestamp(settings.history_start_date),
            now - pd.Timedelta(days=settings.recovery_lookback_days),
        )
        observation_gaps = _missing_observation_ranges(lookback_start, now, lat, lon, settings)
        summary["observation_gaps"] = _format_gap_ranges(observation_gaps)

        if observation_gaps:
            recovery_start = observation_gaps[0]["start"]
        elif latest_observation_time < now:
            recovery_start = latest_observation_time + step
        else:
            recovery_start = None

        if recovery_start is not None:
            summary["historical"] = _write_observation_backfill(
                start_time=recovery_start,
                end_time=now,
                lat=lat,
                lon=lon,
                settings=settings,
            )
        else:
            summary["historical"] = {"observations_written": 0, "skipped": True}

    live_summary = update_realtime_and_forecast(settings=settings, lat=lat, lon=lon)
    summary["live"] = live_summary

    issue_time = _to_utc_timestamp(live_summary["issue_time"], settings=settings, round_to_interval=True)
    forecast_start = issue_time + step
    forecast_end = issue_time + pd.Timedelta(days=settings.forecast_days)
    forecast_gaps = _missing_forecast_ranges(issue_time, forecast_start, forecast_end, lat, lon, settings)
    summary["forecast_gaps"] = _format_gap_ranges(forecast_gaps)
    return summary


def backfill_historical_data(
    start_date: str | None = None,
    end_date: str | None = None,
    settings: Settings | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat, lon = _resolve_location(lat, lon, settings)
    ensure_schema(settings)

    start_date = start_date or settings.history_start_date
    end_date = end_date or pd.Timestamp.now(tz=timezone.utc).floor(settings.interval).strftime("%Y-%m-%d")

    logger.info("Starting historical backfill for lat=%s lon=%s from %s to %s", lat, lon, start_date, end_date)
    result = _write_observation_backfill(
        start_time=_to_utc_timestamp(start_date),
        end_time=_to_window_end(end_date, settings),
        lat=lat,
        lon=lon,
        settings=settings,
    )
    logger.info("Historical backfill finished with %s rows written", result["observations_written"])
    return {"lat": lat, "lon": lon, **result}


def update_realtime_and_forecast(
    settings: Settings | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat, lon = _resolve_location(lat, lon, settings)
    ensure_schema(settings)

    issue_time, actuals, forecasts = fetch_weather(lat, lon, settings)
    actual_rows = upsert_dataframe(
        "weather_observations",
        _prepare_observations(actuals, lat, lon),
        conflict_columns=["time", "lat", "lon"],
        settings=settings,
    )
    forecast_rows = upsert_dataframe(
        "weather_forecast",
        _prepare_forecast(
            issue_time=issue_time,
            frame=forecasts,
            lat=lat,
            lon=lon,
            forecast_type="provider",
            source="open_meteo",
        ),
        conflict_columns=["issue_time", "forecast_time", "lat", "lon", "forecast_type", "source", "model_version"],
        settings=settings,
    )

    logger.info(
        "Realtime update wrote %s actual rows and %s provider forecast rows for lat=%s lon=%s",
        actual_rows,
        forecast_rows,
        lat,
        lon,
    )
    return {
        "issue_time": issue_time.isoformat(),
        "lat": lat,
        "lon": lon,
        "actual_rows_written": actual_rows,
        "external_forecast_rows_written": forecast_rows,
    }


def run_collection_cycle(
    settings: Settings | None = None,
    full_backfill: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat, lon = _resolve_location(lat, lon, settings)
    ensure_schema(settings)

    has_data = False
    if table_exists("weather_observations", settings):
        count_query = f"""
            SELECT COUNT(*) AS row_count
            FROM {settings.postgres_schema}.weather_observations
            WHERE lat = :lat
              AND lon = :lon
        """
        count_frame = fetch_dataframe(count_query, {"lat": lat, "lon": lon}, settings=settings)
        has_data = not count_frame.empty and int(count_frame.iloc[0]["row_count"]) > 0

    return recover_missing_data(
        settings=settings,
        full_backfill=full_backfill or not has_data,
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
    )
