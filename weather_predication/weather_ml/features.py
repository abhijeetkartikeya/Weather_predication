from __future__ import annotations

import numpy as np
import pandas as pd

from .db import ensure_schema, fetch_dataframe, upsert_dataframe
from .logging_utils import get_logger
from .settings import Settings, load_settings

logger = get_logger("weather_ml.features")


def _impute_observations(observations: pd.DataFrame) -> pd.DataFrame:
    filled = observations.copy()
    fallback_map = {
        "nasa_ghi": "shortwave_radiation",
        "nasa_dhi": "diffuse_radiation",
        "nasa_temperature": "temperature_2m",
        "nasa_wind_speed": "wind_speed_10m",
        "nasa_wind_direction": "wind_direction_10m",
        "nasa_humidity": "relative_humidity_2m",
        "nasa_cloud_cover": "cloud_cover",
        "nasa_pressure": "pressure_msl",
    }
    for target, source in fallback_map.items():
        if target in filled.columns and source in filled.columns:
            filled[target] = filled[target].fillna(filled[source])
    return filled.ffill().bfill()


def load_observations(
    settings: Settings | None = None,
    lookback_rows: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    start_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    lat = float(settings.latitude if lat is None else lat)
    lon = float(settings.longitude if lon is None else lon)
    limit_clause = f"LIMIT {int(lookback_rows)}" if lookback_rows else ""
    start_filter = "AND time >= :start_time" if start_time is not None else ""
    query = f"""
        SELECT *
        FROM (
            SELECT time, lat, lon, {", ".join(settings.observation_columns)}
            FROM {settings.postgres_schema}.weather_observations
            WHERE lat = :lat
              AND lon = :lon
              {start_filter}
            ORDER BY time DESC
            {limit_clause}
        ) recent
        ORDER BY time
    """
    params = {"lat": lat, "lon": lon}
    if start_time is not None:
        start_timestamp = pd.Timestamp(start_time)
        if start_timestamp.tzinfo is None:
            start_timestamp = start_timestamp.tz_localize("UTC")
        else:
            start_timestamp = start_timestamp.tz_convert("UTC")
        params["start_time"] = start_timestamp.to_pydatetime()
    frame = fetch_dataframe(query, params=params, settings=settings, parse_dates=["time"])
    if frame.empty:
        return frame
    return frame.set_index("time").sort_index()


def add_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + (index.minute / 60.0)
    day_of_week = index.dayofweek
    day_of_year = index.dayofyear

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7.0),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7.0),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "is_weekend": index.dayofweek.isin([5, 6]).astype(int),
        },
        index=index,
    )


def build_feature_matrix(
    observations: pd.DataFrame,
    settings: Settings | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or load_settings()
    if observations.empty:
        return pd.DataFrame(), pd.DataFrame()

    observations = _impute_observations(observations.copy().sort_index())
    observations = observations.reindex(columns=settings.observation_columns)
    feature_parts: list[pd.DataFrame] = [add_temporal_features(observations.index)]

    for column in settings.observation_columns:
        series = observations[column]
        shifted = series.shift(1)
        column_features = {
            f"{column}_lag_{lag}": series.shift(lag)
            for lag in settings.lag_steps
        }
        for window in settings.rolling_windows:
            rolling = shifted.rolling(window=window, min_periods=max(1, window // 2))
            column_features[f"{column}_roll_mean_{window}"] = rolling.mean()
            column_features[f"{column}_roll_std_{window}"] = rolling.std()
        feature_parts.append(pd.DataFrame(column_features, index=observations.index))

    if "wind_direction_10m" in observations.columns:
        radians = np.deg2rad(observations["wind_direction_10m"].shift(1).fillna(0))
        feature_parts.append(
            pd.DataFrame(
                {
                    "wind_direction_sin_lag1": np.sin(radians),
                    "wind_direction_cos_lag1": np.cos(radians),
                },
                index=observations.index,
            )
        )

    feature_frame = pd.concat(feature_parts, axis=1)
    target_frame = observations.loc[:, settings.target_columns].copy()
    combined = feature_frame.join(target_frame, how="left")
    if dropna:
        combined = combined.dropna()

    feature_columns = [column for column in combined.columns if column not in settings.target_columns]
    target_columns = list(settings.target_columns)
    return combined[feature_columns], combined[target_columns]


def build_single_feature_row(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    max_required_rows = max(settings.lag_steps + settings.rolling_windows) + 5
    window = history.loc[:timestamp].tail(max_required_rows).copy()
    features, _ = build_feature_matrix(window, settings=settings, dropna=False)
    row = features.loc[[timestamp]]
    return row.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def materialize_feature_store(
    settings: Settings | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat = float(settings.latitude if lat is None else lat)
    lon = float(settings.longitude if lon is None else lon)
    ensure_schema(settings)

    warmup_steps = max(settings.lag_steps + settings.rolling_windows) + 16
    latest_feature_query = f"""
        SELECT MAX(time) AS latest_value
        FROM {settings.postgres_schema}.weather_feature_store
        WHERE lat = :lat
          AND lon = :lon
          AND feature_set_version = :feature_set_version
    """
    latest_feature_frame = fetch_dataframe(
        latest_feature_query,
        {"lat": lat, "lon": lon, "feature_set_version": settings.feature_set_version},
        settings=settings,
        parse_dates=["latest_value"],
    )
    latest_feature_time = None
    if not latest_feature_frame.empty and latest_feature_frame["latest_value"].notna().any():
        latest_feature_time = pd.Timestamp(latest_feature_frame.iloc[0]["latest_value"])
    start_time = None
    if latest_feature_time is not None:
        start_time = latest_feature_time - (pd.Timedelta(minutes=settings.schedule_minutes) * warmup_steps)
    else:
        start_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=min(settings.training_lookback_days, 30))

    observations = load_observations(settings=settings, lat=lat, lon=lon, start_time=start_time)
    features, targets = build_feature_matrix(observations, settings=settings, dropna=True)

    if features.empty:
        logger.warning("Feature store skipped because there is not enough observation history yet")
        return {"feature_rows_written": 0}

    if latest_feature_time is not None:
        mask = features.index > latest_feature_time
        features = features.loc[mask]
        targets = targets.loc[mask]

    if features.empty:
        return {"feature_rows_written": 0, "lat": lat, "lon": lon, "incremental": True}

    records = [
        {
            "time": timestamp.to_pydatetime(),
            "lat": lat,
            "lon": lon,
            "feature_set_version": settings.feature_set_version,
            "features": {key: float(value) for key, value in row.items()},
            "targets": {key: float(value) for key, value in targets.loc[timestamp].items()},
        }
        for timestamp, row in features.iterrows()
    ]

    frame = pd.DataFrame(records)
    rows = upsert_dataframe(
        "weather_feature_store",
        frame,
        conflict_columns=["time", "lat", "lon", "feature_set_version"],
        settings=settings,
    )
    logger.info("Materialized %s feature rows for lat=%s lon=%s", rows, lat, lon)
    return {"feature_rows_written": rows, "lat": lat, "lon": lon}
