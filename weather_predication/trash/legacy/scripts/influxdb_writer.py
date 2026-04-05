"""
InfluxDB Writer for Weather Forecasting Pipeline
====================================================
Writes 15-minute forecast predictions to InfluxDB for
visualization in Grafana or InfluxDB's built-in explorer.

Usage:
    from influxdb_writer import write_forecasts_to_influxdb
    write_forecasts_to_influxdb(forecast_df)
"""

import pandas as pd
from datetime import datetime

from config import (
    INFLUXDB_BUCKET,
    INFLUXDB_MEASUREMENT,
    INFLUXDB_ORG,
    INFLUXDB_TOKEN,
    INFLUXDB_URL,
    LOCATION_NAME,
    TARGET_DISPLAY_NAMES,
)
from utils import setup_logger

logger = setup_logger("influxdb")

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logger.warning("influxdb-client not installed. Run: pip install influxdb-client")


def get_influxdb_client():
    """Create and return an InfluxDB client."""
    if not INFLUXDB_AVAILABLE:
        raise ImportError("influxdb-client package not installed")
    return InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
    )


def ensure_bucket_exists():
    """Create the bucket if it does not exist."""
    client = get_influxdb_client()
    buckets_api = client.buckets_api()

    existing = buckets_api.find_bucket_by_name(INFLUXDB_BUCKET)
    if existing is None:
        logger.info(f"Creating InfluxDB bucket: {INFLUXDB_BUCKET}")
        from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules
        retention = BucketRetentionRules(type="expire", every_seconds=0)  # infinite
        buckets_api.create_bucket(
            bucket_name=INFLUXDB_BUCKET,
            retention_rules=retention,
            org=INFLUXDB_ORG,
        )
        logger.info(f"✓ Bucket created: {INFLUXDB_BUCKET}")
    else:
        logger.info(f"Bucket already exists: {INFLUXDB_BUCKET}")

    client.close()


def write_forecasts_to_influxdb(forecast_df: pd.DataFrame):
    """
    Write forecast predictions to InfluxDB at 15-min intervals.

    Each row in forecast_df becomes one InfluxDB point with:
    - measurement: weather_predictions
    - time: forecast_time (the future time being predicted)
    - tags: target, horizon, location
    - fields: predicted_value

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output from predict.py's generate_forecast().
        Must have columns: forecast_time, target, horizon, predicted_value
    """
    if not INFLUXDB_AVAILABLE:
        logger.error("influxdb-client not installed. Skipping InfluxDB write.")
        return

    if forecast_df.empty:
        logger.warning("Empty forecast DataFrame, nothing to write.")
        return

    client = get_influxdb_client()
    write_api = client.write_api(write_options=SYNCHRONOUS)

    points = []
    for _, row in forecast_df.iterrows():
        forecast_time = pd.to_datetime(row["forecast_time"])

        point = (
            Point(INFLUXDB_MEASUREMENT)
            .tag("target", row["target"])
            .tag("target_display", row.get("target_display", row["target"]))
            .tag("horizon", row["horizon"])
            .tag("location", LOCATION_NAME)
            .field("predicted_value", float(row["predicted_value"]))
            .time(forecast_time, WritePrecision.S)
        )
        points.append(point)

    # Write in batches of 5000
    batch_size = 5000
    total_written = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        write_api.write(bucket=INFLUXDB_BUCKET, record=batch)
        total_written += len(batch)

    logger.info(f"✓ Wrote {total_written:,} points to InfluxDB ({INFLUXDB_BUCKET})")

    client.close()


def write_actuals_to_influxdb(df: pd.DataFrame, target_columns: list):
    """
    Write actual observed weather data to InfluxDB for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Observed data with DatetimeIndex and target columns.
    target_columns : list
        Which columns to write as actuals.
    """
    if not INFLUXDB_AVAILABLE:
        logger.error("influxdb-client not installed. Skipping.")
        return

    client = get_influxdb_client()
    write_api = client.write_api(write_options=SYNCHRONOUS)

    points = []
    for col in target_columns:
        if col not in df.columns:
            continue
        display = TARGET_DISPLAY_NAMES.get(col, col)
        for idx, val in df[col].dropna().items():
            point = (
                Point("weather_actuals")
                .tag("target", col)
                .tag("target_display", display)
                .tag("location", LOCATION_NAME)
                .field("actual_value", float(val))
                .time(idx, WritePrecision.S)
            )
            points.append(point)

    # Write in batches
    batch_size = 5000
    total_written = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        write_api.write(bucket=INFLUXDB_BUCKET, record=batch)
        total_written += len(batch)

    logger.info(f"✓ Wrote {total_written:,} actuals to InfluxDB")

    client.close()


def query_latest_forecasts(target: str = None, horizon: str = None) -> pd.DataFrame:
    """
    Query the latest forecasts from InfluxDB.

    Parameters
    ----------
    target : str, optional
        Filter by target variable.
    horizon : str, optional
        Filter by horizon (24h, 48h, 72h).

    Returns
    -------
    pd.DataFrame
        Recent forecasts.
    """
    if not INFLUXDB_AVAILABLE:
        logger.error("influxdb-client not installed.")
        return pd.DataFrame()

    client = get_influxdb_client()
    query_api = client.query_api()

    filters = ""
    if target:
        filters += f' |> filter(fn: (r) => r["target"] == "{target}")'
    if horizon:
        filters += f' |> filter(fn: (r) => r["horizon"] == "{horizon}")'

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -3d)
        |> filter(fn: (r) => r["_measurement"] == "{INFLUXDB_MEASUREMENT}")
        {filters}
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    '''

    result = query_api.query_data_frame(query)
    client.close()

    return result
