"""
InfluxDB Writer — Write forecast predictions to InfluxDB for Grafana visualization.

Writes each forecast cycle's predictions as time-series points with 15-min resolution.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config import (
    INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET,
    TARGET_COLUMNS,
)

logger = logging.getLogger(__name__)


def _get_client():
    """Get InfluxDB client. Returns None if library not available."""
    try:
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
        )
        return client
    except ImportError:
        logger.error("influxdb-client not installed — pip install influxdb-client")
        return None
    except Exception as e:
        logger.error("InfluxDB connection failed: %s", e)
        return None


def ensure_bucket_exists():
    """Create the bucket if it doesn't exist."""
    client = _get_client()
    if client is None:
        return False
    try:
        buckets_api = client.buckets_api()
        existing = buckets_api.find_bucket_by_name(INFLUXDB_BUCKET)
        if existing is None:
            org = client.organizations_api().find_organizations(org=INFLUXDB_ORG)
            if org:
                buckets_api.create_bucket(
                    bucket_name=INFLUXDB_BUCKET,
                    org_id=org[0].id,
                    retention_rules=[],
                )
                logger.info("Created InfluxDB bucket: %s", INFLUXDB_BUCKET)
        return True
    except Exception as e:
        logger.warning("Could not ensure bucket: %s", e)
        return False
    finally:
        client.close()


def write_forecast_to_influxdb(
    df: pd.DataFrame,
    source: str = "ml_model",
    model_version: str = "lgbm_v2",
) -> bool:
    """
    Write forecast DataFrame to InfluxDB.

    Args:
        df: DataFrame with timestamp + target columns (ghi, dhi, temperature, windspeed, cloud_cover)
        source: Tag value — 'ml_model', 'openmeteo', or 'vedas'
        model_version: Model version tag
    """
    client = _get_client()
    if client is None:
        logger.warning("Skipping InfluxDB write — client unavailable")
        return False

    try:
        from influxdb_client import Point, WritePrecision
        from influxdb_client.client.write_api import SYNCHRONOUS

        write_api = client.write_api(write_options=SYNCHRONOUS)

        points = []
        for _, row in df.iterrows():
            ts = pd.to_datetime(row["timestamp"])
            p = (
                Point("weather_forecast")
                .tag("source", source)
                .tag("model_version", model_version)
                .tag("location", "anand_gujarat")
            )
            for col in TARGET_COLUMNS:
                if col in row and pd.notna(row[col]):
                    p = p.field(col, float(row[col]))
            p = p.time(ts, WritePrecision.S)
            points.append(p)

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)
        logger.info("Wrote %d points to InfluxDB (source=%s)", len(points), source)
        return True

    except Exception as e:
        logger.error("InfluxDB write failed: %s", e)
        return False
    finally:
        client.close()


def write_vedas_to_influxdb(vedas_df: pd.DataFrame) -> bool:
    """Write VEDAS forecast data to InfluxDB for Grafana comparison."""
    if vedas_df is None or vedas_df.empty:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        from influxdb_client import Point, WritePrecision
        from influxdb_client.client.write_api import SYNCHRONOUS

        write_api = client.write_api(write_options=SYNCHRONOUS)
        points = []

        for _, row in vedas_df.iterrows():
            ts = pd.to_datetime(row["timestamp"])
            p = (
                Point("weather_forecast")
                .tag("source", "vedas")
                .tag("location", "anand_gujarat")
            )
            if "vedas_temperature" in row:
                p = p.field("temperature", float(row["vedas_temperature"]))
            if "vedas_solar_potential" in row:
                p = p.field("solar_potential", float(row["vedas_solar_potential"]))
            p = p.time(ts, WritePrecision.S)
            points.append(p)

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)
        logger.info("Wrote %d VEDAS points to InfluxDB", len(points))
        return True
    except Exception as e:
        logger.error("VEDAS InfluxDB write failed: %s", e)
        return False
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ok = ensure_bucket_exists()
    print(f"Bucket ready: {ok}")
