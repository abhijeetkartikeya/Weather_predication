"""
Utility Functions for Weather Forecasting Pipeline
====================================================
Shared helpers: API requests, solar position, data validation, logging.
"""

import logging
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import (
    API_MAX_RETRIES,
    API_RETRY_DELAY,
    API_TIMEOUT,
    LATITUDE,
    LONGITUDE,
)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# ─────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────

def api_request_with_retry(
    url: str,
    params: dict = None,
    max_retries: int = API_MAX_RETRIES,
    retry_delay: int = API_RETRY_DELAY,
    timeout: int = API_TIMEOUT,
    logger: logging.Logger = None,
) -> dict:
    """Make an API GET request with exponential backoff retry."""
    log = logger or setup_logger("api")

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = retry_delay * (2 ** (attempt - 1))
                log.warning(f"Rate limited. Retrying in {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
            elif 500 <= response.status_code < 600:
                wait = retry_delay * (2 ** (attempt - 1))
                log.warning(f"Server error {response.status_code}. Retrying in {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
            else:
                log.error(f"HTTP error: {e}")
                raise

        except requests.exceptions.ConnectionError:
            wait = retry_delay * (2 ** (attempt - 1))
            log.warning(f"Connection error. Retrying in {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = retry_delay * (2 ** (attempt - 1))
            log.warning(f"Request timeout. Retrying in {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


# ─────────────────────────────────────────────
# Solar Position Calculator
# ─────────────────────────────────────────────

def solar_declination(day_of_year: int) -> float:
    """Solar declination angle in radians."""
    return math.radians(23.45) * math.sin(math.radians(360 / 365 * (day_of_year - 81)))


def equation_of_time(day_of_year: int) -> float:
    """Equation of time correction in minutes."""
    b = math.radians(360 / 365 * (day_of_year - 81))
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def solar_hour_angle(hour: float, longitude: float, day_of_year: int, timezone_offset: float = 5.5) -> float:
    """Solar hour angle in radians."""
    lstm = 15 * timezone_offset  # Local Standard Time Meridian
    eot = equation_of_time(day_of_year)
    time_correction = 4 * (longitude - lstm) + eot
    local_solar_time = hour + time_correction / 60
    return math.radians(15 * (local_solar_time - 12))


def solar_zenith_angle(
    latitude: float,
    longitude: float,
    day_of_year: int,
    hour: float,
    timezone_offset: float = 5.5,
) -> float:
    """Solar zenith angle in degrees."""
    lat_rad = math.radians(latitude)
    decl = solar_declination(day_of_year)
    ha = solar_hour_angle(hour, longitude, day_of_year, timezone_offset)
    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    )
    cos_zenith = max(-1, min(1, cos_zenith))
    return math.degrees(math.acos(cos_zenith))


def solar_elevation_angle(
    latitude: float,
    longitude: float,
    day_of_year: int,
    hour: float,
    timezone_offset: float = 5.5,
) -> float:
    """Solar elevation angle in degrees."""
    return 90.0 - solar_zenith_angle(latitude, longitude, day_of_year, hour, timezone_offset)


def day_length_hours(latitude: float, day_of_year: int) -> float:
    """Day length in hours."""
    lat_rad = math.radians(latitude)
    decl = solar_declination(day_of_year)
    cos_ha = -math.tan(lat_rad) * math.tan(decl)
    cos_ha = max(-1, min(1, cos_ha))
    hour_angle = math.degrees(math.acos(cos_ha))
    return 2 * hour_angle / 15


def compute_solar_features(df: pd.DataFrame, lat: float = LATITUDE, lon: float = LONGITUDE) -> pd.DataFrame:
    """Add solar position features to DataFrame (must have DatetimeIndex)."""
    doy = df.index.dayofyear
    hour = df.index.hour + df.index.minute / 60.0

    zenith = np.vectorize(solar_zenith_angle)(lat, lon, doy, hour)
    elevation = 90.0 - zenith
    d_length = np.vectorize(day_length_hours)(lat, doy)

    df["solar_zenith"] = zenith
    df["solar_elevation"] = elevation
    df["day_length"] = d_length
    df["is_daylight"] = (elevation > 0).astype(int)

    return df


# ─────────────────────────────────────────────
# Data Validation
# ─────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, expected_columns: list, logger: logging.Logger = None) -> dict:
    """Validate a DataFrame and return a quality report."""
    log = logger or setup_logger("validation")
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_columns": [c for c in expected_columns if c not in df.columns],
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "date_range": (str(df.index.min()), str(df.index.max())) if isinstance(df.index, pd.DatetimeIndex) else None,
        "duplicated_rows": int(df.duplicated().sum()),
    }

    log.info(f"Dataset: {report['total_rows']:,} rows × {report['total_columns']} columns")
    if report["date_range"]:
        log.info(f"Date range: {report['date_range'][0]} → {report['date_range'][1]}")
    if report["missing_columns"]:
        log.warning(f"Missing columns: {report['missing_columns']}")

    high_null = {k: f"{v:.1f}%" for k, v in report["null_percentage"].items() if v > 5}
    if high_null:
        log.warning(f"High null columns (>5%): {high_null}")

    return report


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame: remove duplicates, handle NaN, clip outliers."""
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="first")]

    # Sort by time
    df = df.sort_index()

    # Forward fill then interpolate remaining NaN (max gap = 6 hours)
    df = df.ffill(limit=6)
    df = df.interpolate(method="linear", limit=12)

    return df


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────

def format_metrics(metrics: dict) -> str:
    """Format model evaluation metrics for display."""
    lines = []
    for target, m in metrics.items():
        lines.append(f"\n  {target}:")
        for metric_name, value in m.items():
            lines.append(f"    {metric_name:>10}: {value:.4f}")
    return "\n".join(lines)
