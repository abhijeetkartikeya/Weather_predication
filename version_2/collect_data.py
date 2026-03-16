"""
Data Collection Module for Weather Forecasting Pipeline
=========================================================
Collects historical weather data from NASA POWER and Open-Meteo APIs,
merges them into a unified dataset for ML training.

Usage:
    python collect_data.py                          # Full collection (2005-yesterday)
    python collect_data.py --start-year 2020        # From 2020
    python collect_data.py --end-year 2023          # Up to 2023
    python collect_data.py --skip-nasa              # Skip NASA POWER (use cached)
    python collect_data.py --skip-openmeteo         # Skip Open-Meteo (use cached)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import (
    DATA_RESOLUTION,
    LATITUDE,
    LONGITUDE,
    DATA_START_YEAR,
    NASA_POWER_BASE_URL,
    NASA_POWER_COMMUNITY,
    NASA_POWER_PARAMETERS,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_PARAMETERS,
    MERGED_DATASET_PATH,
)
from utils import (
    api_request_with_retry,
    clean_dataframe,
    setup_logger,
    validate_dataframe,
)

logger = setup_logger("collect_data")


# ═══════════════════════════════════════════════
# NASA POWER API
# ═══════════════════════════════════════════════

def collect_nasa_power(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Download hourly data from NASA POWER API year by year.

    Parameters
    ----------
    start_year : int
        First year to download.
    end_year : int
        Last year to download.

    Returns
    -------
    pd.DataFrame
        Hourly dataset with DatetimeIndex.
    """
    logger.info(f"╔══ NASA POWER API: Downloading {start_year}–{end_year}")
    params_str = ",".join(NASA_POWER_PARAMETERS)
    all_frames = []

    for year in range(start_year, end_year + 1):
        start = f"{year}0101"
        end_dt = f"{year}1231"

        # For the current year, cap at yesterday
        today = datetime.today()
        if year == today.year:
            yesterday = today - timedelta(days=2)  # NASA POWER has ~2 day lag
            end_dt = yesterday.strftime("%Y%m%d")

        url = (
            f"{NASA_POWER_BASE_URL}?"
            f"parameters={params_str}"
            f"&community={NASA_POWER_COMMUNITY}"
            f"&longitude={LONGITUDE}"
            f"&latitude={LATITUDE}"
            f"&start={start}"
            f"&end={end_dt}"
            f"&format=JSON"
        )

        logger.info(f"  ├── Downloading {year} ...")

        try:
            data = api_request_with_retry(url, logger=logger)
        except Exception as e:
            logger.error(f"  ├── Failed {year}: {e}")
            continue

        if "properties" not in data:
            logger.warning(f"  ├── No data for {year}: {data.get('message', 'unknown error')}")
            continue

        hourly = data["properties"]["parameter"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df.index, format="%Y%m%d%H")
        df.set_index("time", inplace=True)

        # Replace fill values (-999) with NaN
        df = df.replace(-999.0, np.nan)
        df = df.replace(-999, np.nan)

        all_frames.append(df)
        logger.info(f"  ├── {year}: {len(df):,} rows")

    if not all_frames:
        raise RuntimeError("No data collected from NASA POWER API")

    dataset = pd.concat(all_frames)
    dataset = dataset.sort_index()
    dataset = dataset[~dataset.index.duplicated(keep="first")]

    logger.info(f"╚══ NASA POWER total: {len(dataset):,} hourly rows")

    # Resample to 15-min intervals via linear interpolation
    logger.info("  Resampling NASA data to 15-min intervals ...")
    dataset = dataset.resample(DATA_RESOLUTION).interpolate(method="linear")
    logger.info(f"  After resampling: {len(dataset):,} rows")

    # Rename columns to friendlier names
    rename_map = {
        "ALLSKY_SFC_SW_DWN": "nasa_ghi",
        "ALLSKY_SFC_SW_DIFF": "nasa_dhi",
        "T2M": "nasa_temperature",
        "WS10M": "nasa_wind_speed",
        "WD10M": "nasa_wind_direction",
        "RH2M": "nasa_humidity",
        "CLOUD_AMT": "nasa_cloud_cover",
        "PS": "nasa_pressure",
    }
    dataset = dataset.rename(columns=rename_map)

    return dataset


# ═══════════════════════════════════════════════
# Open-Meteo Historical Archive API
# ═══════════════════════════════════════════════

def collect_openmeteo_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical hourly data from Open-Meteo archive API.

    The API accepts large date ranges, so we download in 5-year chunks
    to avoid timeouts.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Hourly dataset with DatetimeIndex.
    """
    logger.info(f"╔══ Open-Meteo Archive: Downloading {start_date} → {end_date}")
    params_str = ",".join(OPENMETEO_PARAMETERS)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_frames = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=365 * 3), end)
        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        url = (
            f"{OPENMETEO_ARCHIVE_URL}?"
            f"latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&start_date={s}&end_date={e}"
            f"&hourly={params_str}"
            f"&timezone=auto"
        )

        logger.info(f"  ├── Downloading {s} → {e} ...")

        try:
            data = api_request_with_retry(url, logger=logger)
        except Exception as exc:
            logger.error(f"  ├── Failed chunk {s}→{e}: {exc}")
            chunk_start = chunk_end + timedelta(days=1)
            continue

        if "hourly" not in data:
            logger.warning(f"  ├── No hourly data for chunk {s}→{e}")
            chunk_start = chunk_end + timedelta(days=1)
            continue

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        all_frames.append(df)
        logger.info(f"  ├── Chunk: {len(df):,} rows")

        chunk_start = chunk_end + timedelta(days=1)

    if not all_frames:
        raise RuntimeError("No data collected from Open-Meteo Archive API")

    dataset = pd.concat(all_frames)
    dataset = dataset.sort_index()
    dataset = dataset[~dataset.index.duplicated(keep="first")]

    logger.info(f"╚══ Open-Meteo total: {len(dataset):,} hourly rows")

    # Resample to 15-min intervals via linear interpolation
    logger.info("  Resampling Open-Meteo data to 15-min intervals ...")
    dataset = dataset.resample(DATA_RESOLUTION).interpolate(method="linear")
    logger.info(f"  After resampling: {len(dataset):,} rows")

    return dataset


# ═══════════════════════════════════════════════
# Merge Datasets
# ═══════════════════════════════════════════════

def merge_datasets(nasa_df: pd.DataFrame, openmeteo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge NASA POWER and Open-Meteo datasets on their DatetimeIndex.

    Open-Meteo is treated as the primary source; NASA POWER fills in
    satellite-derived columns (especially GHI/DHI from a different sensor).

    Parameters
    ----------
    nasa_df : pd.DataFrame
        NASA POWER hourly data.
    openmeteo_df : pd.DataFrame
        Open-Meteo archive hourly data.

    Returns
    -------
    pd.DataFrame
        Merged and cleaned dataset.
    """
    logger.info("Merging datasets ...")

    # Outer join to keep all timestamps
    merged = openmeteo_df.join(nasa_df, how="outer", rsuffix="_nasa_dup")

    # Drop any duplicate columns from the join
    dup_cols = [c for c in merged.columns if c.endswith("_nasa_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # Clean
    merged = clean_dataframe(merged)

    logger.info(f"Merged dataset: {len(merged):,} rows × {len(merged.columns)} columns")
    logger.info(f"Columns: {list(merged.columns)}")

    return merged


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Collect weather data for ML training")
    parser.add_argument("--start-year", type=int, default=DATA_START_YEAR, help="Start year")
    parser.add_argument("--end-year", type=int, default=None, help="End year (default: current)")
    parser.add_argument("--skip-nasa", action="store_true", help="Skip NASA POWER download (use cached file)")
    parser.add_argument("--skip-openmeteo", action="store_true", help="Skip Open-Meteo download (use cached file)")
    parser.add_argument("--output", type=str, default=MERGED_DATASET_PATH, help="Output CSV path")
    args = parser.parse_args()

    end_year = args.end_year or datetime.today().year
    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("  Weather Data Collection Pipeline")
    logger.info(f"  Location: {LATITUDE}°N, {LONGITUDE}°E")
    logger.info(f"  Period: {args.start_year}–{end_year}")
    logger.info("=" * 60)

    # ── NASA POWER ──
    nasa_cache = "nasa_hourly_cache.csv"
    if args.skip_nasa and os.path.exists(nasa_cache):
        logger.info(f"Loading cached NASA data from {nasa_cache}")
        nasa_df = pd.read_csv(nasa_cache, index_col=0, parse_dates=True)
    else:
        nasa_df = collect_nasa_power(args.start_year, end_year)
        nasa_df.to_csv(nasa_cache)
        logger.info(f"Saved NASA cache: {nasa_cache}")

    # ── Open-Meteo ──
    openmeteo_cache = "openmeteo_hourly_cache.csv"
    if args.skip_openmeteo and os.path.exists(openmeteo_cache):
        logger.info(f"Loading cached Open-Meteo data from {openmeteo_cache}")
        openmeteo_df = pd.read_csv(openmeteo_cache, index_col=0, parse_dates=True)
    else:
        start_date = f"{args.start_year}-01-01"
        openmeteo_df = collect_openmeteo_historical(start_date, end_date)
        openmeteo_df.to_csv(openmeteo_cache)
        logger.info(f"Saved Open-Meteo cache: {openmeteo_cache}")

    # ── Validate ──
    from config import TARGET_COLUMNS
    validate_dataframe(openmeteo_df, TARGET_COLUMNS, logger)

    # ── Merge ──
    merged = merge_datasets(nasa_df, openmeteo_df)

    # ── Save ──
    merged.to_csv(args.output)
    logger.info(f"✓ Saved merged dataset: {args.output}")
    logger.info(f"  Shape: {merged.shape}")
    logger.info(f"  Date range: {merged.index.min()} → {merged.index.max()}")

    # ── Summary ──
    null_pct = merged.isnull().sum() / len(merged) * 100
    logger.info("\nNull percentage by column:")
    for col in sorted(null_pct.index):
        if null_pct[col] > 0:
            logger.info(f"  {col:>30}: {null_pct[col]:.2f}%")

    logger.info("\n✓ Data collection complete!")


if __name__ == "__main__":
    main()
