"""
Fetch hourly weather data from the Open-Meteo Archive API.

The API has a practical limit of ~1 year per request, so we loop year by year,
concatenate, and save a single CSV.
"""

import time
import requests
import pandas as pd

from config import (
    API_BASE_URL,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    HOURLY_VARIABLES,
    FETCH_START_DATE,
    FETCH_END_DATE,
    RAW_CSV,
)


MAX_RETRIES = 5
RETRY_BACKOFF = 5  # seconds


def fetch_year(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one chunk (up to a year) from Open-Meteo and return a DataFrame."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            hourly = data["hourly"]
            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"])
            print(f"  Fetched {start_date} → {end_date}  ({len(df)} rows)")
            return df
        except Exception as exc:
            wait = RETRY_BACKOFF * attempt
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed ({exc}). "
                  f"Retrying in {wait}s …")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch data for {start_date} → {end_date}")


def collect_data() -> pd.DataFrame:
    """Download the full date range year-by-year and save to CSV."""
    start_year = int(FETCH_START_DATE[:4])
    end_year = int(FETCH_END_DATE[:4])
    final_end = FETCH_END_DATE

    chunks = []
    for year in range(start_year, end_year + 1):
        chunk_start = f"{year}-01-01"
        # For the last year, cap at the configured end date
        if year == end_year:
            chunk_end = final_end
        else:
            chunk_end = f"{year}-12-31"
        df_chunk = fetch_year(chunk_start, chunk_end)
        chunks.append(df_chunk)
        # Be polite to the API
        time.sleep(1)

    df = pd.concat(chunks, ignore_index=True)
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(RAW_CSV, index=False)
    print(f"\nRaw data saved → {RAW_CSV}  ({len(df)} rows, "
          f"{df['time'].min()} to {df['time'].max()})")
    return df


if __name__ == "__main__":
    collect_data()
