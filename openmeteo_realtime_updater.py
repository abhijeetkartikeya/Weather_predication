import requests
import psycopg2
import pandas as pd
import time
from datetime import datetime
from psycopg2.extras import execute_values

LAT = 22.25
LON = 72.74

FORECAST_DAYS = 7
UPDATE_INTERVAL = 900  # 15 minutes

CSV_PATH = "/Users/kartikeya/Desktop/weather_predication/live_forecast_data.csv"

DB_CONFIG = {
    "host": "localhost",
    "database": "weatherdb",
    "user": "kartikeya",
    "password": ""
}


def fetch_store_export():

    print("Fetching 15-min realtime + forecast...")

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}"
        f"&longitude={LON}"
        f"&minutely_15=temperature_2m,windspeed_10m,cloudcover,shortwave_radiation,diffuse_radiation"
        f"&forecast_days={FORECAST_DAYS}"
        f"&timezone=auto"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "minutely_15" not in data:
        print("API Error:", data)
        return

    minute15 = data["minutely_15"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(minute15["time"]),
        "temperature": minute15["temperature_2m"],
        "windspeed": minute15["windspeed_10m"],
        "cloudcover": minute15["cloudcover"],
        "ghi": minute15["shortwave_radiation"],
        "dhi": minute15["diffuse_radiation"]
    })

    df["latitude"] = LAT
    df["longitude"] = LON

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Bulk upsert (no DELETE, incremental safe)
    records = [
        (
            row.timestamp,
            row.latitude,
            row.longitude,
            row.temperature,
            row.windspeed,
            row.cloudcover,
            row.ghi,
            row.dhi,
        )
        for row in df.itertuples(index=False)
    ]

    insert_query = """
        INSERT INTO weather_live
        (timestamp, latitude, longitude, temperature, windspeed, cloudcover, ghi, dhi)
        VALUES %s
        ON CONFLICT (timestamp, latitude, longitude)
        DO UPDATE SET
            temperature = EXCLUDED.temperature,
            windspeed = EXCLUDED.windspeed,
            cloudcover = EXCLUDED.cloudcover,
            ghi = EXCLUDED.ghi,
            dhi = EXCLUDED.dhi;
    """

    execute_values(cursor, insert_query, records)
    conn.commit()

    # Export fresh CSV from DB
    df_live = pd.read_sql(
        "SELECT timestamp, latitude, longitude, temperature, windspeed, cloudcover, ghi, dhi "
        "FROM weather_live ORDER BY timestamp ASC;",
        conn,
    )

    df_live.to_csv(CSV_PATH, index=False)

    conn.close()

    print("Pipeline updated at:", datetime.now())


if __name__ == "__main__":
    while True:
        try:
            fetch_store_export()
        except Exception as e:
            print("Error:", e)

        time.sleep(UPDATE_INTERVAL)