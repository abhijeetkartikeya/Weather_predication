import requests
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

LAT = 22.25
LON = 72.74

START_YEAR = 2015
END_YEAR = 2026
END_CUTOFF = "2026-03-02 12:00:00"

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": os.getenv("PG_DATABASE", "weatherdb"),
    "user": os.getenv("PG_USER", "kartikeya"),
    "password": os.getenv("PG_PASSWORD", "")
}

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

for year in range(START_YEAR, END_YEAR + 1):

    if year == 2026:
        start_date = "2026-01-01"
        end_date = "2026-03-02"
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    print(f"Fetching {start_date} → {end_date}")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}"
        f"&longitude={LON}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&hourly=temperature_2m,"
        f"windspeed_10m,"
        f"cloudcover,"
        f"shortwave_radiation,"
        f"diffuse_radiation"
        f"&models=era5"
    )

    response = requests.get(url)
    data = response.json()

    if "hourly" not in data:
        print("API Error:", data)
        continue

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "windspeed": hourly["windspeed_10m"],
        "cloudcover": hourly["cloudcover"],
        "ghi": hourly["shortwave_radiation"],
        "dhi": hourly["diffuse_radiation"]
    })

    df = df[df["timestamp"] <= END_CUTOFF]

    # ----------------------------
    # Convert hourly → 15 minutes
    # ----------------------------
    df = df.set_index("timestamp")
    df_15 = df.resample("15min").interpolate(method="time")

    df_15 = df_15.reset_index()
    df_15["latitude"] = LAT
    df_15["longitude"] = LON

    print("15-min rows:", len(df_15))

    for _, row in df_15.iterrows():
        cursor.execute("""
            INSERT INTO weather_historical 
            (timestamp, latitude, longitude, temperature, windspeed, cloudcover, ghi, dhi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            row["timestamp"],
            row["latitude"],
            row["longitude"],
            row["temperature"],
            row["windspeed"],
            row["cloudcover"],
            row["ghi"],
            row["dhi"]
        ))

    conn.commit()

cursor.close()
conn.close()

print("15-minute historical dataset created successfully.")