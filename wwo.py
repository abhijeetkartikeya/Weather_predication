import requests
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
API_KEY = os.getenv("WWO_API_KEY")

LAT = 28.6139
LON = 77.2090

START_DATE = datetime(2015, 1, 1)

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": os.getenv("PG_DATABASE", "weatherdb"),
    "user": os.getenv("PG_USER", "kartikeya"),
    "password": os.getenv("PG_PASSWORD", "")
}

# ================= CONNECT DB =================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# ================= FIND LAST STORED DATE =================
cur.execute("SELECT MAX(timestamp) FROM weather_full_data;")
last_timestamp = cur.fetchone()[0]

if last_timestamp:
    current_date = last_timestamp + timedelta(hours=1)
    print("Resuming from:", current_date)
else:
    current_date = START_DATE
    print("Starting fresh from:", current_date)

END_DATE = datetime.today()

insert_query = """
INSERT INTO weather_full_data
(timestamp, temp, cloud_cover, rain, lat, lon)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp) DO NOTHING;
"""

# ================= LOOP =================
while current_date.date() <= END_DATE.date():

    print("Fetching:", current_date.date())

    url = (
        "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"
        f"?key={API_KEY}"
        f"&q={LAT},{LON}"
        f"&date={current_date.strftime('%Y-%m-%d')}"
        "&tp=1"
        "&format=json"
    )

    response = requests.get(url).json()

    if "data" not in response or "error" in response["data"]:
        print("API error:", response)
        break

    hourly_data = response["data"]["weather"][0]["hourly"]

    for hour in hourly_data:
        time_str = hour["time"].zfill(4)
        timestamp = pd.to_datetime(
            current_date.strftime("%Y-%m-%d") + " " +
            time_str[:2] + ":" + time_str[2:]
        )

        temp = float(hour.get("tempC", 0))
        cloud = float(hour.get("cloudcover", 0))
        rain = float(hour.get("precipMM", 0))

        cur.execute(insert_query,
                    (timestamp, temp, cloud, rain, LAT, LON))

    conn.commit()

    current_date += timedelta(days=1)
    time.sleep(1)

cur.close()
conn.close()

print("Data sync complete.")