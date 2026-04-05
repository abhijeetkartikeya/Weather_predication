import requests
import psycopg2
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
LAT = 22.25
LON = 72.74

START_DATE = "20150101"
END_DATE = (datetime.today() - timedelta(days=7)).strftime("%Y%m%d")

CSV_PATH = "/Users/kartikeya/Desktop/nasa_historical_data.csv"

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": os.getenv("PG_DATABASE", "weatherdb"),
    "user": os.getenv("PG_USER", "kartikeya"),
    "password": os.getenv("PG_PASSWORD", "")
}

# ---------------- FETCH NASA DATA ----------------
print("Fetching NASA POWER historical data...")

url = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    f"?parameters=T2M,WS10M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF,CLOUD_AMT"
    f"&community=RE"
    f"&longitude={LON}"
    f"&latitude={LAT}"
    f"&start={START_DATE}"
    f"&end={END_DATE}"
    f"&format=CSV"
    f"&time-standard=UTC"
)

response = requests.get(url)

if response.status_code != 200:
    print("API Error:")
    print(response.text)
    exit()

# NASA returns metadata lines at top → skip first 13 rows
df = pd.read_csv(StringIO(response.text), skiprows=13)

# ---------------- CLEAN DATA ----------------
df.rename(columns={
    "YEAR": "year",
    "MO": "month",
    "DY": "day",
    "HR": "hour",
    "T2M": "temperature",
    "WS10M": "windspeed",
    "CLOUD_AMT": "cloudcover",
    "ALLSKY_SFC_SW_DWN": "ghi",
    "ALLSKY_SFC_SW_DIFF": "dhi"
}, inplace=True)

df["timestamp"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

df = df[["timestamp", "temperature", "windspeed", "cloudcover", "ghi", "dhi"]]

# ---------------- CONVERT HOURLY → 15 MIN ----------------
df = df.set_index("timestamp")
df_15 = df.resample("15min").interpolate(method="time")
df_15 = df_15.reset_index()

# Remove missing values (-999)
df.replace(-999.0, pd.NA, inplace=True)
df.dropna(inplace=True)

print("Total hourly rows:", len(df))
print("Total 15-min rows:", len(df_15))

# ---------------- INSERT INTO POSTGRES ----------------
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("DELETE FROM weather_historical;")
conn.commit()

for _, row in df_15.iterrows():
    cursor.execute("""
        INSERT INTO weather_historical
        (timestamp, latitude, longitude, temperature, windspeed, cloudcover, ghi, dhi)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """, (
        row["timestamp"],
        LAT,
        LON,
        row["temperature"],
        row["windspeed"],
        row["cloudcover"],
        row["ghi"],
        row["dhi"]
    ))

conn.commit()

# ---------------- EXPORT FULL TABLE TO CSV ----------------
df_export = pd.read_sql(
    """
    SELECT timestamp, latitude, longitude, temperature, windspeed, cloudcover, ghi, dhi
    FROM weather_historical
    ORDER BY timestamp ASC
    """,
    conn
)

df_export.to_csv(CSV_PATH, index=False)

print("CSV exported successfully at:", CSV_PATH)

cursor.close()
conn.close()

print("Done.")