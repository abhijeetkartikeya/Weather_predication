import requests
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
API_KEY = os.getenv("WEATHERSTACK_API_KEY")
LOCATION = "28.6139,77.2090"

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": os.getenv("PG_DATABASE", "weatherdb"),
    "user": os.getenv("PG_USER", "kartikeya"),
    "password": os.getenv("PG_PASSWORD", "")
}

# ================= FETCH REAL-TIME =================
url = (
    "http://api.weatherstack.com/current"
    f"?access_key={API_KEY}"
    f"&query={LOCATION}"
)

response = requests.get(url).json()

if "current" not in response:
    print("API Error:", response)
    exit()

current = response["current"]

timestamp = datetime.utcnow()

temp = current.get("temperature")
feelslike = current.get("feelslike")
humidity = current.get("humidity")
pressure = current.get("pressure")
wind_speed = current.get("wind_speed")
wind_dir = current.get("wind_dir")
cloudcover = current.get("cloudcover")
precipitation = current.get("precip")
uv_index = current.get("uv_index")
visibility = current.get("visibility")
description = current.get("weather_descriptions")[0]

# ================= INSERT INTO POSTGRES =================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

insert_query = """
INSERT INTO weatherstack_realtime
(timestamp, temp, feelslike, humidity, pressure,
 wind_speed, wind_dir, cloudcover, precipitation,
 uv_index, visibility, description)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp) DO NOTHING;
"""

cur.execute(insert_query, (
    timestamp, temp, feelslike, humidity, pressure,
    wind_speed, wind_dir, cloudcover,
    precipitation, uv_index, visibility, description
))

conn.commit()
cur.close()
conn.close()

print("Real-time data inserted successfully.")