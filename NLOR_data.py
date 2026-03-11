import requests
import psycopg2
import pandas as pd
from psycopg2.extras import execute_batch

# ================= CONFIG =================
OPENWEATHER_API_KEY = "REMOVED_SECRET"

LAT = 28.6139
LON = 77.2090

DB_CONFIG = {
    "host": "localhost",
    "database": "weatherdb",
    "user": "kartikeya",
    "password": ""
}

# ================= FETCH SOLAR DATA (NASA POWER - WORKS FOR INDIA) =================
nasa_url = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    "?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF"
    "&community=RE"
    f"&longitude={LON}"
    f"&latitude={LAT}"
    "&start=20200101"
    "&end=20201231"
    "&format=JSON"
)

solar_response = requests.get(nasa_url).json()

if "properties" not in solar_response:
    print("Solar API Error:", solar_response)
    exit()

print("NASA solar data downloaded successfully.")

ghi_data = solar_response["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
dhi_data = solar_response["properties"]["parameter"]["ALLSKY_SFC_SW_DIFF"]

solar_records = []

for timestamp_str in ghi_data:
    ts = pd.to_datetime(timestamp_str, format="%Y%m%d%H")
    ghi_value = ghi_data[timestamp_str]
    dhi_value = dhi_data[timestamp_str]
    solar_records.append((ts, ghi_value, dhi_value))

solar_df = pd.DataFrame(solar_records, columns=["timestamp", "GHI", "DHI"])

# ================= FETCH WEATHER DATA (REAL-TIME) =================
weather_url = (
    "https://api.openweathermap.org/data/2.5/weather"
    f"?lat={LAT}&lon={LON}"
    f"&appid={OPENWEATHER_API_KEY}"
    "&units=metric"
)

weather_response = requests.get(weather_url).json()

if "main" not in weather_response:
    print("Weather API Error:", weather_response)
    exit()

temp = weather_response["main"]["temp"]
cloud_cover = weather_response["clouds"]["all"]
rain = weather_response.get("rain", {}).get("1h", 0)

print("Weather data fetched successfully.")

# ================= PREPARE RECORDS =================
records = [
    (
        row["timestamp"],
        row["GHI"],
        row["DHI"],
        temp,
        cloud_cover,
        rain,
        LAT,
        LON
    )
    for _, row in solar_df.iterrows()
]

# ================= INSERT INTO POSTGRES =================
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
        INSERT INTO weather_full_data
        (timestamp, ghi, dhi, temp, cloud_cover, rain, lat, lon)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    execute_batch(cur, query, records)

    conn.commit()
    cur.close()
    conn.close()

    print("All data inserted successfully.")

except Exception as e:
    print("Database Error:", e)