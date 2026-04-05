import requests
import pandas as pd
from datetime import datetime, timedelta

LAT = 28.6139
LON = 77.2090

start_date = "2005-01-01"
end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

parameters = ",".join([
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation"
])

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={start_date}"
    f"&end_date={end_date}"
    f"&hourly={parameters}"
    "&timezone=auto"
)

print("Downloading dataset...")

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data["hourly"])

df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)

print("Original rows:", len(df))

# Convert to 15-minute interval
df_15min = df.resample("15min").interpolate(method="linear")

print("15-min rows:", len(df_15min))

df_15min.to_csv("weather_solar_2005_to_yesterday_15min.csv")

print("Saved dataset: openmeteo_data_historical.csv")