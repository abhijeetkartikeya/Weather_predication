from meteostat import Point
from meteostat.stations import Stations
from meteostat.hourly import Hourly
from datetime import datetime
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

LAT = 28.6139
LON = 77.2090

START_YEAR = 2015
END_YEAR = datetime.today().year

# ---------------- FIND NEAREST STATION ----------------
stations = Stations()  # create object
stations = stations.nearby(LAT, LON)
station = stations.fetch(1)

if station.empty:
    print("No station found.")
    exit()

station_id = station.index[0]
print("Using station:", station_id)

# ---------------- CONNECT DB ----------------
conn = psycopg2.connect(
    host=os.getenv("PG_HOST", "localhost"),
    database=os.getenv("PG_DATABASE", "weatherdb"),
    user=os.getenv("PG_USER", "kartikeya"),
    password=os.getenv("PG_PASSWORD", "")
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS weather_hourly (
    timestamp TIMESTAMP PRIMARY KEY,
    temp FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    precipitation FLOAT,
    pressure FLOAT
);
""")

insert_query = """
INSERT INTO weather_hourly
(timestamp, temp, humidity, wind_speed, precipitation, pressure)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp) DO NOTHING;
"""

# ---------------- FETCH YEAR BY YEAR ----------------
for year in range(START_YEAR, END_YEAR + 1):

    print("Fetching year:", year)

    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59)

    data = Hourly(station_id, start, end)
    df = data.fetch()

    if df is None or df.empty:
        print("No data for year:", year)
        continue

    print("Rows:", len(df))

    for index, row in df.iterrows():
        cur.execute(insert_query, (
            index,
            row.get('temp'),
            row.get('rhum'),
            row.get('wspd'),
            row.get('prcp'),
            row.get('pres')
        ))

    conn.commit()

cur.close()
conn.close()

print("Data inserted successfully.")