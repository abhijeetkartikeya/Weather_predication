import requests
import pandas as pd
from datetime import datetime, timedelta

LAT = 28.6139
LON = 77.2090

start_year = 2005
end_year = datetime.today().year

parameters = "ALLSKY_SFC_SW_DWN,T2M,WS10M,WD10M,RH2M,CLOUD_AMT,PS"

all_data = []

for year in range(start_year, end_year + 1):

    start = f"{year}0101"
    end = f"{year}1231"

    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters={parameters}&community=RE&longitude={LON}&latitude={LAT}&start={start}&end={end}&format=JSON"

    print("Downloading:", year)

    response = requests.get(url)
    data = response.json()

    if "properties" not in data:
        print("API error:", data)
        continue

    hourly = data["properties"]["parameter"]

    df = pd.DataFrame(hourly)

    df["time"] = pd.to_datetime(df.index, format="%Y%m%d%H")

    df.set_index("time", inplace=True)

    all_data.append(df)

dataset = pd.concat(all_data)

print("Total hourly rows:", len(dataset))

dataset_15min = dataset.resample("15min").interpolate()

print("Total 15min rows:", len(dataset_15min))

dataset_15min.to_csv("nasa_2005_to_yesterday_15min_dataset.csv")

print("Dataset saved.")