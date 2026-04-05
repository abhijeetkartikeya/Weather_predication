import pandas as pd
import requests
url = f"https://vedas.sac.gov.in/VedasServices/rest/ForecastService/forecast15/75.25/25.5"

response = requests.get(url)

print("Status code:", response.status_code)

data = response.json()

print("Type of response:", type(data))

if isinstance(data, list):
    print("Number of records:", len(data))
    df = pd.DataFrame(data)
    print("Columns received:")
    print(df.columns)
else:
    print("Unexpected structure:")
    print(data)