import pandas as pd
from predict import fetch_recent_data, build_features
recent_df = fetch_recent_data(10)
print(recent_df.index.max())
X, _ = build_features(recent_df.copy(), "temperature_2m", 96)
last_rows = X.tail(288)
print("X max index:", last_rows.index.max())
print("Horizon:", 96)
for idx in last_rows.index[-2:]:
    forecast_time = idx + pd.Timedelta(minutes=96*15)
    last_actual_time = recent_df.index.max()
    print("idx:", idx, "forecast:", forecast_time, "last_actual:", last_actual_time)
    print("valid:", last_actual_time < forecast_time <= last_actual_time + pd.Timedelta(hours=24))
