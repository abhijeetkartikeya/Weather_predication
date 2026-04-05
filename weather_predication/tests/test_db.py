import pandas as pd
from datetime import datetime, timedelta
from timescale_writer import write_forecasts_to_timescale, write_actuals_to_timescale

df_actuals = pd.DataFrame({
    "temperature_2m": [20.5, 21.0]
}, index=[datetime.now(), datetime.now() + timedelta(minutes=15)])

df_forecasts = pd.DataFrame([
    {
        "forecast_time": datetime.now() + timedelta(hours=1),
        "target": "temperature_2m",
        "target_display": "Temperature (°C)",
        "horizon": "continuous_3d",
        "predicted_value": 22.0
    }
])

write_actuals_to_timescale(df_actuals, ["temperature_2m"])
write_forecasts_to_timescale(df_forecasts)
print("Finished test script")
