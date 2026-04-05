"""
Configuration for Weather Forecasting ML Pipeline
===================================================
Central configuration for data collection, feature engineering,
model training, and prediction.
"""

import os

# ─────────────────────────────────────────────
# Location
# ─────────────────────────────────────────────
LATITUDE = float(os.getenv("LATITUDE", "28.6139"))
LONGITUDE = float(os.getenv("LONGITUDE", "77.2090"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
LOCATION_NAME = os.getenv("LOCATION_NAME", "New Delhi, India")

def get_location_label(latitude: float = LATITUDE, longitude: float = LONGITUDE, name: str = LOCATION_NAME) -> str:
    return f"{name} ({latitude:.4f}, {longitude:.4f})"

# ─────────────────────────────────────────────
# Data Collection
# ─────────────────────────────────────────────
DATA_START_YEAR = 2005
HISTORICAL_START_DATE = "2005-01-01"

# NASA POWER API
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_COMMUNITY = "RE"
NASA_POWER_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",   # GHI - Global Horizontal Irradiance (W/m²)
    "ALLSKY_SFC_SW_DIFF",  # DHI - Diffuse Horizontal Irradiance (W/m²)
    "T2M",                 # Temperature at 2m (°C)
    "WS10M",               # Wind Speed at 10m (m/s)
    "WD10M",               # Wind Direction at 10m (degrees)
    "RH2M",                # Relative Humidity at 2m (%)
    "CLOUD_AMT",           # Cloud Amount (%)
    "PS",                  # Surface Pressure (kPa)
]

# Open-Meteo Historical Archive API
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_PARAMETERS = [
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
    "diffuse_radiation",
    "direct_normal_irradiance",
]

# ─────────────────────────────────────────────
# Target Variables
# ─────────────────────────────────────────────
TARGET_COLUMNS = [
    "temperature_2m",          # °C
    "wind_speed_10m",          # m/s
    "cloud_cover",             # %
    "shortwave_radiation",     # GHI (W/m²)
    "diffuse_radiation",       # DHI (W/m²)
]

TARGET_DISPLAY_NAMES = {
    "temperature_2m": "Temperature (°C)",
    "wind_speed_10m": "Wind Speed (m/s)",
    "cloud_cover": "Cloud Cover (%)",
    "shortwave_radiation": "GHI (W/m²)",
    "diffuse_radiation": "DHI (W/m²)",
}

# ─────────────────────────────────────────────
# Forecast Horizons (in hours)
# ─────────────────────────────────────────────
# Horizons in number of 15-min steps
FORECAST_HORIZONS = {
    "24h": 96,     # 24 hours × 4 steps/hour = 96 steps
    "48h": 192,    # 48 hours × 4 steps/hour = 192 steps
    "72h": 288,    # 72 hours × 4 steps/hour = 288 steps
}

# ─────────────────────────────────────────────
# Data Resolution
# ─────────────────────────────────────────────
DATA_RESOLUTION = "15min"       # 15-minute intervals
STEPS_PER_HOUR = 4              # 60 / 15 = 4 steps per hour

# ─────────────────────────────────────────────
# Feature Engineering (in 15-min steps)
# ─────────────────────────────────────────────
# These are in NUMBER OF 15-MIN STEPS, not hours
LAG_STEPS = [1, 2, 4, 8, 12, 24, 48, 96, 192, 288]  # 15m,30m,1h,2h,3h,6h,12h,24h,48h,72h
ROLLING_WINDOWS = [4, 8, 24, 48, 96, 192, 288]       # 1h,2h,6h,12h,24h,48h,72h in 15-min steps
DIFF_PERIODS = [1, 4, 96]      # 15-min, 1-hour, day-over-day

# ─────────────────────────────────────────────
# Train/Val/Test Split (temporal)
# ─────────────────────────────────────────────
TRAIN_END_YEAR = 2022       # Train: 2005-2022
VAL_END_YEAR = 2024         # Validation: 2023-2024
# Test: 2025+

# ─────────────────────────────────────────────
# Model Hyperparameters (defaults)
# ─────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 10,
    "num_leaves": 127,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 10,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

# Ensemble weights (LightGBM, XGBoost)
ENSEMBLE_WEIGHTS = {
    "lgbm": 0.55,
    "xgb": 0.45,
}

# Early stopping
EARLY_STOPPING_ROUNDS = 100

# ─────────────────────────────────────────────
# Optuna Hyperparameter Tuning
# ─────────────────────────────────────────────
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 1800  # 30 minutes per target

# ─────────────────────────────────────────────
# Output Paths
# ─────────────────────────────────────────────
MODELS_DIR = "models"
RESULTS_DIR = "results"
FORECAST_OUTPUT_PATH = "forecast_output.csv"
MERGED_DATASET_PATH = "merged_weather_dataset.csv"

# ═══════════════════════════════════════════════
# PostgreSQL Settings
# ═══════════════════════════════════════════════

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "adminpassword123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "weatherdb")
POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

NASA_CACHE_TABLE = "weather_nasa_cache"
OPENMETEO_CACHE_TABLE = "weather_openmeteo_cache"
MERGED_DATASET_TABLE = "weather_merged_data"
TRAINING_REPORT_TABLE = "weather_training_reports"
FORECAST_TABLE = "weather_predictions"
ACTUALS_TABLE = "weather_actuals"

# ─────────────────────────────────────────────
# Scheduler Settings
# ─────────────────────────────────────────────
SCHEDULER_INTERVAL_MINUTES = 15  # Run predictions every 15 minutes

# ─────────────────────────────────────────────
# API Request Settings
# ─────────────────────────────────────────────
API_MAX_RETRIES = 5
API_RETRY_DELAY = 10  # seconds
API_TIMEOUT = 120     # seconds
