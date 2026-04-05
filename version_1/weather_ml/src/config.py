"""
Centralized configuration for the Weather ML Pipeline V2.
All paths, database credentials, model parameters, and constants live here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Data Files
# ──────────────────────────────────────────────
HISTORICAL_CSV = DATA_DIR / "historical_15min_data.csv"
NASA_CSV = DATA_DIR / "nasa_historical_data.csv"
LIVE_FORECAST_CSV = DATA_DIR / "live_forecast_data.csv"
FORECAST_OUTPUT_CSV = PROJECT_ROOT / "forecast_output.csv"

# ──────────────────────────────────────────────
# Location (Anand, Gujarat — all datasets verified)
# ──────────────────────────────────────────────
LATITUDE = 22.25
LONGITUDE = 72.74

# ──────────────────────────────────────────────
# PostgreSQL
# ──────────────────────────────────────────────
DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "database": os.environ.get("PG_DATABASE", "weatherdb"),
    "user": os.environ.get("PG_USER", "kartikeya"),
    "password": os.environ.get("PG_PASSWORD", ""),
}

# ──────────────────────────────────────────────
# InfluxDB
# ──────────────────────────────────────────────
INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "weather-ml-token-secret")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "weather-ml")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "weather_forecast")

# ──────────────────────────────────────────────
# Target Variables
# ──────────────────────────────────────────────
TARGET_COLUMNS = ["ghi", "dhi", "temperature", "windspeed", "cloud_cover"]

# ──────────────────────────────────────────────
# Forecast Horizon — V2: 3 days
# ──────────────────────────────────────────────
FORECAST_STEPS = 288            # 3 days × 24 hours × 4 (15-min intervals)
FORECAST_INTERVAL_MIN = 15
UPDATE_INTERVAL_SEC = 900       # 15 minutes

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
LAG_STEPS = [1, 2, 4, 8, 16, 96, 192, 288]   # added 2-day and 3-day lags
ROLLING_WINDOWS = {
    "1h": 4,
    "3h": 12,
    "6h": 24,
    "12h": 48,     # added 12-hour window
}

# ──────────────────────────────────────────────
# Model Parameters — V2: Fine-tuned
# ──────────────────────────────────────────────
MODEL_FILE = MODEL_DIR / "weather_model_v2.pkl"
EVAL_REPORT_FILE = MODEL_DIR / "evaluation_report_v2.json"

VALIDATION_DAYS = 30

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 511,          # Increased from 255 for more complex trees
    "learning_rate": 0.015,     # Decreased from 0.03 for better accuracy
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_estimators": 5000,       # Increased from 2000
    "early_stopping_rounds": 200, # Increased patience
}

XGBOOST_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "booster": "gbtree",
    "learning_rate": 0.015,     # Decreased from 0.03
    "max_depth": 12,            # Increased from 8
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": 0,
    "n_estimators": 5000,       # Increased from 2000
    "early_stopping_rounds": 200, # Increased patience
}

# ──────────────────────────────────────────────
# Physical Bounds
# ──────────────────────────────────────────────
PHYSICAL_BOUNDS = {
    "ghi": (0, 1400),
    "dhi": (0, 800),
    "temperature": (-50, 60),
    "windspeed": (0, 120),
    "cloud_cover": (0, 100),
}

# ──────────────────────────────────────────────
# Open-Meteo API
# ──────────────────────────────────────────────
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_DAYS = 7

# ──────────────────────────────────────────────
# VEDAS API (ISRO SAC)
# ──────────────────────────────────────────────
VEDAS_FORECAST_URL = f"https://vedas.sac.gov.in/VedasServices/rest/ForecastService/forecast15/{LONGITUDE}/{LATITUDE}"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
