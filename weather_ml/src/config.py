"""
Centralized configuration for the Weather ML Pipeline.
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

# Ensure dirs exist
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
# Location
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
# Target Variables
# ──────────────────────────────────────────────
TARGET_COLUMNS = ["ghi", "dhi", "temperature", "windspeed", "cloud_cover"]

# ──────────────────────────────────────────────
# Forecast Horizon
# ──────────────────────────────────────────────
FORECAST_STEPS = 384          # 4 days × 24 hours × 4 (15-min intervals)
FORECAST_INTERVAL_MIN = 15    # minutes between each forecast step
UPDATE_INTERVAL_SEC = 900     # 15 minutes between forecast refreshes

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
LAG_STEPS = [1, 2, 4, 8, 16, 96]   # 96 = 1 day at 15-min
ROLLING_WINDOWS = {
    "1h": 4,     # 4 × 15 min = 1 hour
    "3h": 12,    # 12 × 15 min = 3 hours
    "6h": 24,    # 24 × 15 min = 6 hours
}

# ──────────────────────────────────────────────
# Model Parameters
# ──────────────────────────────────────────────
MODEL_FILE = MODEL_DIR / "weather_model.pkl"
EVAL_REPORT_FILE = MODEL_DIR / "evaluation_report.json"

VALIDATION_DAYS = 30  # last N days reserved for validation

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
}

# ──────────────────────────────────────────────
# Physical Bounds (for outlier clipping)
# ──────────────────────────────────────────────
PHYSICAL_BOUNDS = {
    "ghi": (0, 1400),          # W/m²
    "dhi": (0, 800),           # W/m²
    "temperature": (-50, 60),  # °C
    "windspeed": (0, 120),     # km/h
    "cloud_cover": (0, 100),   # %
}

# ──────────────────────────────────────────────
# Open-Meteo API
# ──────────────────────────────────────────────
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_DAYS = 7

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
