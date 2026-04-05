"""
Configuration constants for the leakage-free weather/solar forecasting pipeline.
"""

# ─── Location ───────────────────────────────────────────────────────────────
LATITUDE = 28.7041
LONGITUDE = 77.1025
LOCATION_NAME = "Delhi, India"

# ─── Open-Meteo Archive API ────────────────────────────────────────────────
API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "UTC"

# ─── Date ranges ────────────────────────────────────────────────────────────
FETCH_START_DATE = "2000-01-01"
FETCH_END_DATE = "2026-03-24"          # today / latest available

TRAIN_CUTOFF = "2026-01-31 23:59:59"   # inclusive upper bound for training
TEST_START = "2026-02-01"
TEST_END = "2026-03-24"

# ─── Hourly variables to fetch ──────────────────────────────────────────────
HOURLY_VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

# ─── Target variables (same as HOURLY_VARIABLES here) ──────────────────────
TARGET_VARIABLES = HOURLY_VARIABLES.copy()

# ─── Lag features ───────────────────────────────────────────────────────────
# More lags = better pattern capture for recursive forecasting.
# 1,2,3  → short-term persistence (last few hours)
# 6,12   → half-day trends
# 24,48  → diurnal cycle (yesterday, day before yesterday)
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]

# ─── Rolling window sizes (in hours) ─────────────────────────────────────
# Smoothed rolling statistics reduce noise in recursive predictions.
# Applied with shift(1) to prevent leakage — uses PAST values only.
ROLLING_WINDOWS = [6, 12, 24]

# ─── Direct multi-horizon forecasting ──────────────────────────────────
# Maximum forecast horizon in hours.  The model learns to predict directly
# at any horizon from 1 to MAX_HORIZON without recursion.
MAX_HORIZON = 24

# ─── XGBoost hyper-parameters ──────────────────────────────────────────────
# Tuned for stability: lower depth + colsample to reduce overfitting.
XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# ─── Prediction smoothing ─────────────────────────────────────────────────
# Rolling window applied AFTER prediction to reduce jitter.
PREDICTION_SMOOTHING_WINDOW = 5

# ─── File paths ─────────────────────────────────────────────────────────────
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
MODELS_DIR = os.path.join(PIPELINE_DIR, "models")
PLOTS_DIR = os.path.join(PIPELINE_DIR, "plots")

RAW_CSV = os.path.join(DATA_DIR, "raw_weather.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

# Ensure output directories exist
for d in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


def model_path(target: str) -> str:
    """Return the joblib path for a given target variable's model."""
    return os.path.join(MODELS_DIR, f"xgb_{target}.joblib")
