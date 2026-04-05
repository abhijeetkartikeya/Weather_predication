# Weather Parameter Forecasting ML Pipeline — Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Sources](#3-data-sources)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Architecture](#5-model-architecture)
6. [Installation & Setup](#6-installation--setup)
7. [Usage Guide](#7-usage-guide)
8. [Configuration Reference](#8-configuration-reference)
9. [Performance Metrics](#9-performance-metrics)
10. [Troubleshooting](#10-troubleshooting)
11. [API Reference](#11-api-reference)

---

## 1. Project Overview

### Goal
Forecast **5 critical weather parameters** at **3 forecast horizons** (24h, 48h, 72h ahead) using a **LightGBM + XGBoost ensemble** trained on ~20 years of multi-source weather data.

### Target Parameters

| Parameter | Unit | Source Column | Description |
|-----------|------|---------------|-------------|
| **Temperature** | °C | `temperature_2m` | Air temperature at 2 meters above ground |
| **Wind Speed** | m/s | `wind_speed_10m` | Wind speed at 10 meters above ground |
| **Cloud Cover** | % | `cloud_cover` | Total cloud cover percentage |
| **GHI** | W/m² | `shortwave_radiation` | Global Horizontal Irradiance — total solar energy on a horizontal surface |
| **DHI** | W/m² | `diffuse_radiation` | Diffuse Horizontal Irradiance — scattered solar radiation (no direct beam) |

### Forecast Horizons
- **24h** — 1 day ahead
- **48h** — 2 days ahead
- **72h** — 3 days ahead

### Location
**New Delhi, India** — Latitude: 28.6139°N, Longitude: 77.2090°E

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WEATHER FORECAST PIPELINE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │ NASA POWER  │    │ Open-Meteo  │    │  Open-Meteo    │  │
│  │    API      │    │  Archive    │    │  Forecast API  │  │
│  │ (Satellite) │    │ (Reanalysis)│    │  (Live Data)   │  │
│  └──────┬──────┘    └──────┬──────┘    └───────┬────────┘  │
│         │                  │                   │            │
│         └──────────┬───────┘                   │            │
│                    ▼                           │            │
│         ┌──────────────────┐                   │            │
│         │  collect_data.py │                   │            │
│         │  Merge & Clean   │                   │            │
│         └────────┬─────────┘                   │            │
│                  ▼                             │            │
│         ┌──────────────────────┐               │            │
│         │ feature_engineering  │               │            │
│         │ .py                  │               │            │
│         │ • Temporal (cyclical)│               │            │
│         │ • Solar position    │               │            │
│         │ • Lags (1-72h)      │               │            │
│         │ • Rolling stats     │               │            │
│         │ • Interactions      │               │            │
│         └────────┬─────────────┘               │            │
│                  ▼                             │            │
│         ┌──────────────────┐                   │            │
│         │  train_model.py  │                   │            │
│         │ LightGBM+XGBoost │                   │            │
│         │ 5 targets × 3h   │                   │            │
│         │ = 30 Models       │                   │            │
│         └────────┬─────────┘                   │            │
│                  ▼                             ▼            │
│         ┌──────────────────────────────────────────┐        │
│         │             predict.py                   │        │
│         │  Load models → Features → 3-day Forecast │        │
│         └──────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
weather_predication_version2/
├── config.py                    # Central configuration
├── utils.py                     # Shared utilities (API, solar, validation)
├── collect_data.py              # Data collection from APIs
├── feature_engineering.py       # Feature engineering pipeline
├── train_model.py               # Model training (LightGBM + XGBoost)
├── predict.py                   # 3-day forecast generation
├── requirements.txt             # Python dependencies
├── DOCUMENTATION.md             # This file
├── merged_weather_dataset.csv   # [Generated] Merged training dataset
├── models/                      # [Generated] Trained model files
│   ├── temperature_2m_24h_lgb.joblib
│   ├── temperature_2m_24h_xgb.joblib
│   ├── temperature_2m_24h_features.json
│   └── ...  (30 model files total)
├── results/                     # [Generated] Evaluation reports & plots
│   ├── evaluation_report.csv
│   ├── temperature_2m_24h_importance.png
│   └── ...
└── forecast_output.csv          # [Generated] Latest forecast
```

---

## 3. Data Sources

### 3.1 NASA POWER API (Satellite-Based)

**Endpoint:** `https://power.larc.nasa.gov/api/temporal/hourly/point`

NASA's Prediction Of Worldwide Energy Resources (POWER) provides satellite-derived meteorological data. This is particularly valuable for:
- **GHI and DHI** — derived from CERES (Clouds and the Earth's Radiant Energy System) satellite instruments
- **Cloud amount** — from MODIS/CERES radiative flux analysis
- Global coverage with no ground station dependency

**Parameters collected:**

| API Parameter | Description | Unit |
|--------------|-------------|------|
| `ALLSKY_SFC_SW_DWN` | All Sky Surface Shortwave Downward Irradiance (GHI) | W/m² |
| `ALLSKY_SFC_SW_DIFF` | All Sky Surface Shortwave Diffuse Irradiance (DHI) | W/m² |
| `T2M` | Temperature at 2 Meters | °C |
| `WS10M` | Wind Speed at 10 Meters | m/s |
| `WD10M` | Wind Direction at 10 Meters | degrees |
| `RH2M` | Relative Humidity at 2 Meters | % |
| `CLOUD_AMT` | Cloud Amount | % |
| `PS` | Surface Pressure | kPa |

**Data resolution:** Hourly, from 2005 to present  
**Latency:** ~2 days behind real-time  
**Rate limit:** Free, no API key required (reasonable use)

### 3.2 Open-Meteo Historical Archive API (Reanalysis)

**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`

Open-Meteo combines multiple reanalysis models (ERA5, CERRA, etc.) to provide high-quality historical weather data. This is the primary data source for:
- **Temperature, wind, cloud cover** — from ERA5 reanalysis
- **Solar radiation** — from CAMS (Copernicus Atmosphere Monitoring Service) radiation data

**Parameters collected:**

| API Parameter | Description | Unit |
|--------------|-------------|------|
| `temperature_2m` | Air temperature at 2m | °C |
| `relative_humidity_2m` | Relative humidity at 2m | % |
| `dew_point_2m` | Dew point temperature at 2m | °C |
| `pressure_msl` | Mean sea level pressure | hPa |
| `cloud_cover` | Total cloud cover | % |
| `cloud_cover_low` | Low cloud cover (0–2km) | % |
| `cloud_cover_mid` | Mid cloud cover (2–6km) | % |
| `cloud_cover_high` | High cloud cover (>6km) | % |
| `wind_speed_10m` | Wind speed at 10m | m/s |
| `wind_direction_10m` | Wind direction at 10m | degrees |
| `wind_gusts_10m` | Wind gusts at 10m | m/s |
| `shortwave_radiation` | Global Horizontal Irradiance (GHI) | W/m² |
| `direct_radiation` | Direct radiation on horizontal surface | W/m² |
| `diffuse_radiation` | Diffuse Horizontal Irradiance (DHI) | W/m² |
| `direct_normal_irradiance` | Direct Normal Irradiance (DNI) | W/m² |

**Data resolution:** Hourly, from 1940 to present  
**Latency:** ~5 days behind real-time  
**Rate limit:** Free, 10,000 requests/day

### 3.3 Open-Meteo Forecast API (Live Data)

**Endpoint:** `https://api.open-meteo.com/v1/forecast`

Used during **prediction** to fetch the most recent weather observations for building features. Same parameters as the Archive API but with data up to the current hour.

### 3.4 Why Multiple Data Sources?

| Aspect | NASA POWER | Open-Meteo Archive |
|--------|-----------|-------------------|
| **Source** | Satellite (CERES, MERRA-2) | Reanalysis (ERA5, CAMS) |
| **Strength** | Direct satellite observations of radiation | Better wind/temperature from data assimilation |
| **GHI/DHI quality** | High (direct measurement) | Good (model-derived) |
| **Wind quality** | Moderate (satellite-derived) | High (reanalysis-assimilated) |
| **Coverage** | Global, uniform | Global, higher resolution |

The model gains diversity by seeing the same phenomenon from different measurement methodologies, improving robustness.

---

## 4. Feature Engineering

The `feature_engineering.py` module transforms raw hourly data into ML-ready features. All features are designed to capture meteorological patterns relevant to multi-day forecasting.

### 4.1 Temporal Features (Cyclical Encoding)

Raw time components are encoded as sine/cosine pairs to preserve cyclical continuity (e.g., hour 23 is close to hour 0):

```python
hour_sin = sin(2π × hour / 24)     # Hour of day
hour_cos = cos(2π × hour / 24)
month_sin = sin(2π × month / 12)   # Month of year
doy_sin = sin(2π × day / 365.25)   # Day of year (seasonal)
```

**9 features:** `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos`, `week_sin`, `week_cos`, `is_weekend`

### 4.2 Solar Position Features

Computed from astronomical formulas using latitude, longitude, and timestamp:

| Feature | Description | Importance |
|---------|-------------|------------|
| `solar_zenith` | Angle between sun and overhead (0° = directly overhead) | Primary driver of GHI/DHI |
| `solar_elevation` | Complement of zenith (90° - zenith) | Correlates with temperature |
| `day_length` | Hours of daylight | Seasonal energy availability |
| `is_daylight` | Binary: 1 if sun is above horizon | Filters night radiation to 0 |

### 4.3 Lag Features

Past values of each target variable serve as strong predictors of future values:

```
For each target column:
  lag_1h    — 1 hour ago
  lag_3h    — 3 hours ago
  lag_6h    — 6 hours ago
  lag_12h   — 12 hours ago
  lag_24h   — 1 day ago (diurnal cycle)
  lag_48h   — 2 days ago
  lag_72h   — 3 days ago (persistence baseline)
```

**35 features** (5 targets × 7 lags)

### 4.4 Rolling Statistics

Capture trends and variability over time windows:

| Window | Statistics | Purpose |
|--------|-----------|---------|
| 6h | mean, std, min, max | Short-term trends |
| 12h | mean, std, min, max | Half-day patterns |
| 24h | mean, std, min, max | Full diurnal cycle |
| 48h | mean, std, min, max | 2-day weather patterns |
| 72h | mean, std, min, max | Synoptic-scale variability |

**100 features** (5 targets × 5 windows × 4 statistics)

### 4.5 Difference Features

Rate of change captures transitions between weather regimes:

- **Hour-over-hour difference:** Rapid changes (frontal passages)
- **Day-over-day difference:** Diurnal anomalies

### 4.6 Interaction Features

Physically meaningful cross-variable interactions:

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `temp_x_humidity` | temperature × humidity / 100 | Heat index proxy |
| `wind_x_cloud` | wind_speed × cloud_cover / 100 | Convective potential |
| `ghi_x_clearsky` | GHI × (1 - cloud/100) | Clear sky radiation estimate |
| `temp_dewpoint_spread` | temperature - dew_point | Dryness indicator |
| `diffuse_fraction` | DHI / GHI | Fraction of diffuse radiation |

### 4.7 Total Feature Count

| Category | Count |
|----------|-------|
| Temporal | 9 |
| Solar position | 4 |
| Lag features | ~35 |
| Rolling statistics | ~100 |
| Difference features | ~10 |
| Interaction features | 5 |
| Raw auxiliary columns | ~15 |
| **Total** | **~180+** |

---

## 5. Model Architecture

### 5.1 Training Strategy

The system trains **30 independent models**: one for each **(target × horizon)** combination.

```
Targets (5):  temperature_2m, wind_speed_10m, cloud_cover, shortwave_radiation, diffuse_radiation
Horizons (3): 24h, 48h, 72h
Models (2):   LightGBM, XGBoost
─────────────────────────────────────────
Total models: 5 × 3 × 2 = 30
```

### 5.2 Why Separate Models per Horizon?

Each horizon has different optimal features:
- **24h:** Lags and recent trends dominate
- **48h:** Seasonal and synoptic patterns become more important
- **72h:** Climatological features carry more weight

### 5.3 LightGBM Configuration

```python
{
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
    "early_stopping_rounds": 100,
}
```

**Why LightGBM?** Faster training, handles large datasets efficiently, native categorical support, lower memory usage.

### 5.4 XGBoost Configuration

```python
{
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
    "early_stopping_rounds": 100,
}
```

**Why XGBoost?** Excellent regularization, proven performance on structured data, complements LightGBM's splits.

### 5.5 Ensemble Method

Weighted average of LightGBM and XGBoost predictions:

```
final_prediction = 0.55 × LightGBM + 0.45 × XGBoost
```

The slight bias toward LightGBM reflects its typically better performance on weather data with long-tail distributions.

### 5.6 Train/Validation/Test Split

**Temporal split** (no data leakage):

| Split | Years | Purpose |
|-------|-------|---------|
| **Train** | 2005–2022 | Model learning (~157,000 hourly samples) |
| **Validation** | 2023–2024 | Early stopping + hyperparameter selection |
| **Test** | 2025+ | Final unbiased performance estimate |

### 5.7 Optuna Hyperparameter Tuning

When `--tune` flag is used, Optuna performs Bayesian optimization over:
- `learning_rate` (0.01–0.15)
- `max_depth` (5–15)
- `num_leaves` (31–255)
- `min_child_samples` (10–100)
- `subsample` (0.5–1.0)
- `colsample_bytree` (0.5–1.0)
- `reg_alpha` (1e-4–10.0)
- `reg_lambda` (1e-4–10.0)

**50 trials** per target, with 30-minute timeout.

---

## 6. Installation & Setup

### 6.1 Prerequisites

- **Python 3.10+**
- **Internet connection** (for API data download)
- **~4 GB RAM** minimum (8 GB recommended for training)
- **~2 GB disk space** (for datasets + models)

### 6.2 Setup Steps

```bash
# 1. Navigate to project directory
cd weather_predication_version2

# 2. Create virtual environment (if not already created)
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

### 6.3 Verify Installation

```bash
python -c "import lightgbm, xgboost, optuna; print('All packages installed successfully!')"
```

---

## 7. Usage Guide

### 7.1 Step 1: Collect Data

```bash
# Full collection (2005 to yesterday) — takes 30-60 minutes
python collect_data.py

# Faster: Collect only recent years
python collect_data.py --start-year 2020

# Skip NASA if you already have it cached
python collect_data.py --skip-nasa

# Skip Open-Meteo if you already have it cached
python collect_data.py --skip-openmeteo

# Custom output path
python collect_data.py --output my_dataset.csv
```

**Output:** `merged_weather_dataset.csv` (~100MB for full 2005-present)

### 7.2 Step 2: Train Models

```bash
# Full training — all targets, all horizons
python train_model.py

# Train specific targets only
python train_model.py --targets temperature_2m wind_speed_10m

# Train specific horizons only
python train_model.py --horizons 24h 48h

# With Optuna hyperparameter tuning (slower, better results)
python train_model.py --tune

# Use a custom dataset
python train_model.py --dataset custom_dataset.csv
```

**Output:**
- `models/` — 30 model files (`.joblib`) + feature lists (`.json`)
- `results/evaluation_report.csv` — Performance metrics for all models
- `results/*_importance.png` — Feature importance plots

### 7.3 Step 3: Generate Forecasts

```bash
# Full 3-day forecast (all targets)
python predict.py

# Forecast specific targets
python predict.py --targets temperature_2m shortwave_radiation

# Custom output path
python predict.py --output my_forecast.csv
```

**Output:** `forecast_output.csv` with columns:
- `base_time` — Time the prediction was made from
- `forecast_time` — Future time being predicted
- `target` — Parameter name
- `target_display` — Human-readable name
- `horizon` — 24h / 48h / 72h
- `predicted_value` — Forecasted value

---

## 8. Configuration Reference

All settings are in `config.py`. Key parameters:

### Location

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LATITUDE` | 28.6139 | Station latitude (°N) |
| `LONGITUDE` | 77.2090 | Station longitude (°E) |
| `TIMEZONE` | Asia/Kolkata | Timezone string |

### Feature Engineering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LAG_HOURS` | [1,3,6,12,24,48,72] | Lag periods for feature creation |
| `ROLLING_WINDOWS` | [6,12,24,48,72] | Rolling statistics window sizes (hours) |
| `DIFF_PERIODS` | [1, 24] | Difference periods for rate-of-change features |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_END_YEAR` | 2022 | Last year of training data |
| `VAL_END_YEAR` | 2024 | Last year of validation data |
| `EARLY_STOPPING_ROUNDS` | 100 | Patience for early stopping |
| `ENSEMBLE_WEIGHTS` | lgbm:0.55, xgb:0.45 | Model blending weights |
| `OPTUNA_N_TRIALS` | 50 | Number of Optuna tuning trials |

### API Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `API_MAX_RETRIES` | 5 | Maximum retry attempts |
| `API_RETRY_DELAY` | 10 | Base delay between retries (seconds) |
| `API_TIMEOUT` | 120 | Request timeout (seconds) |

---

## 9. Performance Metrics

### 9.1 Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **MAE** | mean(\|y - ŷ\|) | Average absolute error in original units |
| **RMSE** | √mean((y - ŷ)²) | Penalizes large errors more |
| **R²** | 1 - SS_res/SS_tot | Fraction of variance explained (1.0 = perfect) |
| **MAPE** | mean(\|y - ŷ\|/\|y\|) × 100 | Percentage error (undefined near 0) |

### 9.2 Expected Performance Ranges

These are typical ranges based on similar models for Delhi:

| Target | 24h MAE | 48h MAE | 72h MAE | 24h R² |
|--------|---------|---------|---------|--------|
| Temperature (°C) | 1.0–2.0 | 1.5–2.5 | 2.0–3.5 | 0.85–0.95 |
| Wind Speed (m/s) | 0.5–1.2 | 0.7–1.5 | 0.8–1.8 | 0.50–0.75 |
| Cloud Cover (%) | 10–20 | 12–25 | 15–30 | 0.40–0.65 |
| GHI (W/m²) | 30–60 | 40–80 | 50–100 | 0.75–0.90 |
| DHI (W/m²) | 15–30 | 20–40 | 25–50 | 0.60–0.80 |

> **Note:** Actual results depend on training data quality and weather patterns.

### 9.3 Interpreting Results

After training, check `results/evaluation_report.csv`:

```bash
# View evaluation report
cat results/evaluation_report.csv | column -t -s,
```

Key things to look for:
1. **R² > 0.5** for all targets — indicates meaningful predictions
2. **Val vs Test metrics** — if test is much worse, potential distribution shift
3. **Horizon degradation** — 72h should be worse than 24h (expected)

---

## 10. Troubleshooting

### 10.1 Data Collection Issues

| Problem | Solution |
|---------|----------|
| NASA API returns "Rate limited" | Wait 10 minutes, then retry. The script auto-retries with backoff. |
| Open-Meteo returns empty data | Check date ranges; historical data may lag 5 days behind today. |
| API timeout | Increase `API_TIMEOUT` in `config.py` or use `--start-year` for smaller range. |
| Partial download | Use `--skip-nasa` or `--skip-openmeteo` flags to use cached data. |

### 10.2 Training Issues

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `DATA_START_YEAR` to use less data, or reduce `num_leaves`. |
| Very low R² (<0.3) | Check for excessive NaN in dataset. Try `--tune` for better hyperparameters. |
| Training hangs | Check early stopping — if val loss isn't improving, it waits 100 rounds. |
| "Target not in dataset" warning | Ensure `collect_data.py` was run first and output filename matches config. |

### 10.3 Prediction Issues

| Problem | Solution |
|---------|----------|
| "Model files not found" | Train models first: `python train_model.py` |
| "Feature engineering failed" | Open-Meteo Forecast API may be down. Check network connection. |
| Unrealistic predictions | Check if recent data has NaN; the model may be extrapolating. Retrain if needed. |

---

## 11. API Reference

### 11.1 `collect_data.py`

```python
# Programmatic usage
from collect_data import collect_nasa_power, collect_openmeteo_historical, merge_datasets

# Download NASA data for specific years
nasa_df = collect_nasa_power(start_year=2020, end_year=2024)

# Download Open-Meteo data for date range
om_df = collect_openmeteo_historical("2020-01-01", "2024-12-31")

# Merge both sources
merged = merge_datasets(nasa_df, om_df)
merged.to_csv("my_dataset.csv")
```

### 11.2 `feature_engineering.py`

```python
from feature_engineering import build_features, build_all_feature_sets
import pandas as pd

df = pd.read_csv("merged_weather_dataset.csv", index_col=0, parse_dates=True)

# Single target + horizon
X, y = build_features(df, target_col="temperature_2m", horizon_hours=24)

# All targets × all horizons
all_sets = build_all_feature_sets(df)
# all_sets["temperature_2m"]["24h"] → (X, y)
```

### 11.3 `train_model.py`

```python
from train_model import train_all

# Full training
train_all(dataset_path="merged_weather_dataset.csv")

# Selective training with tuning
train_all(
    dataset_path="merged_weather_dataset.csv",
    targets=["temperature_2m", "wind_speed_10m"],
    horizons=["24h"],
    do_tune=True,
)
```

### 11.4 `predict.py`

```python
from predict import generate_forecast, fetch_recent_data

# Generate full forecast
forecast_df = generate_forecast()

# Selective forecast
forecast_df = generate_forecast(
    targets=["temperature_2m"],
    output_path="temp_forecast.csv",
)

# Just fetch recent data
recent = fetch_recent_data(past_days=7)
```

### 11.5 `utils.py`

```python
from utils import (
    solar_zenith_angle,
    solar_elevation_angle,
    day_length_hours,
    compute_solar_features,
    api_request_with_retry,
    validate_dataframe,
    clean_dataframe,
)

# Solar calculations
zenith = solar_zenith_angle(lat=28.6, lon=77.2, day_of_year=172, hour=12.0)
elevation = solar_elevation_angle(lat=28.6, lon=77.2, day_of_year=172, hour=12.0)
daylen = day_length_hours(lat=28.6, day_of_year=172)

# API helper with retry
data = api_request_with_retry("https://api.example.com/data")

# Data validation
report = validate_dataframe(df, expected_columns=["temp", "wind"])
df_clean = clean_dataframe(df)
```

---

## Appendix A: NASA POWER API Documentation

- **Main page:** https://power.larc.nasa.gov/
- **API docs:** https://power.larc.nasa.gov/docs/services/api/
- **Parameters list:** https://power.larc.nasa.gov/#resources
- **Community:** RE (Renewable Energy) includes all solar/wind parameters

## Appendix B: Open-Meteo API Documentation

- **Main page:** https://open-meteo.com/
- **Historical API docs:** https://open-meteo.com/en/docs/historical-weather-api
- **Forecast API docs:** https://open-meteo.com/en/docs
- **Data sources:** ERA5 (ECMWF), CAMS radiation, ICON, GFS

## Appendix C: Modifying for a Different Location

To adapt this pipeline for a different location:

1. Update `config.py`:
   ```python
   LATITUDE = <your_latitude>
   LONGITUDE = <your_longitude>
   TIMEZONE = "<your_timezone>"
   LOCATION_NAME = "<your_location>"
   ```

2. Re-run data collection:
   ```bash
   python collect_data.py
   ```

3. Re-train models:
   ```bash
   python train_model.py --tune
   ```

4. If your location has different climate patterns (e.g., polar, tropical), adjust:
   - `ROLLING_WINDOWS` for different dominant weather scales
   - `TRAIN_END_YEAR` / `VAL_END_YEAR` if less historical data is available
   - `ENSEMBLE_WEIGHTS` after comparing individual model performance

---

*Documentation generated for Weather Forecasting ML Pipeline v2.0*
*Location: New Delhi, India (28.6139°N, 77.2090°E)*
