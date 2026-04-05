# Weather Forecasting ML Pipeline — Documentation

## 1. Introduction

A production-grade machine learning pipeline for **short-term weather forecasting** to support solar power generation planning. The system predicts 5 weather variables at **15-minute resolution** for the **next 3 days** (288 time steps).

**Target Variables:**
- GHI (Global Horizontal Irradiance) — W/m²
- DHI (Diffuse Horizontal Irradiance) — W/m²
- Temperature — °C
- Wind Speed — km/h
- Cloud Cover — %

**Location:** Anand, Gujarat, India (22.25°N, 72.74°E)

---

## 2. Project Ecosystem & Data Ingestion System

While this documentation focuses primarily on the `weather_ml` pipeline, it is important to understand the outer project ecosystem that feeds data into it. The root directory (`Weather_predication/`) contains several standalone Python scripts responsible for fetching historical and real-time weather data from disparate sources into a central PostgreSQL database.

### 2.1 Historical Data Loaders

*   **`historical_loader.py`**: Fetches weather data (temperature, windspeed, cloud cover, radiation) from the **Open-Meteo Archive API** (2015 - 2026). Interpolates hourly data to **15-minute frequency** and inserts it into the `weather_historical` Postgres table.
*   **`nasa_historical_loader.py`**: Fetches hourly solar and weather data from the **NASA POWER API** (2015 - Present). Interpolates to 15-minute frequency, inserts into `weather_historical`, and exports backups to `nasa_historical_data.csv`.
*   **`wwo.py`**: Fetches historical hourly data from **WorldWeatherOnline Premium API**, querying day-by-day from 2015 to the current day. Populates the `weather_full_data` table.
*   **`meteostat1.py`**: Locates the nearest weather station using **Meteostat**, iterates from 2015 to the current year, and populates the `weather_hourly` table.

### 2.2 Real-time & Forecast Updaters

*   **`openmeteo_realtime_updater.py`**: Runs in an infinite loop (every 15 minutes). Fetches 15-minute resolution realtime weather and a 7-day forecast from **Open-Meteo**. Upserts into the `weather_live` table and exports `live_forecast_data.csv`. This provides the live alignment features for ML inferences.
*   **`weather_stack.py`**: Fetches real-time point-in-time weather data from the **WeatherStack API** and inserts it into `weatherstack_realtime`.
*   **`NLOR_data.py`**: Grabs historical solar data via NASA and current weather snapshots via **OpenWeatherMap**, inserting combined records into `weather_full_data`.

**Database Dependencies:**
All ingestion scripts and the ML pipeline expect a PostgreSQL instance configured via a `.env` file containing: `PG_HOST`, `PG_DATABASE`, `PG_USER`, and `PG_PASSWORD`.

---

## 3. ML System Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Historical  │   │  Open-Meteo   │   │   VEDAS API  │
│  CSV Data    │   │  Forecast API │   │  (ISRO SAC)  │
└──────┬───────┘   └──────┬────────┘   └──────┬───────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌──────────────────────────────────────────────────────┐
│              DATA LOADER & MERGER                     │
│  • Merge 3 CSV sources (392K rows, 2015–2026)       │
│  • Fetch VEDAS 3-day temp + solar from ISRO          │
│  • Fetch Open-Meteo 7-day forecast                    │
│  • Resample to 15-min, interpolate, clip bounds       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING (97 features)          │
│  • Lag features: t-1,2,4,8,16,96,192,288            │
│  • Rolling: mean/std @ 1h, 3h, 6h, 12h              │
│  • Time: cyclical hour/doy, month, daytime, weekday  │
│  • Interactions: GHI/DHI ratio, temp×wind, cloud×GHI │
│  • External: Open-Meteo + VEDAS forecast alignment   │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│         LightGBM MODELS (5 separate regressors)      │
│  • 2000 estimators, lr=0.03, 255 leaves              │
│  • Early stopping (patience=100)                      │
│  • Regularization: L1=0.1, L2=0.1                    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│         RECURSIVE FORECAST (288 steps / 3 days)      │
│  • Predict step t+1, feed back as lag input           │
│  • Guided by Open-Meteo + VEDAS external forecasts   │
│  • Physical bounds clipping per step                  │
└──────┬───────────────┬───────────────┬───────────────┘
       │               │               │
       ▼               ▼               ▼
  ┌─────────┐   ┌───────────┐   ┌───────────┐
  │   CSV   │   │ PostgreSQL│   │  InfluxDB  │
  │  Output │   │  weather_ │   │  weather_  │
  │         │   │ forecast  │   │  forecast  │
  └─────────┘   └───────────┘   └─────┬─────┘
                                      │
                                      ▼
                                ┌───────────┐
                                │  Grafana   │
                                │ Dashboard  │
                                └───────────┘
```

---

## 4. ML Data Sources

| Source | Type | Resolution | Period | Rows |
|--------|------|-----------|--------|------|
| Open-Meteo Archive | Historical | 15-min | 2015–2026 | 391,504 |
| NASA POWER API | Historical | 15-min (resampled) | 2015–2026 | 390,909 |
| Open-Meteo Realtime | Live forecast | 15-min | Rolling 8-day | ~768 |
| VEDAS (ISRO SAC) | Forecast | 15-min | Next 3 days | ~288 |
| Open-Meteo Forecast | Forecast | 15-min | Next 7 days | ~672 |

**Merged Training Dataset:** 392,256 rows after deduplication and resampling.

All datasets verified at **identical coordinates**: 22.25°N, 72.74°E (Anand, Gujarat).

---

## 5. Feature Engineering (97 Features)

### 4.1 Lag Features (40 features)
For each of 5 targets × 8 lag steps: t-1, t-2, t-4, t-8, t-16, t-96, t-192, t-288

### 4.2 Rolling Statistics (40 features)
For each of 5 targets × 4 windows × 2 stats (mean, std):
- 1-hour (4 steps), 3-hour (12 steps), 6-hour (24 steps), 12-hour (48 steps)

### 4.3 Time Features (7 features)
`hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`, `month`, `is_daytime`, `weekday`

### 4.4 Interaction Features (3 features)
- `ghi_dhi_ratio` — solar beam fraction
- `temp_wind_interaction` — wind chill proxy
- `cloud_ghi_interaction` — cloud impact on radiation

### 4.5 External Forecast Features (7 features)
- Open-Meteo: `om_ghi`, `om_dhi`, `om_temperature`, `om_windspeed`, `om_cloud_cover`
- VEDAS: `vedas_temperature`, `vedas_solar_potential`

---

## 6. Model Details

**Algorithm:** LightGBM (Gradient Boosted Decision Trees)

| Parameter | Value |
|-----------|-------|
| `num_leaves` | 255 |
| `learning_rate` | 0.03 |
| `n_estimators` | 2000 |
| `feature_fraction` | 0.8 |
| `bagging_fraction` | 0.8 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 0.1 |
| `early_stopping_rounds` | 100 |
| `min_child_samples` | 20 |

**Training split:** Last 30 days = validation (2,880 samples), rest = training (389,088 samples).

---

## 7. Evaluation Results (V2)

### V1 → V2 Comparison

| Target | V1 MAE | V2 MAE | Improvement | V2 R² |
|--------|--------|--------|-------------|-------|
| **GHI** | 11.82 | **10.69** | ↓ 9.6% | 0.9852 |
| **DHI** | 3.04 | 4.00 | — | 0.9369 |
| **Temperature** | 0.30 | **0.28** | ↓ 4.7% | 0.9899 |
| **Windspeed** | 0.19 | **0.10** | ↓ 44.6% | 0.9967 |
| **Cloud Cover** | 0.39 | **0.33** | ↓ 14.6% | 0.9937 |

Training time: 470 seconds (97 features × 392K rows × 2000 estimators).

### Top Feature Importance

| Target | #1 Feature | #2 Feature | #3 Feature |
|--------|-----------|-----------|-----------|
| GHI | `ghi_dhi_ratio` ✨ | `ghi_lag96` | `ghi_rstd_1h` |
| DHI | `ghi_dhi_ratio` ✨ | `dhi_lag4` | `cloud_ghi_interaction` ✨ |
| Temperature | `temperature_lag4` | `temperature_rstd_1h` | `temperature_rmean_1h` |
| Windspeed | `temp_wind_interaction` ✨ | `windspeed_lag4` | `windspeed_rmean_1h` |
| Cloud Cover | `cloud_cover_lag4` | `cloud_cover_rmean_1h` | `cloud_cover_rstd_1h` |

✨ = New V2 interaction features dominating importance

---

## 8. API Integrations

### 7.1 Open-Meteo
- **Archive API:** Fetches recent 3-day observations as seed data for forecasting
- **Forecast API:** 7-day ahead forecast used as input features (`om_*`)
- **No API key required**

### 7.2 VEDAS (ISRO SAC)
- **Endpoint:** `https://vedas.sac.gov.in/VedasServices/rest/ForecastService/forecast15/{lon}/{lat}`
- **Returns:** 3-day temperature + solar potential at 15-min intervals
- **No API key required**
- **Timestamp format:** `"10Mar:05.30"` → parsed to proper datetime

### 7.3 PostgreSQL
- **Table:** `weather_forecast` (auto-created)
- **Upsert:** `ON CONFLICT` batch insert, 3 retries with exponential backoff
- **Fields:** timestamp, ghi, dhi, temperature, windspeed, cloud_cover, model_version, created_at

### 7.4 InfluxDB
- **Bucket:** `weather_forecast`
- **Measurement:** `weather_forecast`
- **Tags:** `source` (ml_model / openmeteo / vedas), `model_version`, `location`
- **Fields:** ghi, dhi, temperature, windspeed, cloud_cover, solar_potential

---

## 9. Deployment

### 8.1 Docker Compose (Recommended)

```bash
cd weather_ml
docker compose up -d --build
```

**Services:**
| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `influxdb` | influxdb:2.7 | 8086 | Time-series storage |
| `grafana` | grafana/grafana:11.3.0 | 3000 | Dashboard visualization |
| `ml-service` | Custom Python 3.12 | — | Forecast pipeline (15-min loop) |

**Default credentials:**
- Grafana: admin / admin → `http://localhost:3000`
- InfluxDB: admin / adminpassword → `http://localhost:8086`

### 8.2 PM2 (Alternative)

```bash
cd weather_ml
pm2 start ecosystem.config.js
pm2 save && pm2 startup
```

### 8.3 Manual

```bash
# Train
cd weather_ml/src
python train_weather_model.py

# Single forecast
python forecast_service.py --once

# Continuous
python scheduler.py
```

---

## 10. Grafana Dashboard

Pre-provisioned dashboard with 6 panels at 15-minute resolution:

1. **GHI** — ML model (orange) vs Open-Meteo (blue) vs VEDAS (green)
2. **DHI** — ML model vs Open-Meteo overlay
3. **Temperature** — 3-source comparison
4. **Wind Speed** — ML vs Open-Meteo (half-width panel)
5. **Cloud Cover** — ML vs Open-Meteo (half-width panel)
6. **VEDAS Solar Potential** — ISRO forecast overlay

Default time range: `now-3d` to `now+3d` (centered on current time).
Auto-refresh: every 15 minutes.

---

## 11. Project Structure

```
weather_ml/
├── data/                              # Symlinks to source CSVs
│   ├── historical_15min_data.csv
│   ├── nasa_historical_data.csv
│   └── live_forecast_data.csv
├── models/
│   ├── weather_model_v2.pkl           # Trained LightGBM models
│   └── evaluation_report_v2.json      # Metrics + feature importance
├── src/
│   ├── config.py                      # Centralized configuration
│   ├── data_loader.py                 # Multi-source data pipeline
│   ├── feature_engineering.py         # 97-feature pipeline
│   ├── train_weather_model.py         # LightGBM training
│   ├── forecast_service.py            # 288-step recursive forecast
│   ├── influxdb_writer.py             # InfluxDB time-series writer
│   ├── postgres_writer.py             # PostgreSQL writer
│   └── scheduler.py                   # 15-min loop (PM2/Docker)
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/influxdb.yml
│   │   └── dashboards/dashboards.yml
│   └── dashboards/weather_dashboard.json
├── Dockerfile
├── docker-compose.yml
├── ecosystem.config.js               # PM2 config
├── requirements.txt
├── forecast_output.csv               # Latest forecast output
└── DOCUMENTATION.md                  # This file
```

---

## 12. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_HOST` | localhost | PostgreSQL host |
| `PG_DATABASE` | weatherdb | PostgreSQL database |
| `PG_USER` | kartikeya | PostgreSQL user |
| `PG_PASSWORD` | (empty) | PostgreSQL password |
| `INFLUXDB_URL` | http://localhost:8086 | InfluxDB URL |
| `INFLUXDB_TOKEN` | weather-ml-token-secret | InfluxDB admin token |
| `INFLUXDB_ORG` | weather-ml | InfluxDB organization |
| `INFLUXDB_BUCKET` | weather_forecast | InfluxDB bucket |

---

## 13. Future Improvements

- **Direct multi-step training** — train models for specific horizons (1h, 6h, 1d, 2d, 3d) instead of recursive
- **Temporal Fusion Transformer** — deep learning model for improved long-horizon accuracy
- **Automated weekly retraining** — schedule model refresh with latest data
- **Ensemble blending** — weighted average of ML + Open-Meteo + VEDAS predictions
- **Anomaly detection** — flag forecast cycles where predictions deviate significantly from external references
