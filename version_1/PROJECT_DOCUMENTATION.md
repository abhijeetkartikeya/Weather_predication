# Weather Predication Project — Comprehensive Documentation

This document provides a holistic overview of the **Weather Predication** project. The project is divided into two major components:
1. **Multi-Source Data Ingestion System**: A collection of Python scripts fetching historical and real-time weather data from various providers (NASA, Open-Meteo, OpenWeather, etc.) into a PostgreSQL database.
2. **Weather ML Pipeline (`weather_ml` folder)**: A production-grade Machine Learning pipeline that uses the collected data to forecast solar power-related weather variables for the next 4 days at 15-minute intervals.

---

## 1. Project Structure Overivew

```
Weather_predication/
├── .env / .env.example              # Environment variables configuration
├── Data_collection_report.pdf       # Existing data collection reports
├── weather_ml/                      # Core Machine Learning Pipeline
│   ├── src/                         # ML source code (training, forecasting, data loading)
│   ├── models/                      # Saved trained models (LightGBM)
│   ├── grafana/                     # Grafana dashboards & provisioning
│   ├── Dockerfile / docker-compose.yml # Containerization setup
│   └── DOCUMENTATION.md             # Detailed ML-specific documentation
├── historical_loader.py             # Open-Meteo historical data ingestion (2015-2026)
├── nasa_historical_loader.py        # NASA POWER historical data ingestion (2015-present)
├── openmeteo_realtime_updater.py    # Open-Meteo realtime & forecast data upstert (runs every 15m)
├── meteostat1.py                    # Meteostat historical hourly data ingestion
├── NLOR_data.py                     # NASA (solar) + OpenWeather (realtime weather) ingestion
├── weather_stack.py                 # WeatherStack real-time weather API ingestion
└── wwo.py                           # WorldWeatherOnline (WWO) historical data ingestion
```

---

## 2. Multi-Source Data Ingestion System

The root directory contains several standalone scripts to aggregate weather data from diverse APIs. The target location is Anand, Gujarat, India (22.25°N, 72.74°E). All scripts store their outputs locally in CSVs and a local **PostgreSQL** database backend (default database: `weatherdb`).

### 2.1 Historical Data Loaders

*   **`historical_loader.py`**: Fetches weather data (temperature, windspeed, cloud cover, radiation) from **Open-Meteo Archive API** (2015 - 2026). Interpolates hourly data to **15-minute frequency** and inserts it into the `weather_historical` Postgres table.
*   **`nasa_historical_loader.py`**: Fetches hourly solar and weather data from **NASA POWER API** (2015 - Present). Interpolates to 15-minute frequency, inserts into `weather_historical`, and simultaneously exports to `nasa_historical_data.csv`.
*   **`wwo.py`**: Fetches historical hourly data from **WorldWeatherOnline Premium API**, querying day-by-day from 2015 to the current day. Populates the `weather_full_data` table.
*   **`meteostat1.py`**: Locates the nearest weather station using **Meteostat**, iterates from 2015 to the current year, and populates the `weather_hourly` table.

### 2.2 Real-time & Forecast Updaters

*   **`openmeteo_realtime_updater.py`**: Runs in an infinite loop (every 15 minutes). Fetches 15-minute resolution realtime weather and 7-day forecast data from **Open-Meteo**. Upserts into the `weather_live` table and exports `live_forecast_data.csv`. This provides live alignment for ML inferences.
*   **`weather_stack.py`**: Fetches real-time point-in-time weather data from **WeatherStack API** and inserts it into `weatherstack_realtime`.
*   **`NLOR_data.py`**: A hybrid script that grabs 2020 solar data via NASA and the current weather snapshot via **OpenWeatherMap**, inserting the combined records into `weather_full_data`.

**Database Requirements for Ingestion:**
The scripts expect a PostgreSQL instance configured via a `.env` file containing: `PG_HOST`, `PG_DATABASE`, `PG_USER`, and `PG_PASSWORD`.

---

## 3. Machine Learning Forecasting Pipeline (`weather_ml/`)

Located in the `weather_ml` directory, this pipeline is designed specifically for **Solar Power Forecasting**. It predicts structural variables (GHI, DHI, Temperature, Wind Speed, Cloud Cover) for the next **3 to 4 days** at **15-minute intervals** (up to 384 prediction steps).

### 3.1 ML Architecture

1.  **Data Merger (`src/data_loader.py`)**: Combines the collected historical datasets (NASA, Open-Meteo), external forecasts like the ISRO VEDAS API (3-day solar/temp), and Open-Meteo's 7-day forecast.
2.  **Feature Engineering (`src/feature_engineering.py`)**: Creates 97 robust features.
    *   **Lag Features:** Historical lags at various steps (t-1, t-2 ... t-288).
    *   **Rolling Stats:** Mean and Standard deviation for 1h, 3h, 6h, 12h windows.
    *   **Time/Cyclical:** Sine/Cosine embeddings for hour, day of year, month, etc.
    *   **Interactions:** Advanced features like GHI/DHI ratio, temp/wind interaction.
3.  **Model Training (`src/train_weather_model.py`)**: Yields 5 separate **LightGBM** regressors (one per target variable). Trained on the last ~10 years of data.
4.  **Forecasting Engine (`src/forecast_service.py`)**: Operates recursively. It predicts step *t+1*, then feeds that output backward as a lag feature to predict *t+2*, guided sequentially by Open-Meteo and VEDAS external api predictions.
5.  **Data Sinks (`src/postgres_writer.py`, `src/influxdb_writer.py`)**: Writes forecast outputs to PostgreSQL (`weather_forecast`), InfluxDB (for Grafana), and `forecast_output.csv`.

### 3.2 Running the ML Pipeline

**Setup Dependencies:**
```bash
cd weather_ml
pip install -r requirements.txt
```

**Training the Model:** Calculates features over ~391K rows and saves the LightGBM models to `weather_ml/models/weather_model.pkl`.
```bash
cd weather_ml/src
python train_weather_model.py
```

**Running a Forecast (Single Run):**
```bash
cd weather_ml/src
python forecast_service.py --once
```

**Starting the Continuous Service:**
You can manage the 15-minute loop via **PM2**.
```bash
cd weather_ml
./start_pm2.sh
```

### 3.3 Dashboard and Visualization

The pipeline natively supports **Grafana** backed by **InfluxDB** for visualizing the time-series operations.
*   **Launch Services**: `docker compose up -d influxdb grafana` (from `weather_ml/`).
*   **Grafana Access**: `http://localhost:3000` (Default: `admin` / `admin`).
*   **Pre-provisioned Dashboard**: Shows 6 custom comparative panels analyzing ML outputs (GHI, DHI, Temperature, Wind Speed, Cloud Cover, VEDAS potential) vs Open-Meteo's reference APIs.

---

## 4. Summary & Integration Flow

The outermost script layers (NASA, Meteostat, WWO, Open-Meteo) function as the **Data Extract & Load phase**. They steadily build local CSVs and PostgreSQL tables mapping comprehensive ground truth and multi-agency perspectives. 

The `weather_ml` inner project builds on this foundation. Once the tables/CSVs are saturated, the `weather_ml` PM2-scheduled services take over—merging these disparate sources, passing them through 97 engineered features and passing inputs into a trained LightGBM suite. This creates a synchronized, highly accurate, and redundant 15-minute resolution weather/solar prediction matrix visualized continuously in Grafana.
