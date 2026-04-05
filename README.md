# Weather Prediction ML System

A production-grade weather forecasting pipeline that collects real-time and historical weather data from multiple APIs, trains XGBoost models, and generates 3-day forecasts at 15-minute granularity. Built with Python, PostgreSQL, Docker Compose, PM2, and Grafana.

## Features

- **Multi-Source Data Ingestion** - Fetches from NASA POWER and Open-Meteo (archive + forecast APIs)
- **XGBoost Forecasting** - Trains per-location models for temperature, wind speed, cloud cover, radiation, and more
- **288-Step Predictions** - Generates 3-day forecasts at 15-minute intervals
- **Automated Pipeline** - PM2-managed scheduler runs the full pipeline every 15 minutes
- **Leakage-Free Pipeline** - Dedicated module with strict train/test separation and leakage validation tests
- **Multi-Location Support** - Trained models for 10+ coordinate pairs across India
- **PostgreSQL Storage** - Raw snapshots, observations, feature store, model registry, and predictions
- **Grafana Dashboards** - Auto-provisioned dashboards with actuals vs. ML forecast vs. provider forecast
- **One-Command Deployment** - `docker-compose up --build` starts the full stack

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | XGBoost, scikit-learn |
| Backend | Python 3, FastAPI, Uvicorn |
| Database | PostgreSQL (raw + feature store + predictions) |
| Visualization | Grafana (provisioned dashboards) |
| Scheduling | PM2 + Python scheduler |
| Deployment | Docker Compose |
| Data Sources | NASA POWER API, Open-Meteo API |

## Architecture

```
+----------------------+   +----------------------+
|   NASA POWER API     |   | Open-Meteo Archive   |
+----------+-----------+   | + Forecast APIs      |
           |               +----------+-----------+
           v                          v
      +------------------------------------+
      |       Python ML Service            |
      |  collect_data.py     (ingestion)   |
      |  feature_engineering.py (features) |
      |  train_model.py      (training)    |
      |  predict.py          (inference)   |
      |  scheduler.py        (orchestrator)|
      +----------------+-------------------+
                       |
                       v
           +-----------------------+
           |     PostgreSQL        |
           | - raw_snapshots       |
           | - observations        |
           | - feature_store       |
           | - model_registry      |
           | - ml_predictions      |
           +-----------+-----------+
                       |
                       v
              +----------------+
              |    Grafana     |
              |  Dashboards    |
              +----------------+
```

## Quick Start

### Docker (Recommended)

```bash
cd weather_predication
docker-compose up --build
```

This starts PostgreSQL, Grafana, and the ML service. The scheduler automatically:
1. Initializes the database schema
2. Backfills historical weather data
3. Engineers features and trains models
4. Generates 3-day forecasts
5. Repeats every 15 minutes

### Local Python

```bash
cd weather_predication
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Initialize database
python write_to_db.py --init-schema

# Run full pipeline
python collect_data.py --full-backfill
python feature_engineering.py
python train_model.py --force
python predict.py

# Or run the orchestrated scheduler
python scheduler.py
```

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `collect_data.py` | Ingest historical + live weather data from NASA POWER and Open-Meteo |
| `feature_engineering.py` | Build engineered features from PostgreSQL observations |
| `train_model.py` | Train XGBoost models per target variable and write metadata |
| `predict.py` | Generate 3-day forecasts (288 steps) and save to PostgreSQL |
| `write_to_db.py` | Initialize schema or upsert CSV data |
| `scheduler.py` | Orchestrate the full pipeline on a 15-minute cadence |
| `setup_postgres.py` | Bootstrap PostgreSQL schema |

## Project Structure

```
weather_predication/
├── weather_ml/                   # Core ML package
│   ├── training.py               # Model training logic
│   └── prediction.py             # Prediction generation
├── leakage_free_pipeline/        # Validated leakage-free ML pipeline
│   ├── config.py                 # Pipeline configuration
│   ├── data_collection.py        # Data fetching
│   ├── feature_engineering.py    # Feature creation
│   ├── model_training.py         # Training with leakage prevention
│   ├── prediction.py             # Inference
│   ├── leakage_tests.py          # Leakage validation tests
│   ├── visualization.py          # Result plotting
│   ├── push_to_grafana.py        # Dashboard integration
│   ├── run_pipeline.py           # End-to-end runner
│   └── plots/                    # Generated comparison plots
├── api/                          # FastAPI REST endpoints
├── grafana/
│   └── dashboards/               # Provisioned Grafana dashboards
├── artifacts/
│   ├── models/                   # Trained model metadata (per location)
│   └── reports/                  # Training reports (per location)
├── tests/                        # Validation helpers
├── docs/                         # Additional documentation
├── collect_data.py               # CLI: data ingestion
├── feature_engineering.py        # CLI: feature store
├── train_model.py                # CLI: model training
├── predict.py                    # CLI: prediction
├── scheduler.py                  # CLI: orchestrated scheduler
├── write_to_db.py                # CLI: schema init / CSV upsert
├── docker-compose.yml            # Full stack deployment
├── Dockerfile                    # ML service image
├── ecosystem.config.js           # PM2 process definition
├── init_schema.sql               # PostgreSQL schema + Grafana views
└── requirements.txt              # Python dependencies
```

## Database Schema

PostgreSQL tables created by `init_schema.sql`:

| Table | Purpose |
|-------|---------|
| `weather_raw_snapshots` | JSONB payloads from API calls (audit trail) |
| `weather_observations` | Merged 15-min historical + real-time observations |
| `weather_external_forecasts` | Provider forecast snapshots from Open-Meteo |
| `weather_feature_store` | Engineered features with aligned targets |
| `weather_model_registry` | Active model artifacts per target variable |
| `weather_training_runs` | Training metadata and performance metrics |
| `weather_ml_predictions` | ML forecasts for the next 3 days |

**Grafana Views:**
- `vw_actuals_long` - Historical actuals in long format
- `vw_latest_provider_forecast_long` - Latest provider forecast
- `vw_latest_ml_forecast_long` - Latest ML forecast

## Target Variables

The system trains separate XGBoost models for each:

- `temperature_2m` - Air temperature at 2m
- `wind_speed_10m` - Wind speed at 10m
- `cloud_cover` - Total cloud cover percentage
- `shortwave_radiation` - Global Horizontal Irradiance (GHI)
- `diffuse_radiation` - Diffuse Horizontal Irradiance (DHI)

## Monitored Locations

Models are trained for 10+ locations across India including:

| Location | Coordinates |
|----------|------------|
| Tamil Nadu | 10.32, 78.66 |
| Andhra Pradesh | 11.32, 78.66 |
| Karnataka | 13.32, 78.66 |
| Maharashtra | 16.04, 74.05 |
| Gujarat (Ahmedabad) | 22.25, 72.74 |
| Delhi NCR | 28.61, 77.21 |
| Rajasthan | 28.70, 74.05 |

## Grafana Dashboard

After startup, access Grafana at `http://localhost:3000` (admin/admin).

The auto-provisioned dashboard includes:
- Real-time weather actuals
- ML forecast for the next 3 days
- Provider forecast baseline comparison
- Metric summary cards
- Time-series panels for each target variable

## Configuration

Key environment variables (`.env`):

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=weather
LATITUDE=22.25
LONGITUDE=72.74
SCHEDULE_MINUTES=15
MODEL_HORIZON_STEPS=288
TARGET_COLUMNS=temperature_2m,wind_speed_10m,cloud_cover,shortwave_radiation,diffuse_radiation
```

## Leakage-Free Pipeline

The `leakage_free_pipeline/` module provides a validated ML pipeline with:
- Strict temporal train/test splits (no future data leakage)
- Automated leakage detection tests
- Comparison plots (actual vs. predicted) for each variable
- Independent data collection and feature engineering

Run it standalone:
```bash
cd leakage_free_pipeline
python run_pipeline.py
```

## License

This project is proprietary. All rights reserved.
