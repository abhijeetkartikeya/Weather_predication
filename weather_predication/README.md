# Weather Forecasting ML System

Production-ready weather forecasting pipeline built around Python, PostgreSQL, PM2, Docker Compose, and Grafana. The system collects historical and live weather data, materializes engineered features, trains XGBoost forecasting models, produces 3-day forecasts at 15-minute granularity, and stores both actuals and predictions in PostgreSQL for dashboarding.

## Overview

- Data sources: NASA POWER and Open-Meteo
- Storage: PostgreSQL for raw snapshots, processed observations, provider forecasts, feature store, training metadata, and ML predictions
- Automation: PM2 supervises a Python scheduler that runs the pipeline every 15 minutes
- Forecasting: XGBoost models generate 288-step forecasts for the next 3 days
- Visualization: Grafana reads directly from PostgreSQL views optimized for dashboard panels
- Deployment: `docker-compose up --build` starts PostgreSQL, Grafana, and the ML service

## Architecture

```text
                    +----------------------+
                    |   NASA POWER API     |
                    +----------+-----------+
                               |
                               v
+----------------------+   +----------------------+   +----------------------+
|  Open-Meteo Archive  |   | Open-Meteo Forecast  |   |      PM2 Runtime     |
|   + Forecast APIs    +-->+  Python ML Service   +-->+  scheduler.py daemon |
+----------------------+   |  - collect_data.py   |   +----------+-----------+
                           |  - feature_engineering.py          |
                           |  - train_model.py                  |
                           |  - predict.py                      |
                           +----------------------+             |
                                                  |             |
                                                  v             v
                                      +-------------------------------+
                                      |          PostgreSQL           |
                                      | raw snapshots                |
                                      | observations                 |
                                      | provider forecasts           |
                                      | feature store                |
                                      | training runs                |
                                      | ML predictions               |
                                      +---------------+---------------+
                                                      |
                                                      v
                                             +----------------+
                                             |    Grafana     |
                                             | dashboards     |
                                             +----------------+
```

## Project Structure

```text
.
├── weather_ml/                   # Active production package
├── grafana/                      # Provisioned dashboards and datasource config
├── docs/                         # Supporting documentation
├── legacy/                       # Archived prototype scripts, datasets, models, plots
├── tests/                        # Validation helpers
├── artifacts/                    # New model outputs and reports
├── logs/                         # PM2/runtime logs
├── collect_data.py               # CLI: data ingestion
├── feature_engineering.py        # CLI: feature store build
├── train_model.py                # CLI: model training
├── predict.py                    # CLI: prediction generation
├── scheduler.py                  # CLI: orchestrated scheduler
├── write_to_db.py                # CLI: schema init / CSV upsert
├── setup_postgres.py             # Compatibility schema bootstrap
├── docker-compose.yml            # One-command stack startup
├── Dockerfile                    # ML service image
├── ecosystem.config.js           # PM2 process definition
├── init_schema.sql               # PostgreSQL schema + Grafana views
├── requirements.txt              # Python dependencies
└── README.md
```

Legacy assets are grouped like this:

```text
legacy/
├── data/
├── models/
├── results/
└── scripts/
```

The active production flow uses the root CLI files plus `weather_ml/`. Everything under `legacy/` is preserved for reference but is not part of the current Docker or PM2 runtime path.

## Database Schema

Main PostgreSQL objects created by `init_schema.sql`:

- `weather_raw_snapshots`: JSONB payloads from NASA POWER and Open-Meteo API calls
- `weather_observations`: merged 15-minute historical and real-time weather observations
- `weather_external_forecasts`: latest provider forecast snapshots from Open-Meteo
- `weather_feature_store`: engineered feature payloads and aligned targets
- `weather_model_registry`: latest active model artifact for each target
- `weather_training_runs`: historical training metadata and metrics
- `weather_ml_predictions`: ML forecasts for the next 3 days
- `vw_actuals_long`: long-format actuals view for Grafana
- `vw_latest_provider_forecast_long`: latest provider forecast view for Grafana
- `vw_latest_ml_forecast_long`: latest ML forecast view for Grafana

## One-Command Startup

1. Open the project directory.
2. Run:

```bash
docker-compose up --build
```

That command:

- starts PostgreSQL
- starts Grafana
- builds the ML service image
- initializes the schema
- launches PM2 inside the ML container
- starts the Python scheduler that updates data every 15 minutes

## Configuration

The project uses `.env` for runtime configuration. Important values:

- `POSTGRES_*`: PostgreSQL connection settings
- `GRAFANA_ADMIN_*`: Grafana admin credentials
- `LATITUDE`, `LONGITUDE`, `LOCATION_NAME`, `TIMEZONE`: site configuration
- `HISTORY_START_DATE`: first date used for historical backfill
- `SCHEDULE_MINUTES`: scheduler cadence, default `15`
- `MODEL_HORIZON_STEPS`: default `288` for 3 days of 15-minute predictions
- `TARGET_COLUMNS`: target variables trained and forecasted

Use `.env.example` as the template for non-default environments.

## Pipeline Scripts

Each stage also works as a standalone command:

```bash
python write_to_db.py --init-schema
python collect_data.py --full-backfill
python feature_engineering.py
python train_model.py --force
python predict.py
python scheduler.py --once --full-backfill --force-retrain
```

Purpose of each script:

- `collect_data.py`: backfills historical data and updates live/provider forecast data
- `feature_engineering.py`: creates the feature store from PostgreSQL observations
- `train_model.py`: trains XGBoost models and writes metadata to PostgreSQL
- `predict.py`: generates 3-day forecasts at 15-minute intervals and saves them to PostgreSQL
- `write_to_db.py`: initializes schema and can upsert CSV files into a target table
- `scheduler.py`: runs the full orchestration flow once or continuously

## PM2 Scheduling

PM2 is used inside the `ml-service` container. The container command runs:

```bash
python write_to_db.py --init-schema && pm2-runtime start ecosystem.config.js
```

`ecosystem.config.js` starts `scheduler.py`, and the Python scheduler:

- runs one full cycle immediately on container startup
- performs an initial historical backfill on the first run
- sleeps until the next 15-minute boundary
- repeats collection, feature materialization, retraining checks, and prediction

This gives you continuous 15-minute updates while PM2 keeps the process alive and restarts it if it crashes.

On every scheduler startup, the first cycle also runs automatic recovery:

- scans PostgreSQL for missing 15-minute observation buckets in UTC
- backfills missed actuals from Open-Meteo archive plus NASA POWER
- refreshes the latest provider forecast horizon
- regenerates the latest ML forecast after the recovered provider snapshot is available

Operational SQL checks and PM2 host-boot commands are documented in `docs/RECOVERY.md`.

## Grafana

After startup:

- Grafana URL: `http://localhost:3000`
- Username: `admin`
- Password: `admin`

The dashboard is provisioned automatically and includes:

- real-time actuals
- latest ML forecast for the next 3 days
- latest provider forecast baseline
- latest metric summary cards

Use Grafana's built-in time picker for time range selection and the `Metric` variable to filter displayed series.

## SQL Notes For Visualization

The dashboard reads from PostgreSQL views that are already optimized for Grafana:

- `vw_actuals_long`
- `vw_latest_provider_forecast_long`
- `vw_latest_ml_forecast_long`

If you want to build your own panels, these query shapes are a good baseline:

```sql
SELECT time, metric, value
FROM vw_actuals_long
WHERE metric = 'temperature_2m'
  AND time BETWEEN NOW() - INTERVAL '7 days' AND NOW()
ORDER BY time;
```

```sql
SELECT time, metric, value
FROM vw_latest_ml_forecast_long
WHERE metric = 'shortwave_radiation'
ORDER BY time;
```

## Step-by-Step Execution Guide

### Docker path

```bash
docker-compose up --build
docker-compose ps
docker-compose logs -f ml-service
```

### Local Python path

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python write_to_db.py --init-schema
python collect_data.py --full-backfill
python feature_engineering.py
python train_model.py --force
python predict.py
python scheduler.py
```

## Output Artifacts

Generated files are written to:

- `artifacts/models/`: trained XGBoost models and metadata files
- `artifacts/reports/latest_training_report.json`: latest training summary
- `artifacts/reports/latest_predictions.csv`: latest prediction export
- `logs/`: PM2 scheduler logs

## Production Notes

- The current design stores raw API payloads in PostgreSQL JSONB for auditability.
- Historical data is merged into a canonical `weather_observations` table at 15-minute resolution.
- Predictions are versioned by `model_version` and `issue_time`, so historical forecast snapshots are preserved.
- Grafana reads only from PostgreSQL; there is no secondary time-series database dependency.
- Model retraining is triggered automatically when the latest training run becomes older than `RETRAIN_INTERVAL_HOURS`.
