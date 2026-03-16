# Weather ML Pipeline — Solar Power Forecasting

A production-grade machine learning pipeline that predicts weather variables (GHI, DHI, temperature, windspeed, cloud cover) for the next **4 days at 15-minute resolution** (384 predictions) to support solar power forecasting.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                 Data Sources                     │
├──────────────┬───────────┬───────────────────────┤
│ Historical   │ NASA      │ Open-Meteo Forecast   │
│ CSV (2015+)  │ POWER CSV │ API (real-time)       │
└──────┬───────┴─────┬─────┴──────────┬────────────┘
       │             │                │
       ▼             ▼                ▼
 ┌─────────────────────────────────────────────┐
 │         data_loader.py                      │
 │  Merge → Clean → Resample → Normalize      │
 └────────────────────┬────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────┐
 │      feature_engineering.py                 │
 │  Lags → Rolling → Time → Open-Meteo align  │
 └────────────────────┬────────────────────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
 ┌───────────────────┐ ┌────────────────────┐
 │ train_weather_    │ │ forecast_service.py│
 │ model.py          │ │ Recursive 384-step │
 │ LightGBM ×5      │ │ prediction         │
 └───────────────────┘ └────────┬───────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              forecast_   PostgreSQL   PM2 scheduler
              output.csv  weather_     (every 15 min)
                          forecast
```

## Quick Start

### 1. Install Dependencies

```bash
cd weather_ml
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd src
python train_weather_model.py
```

This will:
- Load and merge historical datasets (~391K rows)
- Create time-series features (lag, rolling, time)
- Train 5 LightGBM regressors (one per target)
- Save model to `models/weather_model.pkl`
- Save evaluation report to `models/evaluation_report.json`

### 3. Run a Single Forecast

```bash
cd src
python forecast_service.py --once
```

Generates `forecast_output.csv` with 384 ML predictions + Open-Meteo reference.

### 4. Start InfluxDB + Grafana

```bash
cd weather_ml
docker compose up -d influxdb grafana
```

Services:
- InfluxDB: `http://localhost:8086`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

The Compose file keeps `ml-service` behind the optional `ml-service` profile so it does not duplicate the PM2 scheduler.

### 5. Start Continuous Service (PM2)

```bash
# From the weather_ml directory
mkdir -p logs
./start_pm2.sh
PM2_HOME=$PWD/.pm2 pm2 save
```

Check status:
```bash
PM2_HOME=$PWD/.pm2 pm2 status
PM2_HOME=$PWD/.pm2 pm2 logs weather_forecaster
```

## Output Format

### forecast_output.csv

| Column | Description |
|--------|-------------|
| `timestamp` | Forecast datetime (15-min intervals) |
| `GHI_pred` | Global Horizontal Irradiance (W/m²) |
| `DHI_pred` | Diffuse Horizontal Irradiance (W/m²) |
| `temperature_pred` | Temperature (°C) |
| `windspeed_pred` | Wind speed (km/h) |
| `cloud_cover_pred` | Cloud cover (%) |
| `source` | `ml_model` or `openmeteo_reference` |

### PostgreSQL Table

```sql
weather_forecast (
    timestamp, ghi, dhi, temperature, windspeed, cloud_cover,
    model_version, created_at
)
```

## Model Details

- **Algorithm**: LightGBM (Gradient Boosted Decision Trees)
- **Strategy**: One model per target variable, recursive multi-step
- **Features**: 67+ features including lag values, rolling stats, cyclical time encoding, Open-Meteo forecast alignment
- **Evaluation**: MAE, RMSE, R² per target on 30-day holdout set

## Configuration

All settings are centralized in `src/config.py`:
- Location: Anand, Gujarat (22.25°N, 72.74°E)
- Database: PostgreSQL `weatherdb`
- Forecast horizon: 384 steps (4 days)
- Update interval: 15 minutes

## File Structure

```
weather_ml/
├── data/                        # Symlinks to source CSVs
├── models/
│   ├── weather_model.pkl        # Trained LightGBM models
│   └── evaluation_report.json   # Metrics + feature importance
├── src/
│   ├── config.py                # Centralized configuration
│   ├── data_loader.py           # Multi-source data pipeline
│   ├── feature_engineering.py   # Feature creation
│   ├── train_weather_model.py   # Training pipeline
│   ├── forecast_service.py      # Inference + export
│   ├── postgres_writer.py       # Database integration
│   └── scheduler.py             # PM2 entry point
├── forecast_output.csv          # Latest forecast
├── ecosystem.config.js          # PM2 config
├── requirements.txt
└── README.md
```
