# Quick Start Guide

## ⚡ One-Line Startup

```bash
./start.sh
```

This will:
1. ✓ Start PostgreSQL + InfluxDB + Grafana (Docker)
2. ✓ Create database tables
3. ✓ Start PM2 forecast scheduler
4. ✓ Show dashboard URLs and credentials

---

## 🎯 View Dashboard

**Grafana:** http://localhost:3000
- **User:** admin
- **Password:** admin

---

## 📋 What Gets Started

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Data storage (weatherdb) |
| InfluxDB | 8086 | Metrics storage |
| Grafana | 3000 | Dashboard & visualization |
| PM2 Scheduler | - | Runs forecasts every 15 min |

---

## 📦 PM2 Services

Three services run automatically:

1. **forecast_scheduler** - Updates predictions every 15 minutes (always running)
2. **data_collector** - Fetches historical weather data (run once)
3. **model_trainer** - Trains ML models (run once)

```bash
pm2 list              # Show all services
pm2 logs              # View all logs
pm2 stop scheduler    # Stop specific service
pm2 restart all       # Restart services
pm2 delete ecosystem.config.js  # Delete all
```

---

## 🛑 Stop Everything

```bash
./stop.sh
```

---

## 🔧 Manual Commands

Run data collection manually:
```bash
pm2 start data_collector --name "collect_data"
```

Run model training manually:
```bash
pm2 start model_trainer --name "train_models"
```

View forecast results:
```bash
psql -h localhost -U postgres -d weatherdb
SELECT * FROM weather_predictions ORDER BY forecast_time DESC LIMIT 10;
```

---

## 📝 Logs

```bash
tail -f logs/scheduler-out.log     # Forecast scheduler
tail -f logs/collector-out.log     # Data collector
tail -f logs/trainer-out.log       # Model trainer
pm2 logs                            # All PM2 logs
```

---

## 🆘 Troubleshooting

**Docker not running:**
```bash
docker ps  # Check if Docker is running
# If not, start Docker app
```

**PostgreSQL connection failed:**
```bash
docker exec weather_timescaledb pg_isready -U postgres -d weatherdb
```

**PM2 services not starting:**
```bash
pm2 start ecosystem.config.js --update-env
```

**Reset everything:**
```bash
./stop.sh
docker volume prune -f
./start.sh
```

---

## 📚 Files

- **start.sh** - Full startup script
- **stop.sh** - Shutdown script
- **setup_postgres.py** - Database initialization
- **ecosystem.config.js** - PM2 configuration
- **docker-compose.yml** - Docker services
- **requirements.txt** - Python dependencies
- **config.py** - Configuration (update for your location)

---

## 🌍 Change Location

Edit `config.py`:
```python
LATITUDE = 28.6139      # Your latitude
LONGITUDE = 77.2090     # Your longitude
LOCATION_NAME = "New Delhi, India"  # Your location
```

Then update in `ecosystem.config.js` and `docker-compose.yml` if needed.

---

**Status:** Version 2 is production-ready with PostgreSQL backend, automated forecasting, and Grafana dashboard.
