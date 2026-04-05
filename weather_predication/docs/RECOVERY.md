# Recovery and Backfill Runbook

This pipeline stores actual weather data in `weather_observations` and forecast snapshots in `weather_forecast`.

## Recovery Flow

1. `scheduler.py` starts immediately when the container or PM2 process comes back.
2. `run_collection_cycle()` calls `recover_missing_data()`.
3. The collector reads the latest UTC timestamp for the configured latitude and longitude from PostgreSQL.
4. It scans the recent recovery window for missing 15-minute buckets with `generate_series(...)`.
5. If it finds gaps, it fetches archive data from Open-Meteo and NASA for the missing period and upserts into `weather_observations`.
6. It then fetches the current Open-Meteo live payload, writes the latest actuals, and writes a fresh provider forecast horizon into `weather_forecast`.
7. Feature materialization runs incrementally from the repaired observations.
8. Training runs only when models are stale or explicitly forced.
9. Prediction regenerates the latest ML forecast horizon from the repaired provider snapshot.

All writes use UPSERT semantics, so re-running the same recovery window is safe.

## Important Forecast Note

The current automatic recovery fully backfills missed actual observations and restores the latest provider forecast horizon after downtime.

If you need every historical provider `issue_time` snapshot for the outage period, you need an issue-based archived forecast feed. Open-Meteo archive endpoints used in this project are time-series based, so they are suitable for repairing actual history and restoring the latest operational forecast state, but not for perfectly reconstructing every missed historical forecast issuance.

## SQL Checks

Latest actual timestamp for one location:

```sql
SELECT MAX(time) AS latest_actual_time
FROM public.weather_observations
WHERE lat = 22.25
  AND lon = 72.74;
```

Latest provider forecast issue for one location:

```sql
SELECT MAX(issue_time) AS latest_provider_issue_time
FROM public.weather_forecast
WHERE lat = 22.25
  AND lon = 72.74
  AND forecast_type = 'provider'
  AND source = 'open_meteo';
```

Missing actual 15-minute ranges:

```sql
WITH expected AS (
    SELECT generate_series(
        TIMESTAMPTZ '2026-03-18 00:00:00+00',
        TIMESTAMPTZ '2026-03-22 12:00:00+00',
        INTERVAL '15 minutes'
    ) AS bucket_time
),
missing AS (
    SELECT e.bucket_time
    FROM expected e
    LEFT JOIN public.weather_observations o
        ON o.time = e.bucket_time
       AND o.lat = 22.25
       AND o.lon = 72.74
    WHERE o.time IS NULL
),
grouped AS (
    SELECT
        bucket_time,
        bucket_time - (ROW_NUMBER() OVER (ORDER BY bucket_time) * INTERVAL '15 minutes') AS gap_group
    FROM missing
)
SELECT
    MIN(bucket_time) AS gap_start,
    MAX(bucket_time) AS gap_end
FROM grouped
GROUP BY gap_group
ORDER BY gap_start;
```

Missing forecast buckets in the latest provider snapshot:

```sql
WITH latest_issue AS (
    SELECT MAX(issue_time) AS issue_time
    FROM public.weather_forecast
    WHERE lat = 22.25
      AND lon = 72.74
      AND forecast_type = 'provider'
      AND source = 'open_meteo'
),
expected AS (
    SELECT generate_series(
        (SELECT issue_time + INTERVAL '15 minutes' FROM latest_issue),
        (SELECT issue_time + INTERVAL '3 days' FROM latest_issue),
        INTERVAL '15 minutes'
    ) AS forecast_time
)
SELECT e.forecast_time
FROM expected e
LEFT JOIN public.weather_forecast wf
    ON wf.issue_time = (SELECT issue_time FROM latest_issue)
   AND wf.forecast_time = e.forecast_time
   AND wf.lat = 22.25
   AND wf.lon = 72.74
   AND wf.forecast_type = 'provider'
   AND wf.source = 'open_meteo'
   AND wf.model_version = ''
WHERE wf.forecast_time IS NULL
ORDER BY e.forecast_time;
```

## PM2 Persistence Outside Docker

If you run PM2 directly on the host instead of inside Docker, enable boot persistence once:

```bash
pm2 start ecosystem.config.js --update-env
pm2 save
pm2 startup systemd
```

After `pm2 startup systemd`, PM2 prints one command that must be run with elevated privileges. Run that command once, then run `pm2 save` again if needed.

## Docker Expectations

`docker-compose.yml` now uses `restart: always` for PostgreSQL, Grafana, and the ML service. On host reboot:

1. Docker restarts the containers.
2. The ML container runs `python write_to_db.py --init-schema && pm2-runtime start ecosystem.config.js`.
3. PM2 launches `scheduler.py`.
4. The first scheduler cycle automatically runs recovery before the normal 15-minute loop continues.
