CREATE TABLE IF NOT EXISTS weather_raw_snapshots (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    interval_start TIMESTAMPTZ,
    interval_end TIMESTAMPTZ,
    payload JSONB NOT NULL
);

ALTER TABLE weather_raw_snapshots ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
ALTER TABLE weather_raw_snapshots ADD COLUMN IF NOT EXISTS lon DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_weather_raw_snapshots_source_type
    ON weather_raw_snapshots (source, dataset_type, requested_at DESC);

CREATE INDEX IF NOT EXISTS idx_weather_raw_snapshots_location
    ON weather_raw_snapshots (lat, lon, requested_at DESC);

CREATE TABLE IF NOT EXISTS weather_observations (
    time TIMESTAMPTZ NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    temperature_2m DOUBLE PRECISION,
    relative_humidity_2m DOUBLE PRECISION,
    dew_point_2m DOUBLE PRECISION,
    pressure_msl DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    cloud_cover_low DOUBLE PRECISION,
    cloud_cover_mid DOUBLE PRECISION,
    cloud_cover_high DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    wind_direction_10m DOUBLE PRECISION,
    wind_gusts_10m DOUBLE PRECISION,
    shortwave_radiation DOUBLE PRECISION,
    direct_radiation DOUBLE PRECISION,
    diffuse_radiation DOUBLE PRECISION,
    direct_normal_irradiance DOUBLE PRECISION,
    nasa_ghi DOUBLE PRECISION,
    nasa_dhi DOUBLE PRECISION,
    nasa_temperature DOUBLE PRECISION,
    nasa_wind_speed DOUBLE PRECISION,
    nasa_wind_direction DOUBLE PRECISION,
    nasa_humidity DOUBLE PRECISION,
    nasa_cloud_cover DOUBLE PRECISION,
    nasa_pressure DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (time, lat, lon)
);

ALTER TABLE weather_observations ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
ALTER TABLE weather_observations ADD COLUMN IF NOT EXISTS lon DOUBLE PRECISION;
ALTER TABLE weather_observations ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE weather_observations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'weather_observations_pkey'
          AND conrelid = 'weather_observations'::regclass
    ) THEN
        ALTER TABLE weather_observations DROP CONSTRAINT weather_observations_pkey;
    END IF;
EXCEPTION
    WHEN undefined_table THEN NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_observations_time_lat_lon
    ON weather_observations (time, lat, lon);

CREATE INDEX IF NOT EXISTS idx_weather_observations_time_lat_lon_desc
    ON weather_observations (time DESC, lat, lon);

CREATE INDEX IF NOT EXISTS idx_weather_observations_lat_lon_time_desc
    ON weather_observations (lat, lon, time DESC);

CREATE TABLE IF NOT EXISTS weather_external_forecasts (
    issue_time TIMESTAMPTZ NOT NULL,
    forecast_time TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL DEFAULT 'open_meteo',
    temperature_2m DOUBLE PRECISION,
    relative_humidity_2m DOUBLE PRECISION,
    dew_point_2m DOUBLE PRECISION,
    pressure_msl DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    cloud_cover_low DOUBLE PRECISION,
    cloud_cover_mid DOUBLE PRECISION,
    cloud_cover_high DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    wind_direction_10m DOUBLE PRECISION,
    wind_gusts_10m DOUBLE PRECISION,
    shortwave_radiation DOUBLE PRECISION,
    direct_radiation DOUBLE PRECISION,
    diffuse_radiation DOUBLE PRECISION,
    direct_normal_irradiance DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (issue_time, forecast_time, source)
);

CREATE INDEX IF NOT EXISTS idx_weather_external_forecasts_forecast_time
    ON weather_external_forecasts (forecast_time);

CREATE INDEX IF NOT EXISTS idx_weather_external_forecasts_issue_time_desc
    ON weather_external_forecasts (issue_time DESC);

CREATE TABLE IF NOT EXISTS weather_forecast (
    issue_time TIMESTAMPTZ NOT NULL,
    forecast_time TIMESTAMPTZ NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    forecast_type TEXT NOT NULL,
    source TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT '',
    temperature_2m DOUBLE PRECISION,
    relative_humidity_2m DOUBLE PRECISION,
    dew_point_2m DOUBLE PRECISION,
    pressure_msl DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    cloud_cover_low DOUBLE PRECISION,
    cloud_cover_mid DOUBLE PRECISION,
    cloud_cover_high DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    wind_direction_10m DOUBLE PRECISION,
    wind_gusts_10m DOUBLE PRECISION,
    shortwave_radiation DOUBLE PRECISION,
    direct_radiation DOUBLE PRECISION,
    diffuse_radiation DOUBLE PRECISION,
    direct_normal_irradiance DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_weather_forecast_type CHECK (forecast_type IN ('provider', 'ml')),
    PRIMARY KEY (issue_time, forecast_time, lat, lon, forecast_type, source, model_version)
);

CREATE INDEX IF NOT EXISTS idx_weather_forecast_time_lat_lon
    ON weather_forecast (forecast_time, lat, lon);

CREATE INDEX IF NOT EXISTS idx_weather_forecast_issue_type_lat_lon
    ON weather_forecast (issue_time DESC, forecast_type, lat, lon);

CREATE INDEX IF NOT EXISTS idx_weather_forecast_location_window
    ON weather_forecast (lat, lon, forecast_time DESC);

CREATE TABLE IF NOT EXISTS weather_feature_store (
    time TIMESTAMPTZ NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    feature_set_version TEXT NOT NULL,
    features JSONB NOT NULL,
    targets JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE weather_feature_store ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
ALTER TABLE weather_feature_store ADD COLUMN IF NOT EXISTS lon DOUBLE PRECISION;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'weather_feature_store_pkey'
          AND conrelid = 'weather_feature_store'::regclass
    ) THEN
        ALTER TABLE weather_feature_store DROP CONSTRAINT weather_feature_store_pkey;
    END IF;
EXCEPTION
    WHEN undefined_table THEN NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_feature_store_time_lat_lon_version
    ON weather_feature_store (time, lat, lon, feature_set_version);

CREATE INDEX IF NOT EXISTS idx_weather_feature_store_location_time
    ON weather_feature_store (lat, lon, time DESC);

CREATE TABLE IF NOT EXISTS weather_model_registry (
    model_name TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    feature_set_version TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    trained_at TIMESTAMPTZ NOT NULL,
    metrics JSONB NOT NULL
);

ALTER TABLE weather_model_registry ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
ALTER TABLE weather_model_registry ADD COLUMN IF NOT EXISTS lon DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS weather_training_runs (
    model_version TEXT NOT NULL,
    target TEXT NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    trained_at TIMESTAMPTZ NOT NULL,
    train_rows INTEGER NOT NULL,
    validation_rows INTEGER NOT NULL,
    metrics JSONB NOT NULL
);

ALTER TABLE weather_training_runs ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
ALTER TABLE weather_training_runs ADD COLUMN IF NOT EXISTS lon DOUBLE PRECISION;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'weather_training_runs_pkey'
          AND conrelid = 'weather_training_runs'::regclass
    ) THEN
        ALTER TABLE weather_training_runs DROP CONSTRAINT weather_training_runs_pkey;
    END IF;
EXCEPTION
    WHEN undefined_table THEN NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_training_runs_model_target_lat_lon
    ON weather_training_runs (model_version, target, lat, lon);

CREATE INDEX IF NOT EXISTS idx_weather_training_runs_trained_at_desc
    ON weather_training_runs (trained_at DESC);

CREATE TABLE IF NOT EXISTS weather_ml_predictions (
    issue_time TIMESTAMPTZ NOT NULL,
    forecast_time TIMESTAMPTZ NOT NULL,
    target TEXT NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (issue_time, forecast_time, target, model_version)
);

CREATE INDEX IF NOT EXISTS idx_weather_ml_predictions_forecast_time
    ON weather_ml_predictions (forecast_time);

CREATE INDEX IF NOT EXISTS idx_weather_ml_predictions_issue_time_desc
    ON weather_ml_predictions (issue_time DESC);

-- Simplified predictions table matching API spec
CREATE TABLE IF NOT EXISTS weather_predictions (
    id BIGSERIAL PRIMARY KEY,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    predicted_temperature DOUBLE PRECISION,
    predicted_cloud_cover DOUBLE PRECISION,
    predicted_wind_speed DOUBLE PRECISION,
    predicted_ghi DOUBLE PRECISION,
    predicted_dhi DOUBLE PRECISION,
    model_version TEXT,
    issue_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_predictions_loc_time
    ON weather_predictions (latitude, longitude, timestamp, issue_time);

CREATE INDEX IF NOT EXISTS idx_weather_predictions_loc_time_desc
    ON weather_predictions (latitude, longitude, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_weather_predictions_created_at
    ON weather_predictions (created_at DESC);

DROP VIEW IF EXISTS vw_actuals_long;
CREATE OR REPLACE VIEW vw_actuals_long AS
SELECT time, lat, lon, 'temperature_2m'::TEXT AS metric, temperature_2m AS value FROM weather_observations
UNION ALL
SELECT time, lat, lon, 'cloud_cover', cloud_cover FROM weather_observations
UNION ALL
SELECT time, lat, lon, 'wind_speed_10m', wind_speed_10m FROM weather_observations
UNION ALL
SELECT time, lat, lon, 'shortwave_radiation', shortwave_radiation FROM weather_observations
UNION ALL
SELECT time, lat, lon, 'diffuse_radiation', diffuse_radiation FROM weather_observations;

DROP VIEW IF EXISTS vw_latest_provider_forecast_long;
CREATE OR REPLACE VIEW vw_latest_provider_forecast_long AS
WITH latest_issue AS (
    SELECT lat, lon, MAX(issue_time) AS issue_time
    FROM weather_forecast
    WHERE forecast_type = 'provider'
    GROUP BY lat, lon
)
SELECT wf.forecast_time AS time, wf.lat, wf.lon, 'temperature_2m'::TEXT AS metric, wf.temperature_2m AS value, wf.issue_time, wf.source
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'provider'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'cloud_cover', wf.cloud_cover, wf.issue_time, wf.source
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'provider'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'wind_speed_10m', wf.wind_speed_10m, wf.issue_time, wf.source
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'provider'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'shortwave_radiation', wf.shortwave_radiation, wf.issue_time, wf.source
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'provider'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'diffuse_radiation', wf.diffuse_radiation, wf.issue_time, wf.source
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'provider';

DROP VIEW IF EXISTS vw_latest_ml_forecast_long;
CREATE OR REPLACE VIEW vw_latest_ml_forecast_long AS
WITH latest_issue AS (
    SELECT lat, lon, MAX(issue_time) AS issue_time
    FROM weather_forecast
    WHERE forecast_type = 'ml'
    GROUP BY lat, lon
)
SELECT wf.forecast_time AS time, wf.lat, wf.lon, 'temperature_2m'::TEXT AS metric, wf.temperature_2m AS value, wf.issue_time, wf.model_version
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'ml'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'cloud_cover', wf.cloud_cover, wf.issue_time, wf.model_version
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'ml'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'wind_speed_10m', wf.wind_speed_10m, wf.issue_time, wf.model_version
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'ml'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'shortwave_radiation', wf.shortwave_radiation, wf.issue_time, wf.model_version
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'ml'
UNION ALL
SELECT wf.forecast_time, wf.lat, wf.lon, 'diffuse_radiation', wf.diffuse_radiation, wf.issue_time, wf.model_version
FROM weather_forecast wf
JOIN latest_issue li
  ON wf.lat = li.lat
 AND wf.lon = li.lon
 AND wf.issue_time = li.issue_time
WHERE wf.forecast_type = 'ml';
