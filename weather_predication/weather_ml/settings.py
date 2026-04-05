from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def location_token(lat: float, lon: float) -> str:
    return f"lat_{lat:.4f}_lon_{lon:.4f}".replace("-", "m").replace(".", "_")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    timezone: str = os.getenv("TIMEZONE", "Asia/Kolkata")
    location_name: str = os.getenv("LOCATION_NAME", "New Delhi, India")
    latitude: float = float(os.getenv("LATITUDE", "28.6139"))
    longitude: float = float(os.getenv("LONGITUDE", "77.2090"))

    postgres_host: str = os.getenv("POSTGRES_HOST", "postgres")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "weather_ml")
    postgres_user: str = os.getenv("POSTGRES_USER", "weather")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "weatherpass")
    postgres_schema: str = os.getenv("POSTGRES_SCHEMA", "public")

    schedule_minutes: int = int(os.getenv("SCHEDULE_MINUTES", "15"))
    history_start_date: str = os.getenv("HISTORY_START_DATE", "2021-01-01")
    history_chunk_days: int = int(os.getenv("HISTORY_CHUNK_DAYS", "90"))
    openmeteo_past_days: int = int(os.getenv("OPENMETEO_PAST_DAYS", "2"))
    forecast_days: int = int(os.getenv("FORECAST_DAYS", "3"))
    recovery_lookback_days: int = int(os.getenv("RECOVERY_LOOKBACK_DAYS", "14"))

    model_algorithm: str = os.getenv("MODEL_ALGORITHM", "xgboost")
    model_horizon_steps: int = int(os.getenv("MODEL_HORIZON_STEPS", "288"))
    retrain_interval_hours: int = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
    validation_days: int = int(os.getenv("VALIDATION_DAYS", "14"))
    training_lookback_days: int = int(os.getenv("TRAINING_LOOKBACK_DAYS", "180"))
    model_random_state: int = int(os.getenv("MODEL_RANDOM_STATE", "42"))
    feature_set_version: str = os.getenv("FEATURE_SET_VERSION", "v1")

    api_timeout_seconds: int = int(os.getenv("API_TIMEOUT_SECONDS", "60"))
    api_max_retries: int = int(os.getenv("API_MAX_RETRIES", "5"))
    api_retry_backoff_seconds: float = float(os.getenv("API_RETRY_BACKOFF_SECONDS", "2.0"))

    target_columns: tuple[str, ...] = field(
        default_factory=lambda: _split_csv(
            os.getenv(
                "TARGET_COLUMNS",
                "temperature_2m,cloud_cover,wind_speed_10m,shortwave_radiation,diffuse_radiation",
            )
        )
    )
    observation_columns: tuple[str, ...] = field(
        default_factory=lambda: _split_csv(
            os.getenv(
                "OBSERVATION_COLUMNS",
                ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "dew_point_2m",
                        "pressure_msl",
                        "cloud_cover",
                        "cloud_cover_low",
                        "cloud_cover_mid",
                        "cloud_cover_high",
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "wind_gusts_10m",
                        "shortwave_radiation",
                        "direct_radiation",
                        "diffuse_radiation",
                        "direct_normal_irradiance",
                        "nasa_ghi",
                        "nasa_dhi",
                        "nasa_temperature",
                        "nasa_wind_speed",
                        "nasa_wind_direction",
                        "nasa_humidity",
                        "nasa_cloud_cover",
                        "nasa_pressure",
                    ]
                ),
            )
        )
    )
    lag_steps: tuple[int, ...] = field(
        default_factory=lambda: tuple(int(item) for item in _split_csv(os.getenv("LAG_STEPS", "1,2,4,8,16,96,192")))
    )
    rolling_windows: tuple[int, ...] = field(
        default_factory=lambda: tuple(int(item) for item in _split_csv(os.getenv("ROLLING_WINDOWS", "4,16,96")))
    )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def interval(self) -> str:
        return f"{self.schedule_minutes}min"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "artifacts" / "models"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "artifacts" / "reports"

    @property
    def schema_path(self) -> Path:
        return self.project_root / "init_schema.sql"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    settings = Settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
