from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from weather_ml.db import ensure_schema, fetch_dataframe
from weather_ml.logging_utils import get_logger
from weather_ml.orchestration import (
    INDIA_LAT_MAX,
    INDIA_LAT_MIN,
    INDIA_LON_MAX,
    INDIA_LON_MIN,
    fetch_stored_predictions,
    run_ondemand_prediction,
    validate_india_location,
)
from weather_ml.settings import load_settings

logger = get_logger("weather_ml.api")

# Track running prediction jobs per location
_running_jobs: dict[str, threading.Event] = {}
_job_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    ensure_schema(settings)
    logger.info("API started — schema verified")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Weather Prediction API",
    description="72-hour weather forecasting for any location in India at 15-minute intervals",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=INDIA_LAT_MIN, le=INDIA_LAT_MAX, description="Latitude (India: 6-37°N)")
    longitude: float = Field(..., ge=INDIA_LON_MIN, le=INDIA_LON_MAX, description="Longitude (India: 68-98°E)")


class ForecastStep(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    predicted_temperature: float | None = None
    predicted_cloud_cover: float | None = None
    predicted_wind_speed: float | None = None
    predicted_ghi: float | None = None
    predicted_dhi: float | None = None
    issue_time: str | None = None
    model_version: str | None = None


class PredictResponse(BaseModel):
    status: str
    latitude: float
    longitude: float
    forecast_hours: int = 72
    interval_minutes: int = 15
    total_steps: int
    predictions: list[ForecastStep]


class PredictAcceptedResponse(BaseModel):
    status: str = "accepted"
    message: str
    latitude: float
    longitude: float


class DataResponse(BaseModel):
    status: str = "ok"
    count: int
    predictions: list[ForecastStep]


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    database: str = "unknown"


class LocationsResponse(BaseModel):
    status: str = "ok"
    locations: list[dict]


# ── Background job runner ────────────────────────────────────────────────────


def _location_key(lat: float, lon: float) -> str:
    return f"{lat:.4f}_{lon:.4f}"


def _run_prediction_job(lat: float, lon: float) -> None:
    key = _location_key(lat, lon)
    try:
        logger.info("Background prediction started for lat=%s lon=%s", lat, lon)
        run_ondemand_prediction(lat, lon)
        logger.info("Background prediction completed for lat=%s lon=%s", lat, lon)
    except Exception as exc:
        logger.exception("Background prediction failed for lat=%s lon=%s: %s", lat, lon, exc)
    finally:
        with _job_lock:
            _running_jobs.pop(key, None)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and database health."""
    settings = load_settings()
    db_status = "unknown"
    try:
        frame = fetch_dataframe("SELECT 1 AS ok", settings=settings)
        db_status = "connected" if not frame.empty else "error"
    except Exception:
        db_status = "disconnected"
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        timestamp=datetime.utcnow().isoformat() + "Z",
        database=db_status,
    )


@app.post("/predict", response_model=PredictResponse | PredictAcceptedResponse, status_code=200)
def predict_weather(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Generate 72-hour weather forecast for any Indian location.

    If predictions already exist, returns them immediately.
    If not, triggers a background pipeline and returns existing data or an accepted status.
    """
    lat = round(request.latitude, 4)
    lon = round(request.longitude, 4)

    try:
        validate_india_location(lat, lon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Check for existing predictions first
    existing = fetch_stored_predictions(lat=lat, lon=lon, settings=load_settings())

    key = _location_key(lat, lon)
    with _job_lock:
        job_running = key in _running_jobs

    if existing:
        # Return existing predictions; trigger refresh in background if not already running
        if not job_running:
            with _job_lock:
                _running_jobs[key] = threading.Event()
            background_tasks.add_task(_run_prediction_job, lat, lon)

        steps = [ForecastStep(**row) for row in existing]
        return PredictResponse(
            status="ok",
            latitude=lat,
            longitude=lon,
            total_steps=len(steps),
            predictions=steps,
        )

    # No existing predictions — run synchronously for first request
    if job_running:
        return PredictAcceptedResponse(
            message="Prediction pipeline is already running for this location. Check back shortly.",
            latitude=lat,
            longitude=lon,
        )

    try:
        results = run_ondemand_prediction(lat, lon)
        steps = [ForecastStep(**row) for row in results]
        return PredictResponse(
            status="ok",
            latitude=lat,
            longitude=lon,
            total_steps=len(steps),
            predictions=steps,
        )
    except Exception as exc:
        logger.exception("Prediction failed for lat=%s lon=%s", lat, lon)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {exc}")


@app.get("/data", response_model=DataResponse)
def get_stored_data(
    latitude: float | None = Query(None, ge=INDIA_LAT_MIN, le=INDIA_LAT_MAX, description="Filter by latitude"),
    longitude: float | None = Query(None, ge=INDIA_LON_MIN, le=INDIA_LON_MAX, description="Filter by longitude"),
    limit: int = Query(288, ge=1, le=2880, description="Max rows to return"),
):
    """Retrieve stored predictions from the database, optionally filtered by location."""
    try:
        lat = round(latitude, 4) if latitude is not None else None
        lon = round(longitude, 4) if longitude is not None else None
        results = fetch_stored_predictions(lat=lat, lon=lon, limit=limit, settings=load_settings())
        steps = [ForecastStep(**row) for row in results]
        return DataResponse(count=len(steps), predictions=steps)
    except Exception as exc:
        logger.exception("Failed to fetch stored predictions")
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")


@app.get("/trigger")
def trigger_prediction(
    background_tasks: BackgroundTasks,
    latitude: float = Query(..., ge=INDIA_LAT_MIN, le=INDIA_LAT_MAX),
    longitude: float = Query(..., ge=INDIA_LON_MIN, le=INDIA_LON_MAX),
):
    """
    Lightweight trigger: if no predictions exist for this location, start the pipeline
    in the background. Returns immediately. Designed to be called by Grafana or cron.
    """
    lat = round(latitude, 4)
    lon = round(longitude, 4)
    key = _location_key(lat, lon)

    existing = fetch_stored_predictions(lat=lat, lon=lon, limit=1, settings=load_settings())
    if existing:
        return {"status": "exists", "latitude": lat, "longitude": lon, "predictions": len(existing)}

    with _job_lock:
        if key in _running_jobs:
            return {"status": "already_running", "latitude": lat, "longitude": lon}
        _running_jobs[key] = threading.Event()

    background_tasks.add_task(_run_prediction_job, lat, lon)
    return {"status": "triggered", "latitude": lat, "longitude": lon, "message": "Pipeline started in background. Data will appear in a few minutes."}


@app.get("/locations", response_model=LocationsResponse)
def list_locations():
    """List all locations that have predictions stored."""
    settings = load_settings()
    try:
        query = f"""
            SELECT DISTINCT latitude, longitude,
                   COUNT(*) AS prediction_count,
                   MAX(created_at) AS last_updated
            FROM {settings.postgres_schema}.weather_predictions
            GROUP BY latitude, longitude
            ORDER BY last_updated DESC
        """
        frame = fetch_dataframe(query, settings=settings, parse_dates=["last_updated"])
        if frame.empty:
            return LocationsResponse(locations=[])
        frame["last_updated"] = frame["last_updated"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return LocationsResponse(locations=frame.to_dict(orient="records"))
    except Exception as exc:
        logger.exception("Failed to list locations")
        raise HTTPException(status_code=500, detail=str(exc))
