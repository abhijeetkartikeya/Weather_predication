from __future__ import annotations

import json
from datetime import timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .db import ensure_schema, fetch_dataframe, upsert_dataframe
from .features import build_feature_matrix, load_observations
from .logging_utils import get_logger
from .settings import Settings, load_settings, location_token

logger = get_logger("weather_ml.training")


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _split_temporal(features: pd.DataFrame, targets: pd.Series, settings: Settings):
    validation_cutoff = features.index.max() - pd.Timedelta(days=settings.validation_days)
    train_mask = features.index < validation_cutoff
    val_mask = features.index >= validation_cutoff

    x_train = features.loc[train_mask]
    y_train = targets.loc[train_mask]
    x_val = features.loc[val_mask]
    y_val = targets.loc[val_mask]

    if x_train.empty or x_val.empty:
        split_index = int(len(features) * 0.8)
        x_train = features.iloc[:split_index]
        y_train = targets.iloc[:split_index]
        x_val = features.iloc[split_index:]
        y_val = targets.iloc[split_index:]

    return x_train, x_val, y_train, y_val


def models_are_stale(
    settings: Settings | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> bool:
    settings = settings or load_settings()
    lat = float(settings.latitude if lat is None else lat)
    lon = float(settings.longitude if lon is None else lon)
    query = f"""
        SELECT MAX(trained_at) AS latest_value
        FROM {settings.postgres_schema}.weather_training_runs
        WHERE lat = :lat
          AND lon = :lon
    """
    latest_training = fetch_dataframe(query, {"lat": lat, "lon": lon}, settings=settings, parse_dates=["latest_value"])
    if latest_training.empty or latest_training["latest_value"].isna().all():
        return True
    age = pd.Timestamp.now(tz=timezone.utc) - pd.Timestamp(latest_training.iloc[0]["latest_value"])
    return age > pd.Timedelta(hours=settings.retrain_interval_hours)


def train_models(
    settings: Settings | None = None,
    force: bool = False,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    settings = settings or load_settings()
    lat = float(settings.latitude if lat is None else lat)
    lon = float(settings.longitude if lon is None else lon)
    ensure_schema(settings)

    start_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=settings.training_lookback_days)
    observations = load_observations(settings=settings, lat=lat, lon=lon, start_time=start_time)
    features, targets = build_feature_matrix(observations, settings=settings, dropna=True)
    if features.empty:
        raise RuntimeError("Not enough observation history available to train models.")

    model_version = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    location_key = location_token(lat, lon)
    reports: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []

    for target in settings.target_columns:
        logger.info("Training model for %s at lat=%s lon=%s", target, lat, lon)
        x_train, x_val, y_train, y_val = _split_temporal(features, targets[target], settings)

        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=settings.model_random_state,
            n_jobs=2,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        predictions = model.predict(x_val)
        metrics = _metrics(y_val, predictions)

        model_path = settings.models_dir / f"{target}_{location_key}_xgb.joblib"
        metadata_path = settings.models_dir / f"{target}_{location_key}_metadata.json"
        joblib.dump(model, model_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "target": target,
                    "lat": lat,
                    "lon": lon,
                    "model_version": model_version,
                    "algorithm": settings.model_algorithm,
                    "feature_names": list(features.columns),
                    "feature_set_version": settings.feature_set_version,
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        reports.append(
            {
                "model_version": model_version,
                "target": target,
                "lat": lat,
                "lon": lon,
                "trained_at": pd.Timestamp.utcnow().to_pydatetime(),
                "train_rows": int(len(x_train)),
                "validation_rows": int(len(x_val)),
                "metrics": metrics,
            }
        )
        registry_rows.append(
            {
                "model_name": f"xgboost_{target}_{location_key}",
                "model_version": model_version,
                "algorithm": settings.model_algorithm,
                "feature_set_version": settings.feature_set_version,
                "artifact_path": str(model_path.relative_to(settings.project_root)),
                "lat": lat,
                "lon": lon,
                "trained_at": pd.Timestamp.utcnow().to_pydatetime(),
                "metrics": metrics,
            }
        )
        logger.info("Finished %s with MAE %.4f", target, metrics["mae"])

    upsert_dataframe("weather_model_registry", pd.DataFrame(registry_rows), ["model_name"], settings)
    upsert_dataframe("weather_training_runs", pd.DataFrame(reports), ["model_version", "target", "lat", "lon"], settings)

    summary_path = settings.reports_dir / f"latest_training_report_{location_key}.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    **report,
                    "trained_at": pd.Timestamp(report["trained_at"]).isoformat(),
                }
                for report in reports
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model_version": model_version,
        "lat": lat,
        "lon": lon,
        "targets_trained": len(settings.target_columns),
        "report_path": str(summary_path),
        "forced": force,
    }
