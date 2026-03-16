"""
Model Training Module for Weather Forecasting Pipeline
=========================================================
Trains LightGBM + XGBoost ensemble models for each target × horizon.

Usage:
    python train_model.py                                  # Full training
    python train_model.py --targets temperature_2m         # Single target
    python train_model.py --horizons 24h                   # Single horizon
    python train_model.py --tune                           # With Optuna tuning
    python train_model.py --dataset custom_dataset.csv     # Custom dataset
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    EARLY_STOPPING_ROUNDS,
    ENSEMBLE_WEIGHTS,
    FORECAST_HORIZONS,
    LIGHTGBM_PARAMS,
    MERGED_DATASET_PATH,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    RESULTS_DIR,
    TARGET_COLUMNS,
    TARGET_DISPLAY_NAMES,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    XGBOOST_PARAMS,
)
from feature_engineering import build_features
from utils import format_metrics, setup_logger

warnings.filterwarnings("ignore", category=UserWarning)
logger = setup_logger("train")


# ═══════════════════════════════════════════════
# Data Splitting
# ═══════════════════════════════════════════════

def temporal_split(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train/val/test by year.

    - Train: ≤ TRAIN_END_YEAR (2022)
    - Val:   TRAIN_END_YEAR+1 to VAL_END_YEAR (2023-2024)
    - Test:  > VAL_END_YEAR (2025+)
    """
    train_mask = X.index.year <= TRAIN_END_YEAR
    val_mask = (X.index.year > TRAIN_END_YEAR) & (X.index.year <= VAL_END_YEAR)
    test_mask = X.index.year > VAL_END_YEAR

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"  Split: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate MAE, RMSE, R², and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — avoid div by zero
    mask = np.abs(y_true) > 1e-6
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}


# ═══════════════════════════════════════════════
# Train LightGBM
# ═══════════════════════════════════════════════

def train_lightgbm(
    X_train, y_train, X_val, y_val, params: dict = None
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor with early stopping."""
    p = {**LIGHTGBM_PARAMS, **(params or {})}

    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


# ═══════════════════════════════════════════════
# Train XGBoost
# ═══════════════════════════════════════════════

def train_xgboost(
    X_train, y_train, X_val, y_val, params: dict = None
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with early stopping."""
    p = {**XGBOOST_PARAMS, **(params or {})}

    model = xgb.XGBRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


# ═══════════════════════════════════════════════
# Optuna Hyperparameter Tuning
# ═══════════════════════════════════════════════

def tune_lightgbm(X_train, y_train, X_val, y_val) -> dict:
    """Tune LightGBM hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = train_lightgbm(X_train, y_train, X_val, y_val, params)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)

    logger.info(f"  Optuna best MAE: {study.best_value:.4f}")
    return study.best_params


def tune_xgboost(X_train, y_train, X_val, y_val) -> dict:
    """Tune XGBoost hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = train_xgboost(X_train, y_train, X_val, y_val, params)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)

    logger.info(f"  Optuna best MAE: {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════════
# Ensemble Prediction
# ═══════════════════════════════════════════════

def ensemble_predict(lgb_model, xgb_model, X: pd.DataFrame) -> np.ndarray:
    """Weighted ensemble prediction."""
    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    w_lgb = ENSEMBLE_WEIGHTS["lgbm"]
    w_xgb = ENSEMBLE_WEIGHTS["xgb"]
    return w_lgb * lgb_pred + w_xgb * xgb_pred


# ═══════════════════════════════════════════════
# Feature Importance Plot
# ═══════════════════════════════════════════════

def plot_feature_importance(model, feature_names: list, title: str, path: str, top_n: int = 25):
    """Save a feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="#4A90D9")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════

def train_all(
    dataset_path: str,
    targets: list = None,
    horizons: list = None,
    do_tune: bool = False,
):
    """
    Train all target × horizon models.

    Parameters
    ----------
    dataset_path : str
        Path to merged weather dataset CSV.
    targets : list, optional
        Subset of TARGET_COLUMNS to train.
    horizons : list, optional
        Subset of FORECAST_HORIZONS keys to train.
    do_tune : bool
        Whether to run Optuna hyperparameter tuning.
    """
    # Setup directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path} ...")
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    logger.info(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    targets = targets or TARGET_COLUMNS
    horizons_dict = {k: v for k, v in FORECAST_HORIZONS.items() if (horizons is None or k in horizons)}

    all_results = []

    for target in targets:
        if target not in df.columns:
            logger.warning(f"Target {target} not in dataset, skipping.")
            continue

        for horizon_name, horizon_hours in horizons_dict.items():
            logger.info("=" * 60)
            logger.info(f"  Training: {TARGET_DISPLAY_NAMES.get(target, target)} @ {horizon_name}")
            logger.info("=" * 60)

            # Build features
            X, y = build_features(df, target, horizon_hours)

            if len(X) < 1000:
                logger.warning(f"  Insufficient data ({len(X)} rows), skipping.")
                continue

            # Split
            X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y)

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("  Empty train or val split, skipping.")
                continue

            # Ensure column names are strings (LightGBM/XGBoost requirement)
            X_train.columns = X_train.columns.astype(str)
            X_val.columns = X_val.columns.astype(str)
            X_test.columns = X_test.columns.astype(str)

            # ── Optuna tuning ──
            lgb_params = {}
            xgb_params = {}
            if do_tune:
                logger.info("  Tuning LightGBM ...")
                lgb_params = tune_lightgbm(X_train, y_train, X_val, y_val)
                logger.info("  Tuning XGBoost ...")
                xgb_params = tune_xgboost(X_train, y_train, X_val, y_val)

            # ── Train LightGBM ──
            logger.info("  Training LightGBM ...")
            lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, lgb_params)

            # ── Train XGBoost ──
            logger.info("  Training XGBoost ...")
            xgb_model = train_xgboost(X_train, y_train, X_val, y_val, xgb_params)

            # ── Evaluate ──
            results_row = {"target": target, "horizon": horizon_name}

            for split_name, X_split, y_split in [
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                if len(X_split) == 0:
                    continue

                lgb_pred = lgb_model.predict(X_split)
                xgb_pred = xgb_model.predict(X_split)
                ens_pred = ensemble_predict(lgb_model, xgb_model, X_split)

                lgb_metrics = evaluate(y_split.values, lgb_pred)
                xgb_metrics = evaluate(y_split.values, xgb_pred)
                ens_metrics = evaluate(y_split.values, ens_pred)

                for prefix, metrics in [("lgb", lgb_metrics), ("xgb", xgb_metrics), ("ens", ens_metrics)]:
                    for m_name, m_val in metrics.items():
                        results_row[f"{split_name}_{prefix}_{m_name}"] = m_val

                logger.info(f"\n  {split_name.upper()} Results:")
                logger.info(f"    LightGBM → MAE={lgb_metrics['MAE']:.4f}, R²={lgb_metrics['R²']:.4f}")
                logger.info(f"    XGBoost  → MAE={xgb_metrics['MAE']:.4f}, R²={xgb_metrics['R²']:.4f}")
                logger.info(f"    Ensemble → MAE={ens_metrics['MAE']:.4f}, R²={ens_metrics['R²']:.4f}")

            all_results.append(results_row)

            # ── Save models ──
            model_prefix = f"{target}_{horizon_name}"
            lgb_path = os.path.join(MODELS_DIR, f"{model_prefix}_lgb.joblib")
            xgb_path = os.path.join(MODELS_DIR, f"{model_prefix}_xgb.joblib")
            joblib.dump(lgb_model, lgb_path)
            joblib.dump(xgb_model, xgb_path)
            logger.info(f"  Saved: {lgb_path}, {xgb_path}")

            # ── Save feature names ──
            feat_path = os.path.join(MODELS_DIR, f"{model_prefix}_features.json")
            with open(feat_path, "w") as f:
                json.dump(list(X_train.columns), f)

            # ── Feature importance plot ──
            plot_path = os.path.join(RESULTS_DIR, f"{model_prefix}_importance.png")
            plot_feature_importance(
                lgb_model,
                list(X_train.columns),
                f"{TARGET_DISPLAY_NAMES.get(target, target)} @ {horizon_name} — Feature Importance",
                plot_path,
            )

            # ── Save tuning params if used ──
            if do_tune:
                params_path = os.path.join(MODELS_DIR, f"{model_prefix}_tuned_params.json")
                with open(params_path, "w") as f:
                    json.dump({"lgb": lgb_params, "xgb": xgb_params}, f, indent=2)

    # ── Save evaluation report ──
    if all_results:
        report_df = pd.DataFrame(all_results)
        report_path = os.path.join(RESULTS_DIR, "evaluation_report.csv")
        report_df.to_csv(report_path, index=False)
        logger.info(f"\n✓ Evaluation report saved: {report_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("  TRAINING SUMMARY")
        logger.info("=" * 60)
        for _, row in report_df.iterrows():
            t = TARGET_DISPLAY_NAMES.get(row["target"], row["target"])
            h = row["horizon"]
            test_cols = [c for c in row.index if c.startswith("test_ens_")]
            if test_cols:
                mae = row.get("test_ens_MAE", np.nan)
                r2 = row.get("test_ens_R²", np.nan)
                logger.info(f"  {t:>25} @ {h}: MAE={mae:.4f}, R²={r2:.4f}")

    logger.info("\n✓ Training complete!")


# ═══════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train weather forecasting models")
    parser.add_argument("--dataset", type=str, default=MERGED_DATASET_PATH, help="Path to merged dataset CSV")
    parser.add_argument("--targets", nargs="+", default=None, help="List of targets to train")
    parser.add_argument("--horizons", nargs="+", default=None, help="List of horizons (e.g. 24h 48h)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()

    train_all(
        dataset_path=args.dataset,
        targets=args.targets,
        horizons=args.horizons,
        do_tune=args.tune,
    )


if __name__ == "__main__":
    main()
