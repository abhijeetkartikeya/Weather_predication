#!/usr/bin/env python3
"""
Main orchestrator — runs the full leakage-free weather forecasting pipeline.

Steps:
  1. Collect data from Open-Meteo Archive API
  2. Engineer lag + calendar features
  3. Train XGBoost models (one per target)
  4. Generate recursive (autoregressive) predictions
  5. Visualise actual vs. predicted
  6. Run leakage-detection tests
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the pipeline directory is on sys.path so sibling imports work
# regardless of working directory.
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from config import RAW_CSV, TARGET_VARIABLES, PREDICTIONS_CSV
from feature_engineering import _ALIAS


def main():
    wall_start = time.time()

    # ── 1. Data collection ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 / 6 : Data Collection")
    print("=" * 70)
    from data_collection import collect_data
    if os.path.exists(RAW_CSV):
        print(f"Raw CSV already exists ({RAW_CSV}). Skipping download.")
        print("  (Delete the file to re-download.)")
        raw_df = pd.read_csv(RAW_CSV, parse_dates=["time"])
    else:
        raw_df = collect_data()

    # ── 2. Feature engineering ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 / 6 : Feature Engineering")
    print("=" * 70)
    from feature_engineering import engineer_features
    feat_df = engineer_features(raw_df)

    # ── 3. Model training ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 / 6 : Model Training")
    print("=" * 70)
    from model_training import train_models
    train_df, test_df = train_models(feat_df)

    # ── 4. Direct multi-horizon prediction ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 / 6 : Direct Multi-Horizon Prediction (no leakage)")
    print("=" * 70)
    from prediction import generate_predictions
    pred_df = generate_predictions(train_df, test_df)

    # ── 5. Visualisation ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 / 6 : Visualisation")
    print("=" * 70)
    from visualization import plot_results
    plot_results(pred_df)

    # ── 6. Leakage tests ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 / 6 : Leakage Detection Tests")
    print("=" * 70)
    from leakage_tests import run_all_tests
    tests_passed = run_all_tests()

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - wall_start
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total wall time: {elapsed / 60:.1f} minutes")
    print(f"Raw data rows:   {len(raw_df)}")
    print(f"Feature rows:    {len(feat_df)}")
    print(f"Test predictions:{len(pred_df)}")

    print("\nDirect multi-horizon forecast metrics on test set:")
    for var in TARGET_VARIABLES:
        alias = _ALIAS[var]
        actual = pred_df[f"actual_{alias}"]
        predicted = pred_df[f"predicted_{alias}"]
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        print(f"  {var:25s}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  R2={r2:7.4f}")

    print(f"\nLeakage tests: {'ALL PASSED' if tests_passed else 'SOME FAILED'}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
