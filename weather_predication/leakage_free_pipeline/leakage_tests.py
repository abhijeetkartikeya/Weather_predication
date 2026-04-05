"""
Automated leakage-detection tests.

Each test prints PASS or FAIL and returns a boolean.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    FEATURES_CSV,
    TRAIN_CUTOFF,
    TARGET_VARIABLES,
    XGB_PARAMS,
)
from feature_engineering import get_base_feature_columns as get_feature_columns


def _load_splits():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["time"])
    train = df[df["time"] <= TRAIN_CUTOFF].copy()
    test = df[df["time"] > TRAIN_CUTOFF].copy()
    return train, test


# ────────────────────────────────────────────────────────────────────────────
# Test 1: Shuffle-target sanity check
# If the model has no leakage, shuffling the target should destroy its
# ability to predict.  If metrics barely change, something is suspicious.
# ────────────────────────────────────────────────────────────────────────────
def test_shuffle_target() -> bool:
    print("\n[Test 1] Shuffle-target sanity check")
    train, test = _load_splits()
    feat_cols = get_feature_columns()
    all_pass = True

    for target in TARGET_VARIABLES:
        # Normal model
        m_normal = XGBRegressor(n_estimators=100, max_depth=4,
                                learning_rate=0.1, random_state=42, n_jobs=-1)
        m_normal.fit(train[feat_cols], train[target], verbose=False)
        r2_normal = r2_score(test[target], m_normal.predict(test[feat_cols]))

        # Shuffled model
        y_shuffled = train[target].sample(frac=1, random_state=0).values
        m_shuffled = XGBRegressor(n_estimators=100, max_depth=4,
                                  learning_rate=0.1, random_state=42, n_jobs=-1)
        m_shuffled.fit(train[feat_cols], y_shuffled, verbose=False)
        r2_shuffled = r2_score(test[target], m_shuffled.predict(test[feat_cols]))

        degradation = r2_normal - r2_shuffled
        passed = degradation > 0.05  # shuffled should be substantially worse
        status = "PASS" if passed else "FAIL"
        print(f"  {target}: R2 normal={r2_normal:.4f}  "
              f"R2 shuffled={r2_shuffled:.4f}  "
              f"degradation={degradation:.4f}  [{status}]")
        if not passed:
            all_pass = False

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ────────────────────────────────────────────────────────────────────────────
# Test 2: Feature-target correlation check
# No single feature should correlate ~1.0 with a target — that usually
# means the target (or a trivial transform of it) leaked into the features.
# ────────────────────────────────────────────────────────────────────────────
def test_correlation() -> bool:
    print("\n[Test 2] Feature-target correlation check")
    train, _ = _load_splits()
    feat_cols = get_feature_columns()
    all_pass = True

    for target in TARGET_VARIABLES:
        for fc in feat_cols:
            corr = train[fc].corr(train[target])
            if abs(corr) > 0.99:
                print(f"  FAIL: |corr({fc}, {target})| = {abs(corr):.4f} > 0.99")
                all_pass = False

    if all_pass:
        print("  No feature has |correlation| > 0.99 with any target.")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ────────────────────────────────────────────────────────────────────────────
# Test 3: Realism check
# Real weather models should NOT achieve R2 >= 0.99 on an out-of-sample
# multi-week horizon.  If they do, there is almost certainly leakage.
# ────────────────────────────────────────────────────────────────────────────
def test_not_too_perfect() -> bool:
    print("\n[Test 3] Realism check (R2 < 0.99 on test set)")
    train, test = _load_splits()
    feat_cols = get_feature_columns()
    all_pass = True

    for target in TARGET_VARIABLES:
        m = XGBRegressor(n_estimators=100, max_depth=4,
                         learning_rate=0.1, random_state=42, n_jobs=-1)
        m.fit(train[feat_cols], train[target], verbose=False)
        r2 = r2_score(test[target], m.predict(test[feat_cols]))
        passed = r2 < 0.99
        status = "PASS" if passed else "FAIL"
        print(f"  {target}: R2={r2:.4f}  [{status}]")
        if not passed:
            all_pass = False

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def run_all_tests() -> bool:
    print("=" * 60)
    print("LEAKAGE DETECTION TESTS")
    print("=" * 60)
    results = [
        test_shuffle_target(),
        test_correlation(),
        test_not_too_perfect(),
    ]
    print("\n" + "=" * 60)
    all_ok = all(results)
    print(f"FINAL VERDICT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    run_all_tests()
