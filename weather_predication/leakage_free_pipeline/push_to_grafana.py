"""
Push leakage-free pipeline predictions + actuals to PostgreSQL
and create a dedicated Grafana dashboard.
"""
import os, sys
import pandas as pd
from sqlalchemy import create_engine, text

# ── DB connection ────────────────────────────────────────────────────────────
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5435")
DB_NAME = os.getenv("POSTGRES_DB", "weather_ml_v2")
DB_USER = os.getenv("POSTGRES_USER", "weather")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "weatherpass")

ENGINE = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ── 1. Create table ─────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS public.leakage_free_forecast (
    time          TIMESTAMPTZ NOT NULL,
    variable      TEXT        NOT NULL,
    actual        DOUBLE PRECISION,
    predicted     DOUBLE PRECISION,
    residual      DOUBLE PRECISION,
    PRIMARY KEY (time, variable)
);
"""

# ── 2. Also create a training metrics table ──────────────────────────────────
CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS public.leakage_free_metrics (
    variable      TEXT PRIMARY KEY,
    train_mae     DOUBLE PRECISION,
    train_rmse    DOUBLE PRECISION,
    train_r2      DOUBLE PRECISION,
    test_mae      DOUBLE PRECISION,
    test_rmse     DOUBLE PRECISION,
    test_r2       DOUBLE PRECISION
);
"""

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pred_path = os.path.join(data_dir, "predictions.csv")

    if not os.path.exists(pred_path):
        print("ERROR: predictions.csv not found. Run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(pred_path, parse_dates=["time"])
    print(f"Loaded {len(df)} prediction rows")

    # ── Reshape to long format ───────────────────────────────────────────────
    records = []
    var_map = {
        "temperature_2m": ("actual_temp", "predicted_temp"),
        "cloud_cover":    ("actual_cloud", "predicted_cloud"),
        "wind_speed_10m": ("actual_wind", "predicted_wind"),
        "shortwave_radiation": ("actual_ghi", "predicted_ghi"),
    }

    for _, row in df.iterrows():
        for var, (act_col, pred_col) in var_map.items():
            actual = float(row[act_col]) if pd.notna(row[act_col]) else None
            predicted = float(row[pred_col]) if pd.notna(row[pred_col]) else None
            residual = (actual - predicted) if actual is not None and predicted is not None else None
            records.append({
                "time": row["time"],
                "variable": var,
                "actual": actual,
                "predicted": predicted,
                "residual": residual,
            })

    long_df = pd.DataFrame(records)
    print(f"Reshaped to {len(long_df)} rows (4 variables × {len(df)} timesteps)")

    # ── Write to PostgreSQL ──────────────────────────────────────────────────
    with ENGINE.begin() as conn:
        conn.execute(text(CREATE_TABLE))
        conn.execute(text(CREATE_METRICS))
        # Clear old data
        conn.execute(text("DELETE FROM public.leakage_free_forecast"))
        conn.execute(text("DELETE FROM public.leakage_free_metrics"))

    long_df.to_sql("leakage_free_forecast", ENGINE, schema="public",
                    if_exists="append", index=False, method="multi", chunksize=500)
    print(f"Wrote {len(long_df)} rows to public.leakage_free_forecast")

    # ── Write metrics ────────────────────────────────────────────────────────
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    metrics_rows = []
    for var, (act_col, pred_col) in var_map.items():
        actual = df[act_col].dropna()
        predicted = df[pred_col].loc[actual.index]
        mae = mean_absolute_error(actual, predicted)
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r2 = r2_score(actual, predicted)
        metrics_rows.append({
            "variable": var,
            "train_mae": None, "train_rmse": None, "train_r2": None,
            "test_mae": mae, "test_rmse": rmse, "test_r2": r2,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_sql("leakage_free_metrics", ENGINE, schema="public",
                      if_exists="append", index=False)
    print(f"Wrote metrics for {len(metrics_rows)} variables")

    # ── Verify ───────────────────────────────────────────────────────────────
    with ENGINE.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM public.leakage_free_forecast")).scalar()
        print(f"\nVerification: {count} rows in leakage_free_forecast table")
        sample = conn.execute(text(
            "SELECT * FROM public.leakage_free_forecast ORDER BY time LIMIT 4"
        )).fetchall()
        for row in sample:
            print(f"  {row}")

    print("\n✅ Data pushed to PostgreSQL. Ready for Grafana.")


if __name__ == "__main__":
    main()
