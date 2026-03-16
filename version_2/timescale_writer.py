"""
TimescaleDB Writer for Weather Forecasting Pipeline
====================================================
Writes 15-minute forecast predictions and actuals to TimescaleDB.

Usage:
    from timescale_writer import write_forecasts_to_timescale
    write_forecasts_to_timescale(forecast_df)
"""

import pandas as pd
from sqlalchemy import create_engine, text

from config import (
    LOCATION_NAME,
    POSTGRES_URL,
    TARGET_DISPLAY_NAMES,
)
from utils import setup_logger

logger = setup_logger("timescale")

def get_engine():
    return create_engine(POSTGRES_URL)


def ensure_hypertables_exist():
    """Create the tables and convert them to TimescaleDB hypertables if they don't exist."""
    engine = get_engine()
    
    # We must use engine.begin() for SQLAlchemy 2.0 to auto-commit DDL
    with engine.begin() as conn:
        # Create extension if not exists
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

        # Table for predictions
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_predictions (
                time TIMESTAMPTZ NOT NULL,
                target TEXT NOT NULL,
                target_display TEXT NOT NULL,
                horizon TEXT NOT NULL,
                location TEXT NOT NULL,
                predicted_value DOUBLE PRECISION NOT NULL
            );
        """))

        # Convert to hypertable if not already converted
        conn.execute(text("""
            SELECT create_hypertable('weather_predictions', 'time', if_not_exists => TRUE);
        """))

        # Table for actuals
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_actuals (
                time TIMESTAMPTZ NOT NULL,
                target TEXT NOT NULL,
                target_display TEXT NOT NULL,
                location TEXT NOT NULL,
                actual_value DOUBLE PRECISION NOT NULL
            );
        """))
        
        conn.execute(text("""
            SELECT create_hypertable('weather_actuals', 'time', if_not_exists => TRUE);
        """))


def write_forecasts_to_timescale(forecast_df: pd.DataFrame):
    """
    Write forecast predictions to TimescaleDB.
    """
    if forecast_df.empty:
        logger.warning("Empty forecast DataFrame, nothing to write.")
        return

    try:
        ensure_hypertables_exist()
        
        # Format the df to match schema
        df_to_write = forecast_df.copy()
        df_to_write = df_to_write.rename(columns={"forecast_time": "time"})
        
        # We only need specific columns
        cols = ["time", "target", "target_display", "horizon", "predicted_value"]
        df_to_write = df_to_write[cols]
        df_to_write["location"] = LOCATION_NAME
        
        # Write to DB
        engine = get_engine()
        df_to_write.to_sql(
            "weather_predictions", 
            engine, 
            if_exists="append", 
            index=False,
            method="multi"
        )
        
        logger.info(f"✓ Wrote {len(df_to_write):,} points to TimescaleDB (weather_predictions)")
        
    except Exception as e:
        logger.error(f"Failed to write forecasts to TimescaleDB: {e}")


def write_actuals_to_timescale(df: pd.DataFrame, target_columns: list):
    """
    Write actual observed weather data to TimescaleDB.
    """
    if df.empty:
        return
        
    try:
        ensure_hypertables_exist()
        
        records = []
        for col in target_columns:
            if col not in df.columns:
                continue
            display = TARGET_DISPLAY_NAMES.get(col, col)
            for idx, val in df[col].dropna().items():
                records.append({
                    "time": idx,
                    "target": col,
                    "target_display": display,
                    "location": LOCATION_NAME,
                    "actual_value": float(val)
                })
                
        if not records:
            return
            
        df_to_write = pd.DataFrame(records)
        
        engine = get_engine()
        df_to_write.to_sql(
            "weather_actuals", 
            engine, 
            if_exists="append", 
            index=False,
            method="multi"
        )
        
        logger.info(f"✓ Wrote {len(df_to_write):,} actuals to TimescaleDB (weather_actuals)")
        
    except Exception as e:
        logger.error(f"Failed to write actuals to TimescaleDB: {e}")
