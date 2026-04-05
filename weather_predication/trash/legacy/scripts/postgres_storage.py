"""
Shared PostgreSQL storage helpers for version_2.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from config import (
    ACTUALS_TABLE,
    FORECAST_TABLE,
    MERGED_DATASET_TABLE,
    NASA_CACHE_TABLE,
    OPENMETEO_CACHE_TABLE,
    POSTGRES_URL,
    TRAINING_REPORT_TABLE,
)
from utils import setup_logger

logger = setup_logger("postgres")

TIME_SERIES_TABLES = {
    NASA_CACHE_TABLE,
    OPENMETEO_CACHE_TABLE,
    MERGED_DATASET_TABLE,
}


def get_engine():
    return create_engine(POSTGRES_URL)


def table_exists(table_name: str) -> bool:
    engine = get_engine()
    return inspect(engine).has_table(table_name)


def ensure_application_tables() -> None:
    """Create application tables used by the pipeline if they do not exist."""
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TRAINING_REPORT_TABLE} (
                run_at TIMESTAMPTZ NOT NULL,
                target TEXT NOT NULL,
                horizon TEXT NOT NULL,
                metrics JSONB NOT NULL,
                PRIMARY KEY (run_at, target, horizon)
            );
        """))

        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TRAINING_REPORT_TABLE}_target_horizon ON {TRAINING_REPORT_TABLE} (target, horizon);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{FORECAST_TABLE}_time ON {FORECAST_TABLE} (time);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{ACTUALS_TABLE}_time ON {ACTUALS_TABLE} (time);"))


def write_time_series(table_name: str, df: pd.DataFrame, if_exists: str = "replace") -> None:
    """Write a DataFrame with a DatetimeIndex to PostgreSQL."""
    if df.empty:
        logger.warning("Table %s not written because the DataFrame is empty", table_name)
        return

    df_to_write = df.copy()
    df_to_write.index.name = "time"
    engine = get_engine()
    df_to_write.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=True,
        chunksize=10000,
        method="multi",
    )
    logger.info("Saved %,d rows to PostgreSQL table %s", len(df_to_write), table_name)


def read_time_series(table_name: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load a PostgreSQL time-series table into a DataFrame."""
    if not table_exists(table_name):
        raise FileNotFoundError(f"PostgreSQL table not found: {table_name}")

    selected = "*"
    if columns:
        selected = ", ".join(["time", *columns])

    engine = get_engine()
    query = text(f"SELECT {selected} FROM {table_name} ORDER BY time")
    df = pd.read_sql(query, engine, parse_dates=["time"])
    df = df.set_index("time")
    return df


def write_training_report(report_df: pd.DataFrame) -> None:
    """Persist the evaluation report as JSON metrics rows."""
    if report_df.empty:
        logger.warning("Training report is empty, nothing to store")
        return

    ensure_application_tables()

    run_at = pd.Timestamp.utcnow()
    records = []
    for _, row in report_df.iterrows():
        metrics = {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in {"target", "horizon"}}
        records.append(
            {
                "run_at": run_at,
                "target": row["target"],
                "horizon": row["horizon"],
                "metrics": metrics,
            }
        )

    pd.DataFrame(records).to_sql(
        TRAINING_REPORT_TABLE,
        get_engine(),
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi",
    )
    logger.info("Saved %,d training report rows to PostgreSQL table %s", len(records), TRAINING_REPORT_TABLE)
