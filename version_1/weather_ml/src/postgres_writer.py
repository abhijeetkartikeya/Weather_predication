"""
PostgreSQL Writer — Create forecast table and insert prediction results.

Handles:
  - Auto-creation of weather_forecast table
  - Batch upsert of forecast rows
  - Connection retry with exponential backoff
  - Graceful failure (logs warning if DB unavailable)
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from config import DB_CONFIG, TARGET_COLUMNS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# SQL Statements
# ──────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS weather_forecast (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    ghi             DOUBLE PRECISION,
    dhi             DOUBLE PRECISION,
    temperature     DOUBLE PRECISION,
    windspeed       DOUBLE PRECISION,
    cloud_cover     DOUBLE PRECISION,
    model_version   VARCHAR(64),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (timestamp, model_version)
);

CREATE INDEX IF NOT EXISTS idx_forecast_ts
    ON weather_forecast (timestamp);

CREATE INDEX IF NOT EXISTS idx_forecast_created
    ON weather_forecast (created_at);
"""

UPSERT_SQL = """
INSERT INTO weather_forecast
    (timestamp, ghi, dhi, temperature, windspeed, cloud_cover, model_version, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp, model_version)
DO UPDATE SET
    ghi         = EXCLUDED.ghi,
    dhi         = EXCLUDED.dhi,
    temperature = EXCLUDED.temperature,
    windspeed   = EXCLUDED.windspeed,
    cloud_cover = EXCLUDED.cloud_cover,
    created_at  = EXCLUDED.created_at;
"""


# ──────────────────────────────────────────────────────────────
# Connection helper
# ──────────────────────────────────────────────────────────────


def _get_connection(max_retries: int = 3, base_delay: float = 2.0):
    """
    Get a psycopg2 connection with retry logic.
    Returns None if all retries fail.
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed — pip install psycopg2-binary")
        return None

    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            logger.debug("PostgreSQL connected (attempt %d)", attempt)
            return conn
        except psycopg2.OperationalError as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "DB connection failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, max_retries, e, delay
            )
            if attempt < max_retries:
                time.sleep(delay)

    logger.error("Could not connect to PostgreSQL after %d attempts", max_retries)
    return None


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def ensure_table_exists() -> bool:
    """Create the weather_forecast table if it doesn't exist. Returns True on success."""
    conn = _get_connection()
    if conn is None:
        return False

    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        logger.info("weather_forecast table ensured")
        return True
    except Exception as e:
        logger.error("Failed to create table: %s", e)
        conn.rollback()
        return False
    finally:
        conn.close()


def insert_forecast(
    df: pd.DataFrame,
    model_version: str = "lgbm_v1",
) -> bool:
    """
    Insert forecast DataFrame into weather_forecast table.

    Expected columns: timestamp, ghi, dhi, temperature, windspeed, cloud_cover

    Returns True on success, False on failure.
    """
    conn = _get_connection()
    if conn is None:
        logger.warning("Skipping PostgreSQL insert — DB unavailable")
        return False

    try:
        ensure_table_exists()
        now = datetime.utcnow()

        records = []
        for _, row in df.iterrows():
            records.append((
                row["timestamp"],
                row.get("ghi"),
                row.get("dhi"),
                row.get("temperature"),
                row.get("windspeed"),
                row.get("cloud_cover"),
                model_version,
                now,
            ))

        with conn.cursor() as cur:
            from psycopg2.extras import execute_batch
            execute_batch(cur, UPSERT_SQL, records, page_size=500)

        conn.commit()
        logger.info("Inserted %d forecast rows (model=%s)", len(records), model_version)
        return True

    except Exception as e:
        logger.error("PostgreSQL insert failed: %s", e)
        conn.rollback()
        return False
    finally:
        conn.close()


def read_latest_forecast(model_version: str = "lgbm_v1") -> Optional[pd.DataFrame]:
    """Read the most recent forecast batch from the database."""
    conn = _get_connection()
    if conn is None:
        return None

    try:
        query = """
            SELECT timestamp, ghi, dhi, temperature, windspeed, cloud_cover,
                   model_version, created_at
            FROM weather_forecast
            WHERE model_version = %s
              AND created_at = (
                  SELECT MAX(created_at)
                  FROM weather_forecast
                  WHERE model_version = %s
              )
            ORDER BY timestamp ASC;
        """
        df = pd.read_sql(query, conn, params=(model_version, model_version))
        logger.info("Read %d rows from weather_forecast", len(df))
        return df
    except Exception as e:
        logger.error("Failed to read forecast: %s", e)
        return None
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ok = ensure_table_exists()
    if ok:
        print("✓ weather_forecast table ready")
    else:
        print("✗ Could not create table (DB might be unavailable)")
