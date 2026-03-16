"""
Scheduler — Auto-update every 15 minutes
==========================================
Runs the prediction pipeline every 15 minutes and writes
forecasts to InfluxDB. Also fetches latest actuals for comparison.

Usage:
    python scheduler.py                  # Start the 15-min auto-update loop
    python scheduler.py --once           # Run once and exit
    python scheduler.py --interval 30    # Custom interval in minutes
"""

import argparse
import signal
import sys
import time
import traceback
from datetime import datetime

from config import (
    SCHEDULER_INTERVAL_MINUTES,
    TARGET_COLUMNS,
)
from utils import setup_logger

logger = setup_logger("scheduler")

# Graceful shutdown flag
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    logger.info("\n⏹  Shutdown signal received. Finishing current cycle ...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_prediction_cycle():
    """
    Execute one prediction cycle:
    1. Fetch latest data from Open-Meteo
    2. Run prediction pipeline
    3. Write forecasts to InfluxDB
    4. Write latest actuals to InfluxDB
    """
    logger.info("=" * 60)
    logger.info(f"  Prediction Cycle @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    cycle_start = time.time()

    try:
        # Step 1: Generate forecast
        from predict import generate_forecast, fetch_recent_data
        forecast_df = generate_forecast()

        if forecast_df.empty:
            logger.warning("No forecasts generated this cycle.")
            return

        # Step 2: Write forecasts to Database
        from timescale_writer import write_forecasts_to_timescale, write_actuals_to_timescale
        write_forecasts_to_timescale(forecast_df)

        # Step 3: Write latest actuals for comparison
        # The user requested 30 days of past data on InfluxDB.
        try:
            recent_df = fetch_recent_data(past_days=30)
            write_actuals_to_timescale(recent_df, TARGET_COLUMNS)
        except Exception as e:
            logger.warning(f"Could not write actuals: {e}")

        elapsed = time.time() - cycle_start
        logger.info(f"\n✓ Cycle complete in {elapsed:.1f}s")
        logger.info(f"  Forecasts written: {len(forecast_df):,} points")

    except Exception as e:
        logger.error(f"Cycle failed: {e}")
        traceback.print_exc()


def run_scheduler(interval_minutes: int = None, run_once: bool = False):
    """
    Main scheduler loop.

    Parameters
    ----------
    interval_minutes : int
        Minutes between each prediction cycle. Default: 15.
    run_once : bool
        If True, run one cycle and exit.
    """
    global _shutdown
    interval = interval_minutes or SCHEDULER_INTERVAL_MINUTES

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║   Weather Forecast Scheduler — 15-Min Auto-Update   ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"  Interval: every {interval} minutes")
    logger.info(f"  Press Ctrl+C to stop gracefully")
    logger.info("")

    # Ensure Database tables exist
    try:
        from timescale_writer import ensure_hypertables_exist
        ensure_hypertables_exist()
    except Exception as e:
        logger.warning(f"Could not check/create TimescaleDB tables: {e}")

    cycle_count = 0

    while not _shutdown:
        cycle_count += 1
        logger.info(f"\n── Cycle #{cycle_count} ──")

        run_prediction_cycle()

        if run_once:
            logger.info("Single run mode. Exiting.")
            break

        # Wait for next interval
        next_run = datetime.now().strftime("%H:%M:%S")
        logger.info(f"\n⏳ Next cycle in {interval} minutes (waiting until {next_run}) ...")

        # Sleep in small increments for responsive shutdown
        for _ in range(interval * 60):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("\n✓ Scheduler stopped.")


def main():
    parser = argparse.ArgumentParser(description="Weather forecast scheduler")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=None, help="Interval in minutes (default: 15)")
    args = parser.parse_args()

    run_scheduler(interval_minutes=args.interval, run_once=args.once)


if __name__ == "__main__":
    main()
