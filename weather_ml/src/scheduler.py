"""
Scheduler — Continuous 15-minute forecast loop for PM2 deployment.

This is the entry point for the PM2 process. It runs forecast_service.run_forecast()
every 15 minutes (900 seconds) in an infinite loop with full error handling.

Usage:
    python scheduler.py
    pm2 start scheduler.py --interpreter python3 --name weather_forecaster
"""

import logging
import signal
import sys
import time
from datetime import datetime

from config import UPDATE_INTERVAL_SEC

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

# Graceful shutdown flag
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.info("Received signal %s — shutting down gracefully …", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────


def main():
    """Run the forecast service in an infinite loop."""
    # Late import to catch import errors early
    from forecast_service import run_forecast

    logger.info("=" * 60)
    logger.info("WEATHER FORECAST SCHEDULER STARTED")
    logger.info("Update interval: %d seconds (%d minutes)",
                UPDATE_INTERVAL_SEC, UPDATE_INTERVAL_SEC // 60)
    logger.info("=" * 60)

    cycle_count = 0
    error_count = 0
    max_consecutive_errors = 10

    while not _shutdown:
        cycle_count += 1

        logger.info("─── Cycle %d at %s ───", cycle_count, datetime.utcnow().isoformat())

        try:
            success = run_forecast()
            if success:
                error_count = 0  # Reset on success
                logger.info("Cycle %d completed successfully", cycle_count)
            else:
                error_count += 1
                logger.warning("Cycle %d returned False (error_count=%d)",
                               cycle_count, error_count)

        except Exception as e:
            error_count += 1
            logger.exception("Cycle %d failed (error_count=%d): %s",
                             cycle_count, error_count, e)

        # Safety: if too many consecutive errors, increase sleep
        if error_count >= max_consecutive_errors:
            logger.error(
                "Reached %d consecutive errors — sleeping 1 hour before retry",
                error_count
            )
            time.sleep(3600)
            error_count = 0
            continue

        # Normal sleep
        if not _shutdown:
            logger.info("Sleeping %d seconds until next cycle …", UPDATE_INTERVAL_SEC)
            # Sleep in small increments to allow graceful shutdown
            for _ in range(UPDATE_INTERVAL_SEC):
                if _shutdown:
                    break
                time.sleep(1)

    logger.info("Scheduler stopped after %d cycles", cycle_count)


if __name__ == "__main__":
    main()
