"""
Scheduler V2 — Entry point for PM2 / Docker.
Runs forecast every 15 minutes with graceful shutdown.
"""

import logging
import signal
import sys
import time

from config import UPDATE_INTERVAL_SEC, LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("scheduler")

SHUTDOWN = False
MAX_CONSECUTIVE_ERRORS = 5


def _handle_signal(signum, frame):
    global SHUTDOWN
    logger.info("Received signal %d — shutting down gracefully …", signum)
    SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main():
    from forecast_service import run_forecast

    logger.info("=" * 60)
    logger.info("Weather Forecast Scheduler V2 starting")
    logger.info("Cycle interval: %d seconds", UPDATE_INTERVAL_SEC)
    logger.info("=" * 60)

    cycle = 0
    consecutive_errors = 0

    while not SHUTDOWN:
        cycle += 1
        logger.info("─── Cycle %d ───", cycle)

        try:
            ok = run_forecast()
            if ok:
                consecutive_errors = 0
                logger.info("Cycle %d completed successfully", cycle)
            else:
                consecutive_errors += 1
                logger.warning("Cycle %d returned failure (%d consecutive errors)",
                               cycle, consecutive_errors)
        except Exception as e:
            consecutive_errors += 1
            logger.error("Cycle %d error: %s (%d consecutive errors)",
                         cycle, e, consecutive_errors, exc_info=True)

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.critical("Too many errors — backing off to 5 minutes")
            sleep_time = 300
        else:
            sleep_time = UPDATE_INTERVAL_SEC

        logger.info("Sleeping %d seconds until next cycle …", sleep_time)
        for _ in range(sleep_time):
            if SHUTDOWN:
                break
            time.sleep(1)

    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
