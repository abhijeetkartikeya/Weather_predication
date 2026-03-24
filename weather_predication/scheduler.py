from __future__ import annotations

import argparse

from weather_ml.orchestration import run_scheduler
from weather_ml.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PM2-supervised weather pipeline scheduler.")
    parser.add_argument("--once", action="store_true", help="Run a single collection-feature-train-predict cycle.")
    parser.add_argument("--full-backfill", action="store_true", help="Run historical backfill before the first cycle.")
    parser.add_argument("--force-retrain", action="store_true", help="Force training during the next cycle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scheduler(
        settings=load_settings(),
        once=args.once,
        full_backfill=args.full_backfill,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()
