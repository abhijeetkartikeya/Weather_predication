from __future__ import annotations

import argparse

from weather_ml.collection import run_collection_cycle
from weather_ml.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect historical, realtime, and provider forecast weather data.")
    parser.add_argument("--full-backfill", action="store_true", help="Run a historical backfill before the live update.")
    parser.add_argument("--start-date", help="Historical backfill start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Historical backfill end date in YYYY-MM-DD format.")
    parser.add_argument("--lat", type=float, help="Latitude override for the collection cycle.")
    parser.add_argument("--lon", type=float, help="Longitude override for the collection cycle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_collection_cycle(
        settings=load_settings(),
        full_backfill=args.full_backfill,
        start_date=args.start_date,
        end_date=args.end_date,
        lat=args.lat,
        lon=args.lon,
    )


if __name__ == "__main__":
    main()
