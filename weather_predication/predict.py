from __future__ import annotations

import argparse

from weather_ml.prediction import generate_predictions
from weather_ml.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the next 3 days of 15-minute ML predictions.")
    parser.add_argument("--output", help="Optional output CSV path relative to the project root.")
    parser.add_argument("--lat", type=float, help="Latitude override for prediction generation.")
    parser.add_argument("--lon", type=float, help="Longitude override for prediction generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_predictions(settings=load_settings(), output_path=args.output, lat=args.lat, lon=args.lon)


if __name__ == "__main__":
    main()
