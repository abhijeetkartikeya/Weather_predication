from __future__ import annotations

import argparse

from weather_ml.features import materialize_feature_store
from weather_ml.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the JSON feature store for a location.")
    parser.add_argument("--lat", type=float, help="Latitude override for feature materialization.")
    parser.add_argument("--lon", type=float, help="Longitude override for feature materialization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_feature_store(load_settings(), lat=args.lat, lon=args.lon)


if __name__ == "__main__":
    main()
