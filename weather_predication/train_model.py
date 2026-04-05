from __future__ import annotations

import argparse

from weather_ml.settings import load_settings
from weather_ml.training import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost weather forecasting models.")
    parser.add_argument("--force", action="store_true", help="Force a retrain even if models are fresh.")
    parser.add_argument("--lat", type=float, help="Latitude override for model training.")
    parser.add_argument("--lon", type=float, help="Longitude override for model training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(settings=load_settings(), force=args.force, lat=args.lat, lon=args.lon)


if __name__ == "__main__":
    main()
