from __future__ import annotations

from weather_ml.db import ensure_schema
from weather_ml.settings import load_settings


def main() -> None:
    ensure_schema(load_settings())


if __name__ == "__main__":
    main()
