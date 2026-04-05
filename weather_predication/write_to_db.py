from __future__ import annotations

import argparse

import pandas as pd

from weather_ml.db import ensure_schema, upsert_dataframe
from weather_ml.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL schema or write a CSV file into a table.")
    parser.add_argument("--init-schema", action="store_true", help="Create the PostgreSQL schema objects.")
    parser.add_argument("--table", help="Target PostgreSQL table name.")
    parser.add_argument("--csv", help="CSV file to import.")
    parser.add_argument("--time-column", default="time", help="Timestamp column name in the CSV file.")
    parser.add_argument(
        "--conflict-columns",
        default="time",
        help="Comma-separated columns to use for PostgreSQL upserts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    if args.init_schema:
        ensure_schema(settings)

    if args.table and args.csv:
        frame = pd.read_csv(args.csv)
        if args.time_column in frame.columns:
            frame[args.time_column] = pd.to_datetime(frame[args.time_column], utc=True)
        conflict_columns = [item.strip() for item in args.conflict_columns.split(",") if item.strip()]
        upsert_dataframe(args.table, frame, conflict_columns=conflict_columns, settings=settings)


if __name__ == "__main__":
    main()
