#!/usr/bin/env bash

set -eu

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PM2_HOME="$PROJECT_DIR/.pm2"

exec pm2 delete weather_forecaster
