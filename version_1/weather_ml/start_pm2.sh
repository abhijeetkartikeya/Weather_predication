#!/usr/bin/env bash

set -eu

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PM2_HOME="$PROJECT_DIR/.pm2"

mkdir -p "$PROJECT_DIR/logs" "$PM2_HOME"

exec pm2 start "$PROJECT_DIR/ecosystem.config.js"
