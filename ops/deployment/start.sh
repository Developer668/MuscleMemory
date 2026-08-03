#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
ENV_FILE=${MM_DEPLOY_ENV_FILE:-"$ROOT_DIR/.env.backend.local"}

cd "$ROOT_DIR"
python3 -m ops.deployment.environment "$ENV_FILE"
docker compose --env-file "$ENV_FILE" up --detach --build --wait --wait-timeout 180

# Provider readiness is append plus consumer readback, not an open TCP socket.
docker compose --env-file "$ENV_FILE" exec -T \
  -e MUSCLE_MEMORY_TELEMETRY_SPOOL=/tmp/laserdata-readiness.sqlite3 \
  api python -m ops.sponsors.verify_laserdata

CONFIGURED_API_PORT=$(sed -n 's/^MM_API_PORT=//p' "$ENV_FILE")
MM_API_PORT=${MM_API_PORT:-${CONFIGURED_API_PORT:-8000}}
python3 -m ops.deployment.smoke \
  --url "http://127.0.0.1:${MM_API_PORT}/api/v1/health" \
  --timeout 90 \
  --require-provider LaserData \
  --require-provider FalkorDB

printf '%s\n' "Muscle Memory backend is running at http://127.0.0.1:${MM_API_PORT}"
printf '%s\n' "Local operator credentials remain in $ENV_FILE"
