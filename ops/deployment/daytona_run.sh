#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
DATA_DIR=${MM_DAYTONA_DATA_DIR:-/data}

if [ ! -d "$DATA_DIR" ] || [ ! -w "$DATA_DIR" ]; then
  printf '%s\n' "Daytona data volume is missing or not writable: $DATA_DIR" >&2
  exit 1
fi

mkdir -p \
  "$DATA_DIR/assets/approvals" \
  "$DATA_DIR/assets/cache" \
  "$DATA_DIR/coordinator" \
  "$DATA_DIR/graph" \
  "$DATA_DIR/logs" \
  "$DATA_DIR/run" \
  "$DATA_DIR/telemetry"

export MM_API_BACKEND_FACTORY=${MM_API_BACKEND_FACTORY:-muscle_memory.runtime:create_api_backend}
export MM_API_HOST=${MM_API_HOST:-0.0.0.0}
export MM_API_PORT=${MM_API_PORT:-8000}
export MM_API_LOG_LEVEL=${MM_API_LOG_LEVEL:-info}
export MUSCLE_MEMORY_COORDINATOR_DB_PATH=${MUSCLE_MEMORY_COORDINATOR_DB_PATH:-$DATA_DIR/coordinator/coordinator.sqlite3}
export MUSCLE_MEMORY_TELEMETRY_SPOOL=${MUSCLE_MEMORY_TELEMETRY_SPOOL:-$DATA_DIR/telemetry/laserdata-spool.sqlite3}
export MUSCLE_MEMORY_FALKORDB_CACHE_PATH=${MUSCLE_MEMORY_FALKORDB_CACHE_PATH:-$DATA_DIR/graph/falkordb-events.jsonl}
export MM_ASSET_CACHE_DIR=${MM_ASSET_CACHE_DIR:-$DATA_DIR/assets/cache}
export MM_ASSET_APPROVAL_LEDGER_DIR=${MM_ASSET_APPROVAL_LEDGER_DIR:-$DATA_DIR/assets/approvals}
export UV_NO_PROGRESS=1

cd "$ROOT_DIR"
if [ "${MM_DAYTONA_SKIP_PREPARE:-0}" != "1" ]; then
  uv sync --frozen --no-dev
  uv run --frozen --no-sync mm-verify-robot
fi

exec uv run --frozen --no-sync python -m ops.api.serve
