#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
DATA_DIR=${MM_DAYTONA_DATA_DIR:-/data}

if [ ! -d "$DATA_DIR" ] || [ ! -w "$DATA_DIR" ]; then
  printf '%s\n' "Daytona data volume is missing or not writable: $DATA_DIR" >&2
  exit 1
fi

HELDOUT_CONFIG_COUNT=0
[ -n "${MM_HELDOUT_EVALUATION_ARTIFACT:-}" ] && HELDOUT_CONFIG_COUNT=$((HELDOUT_CONFIG_COUNT + 1))
[ -n "${MM_HELDOUT_EVALUATION_ARTIFACT_SHA256:-}" ] && HELDOUT_CONFIG_COUNT=$((HELDOUT_CONFIG_COUNT + 1))
[ -n "${MM_HELDOUT_CANDIDATE_CHECKPOINT:-}" ] && HELDOUT_CONFIG_COUNT=$((HELDOUT_CONFIG_COUNT + 1))
[ -n "${MM_HELDOUT_EVALUATED_AT:-}" ] && HELDOUT_CONFIG_COUNT=$((HELDOUT_CONFIG_COUNT + 1))
if [ "$HELDOUT_CONFIG_COUNT" -ne 0 ] && [ "$HELDOUT_CONFIG_COUNT" -ne 4 ]; then
  printf '%s\n' "all four MM_HELDOUT_* values are required together" >&2
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
if [ ! -f "$ROOT_DIR/frontend/dist/index.html" ]; then
  printf '%s\n' "production frontend build is missing: $ROOT_DIR/frontend/dist/index.html" >&2
  exit 1
fi
if [ "${MM_DAYTONA_SKIP_PREPARE:-0}" != "1" ]; then
  uv sync --frozen --no-dev
  uv run --frozen --no-sync mm-verify-robot
fi

exec uv run --frozen --no-sync python -m ops.api.serve
