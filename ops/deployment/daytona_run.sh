#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
STATE_DIR=${MM_DAYTONA_STATE_DIR:-/home/daytona/mm-data}
SNAPSHOT_DIR=${MM_DAYTONA_SNAPSHOT_DIR:-/data/muscle-memory-snapshots}

case "$STATE_DIR/" in
  /data/*)
    printf '%s\n' "mutable Daytona state must not use /data FUSE: $STATE_DIR" >&2
    exit 1
    ;;
esac
case "$SNAPSHOT_DIR/" in
  /data/*) ;;
  *)
    printf '%s\n' "immutable Daytona snapshots must use /data: $SNAPSHOT_DIR" >&2
    exit 1
    ;;
esac

mkdir -p "$STATE_DIR" "$SNAPSHOT_DIR"
if [ ! -w "$STATE_DIR" ]; then
  printf '%s\n' "Daytona sandbox state is not writable: $STATE_DIR" >&2
  exit 1
fi
if [ ! -w "$SNAPSHOT_DIR" ]; then
  printf '%s\n' "Daytona snapshot volume is not writable: $SNAPSHOT_DIR" >&2
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
  "$STATE_DIR/assets/approvals" \
  "$STATE_DIR/assets/cache" \
  "$STATE_DIR/coordinator" \
  "$STATE_DIR/graph" \
  "$STATE_DIR/logs" \
  "$STATE_DIR/run" \
  "$STATE_DIR/telemetry"

export MM_API_BACKEND_FACTORY=${MM_API_BACKEND_FACTORY:-muscle_memory.runtime:create_api_backend}
export MM_API_HOST=${MM_API_HOST:-0.0.0.0}
export MM_API_PORT=${MM_API_PORT:-8000}
export MM_API_LOG_LEVEL=${MM_API_LOG_LEVEL:-info}
export MUSCLE_MEMORY_COORDINATOR_DB_PATH=${MUSCLE_MEMORY_COORDINATOR_DB_PATH:-$STATE_DIR/coordinator/coordinator.sqlite3}
export MUSCLE_MEMORY_TELEMETRY_SPOOL=${MUSCLE_MEMORY_TELEMETRY_SPOOL:-$STATE_DIR/telemetry/laserdata-spool.sqlite3}
export MUSCLE_MEMORY_FALKORDB_CACHE_PATH=${MUSCLE_MEMORY_FALKORDB_CACHE_PATH:-$STATE_DIR/graph/falkordb-events.jsonl}
export MM_ASSET_CACHE_DIR=${MM_ASSET_CACHE_DIR:-$STATE_DIR/assets/cache}
export MM_ASSET_APPROVAL_LEDGER_DIR=${MM_ASSET_APPROVAL_LEDGER_DIR:-$STATE_DIR/assets/approvals}
export UV_NO_PROGRESS=1

cd "$ROOT_DIR"
python3 -m ops.deployment.daytona_state preflight \
  --state-dir "$STATE_DIR" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --mutable-path "$MUSCLE_MEMORY_COORDINATOR_DB_PATH" \
  --mutable-path "$MUSCLE_MEMORY_TELEMETRY_SPOOL" \
  --mutable-path "$MUSCLE_MEMORY_FALKORDB_CACHE_PATH" \
  --mutable-path "$MM_ASSET_CACHE_DIR" \
  --mutable-path "$MM_ASSET_APPROVAL_LEDGER_DIR"
if [ ! -f "$ROOT_DIR/frontend/dist/index.html" ]; then
  printf '%s\n' "production frontend build is missing: $ROOT_DIR/frontend/dist/index.html" >&2
  exit 1
fi
if [ "${MM_DAYTONA_SKIP_PREPARE:-0}" != "1" ]; then
  uv sync --frozen --no-dev
  uv run --frozen --no-sync mm-verify-robot
fi

exec uv run --frozen --no-sync python -m ops.api.serve
