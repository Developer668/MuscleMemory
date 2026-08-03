#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  printf '%s\n' "usage: $0 <40-character-commit-sha>" >&2
  exit 2
fi

ROOT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT_DIR"

REVISION=$1
SANDBOX=${MM_DAYTONA_SANDBOX:-muscle-memory-backend}
REPOSITORY_URL=https://github.com/Developer668/MuscleMemory.git
REPOSITORY_DIR=/home/daytona/MuscleMemory
PORT=${MM_DAYTONA_API_PORT:-8000}
PREVIEW_EXPIRES=${MM_DAYTONA_PREVIEW_EXPIRES:-3600}
REQUIRE_SPONSORS=${MM_DAYTONA_REQUIRE_SPONSORS:-0}
PRODUCTION_HEALTH_URL=${MM_PRODUCTION_HEALTH_URL:-}

case "$REVISION" in
  ''|*[!0-9a-fA-F]*) printf '%s\n' "revision must be a full hexadecimal commit SHA" >&2; exit 2 ;;
esac
if [ "${#REVISION}" -ne 40 ]; then
  printf '%s\n' "revision must be a full 40-character commit SHA" >&2
  exit 2
fi
case "$PORT" in
  ''|*[!0-9]*) printf '%s\n' "MM_DAYTONA_API_PORT must be an integer" >&2; exit 2 ;;
esac
case "$PREVIEW_EXPIRES" in
  ''|*[!0-9]*) printf '%s\n' "MM_DAYTONA_PREVIEW_EXPIRES must be an integer" >&2; exit 2 ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  printf '%s\n' "MM_DAYTONA_API_PORT must be between 1 and 65535" >&2
  exit 2
fi
if [ "$PREVIEW_EXPIRES" -lt 1 ] || [ "$PREVIEW_EXPIRES" -gt 86400 ]; then
  printf '%s\n' "MM_DAYTONA_PREVIEW_EXPIRES must be between 1 and 86400" >&2
  exit 2
fi
if [ "$REQUIRE_SPONSORS" != "0" ] && [ "$REQUIRE_SPONSORS" != "1" ]; then
  printf '%s\n' "MM_DAYTONA_REQUIRE_SPONSORS must be 0 or 1" >&2
  exit 2
fi

INFO_FILE=$(mktemp)
PROCESS_STARTED=0
PROCESS_START_ATTEMPTED=0
PREVIOUS_REVISION=
PREVIOUS_PROCESS_STOPPED=0
ROLLBACK_REQUIRED=0
DEPLOYMENT_COMPLETE=0

stop_api() {
  if daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 60 -- \
    uv run --frozen --no-sync python -m ops.deployment.daytona_process --stop \
      --state-dir /home/daytona/mm-data \
      --snapshot-dir /data/muscle-memory-snapshots; then
    return 0
  fi
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 60 -- \
    uv run --frozen --no-sync python -m ops.deployment.daytona_process --stop \
      --data-dir /home/daytona/mm-data
}

start_api() {
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 60 -- \
    env MM_API_HOST=0.0.0.0 \
    uv run --frozen --no-sync python -m ops.deployment.daytona_process --port "$PORT"
}

restore_previous_revision() {
  printf '%s\n' "Deployment failed; restoring Daytona revision $PREVIOUS_REVISION" >&2
  if [ "$PROCESS_START_ATTEMPTED" -eq 1 ] || [ "$PREVIOUS_PROCESS_STOPPED" -eq 0 ]; then
    stop_api >/dev/null 2>&1 || true
  fi
  daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" reset --hard HEAD || return 1
  daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" clean -ffdx || return 1
  daytona exec "$SANDBOX" -- \
    git -C "$REPOSITORY_DIR" checkout --detach "$PREVIOUS_REVISION" || return 1
  daytona exec "$SANDBOX" -- \
    git -C "$REPOSITORY_DIR" reset --hard "$PREVIOUS_REVISION" || return 1
  daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" clean -ffdx || return 1
  restored_revision=$(
    daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git rev-parse HEAD
  ) || return 1
  if [ "$restored_revision" != "$PREVIOUS_REVISION" ]; then
    printf '%s\n' "Daytona rollback checkout does not match the previous revision" >&2
    return 1
  fi
  if [ -n "$(daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git status --porcelain --untracked-files=all)" ]; then
    printf '%s\n' "Daytona rollback checkout is dirty" >&2
    return 1
  fi
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 900 -- \
    uv sync --frozen --no-dev || return 1
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR/frontend" --timeout 300 -- \
    npm ci --no-audit --no-fund || return 1
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR/frontend" --timeout 300 -- \
    npm run build || return 1
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 120 -- \
    uv run --frozen --no-sync mm-verify-robot || return 1
  start_api || return 1
  printf '%s\n' "Restored previous Daytona revision: $PREVIOUS_REVISION" >&2
}

cleanup() {
  status=$?
  trap - EXIT HUP INT TERM
  rm -f "$INFO_FILE"
  if [ "$status" -ne 0 ]; then
    set +e
    if [ "$ROLLBACK_REQUIRED" -eq 1 ] && [ "$DEPLOYMENT_COMPLETE" -eq 0 ]; then
      if ! restore_previous_revision; then
        printf '%s\n' "FATAL: failed to restore previous Daytona revision $PREVIOUS_REVISION" >&2
      fi
    elif [ "$PROCESS_STARTED" -eq 1 ] || [ "$PROCESS_START_ATTEMPTED" -eq 1 ]; then
      stop_api >/dev/null 2>&1 || true
    fi
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
daytona info "$SANDBOX" --format json >"$INFO_FILE"
python3 -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
errors = []
if payload.get("state") != "started":
    errors.append("sandbox is not started")
if payload.get("autoStopInterval") != 0:
    errors.append("auto-stop must be disabled")
if payload.get("autoArchiveInterval") not in {-1, 43200}:
    errors.append("auto-archive must be disabled or set to the Daytona 30-day maximum")
if payload.get("autoDeleteInterval") != -1:
    errors.append("auto-delete must be disabled")
if payload.get("public") is not True:
    errors.append("sandbox must expose a public preview")
if not any(item.get("mountPath") == "/data" for item in payload.get("volumes", [])):
    errors.append("persistent /data volume is missing")
if errors:
    raise SystemExit("invalid Daytona runtime: " + "; ".join(errors))
print(payload["id"])
' "$INFO_FILE"

if ! daytona exec "$SANDBOX" -- test -d "$REPOSITORY_DIR/.git"; then
  daytona exec "$SANDBOX" -- git clone --filter=blob:none "$REPOSITORY_URL" "$REPOSITORY_DIR"
else
  PREVIOUS_REVISION=$(daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git rev-parse HEAD)
  case "$PREVIOUS_REVISION" in
    ''|*[!0-9a-f]*)
      printf '%s\n' "current Daytona checkout has no restorable revision" >&2
      exit 1
      ;;
  esac
  if [ "${#PREVIOUS_REVISION}" -ne 40 ]; then
    printf '%s\n' "current Daytona checkout has no full restorable revision" >&2
    exit 1
  fi
  if [ -n "$(daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git status --porcelain --untracked-files=all)" ]; then
    printf '%s\n' "current Daytona checkout is dirty; refusing to stop verified production" >&2
    exit 1
  fi
fi
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" fetch --depth 1 origin "$REVISION"
RESOLVED_REVISION=$(daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" rev-parse FETCH_HEAD)
if [ -n "$PREVIOUS_REVISION" ]; then
  ROLLBACK_REQUIRED=1
  if ! stop_api >/dev/null 2>&1; then
    printf '%s\n' "failed to stop and snapshot the previous Daytona process" >&2
    exit 1
  fi
  PREVIOUS_PROCESS_STOPPED=1
fi
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" reset --hard HEAD
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" clean -ffdx
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" checkout --detach "$RESOLVED_REVISION"
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" reset --hard "$RESOLVED_REVISION"
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" clean -ffdx
CHECKED_OUT_REVISION=$(daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git rev-parse HEAD)
if [ "$CHECKED_OUT_REVISION" != "$RESOLVED_REVISION" ]; then
  printf '%s\n' "Daytona checkout does not match the requested revision" >&2
  exit 1
fi
if [ -n "$(daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git status --porcelain --untracked-files=all)" ]; then
  printf '%s\n' "Daytona checkout is dirty after exact revision selection" >&2
  exit 1
fi
if [ -n "$(daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" -- git clean -ndx)" ]; then
  printf '%s\n' "Daytona checkout contains ignored deployment drift" >&2
  exit 1
fi

daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 900 -- uv sync --frozen --no-dev
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR/frontend" --timeout 300 -- \
  npm ci --no-audit --no-fund
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR/frontend" --timeout 300 -- \
  npm run build
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 120 -- \
  uv run --frozen --no-sync mm-verify-robot
PROCESS_START_ATTEMPTED=1
start_api
PROCESS_STARTED=1

if [ "$REQUIRE_SPONSORS" = "1" ]; then
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 120 -- \
    uv run --frozen --no-sync python -m ops.sponsors.verify_laserdata
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 180 -- \
    uv run --frozen --no-sync python -m ops.sponsors.verify_rocketride
fi

DISCOVERY_URL=$(daytona preview-url "$SANDBOX" --port "$PORT" --expires "$PREVIEW_EXPIRES")
PUBLIC_ORIGIN=${DISCOVERY_URL%%\?*}
case "$PUBLIC_ORIGIN" in
  https://*) ;;
  *) printf '%s\n' "Daytona did not return an HTTPS public origin" >&2; exit 1 ;;
esac
HEALTH_URL=${PUBLIC_ORIGIN%/}/api/v1/health
if [ "$REQUIRE_SPONSORS" = "1" ]; then
  python3 -m ops.deployment.smoke \
    --url "$HEALTH_URL" \
    --timeout 180 \
    --require-provider LaserData \
    --require-provider FalkorDB
  printf '%s\n' "Live LaserData and RocketRide probes passed; Guild review evidence remains workflow-scoped."
else
  python3 -m ops.deployment.smoke --url "$HEALTH_URL" --timeout 180
  printf '%s\n' "Runtime-only smoke passed; sponsor providers were not required."
fi

if [ -n "$PRODUCTION_HEALTH_URL" ]; then
  case "$PRODUCTION_HEALTH_URL" in
    https://*) ;;
    *) printf '%s\n' "MM_PRODUCTION_HEALTH_URL must use HTTPS" >&2; exit 2 ;;
  esac
  python3 -m ops.deployment.smoke --url "$PRODUCTION_HEALTH_URL" --timeout 180
  printf '%s\n' "Production-domain smoke passed: $PRODUCTION_HEALTH_URL"
fi

DEPLOYMENT_COMPLETE=1
printf '%s\n' "Daytona sandbox: $SANDBOX"
printf '%s\n' "Revision: $RESOLVED_REVISION"
printf '%s\n' "Public origin: $PUBLIC_ORIGIN"
