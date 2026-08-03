#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  printf '%s\n' "usage: $0 <40-character-commit-sha>" >&2
  exit 2
fi

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT_DIR"

REVISION=$1
SANDBOX=${MM_DAYTONA_SANDBOX:-muscle-memory-backend}
REPOSITORY_URL=https://github.com/Developer668/MuscleMemory.git
REPOSITORY_DIR=/home/daytona/MuscleMemory
PORT=${MM_DAYTONA_API_PORT:-8000}
PREVIEW_EXPIRES=${MM_DAYTONA_PREVIEW_EXPIRES:-3600}
REQUIRE_SPONSORS=${MM_DAYTONA_REQUIRE_SPONSORS:-0}

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
trap 'rm -f "$INFO_FILE"' EXIT HUP INT TERM
daytona info "$SANDBOX" --format json >"$INFO_FILE"
python3 -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
errors = []
if payload.get("state") != "started":
    errors.append("sandbox is not started")
if payload.get("autoStopInterval") != 0:
    errors.append("auto-stop must be disabled")
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
fi
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" fetch --depth 1 origin "$REVISION"
RESOLVED_REVISION=$(daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" rev-parse FETCH_HEAD)
daytona exec "$SANDBOX" -- git -C "$REPOSITORY_DIR" checkout --detach "$RESOLVED_REVISION"

daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 900 -- uv sync --frozen --no-dev
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR/frontend" --timeout 300 -- \
  npm ci --no-audit --no-fund
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR/frontend" --timeout 300 -- \
  npm run build
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 120 -- \
  uv run --frozen --no-sync mm-verify-robot
daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 60 -- \
  uv run --frozen --no-sync python -m ops.deployment.daytona_process --port "$PORT"

if [ "$REQUIRE_SPONSORS" = "1" ]; then
  daytona exec "$SANDBOX" --cwd "$REPOSITORY_DIR" --timeout 120 -- \
    uv run --frozen --no-sync python -m ops.sponsors.verify_laserdata
fi

PREVIEW_URL=$(daytona preview-url "$SANDBOX" --port "$PORT" --expires "$PREVIEW_EXPIRES")
HEALTH_URL=${PREVIEW_URL%/}/api/v1/health
if [ "$REQUIRE_SPONSORS" = "1" ]; then
  python3 -m ops.deployment.smoke \
    --url "$HEALTH_URL" \
    --timeout 180 \
    --require-provider LaserData \
    --require-provider FalkorDB \
    --require-provider guild.ai \
    --require-provider rocketride.ai
else
  python3 -m ops.deployment.smoke --url "$HEALTH_URL" --timeout 180
  printf '%s\n' "Runtime-only smoke passed; sponsor providers were not required."
fi

printf '%s\n' "Daytona sandbox: $SANDBOX"
printf '%s\n' "Revision: $RESOLVED_REVISION"
printf '%s\n' "Preview: $PREVIEW_URL"
