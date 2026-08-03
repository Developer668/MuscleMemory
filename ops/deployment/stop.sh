#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
ENV_FILE=${MM_DEPLOY_ENV_FILE:-"$ROOT_DIR/.env.backend.local"}

if [ ! -f "$ENV_FILE" ]; then
  printf '%s\n' "Deployment environment not found: $ENV_FILE" >&2
  exit 2
fi

cd "$ROOT_DIR"
if [ "${1:-}" = "--purge" ]; then
  docker compose --env-file "$ENV_FILE" down --volumes --remove-orphans
else
  docker compose --env-file "$ENV_FILE" down --remove-orphans
fi
