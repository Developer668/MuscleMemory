"""Create and validate the ignored secret environment for the local stack."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import stat
from pathlib import Path

DEFAULT_ENV_PATH = Path(".env.backend.local")
DEFAULT_BACKEND_FACTORY = "muscle_memory.runtime:create_api_backend"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = frozenset(
    {
        "COMPOSE_PROJECT_NAME",
        "FALKORDB_PASSWORD",
        "IGGY_ROOT_PASSWORD",
        "MM_API_BACKEND_FACTORY",
        "MM_API_OPERATOR_TOKEN",
        "MM_API_OPERATOR_TOKEN_SHA256",
    }
)


class DeploymentEnvironmentError(RuntimeError):
    """The local secret environment is missing or unsafe to consume."""


def _secret() -> str:
    return secrets.token_urlsafe(32)


def parse_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or any(character.isspace() for character in key):
            raise DeploymentEnvironmentError(
                f"invalid environment assignment on line {line_number}"
            )
        if key in values:
            raise DeploymentEnvironmentError(f"duplicate environment key: {key}")
        values[key] = value
    return values


def validate_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DeploymentEnvironmentError(f"deployment environment does not exist: {path}")
    permissions = stat.S_IMODE(path.stat().st_mode)
    if permissions & 0o077:
        raise DeploymentEnvironmentError(
            f"deployment environment must not be group/world accessible: {path}"
        )
    values = parse_environment(path)
    missing = sorted(_REQUIRED_KEYS - values.keys())
    if missing:
        raise DeploymentEnvironmentError(
            f"deployment environment is missing required keys: {', '.join(missing)}"
        )
    expected_hash = hashlib.sha256(values["MM_API_OPERATOR_TOKEN"].encode("utf-8")).hexdigest()
    actual_hash = values["MM_API_OPERATOR_TOKEN_SHA256"]
    if not _HASH_PATTERN.fullmatch(actual_hash) or actual_hash != expected_hash:
        raise DeploymentEnvironmentError("operator token digest does not match the token")
    if values["MM_API_BACKEND_FACTORY"] != DEFAULT_BACKEND_FACTORY:
        raise DeploymentEnvironmentError(
            "local deployment backend factory differs from the production composition root"
        )
    return values


def create_environment(path: Path) -> dict[str, str]:
    """Create a new 0600 file without overwriting an existing secret environment."""

    operator_token = _secret()
    operator_digest = hashlib.sha256(operator_token.encode("utf-8")).hexdigest()
    content = "\n".join(
        (
            "# Generated local-development secrets. Do not commit or share this file.",
            "COMPOSE_PROJECT_NAME=muscle-memory",
            "MM_API_BIND_ADDRESS=127.0.0.1",
            "MM_PROVIDER_BIND_ADDRESS=127.0.0.1",
            "MM_API_PORT=8000",
            f"MM_API_BACKEND_FACTORY={DEFAULT_BACKEND_FACTORY}",
            "MM_API_LOG_LEVEL=info",
            f"MM_API_OPERATOR_TOKEN={operator_token}",
            f"MM_API_OPERATOR_TOKEN_SHA256={operator_digest}",
            "IGGY_ROOT_USERNAME=iggy",
            f"IGGY_ROOT_PASSWORD={_secret()}",
            "IGGY_TCP_PORT=8090",
            f"FALKORDB_PASSWORD={_secret()}",
            "FALKORDB_PORT=6379",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DeploymentEnvironmentError(
            f"refusing to overwrite existing deployment environment: {path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return validate_environment(path)


def ensure_environment(path: Path) -> tuple[dict[str, str], bool]:
    if path.exists():
        return validate_environment(path), False
    return create_environment(path), True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or validate the ignored local backend environment."
    )
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_ENV_PATH)
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    _, created = ensure_environment(path)
    action = "created" if created else "validated"
    print(f"{action} local deployment environment: {path}")


if __name__ == "__main__":
    main()
