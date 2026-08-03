from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ops.deployment.environment import (  # noqa: E402
    DEFAULT_BACKEND_FACTORY,
    DeploymentEnvironmentError,
    ensure_environment,
    validate_environment,
)
from ops.deployment.smoke import SmokeError, validate_health  # noqa: E402

HASH_A = "a" * 64


def test_local_environment_is_secret_idempotent_and_self_consistent(tmp_path: Path) -> None:
    path = tmp_path / ".env.backend.local"

    first, created = ensure_environment(path)
    second, created_again = ensure_environment(path)

    assert created is True
    assert created_again is False
    assert first == second
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert first["MM_API_BACKEND_FACTORY"] == DEFAULT_BACKEND_FACTORY
    assert (
        first["MM_API_OPERATOR_TOKEN_SHA256"]
        == hashlib.sha256(first["MM_API_OPERATOR_TOKEN"].encode("utf-8")).hexdigest()
    )
    assert first["FALKORDB_PASSWORD"] != first["IGGY_ROOT_PASSWORD"]


def test_local_environment_rejects_unsafe_permissions(tmp_path: Path) -> None:
    path = tmp_path / ".env.backend.local"
    ensure_environment(path)
    path.chmod(0o644)

    with pytest.raises(DeploymentEnvironmentError, match="group/world accessible"):
        validate_environment(path)


def test_smoke_requires_named_ready_providers() -> None:
    payload = {
        "state": "healthy",
        "providers": [
            {"provider": "LaserData", "state": "end_to_end_verified"},
            {"provider": "FalkorDB", "state": "healthy"},
        ],
    }

    states = validate_health(
        payload,
        required_providers=("LaserData", "FalkorDB"),
    )

    assert states == {"LaserData": "end_to_end_verified", "FalkorDB": "healthy"}


def test_smoke_rejects_a_configured_but_unverified_provider() -> None:
    payload = {
        "state": "degraded",
        "providers": [{"provider": "LaserData", "state": "configured"}],
    }

    with pytest.raises(SmokeError, match="LaserData=configured"):
        validate_health(payload, required_providers=("LaserData",))


def test_dockerfile_is_locked_and_runs_without_root() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "python:3.12.13-slim-bookworm@sha256:" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.11.24@sha256:" in dockerfile
    assert dockerfile.count("uv sync --frozen --no-dev") == 2
    assert "USER ${APP_UID}:${APP_GID}" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "ops.api.serve"]' in dockerfile
    assert "LASERDATA_CONNECTION_STRING=" not in dockerfile
    assert "FALKORDB_PASSWORD=" not in dockerfile


def test_startup_requires_real_laserdata_append_and_readback() -> None:
    start_script = (REPOSITORY_ROOT / "ops/deployment/start.sh").read_text(encoding="utf-8")
    compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    verify_position = start_script.index("ops.sponsors.verify_laserdata")
    smoke_position = start_script.index("ops.deployment.smoke")
    assert verify_position < smoke_position
    assert "LASERDATA_TOPIC: ${LASERDATA_TOPIC:-episode-events-v2}" in compose
    assert "iggy://" not in compose
    assert "@laserdata:8090}" in compose
    assert "laserdatainc/iggy-server:latest@sha256:" in compose
    assert "/var/lib/iggy" in compose


def test_compose_resolves_hardened_persistent_local_stack() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")
    environment = os.environ.copy()
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": "muscle-memory-config-test",
            "FALKORDB_PASSWORD": "falkor-test-secret",
            "IGGY_ROOT_PASSWORD": "iggy-test-secret",
            "MM_API_OPERATOR_TOKEN_SHA256": HASH_A,
            "LASERDATA_CONNECTION_STRING": "iggy://cloud-user:cloud-pass@managed.example:8090",
            "MUSCLE_MEMORY_FALKORDB_URL": "rediss://cloud-user:cloud-pass@managed.example:6379",
        }
    )
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(REPOSITORY_ROOT),
            "-f",
            str(REPOSITORY_ROOT / "docker-compose.yml"),
            "config",
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    services = config["services"]

    assert set(services) == {"api", "falkordb", "laserdata"}
    assert services["api"]["read_only"] is True
    assert services["api"]["cap_drop"] == ["ALL"]
    assert services["api"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["falkordb"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["laserdata"]["ports"][0]["host_ip"] == "127.0.0.1"
    assert services["api"]["environment"]["MM_API_BACKEND_FACTORY"] == DEFAULT_BACKEND_FACTORY
    assert (
        services["api"]["environment"]["LASERDATA_CONNECTION_STRING"]
        == "iggy://cloud-user:cloud-pass@managed.example:8090"
    )
    assert (
        services["api"]["environment"]["MUSCLE_MEMORY_FALKORDB_URL"]
        == "rediss://cloud-user:cloud-pass@managed.example:6379"
    )
    assert "@sha256:" in services["falkordb"]["image"]
    assert "@sha256:" in services["laserdata"]["image"]
    assert services["falkordb"]["healthcheck"]["test"]
    assert services["laserdata"]["healthcheck"]["test"]
    assert set(config["volumes"]) == {
        "asset-cache",
        "coordinator-data",
        "falkordb-data",
        "graph-cache",
        "iggy-data",
        "telemetry-data",
    }
