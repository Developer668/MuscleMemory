from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ops.deployment.daytona_process import stop_process  # noqa: E402
from ops.deployment.daytona_state import (  # noqa: E402
    DaytonaStateError,
    export_snapshot,
    recover_latest,
    reject_fuse_mutable_paths,
    repository_revision,
)
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


def test_daytona_runner_uses_locked_uv_and_persistent_data() -> None:
    runner = (REPOSITORY_ROOT / "ops/deployment/daytona_run.sh").read_text(encoding="utf-8")
    deployment_example = (
        REPOSITORY_ROOT / "config/services/backend-deployment.env.example"
    ).read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev" in runner
    assert "uv run --frozen --no-sync mm-verify-robot" in runner
    assert "exec uv run --frozen --no-sync python -m ops.api.serve" in runner
    assert "MM_DAYTONA_STATE_DIR:-/home/daytona/mm-data" in runner
    assert "MM_DAYTONA_SNAPSHOT_DIR:-/data/muscle-memory-snapshots" in runner
    assert "mutable Daytona state must not use /data FUSE" in runner
    assert "ops.deployment.daytona_state preflight" in runner
    assert "$STATE_DIR/coordinator/coordinator.sqlite3" in runner
    assert "$STATE_DIR/telemetry/laserdata-spool.sqlite3" in runner
    assert "$STATE_DIR/graph/falkordb-events.jsonl" in runner
    assert "MM_API_HOST:-0.0.0.0" in runner
    for name in (
        "MM_HELDOUT_EVALUATION_ARTIFACT",
        "MM_HELDOUT_EVALUATION_ARTIFACT_SHA256",
        "MM_HELDOUT_CANDIDATE_CHECKPOINT",
        "MM_HELDOUT_EVALUATED_AT",
    ):
        assert name in runner
        assert f"{name}=" in deployment_example
    assert "all four MM_HELDOUT_* values are required together" in runner
    assert "MM_STABLE_POLICY_ALIAS=stable" in deployment_example
    assert "docker" not in runner.lower()


def test_daytona_deploy_enforces_cloud_runtime_shape_and_provider_gate() -> None:
    deploy = (REPOSITORY_ROOT / "ops/deployment/daytona_deploy.sh").read_text(encoding="utf-8")

    assert 'payload.get("autoStopInterval") != 0' in deploy
    assert 'payload.get("autoArchiveInterval") not in {-1, 43200}' in deploy
    assert 'payload.get("autoDeleteInterval") != -1' in deploy
    assert 'payload.get("public") is not True' in deploy
    assert 'item.get("mountPath") == "/data"' in deploy
    assert '"${#REVISION}" -ne 40' in deploy
    assert 'reset --hard HEAD' in deploy
    assert "ops.deployment.daytona_process" in deploy
    assert "ops.deployment.daytona_process --stop" in deploy
    assert 'reset --hard "$RESOLVED_REVISION"' in deploy
    assert "clean -ffdx" in deploy
    assert "git status --porcelain --untracked-files=all" in deploy
    assert "git clean -ndx" in deploy
    assert "restore_previous_revision" in deploy
    assert "ROLLBACK_REQUIRED=1" in deploy
    assert "Restored previous Daytona revision" in deploy
    assert deploy.index('fetch --depth 1 origin "$REVISION"') < deploy.index(
        "ROLLBACK_REQUIRED=1"
    )
    assert deploy.index("clean -ffdx") < deploy.index("uv sync --frozen --no-dev")
    assert "nohup" not in deploy
    assert "ops.sponsors.verify_laserdata" in deploy
    assert "ops.sponsors.verify_rocketride" in deploy
    assert "npm ci --no-audit --no-fund" in deploy
    assert "npm run build" in deploy
    assert "PUBLIC_ORIGIN=${DISCOVERY_URL%%\\?*}" in deploy
    for provider in ("LaserData", "FalkorDB"):
        assert f"--require-provider {provider}" in deploy
    for cold_provider in ("guild.ai", "rocketride.ai"):
        assert f"--require-provider {cold_provider}" not in deploy


def test_daytona_deploy_restores_previous_revision_after_prestart_failure(
    tmp_path: Path,
) -> None:
    previous_revision = "a" * 40
    target_revision = "b" * 40
    state_path = tmp_path / "fake-daytona-state.json"
    log_path = tmp_path / "fake-daytona.log"
    state_path.write_text(
        json.dumps(
            {
                "failed_target_install": False,
                "revision": previous_revision,
                "running": True,
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "bin/daytona"
    executable.parent.mkdir()
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

state_path = pathlib.Path(os.environ["FAKE_DAYTONA_STATE"])
log_path = pathlib.Path(os.environ["FAKE_DAYTONA_LOG"])
target_revision = os.environ["FAKE_DAYTONA_TARGET"]
args = sys.argv[1:]

with log_path.open("a", encoding="utf-8") as stream:
    stream.write("CALL " + " ".join(args) + "\\n")

if args[0] == "info":
    print(json.dumps({
        "autoArchiveInterval": 43200,
        "autoDeleteInterval": -1,
        "autoStopInterval": 0,
        "id": "fake-sandbox",
        "public": True,
        "state": "started",
        "volumes": [{"mountPath": "/data"}],
    }))
    raise SystemExit(0)

if args[0] != "exec":
    raise SystemExit(0)

separator = args.index("--", 2)
command = args[separator + 1:]
state = json.loads(state_path.read_text(encoding="utf-8"))

def save() -> None:
    state_path.write_text(json.dumps(state), encoding="utf-8")

def event(value: str) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("EVENT " + value + "\\n")

if command[:2] == ["test", "-d"]:
    raise SystemExit(0)

if command and command[0] == "git":
    if "rev-parse" in command:
        print(target_revision if "FETCH_HEAD" in command else state["revision"])
    elif "checkout" in command and "--detach" in command:
        state["revision"] = command[command.index("--detach") + 1]
        event("checkout:" + state["revision"])
        save()
    elif "reset" in command and "--hard" in command:
        requested = command[command.index("--hard") + 1]
        if requested != "HEAD":
            state["revision"] = requested
            save()
    raise SystemExit(0)

if "ops.deployment.daytona_process" in command and "--stop" in command:
    state["running"] = False
    event("stop:" + state["revision"])
    save()
    raise SystemExit(0)

if "ops.deployment.daytona_process" in command and "--port" in command:
    state["running"] = True
    event("start:" + state["revision"])
    save()
    raise SystemExit(0)

if command[:2] == ["npm", "ci"] and state["revision"] == target_revision:
    if not state["failed_target_install"]:
        state["failed_target_install"] = True
        event("fail-install:" + state["revision"])
        save()
        raise SystemExit(23)

raise SystemExit(0)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_DAYTONA_LOG": str(log_path),
            "FAKE_DAYTONA_STATE": str(state_path),
            "FAKE_DAYTONA_TARGET": target_revision,
            "PATH": f"{executable.parent}{os.pathsep}{environment['PATH']}",
        }
    )

    result = subprocess.run(
        [str(REPOSITORY_ROOT / "ops/deployment/daytona_deploy.sh"), target_revision],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 23, result.stderr
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["revision"] == previous_revision
    assert state["running"] is True
    events = [
        line.removeprefix("EVENT ")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("EVENT ")
    ]
    assert events == [
        f"stop:{previous_revision}",
        f"checkout:{target_revision}",
        f"fail-install:{target_revision}",
        f"checkout:{previous_revision}",
        f"start:{previous_revision}",
    ]


def test_daytona_process_supervisor_guards_against_pid_reuse() -> None:
    supervisor = (REPOSITORY_ROOT / "ops/deployment/daytona_process.py").read_text(encoding="utf-8")
    start_process = supervisor.split("def start_process", maxsplit=1)[1].split(
        "def _parser", maxsplit=1
    )[0]

    assert 'Path(f"/proc/{pid}/stat")' in supervisor
    assert 'payload["start_ticks"]' in supervisor
    assert "start_new_session=True" in supervisor
    assert "os.killpg(process_group_id, signal.SIGTERM)" in supervisor
    assert "os.killpg(process_group_id, signal.SIGKILL)" in supervisor
    assert '"MM_DAYTONA_SKIP_PREPARE": "1"' in supervisor
    assert '"MM_DAYTONA_STATE_DIR": str(state_dir)' in supervisor
    assert '"MM_DAYTONA_SNAPSHOT_DIR": str(snapshot_dir)' in supervisor
    assert "recover_latest(state_dir, snapshot_dir)" in supervisor
    assert "export_snapshot(state_dir, snapshot_dir, revision=revision)" in supervisor
    assert start_process.index("revision = repository_revision(repository)") < start_process.index(
        "stop_process(pid_path)"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Daytona supervisor targets Linux")
def test_daytona_process_supervisor_terminates_the_entire_session(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(60)"
            ),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text())
        start_ticks = _process_start_ticks_for_test(process.pid)
        pid_path = tmp_path / "api.pid.json"
        pid_path.write_text(
            json.dumps({"pid": process.pid, "start_ticks": start_ticks}),
            encoding="utf-8",
        )

        stop_process(pid_path, timeout=2.0)
        process.wait(timeout=5)

        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert not pid_path.exists()
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def _process_start_ticks_for_test(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return int(fields[19])


def test_daytona_state_snapshot_is_create_once_and_recovers_wal_database(
    tmp_path: Path,
) -> None:
    state = tmp_path / "sandbox-state"
    objects = tmp_path / "object-volume"
    database = state / "coordinator/coordinator.sqlite3"
    graph = state / "graph/falkordb-events.jsonl"
    approval = state / "assets/approvals/approval.json"
    database.parent.mkdir(parents=True)
    graph.parent.mkdir(parents=True)
    approval.parent.mkdir(parents=True)
    objects.mkdir()
    graph.write_text('{"event":"one"}\n', encoding="utf-8")
    approval.write_text('{"approved":true}\n', encoding="utf-8")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
    connection.execute("INSERT INTO evidence VALUES ('durable')")
    connection.commit()

    snapshot_id = export_snapshot(
        state,
        objects,
        revision="a" * 40,
        snapshot_id="20260803T120000000000Z-aaaaaaaaaaaa-release",
    )
    connection.close()

    assert snapshot_id is not None
    snapshot = objects / snapshot_id
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "coordinator/coordinator.sqlite3" in paths
    assert "graph/falkordb-events.jsonl" in paths
    assert "assets/approvals/approval.json" in paths
    assert not any(path.endswith(("-wal", "-shm", ".tmp")) for path in paths)
    with pytest.raises(DaytonaStateError, match="already exists"):
        export_snapshot(
            state,
            objects,
            revision="a" * 40,
            snapshot_id=snapshot_id,
        )

    for managed in ("coordinator", "telemetry", "graph", "assets"):
        shutil.rmtree(state / managed, ignore_errors=True)
    (state / "logs").mkdir(parents=True)
    (state / "logs/api.log").write_text("preserved\n", encoding="utf-8")

    assert recover_latest(state, objects) == snapshot_id
    with sqlite3.connect(database) as recovered:
        assert recovered.execute("SELECT value FROM evidence").fetchone() == ("durable",)
    assert graph.read_text(encoding="utf-8") == '{"event":"one"}\n'
    assert (state / "logs/api.log").read_text(encoding="utf-8") == "preserved\n"
    assert recover_latest(state, objects) is None


def test_daytona_rejects_mutable_state_on_object_fuse() -> None:
    with pytest.raises(DaytonaStateError, match="must not use /data FUSE"):
        reject_fuse_mutable_paths((Path("/data/coordinator.sqlite3"),))


def test_daytona_snapshot_provenance_rejects_dirty_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "snapshot@example.test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Snapshot Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("verified\n", encoding="utf-8")
    (repository / ".gitignore").write_text("ignored-build/\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "--message", "verified"],
        cwd=repository,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (repository / "ignored-build").mkdir()
    (repository / "ignored-build/output").write_text("allowed\n", encoding="utf-8")
    assert repository_revision(repository) == revision

    tracked.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(DaytonaStateError, match="dirty"):
        repository_revision(repository)

    tracked.write_text("verified\n", encoding="utf-8")
    (repository / "untracked.txt").write_text("drifted\n", encoding="utf-8")
    with pytest.raises(DaytonaStateError, match="dirty"):
        repository_revision(repository)


def test_daytona_runner_requires_built_operator_console() -> None:
    runner = (REPOSITORY_ROOT / "ops/deployment/daytona_run.sh").read_text(
        encoding="utf-8"
    )

    assert 'frontend/dist/index.html' in runner
    assert 'production frontend build is missing' in runner


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
        "training-data",
    }
