"""Own the long-running API process inside a Daytona sandbox."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from ops.deployment.daytona_state import (
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_STATE_DIR,
    export_snapshot,
    preflight,
    recover_latest,
    repository_revision,
)

STOP_TIMEOUT_SECONDS = 30.0


def _process_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks so a stale PID file cannot kill a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", maxsplit=1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _read_identity(pid_path: Path) -> tuple[int, int] | None:
    try:
        payload = json.loads(pid_path.read_text(encoding="utf-8"))
        return int(payload["pid"]), int(payload["start_ticks"])
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _is_same_process(pid: int, start_ticks: int) -> bool:
    return _process_start_ticks(pid) == start_ticks


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process(pid_path: Path, *, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
    identity = _read_identity(pid_path)
    if identity is None:
        pid_path.unlink(missing_ok=True)
        return

    pid, start_ticks = identity
    if not _is_same_process(pid, start_ticks):
        pid_path.unlink(missing_ok=True)
        return

    try:
        process_group_id = os.getpgid(pid)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return
    if process_group_id != pid:
        raise RuntimeError("Daytona API process is not its recorded session leader")

    os.killpg(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            pid_path.unlink(missing_ok=True)
            return
        time.sleep(0.25)

    if _process_group_exists(process_group_id):
        os.killpg(process_group_id, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)


def start_process(
    repository: Path,
    state_dir: Path,
    snapshot_dir: Path,
    *,
    port: int,
) -> int:
    preflight(state_dir, snapshot_dir)

    runner = repository / "ops" / "deployment" / "daytona_run.sh"
    if not runner.is_file():
        raise RuntimeError(f"Daytona runner is missing: {runner}")

    logs_dir = state_dir / "logs"
    run_dir = state_dir / "run"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_path = run_dir / "api.pid.json"
    revision = repository_revision(repository)
    stop_process(pid_path)
    recover_latest(state_dir, snapshot_dir)
    export_snapshot(state_dir, snapshot_dir, revision=revision)

    environment = os.environ.copy()
    environment.update(
        {
            "MM_API_PORT": str(port),
            "MM_DAYTONA_STATE_DIR": str(state_dir),
            "MM_DAYTONA_SNAPSHOT_DIR": str(snapshot_dir),
            "MM_DAYTONA_SKIP_PREPARE": "1",
        }
    )
    log_path = logs_dir / "api.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [str(runner)],
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )

    time.sleep(1.0)
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            f"Daytona API exited during startup with status {return_code}; see {log_path}"
        )

    start_ticks = _process_start_ticks(process.pid)
    if start_ticks is None:
        process.terminate()
        raise RuntimeError("Daytona API process identity could not be verified")

    temporary_path = pid_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"pid": process.pid, "start_ticks": start_ticks}) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(pid_path)
    return process.pid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="stop the verified process identity without starting a replacement",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = args.repository.resolve()
    state_dir = args.state_dir.resolve()
    snapshot_dir = args.snapshot_dir.resolve()
    if args.stop:
        if state_dir == Path("/data") or state_dir.is_relative_to(Path("/data")):
            raise SystemExit("mutable Daytona state must not use /data FUSE")
        state_dir.mkdir(parents=True, exist_ok=True)
        preflight(state_dir, snapshot_dir)
        revision = repository_revision(repository)
        stop_process(state_dir / "run" / "api.pid.json")
        export_snapshot(
            state_dir,
            snapshot_dir,
            revision=revision,
        )
        print("Daytona API stopped")
        return 0
    preflight(state_dir, snapshot_dir)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    pid = start_process(
        repository,
        state_dir,
        snapshot_dir,
        port=args.port,
    )
    print(f"Daytona API started with PID {pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
