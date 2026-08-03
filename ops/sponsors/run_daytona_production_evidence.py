"""Run production sponsor evidence while safely handing off the Daytona API port."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from ops.deployment.daytona_state import (
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_STATE_DIR,
    repository_revision,
)

DEFAULT_EVIDENCE_ROOT = Path("/data/muscle-memory-sponsor-evidence")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class DaytonaProductionEvidenceError(RuntimeError):
    """The supervised production handoff could not be completed safely."""


@dataclass(frozen=True, slots=True)
class DaytonaEvidenceRun:
    repository: Path
    state_dir: Path
    snapshot_dir: Path
    evidence_root: Path
    expected_revision: str
    output: Path
    approval_request: Path
    port: int
    approval_wait_seconds: float
    maximum_source_episodes: int
    episode_timeout_seconds: float
    provider_timeout_seconds: float


CommandRunner = Callable[[Sequence[str], bool], int]
SignalHandler = Callable[[int, FrameType | None], object] | int | None


def _supervisor_command(config: DaytonaEvidenceRun, *, stop: bool) -> tuple[str, ...]:
    command = (
        sys.executable,
        "-m",
        "ops.deployment.daytona_process",
        "--repository",
        str(config.repository),
        "--state-dir",
        str(config.state_dir),
        "--snapshot-dir",
        str(config.snapshot_dir),
        "--port",
        str(config.port),
    )
    return (*command, "--stop") if stop else command


def _evidence_command(config: DaytonaEvidenceRun) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "ops.sponsors.run_production_evidence",
        "--host",
        "0.0.0.0",
        "--port",
        str(config.port),
        "--expected-revision",
        config.expected_revision,
        "--output",
        str(config.output),
        "--approval-request",
        str(config.approval_request),
        "--approval-wait-seconds",
        str(config.approval_wait_seconds),
        "--maximum-source-episodes",
        str(config.maximum_source_episodes),
        "--episode-timeout-seconds",
        str(config.episode_timeout_seconds),
        "--provider-timeout-seconds",
        str(config.provider_timeout_seconds),
    )


def _run_with_restart(config: DaytonaEvidenceRun, execute: CommandRunner) -> int:
    """Run the handoff and make restart unconditional once stop can begin."""

    primary_error: BaseException | None = None
    primary_status = 1
    try:
        stop_status = execute(_supervisor_command(config, stop=True), False)
        if stop_status != 0:
            primary_status = stop_status
        else:
            primary_status = execute(_evidence_command(config), False)
    except BaseException as exc:
        primary_error = exc
    finally:
        restart_status = execute(_supervisor_command(config, stop=False), True)

    if restart_status != 0:
        detail = (
            f" after {type(primary_error).__name__}"
            if primary_error is not None
            else f" after status {primary_status}"
        )
        raise DaytonaProductionEvidenceError(
            f"Daytona API restart failed with status {restart_status}{detail}"
        )
    if primary_error is not None:
        raise primary_error
    return primary_status


class _SignalRelay:
    """Forward signals to the active child, except during mandatory API restart."""

    def __init__(self, *, repository: Path) -> None:
        self._repository = repository
        self._active: subprocess.Popen[bytes] | None = None
        self._restarting = False
        self.received_signal: int | None = None
        self._previous: dict[int, SignalHandler] = {}

    def install(self) -> None:
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self.received_signal is None:
            self.received_signal = signum
        active = self._active
        if active is None or self._restarting or active.poll() is not None:
            return
        with suppress(ProcessLookupError):
            os.killpg(active.pid, signum)

    def execute(self, command: Sequence[str], restarting: bool) -> int:
        self._restarting = restarting
        try:
            process = subprocess.Popen(
                tuple(command),
                cwd=self._repository,
                env=os.environ.copy(),
                start_new_session=True,
            )
            self._active = process
            return process.wait()
        finally:
            self._active = None
            self._restarting = False


def _new_evidence_path(path: Path, root: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise DaytonaProductionEvidenceError(f"{name} must be below {root}")
    if resolved.exists():
        raise DaytonaProductionEvidenceError(f"immutable {name} already exists: {resolved}")
    return resolved


def _prepare(args: argparse.Namespace) -> DaytonaEvidenceRun:
    repository = args.repository.expanduser().resolve()
    state_dir = args.state_dir.expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    evidence_root = args.evidence_root.expanduser().resolve()
    if not evidence_root.is_relative_to(Path("/data")):
        raise DaytonaProductionEvidenceError(
            "production sponsor evidence must use the persistent /data object volume"
        )
    if _REVISION.fullmatch(args.expected_revision) is None:
        raise DaytonaProductionEvidenceError(
            "expected revision must be a lowercase 40-character commit SHA"
        )
    revision = repository_revision(repository)
    if revision != args.expected_revision:
        raise DaytonaProductionEvidenceError(
            "clean Daytona checkout does not match the expected production revision"
        )
    if not 1 <= args.port <= 65_535:
        raise DaytonaProductionEvidenceError("port must be between 1 and 65535")
    if args.approval_wait_seconds <= 0 or args.episode_timeout_seconds <= 0:
        raise DaytonaProductionEvidenceError("evidence timeouts must be positive")
    if args.provider_timeout_seconds <= 0 or args.maximum_source_episodes < 2:
        raise DaytonaProductionEvidenceError(
            "provider timeout must be positive and at least two source episodes are required"
        )

    output = _new_evidence_path(args.output, evidence_root, "evidence output")
    approval_request = _new_evidence_path(
        args.approval_request,
        evidence_root,
        "approval request",
    )
    if output == approval_request:
        raise DaytonaProductionEvidenceError(
            "evidence output and approval request must use distinct create-once paths"
        )
    reservation = output.with_name(f"{output.name}.reservation")
    if reservation.exists():
        raise DaytonaProductionEvidenceError(
            f"immutable evidence reservation already exists: {reservation}"
        )

    return DaytonaEvidenceRun(
        repository=repository,
        state_dir=state_dir,
        snapshot_dir=snapshot_dir,
        evidence_root=evidence_root,
        expected_revision=args.expected_revision,
        output=output,
        approval_request=approval_request,
        port=args.port,
        approval_wait_seconds=args.approval_wait_seconds,
        maximum_source_episodes=args.maximum_source_episodes,
        episode_timeout_seconds=args.episode_timeout_seconds,
        provider_timeout_seconds=args.provider_timeout_seconds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval-request", type=Path, required=True)
    parser.add_argument("--port", type=int, default=int(os.environ.get("MM_API_PORT", "8000")))
    parser.add_argument("--approval-wait-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-source-episodes", type=int, default=6)
    parser.add_argument("--episode-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--provider-timeout-seconds", type=float, default=600.0)
    return parser


def main() -> int:
    config = _prepare(_parser().parse_args())
    relay = _SignalRelay(repository=config.repository)
    relay.install()
    try:
        status = _run_with_restart(config, relay.execute)
    finally:
        relay.restore()
    if relay.received_signal is not None:
        return 128 + relay.received_signal
    return status


if __name__ == "__main__":
    raise SystemExit(main())
