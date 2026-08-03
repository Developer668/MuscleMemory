from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ops.sponsors.run_daytona_production_evidence import (  # noqa: E402
    DaytonaEvidenceRun,
    DaytonaProductionEvidenceError,
    _evidence_command,
    _run_with_restart,
    _supervisor_command,
)


def _config(tmp_path: Path) -> DaytonaEvidenceRun:
    return DaytonaEvidenceRun(
        repository=tmp_path / "repository",
        state_dir=tmp_path / "state",
        snapshot_dir=tmp_path / "snapshots",
        evidence_root=Path("/data/muscle-memory-sponsor-evidence"),
        expected_revision="a" * 40,
        output=Path("/data/muscle-memory-sponsor-evidence/run.json"),
        approval_request=Path("/data/muscle-memory-sponsor-evidence/approval.json"),
        port=8123,
        approval_wait_seconds=30.0,
        maximum_source_episodes=2,
        episode_timeout_seconds=10.0,
        provider_timeout_seconds=20.0,
    )


@pytest.mark.parametrize("evidence_status", [0, 19])
def test_wrapper_restarts_supervised_api_after_evidence_exit(
    tmp_path: Path,
    evidence_status: int,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[tuple[str, ...], bool]] = []

    def execute(command: Sequence[str], restarting: bool) -> int:
        normalized = tuple(command)
        calls.append((normalized, restarting))
        return evidence_status if "ops.sponsors.run_production_evidence" in normalized else 0

    assert _run_with_restart(config, execute) == evidence_status
    assert calls == [
        (_supervisor_command(config, stop=True), False),
        (_evidence_command(config), False),
        (_supervisor_command(config, stop=False), True),
    ]
    evidence = calls[1][0]
    assert evidence[evidence.index("--port") + 1] == "8123"
    assert evidence[evidence.index("--host") + 1] == "0.0.0.0"


def test_wrapper_restarts_after_stop_failure_without_running_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[tuple[str, ...], bool]] = []

    def execute(command: Sequence[str], restarting: bool) -> int:
        normalized = tuple(command)
        calls.append((normalized, restarting))
        return 7 if len(calls) == 1 else 0

    assert _run_with_restart(config, execute) == 7
    assert calls == [
        (_supervisor_command(config, stop=True), False),
        (_supervisor_command(config, stop=False), True),
    ]


def test_wrapper_restarts_when_evidence_is_interrupted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[tuple[str, ...], bool]] = []

    def execute(command: Sequence[str], restarting: bool) -> int:
        normalized = tuple(command)
        calls.append((normalized, restarting))
        if "ops.sponsors.run_production_evidence" in normalized:
            raise KeyboardInterrupt
        return 0

    with pytest.raises(KeyboardInterrupt):
        _run_with_restart(config, execute)
    assert calls[-1] == (_supervisor_command(config, stop=False), True)


def test_restart_failure_is_never_hidden(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def execute(command: Sequence[str], restarting: bool) -> int:
        del command
        return 23 if restarting else 0

    with pytest.raises(DaytonaProductionEvidenceError, match="restart failed with status 23"):
        _run_with_restart(config, execute)


def test_wrapper_source_has_signal_forwarding_and_mandatory_restart() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ops/sponsors/run_daytona_production_evidence.py"
    ).read_text(encoding="utf-8")

    assert "finally:\n        restart_status = execute" in source
    assert "os.killpg(active.pid, signum)" in source
    assert "if active is None or self._restarting" in source
    assert "start_new_session=True" in source
    assert "repository_revision(repository)" in source
    assert "production sponsor evidence must use the persistent /data object volume" in source
