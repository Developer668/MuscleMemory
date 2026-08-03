from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import pytest

from muscle_memory.training.jobs import (
    FIXED_EXPERT_DATASET_SHA256,
    TaskPolicyTrainingConflictError,
    TaskPolicyTrainingManager,
    TaskPolicyTrainingState,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_training_outputs(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    evidence_path: Path,
    epochs: int,
    seed: int,
    policy_id: str,
    timeout_seconds: float,
) -> None:
    del seed, timeout_seconds
    checkpoint_path.write_bytes(f"checkpoint:{policy_id}".encode())
    payload = {
        "policy_id": policy_id,
        "policy_sha256": _sha256(checkpoint_path),
        "dataset_sha256": _sha256(dataset_path),
        "training_episode_count": 51,
        "validation_episode_count": 13,
        "training_sample_count": 19_565,
        "validation_sample_count": 3_348,
        "best_epoch": epochs,
        "training_command_accuracy": 0.8,
        "validation_command_accuracy": 0.7,
        "validation_loss": 0.1,
        "validation_forward_mae_mps": 0.02,
        "validation_turning_mae_rad_s": 0.03,
        "validation_stop_mae": 0.04,
    }
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")


def _await_terminal(
    manager: TaskPolicyTrainingManager,
    job_id: str,
) -> TaskPolicyTrainingState:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = manager.get(job_id).state
        if state in {TaskPolicyTrainingState.COMPLETED, TaskPolicyTrainingState.FAILED}:
            return state
        time.sleep(0.01)
    raise AssertionError("training job did not reach a terminal state")


def test_training_manager_uses_fixed_dataset_and_creates_immutable_unique_jobs(
    tmp_path: Path,
) -> None:
    manager = TaskPolicyTrainingManager(
        output_root=tmp_path,
        training_function=_write_training_outputs,
    )

    first = manager.start(epochs=2, seed=17)
    assert _await_terminal(manager, first.job_id) is TaskPolicyTrainingState.COMPLETED
    first_complete = manager.get(first.job_id)
    second = manager.start(epochs=3, seed=18)
    assert _await_terminal(manager, second.job_id) is TaskPolicyTrainingState.COMPLETED
    manager.shutdown()

    assert first.job_id != second.job_id
    assert first.policy_id != second.policy_id
    assert first_complete.dataset_sha256 == FIXED_EXPERT_DATASET_SHA256
    assert first_complete.checkpoint_sha256 is not None
    assert first_complete.evidence_sha256 is not None
    assert first_complete.metrics is not None
    assert first_complete.as_json_value()["promotion_status"] == "not_evaluated"
    assert first_complete.as_json_value()["robot_component"] == "high_level_task_policy"
    assert len(tuple(tmp_path.glob("task-policy-*/checkpoint.npz"))) == 2
    assert len(tuple(tmp_path.glob("task-policy-*/training.json"))) == 2


def test_training_manager_rejects_overlap_and_bounds_inputs(tmp_path: Path) -> None:
    started = Event()
    release = Event()

    def blocking_training(**kwargs: object) -> None:
        started.set()
        assert release.wait(timeout=5)
        _write_training_outputs(**kwargs)  # type: ignore[arg-type]

    manager = TaskPolicyTrainingManager(
        output_root=tmp_path,
        training_function=blocking_training,
    )
    active = manager.start(epochs=1, seed=0)
    assert started.wait(timeout=2)

    with pytest.raises(TaskPolicyTrainingConflictError):
        manager.start(epochs=1, seed=1)
    with pytest.raises(ValueError, match="epochs"):
        manager.start(epochs=0, seed=1)
    with pytest.raises(ValueError, match="seed"):
        manager.start(epochs=1, seed=2**31)

    release.set()
    assert _await_terminal(manager, active.job_id) is TaskPolicyTrainingState.COMPLETED
    manager.shutdown()


def test_training_job_api_module_does_not_import_heldout_or_controller_code() -> None:
    audit = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import muscle_memory.training.jobs; "
            "print(any(name.startswith('muscle_memory.evaluation.heldout') "
            "or name.startswith('ops.controller') for name in sys.modules)); "
            "print('muscle_memory.training.expert' in sys.modules)",
        ),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert audit.returncode == 0, audit.stderr
    assert audit.stdout.splitlines() == ["False", "False"]
