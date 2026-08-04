"""Bounded local execution for immutable high-level task-policy training jobs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock, Thread

from muscle_memory.paths import EXPERT_DATASET_V1, REPOSITORY_ROOT

MINIMUM_EPOCHS = 1
MAXIMUM_EPOCHS = 200
MINIMUM_SEED = 0
MAXIMUM_SEED = (2**31) - 1
DEFAULT_JOB_TIMEOUT_SECONDS = 15 * 60
TRAINING_ROOT_ENV = "MM_TASK_POLICY_TRAINING_ROOT"
JOB_MANIFEST_NAME = "job.json"
FIXED_EXPERT_DATASET_SHA256 = (
    "d3c7aa08ae467f0bf17eca13c116037ab2049e0da7d1ff95b2deb489252e20ef"
)


class TaskPolicyTrainingConflictError(RuntimeError):
    """A second task-policy training job was requested while one is active."""


class TaskPolicyTrainingNotFoundError(LookupError):
    """A task-policy training job identifier is unknown to this process."""


class TaskPolicyTrainingState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TaskPolicyTrainingMetrics:
    training_episode_count: int
    validation_episode_count: int
    training_sample_count: int
    validation_sample_count: int
    best_epoch: int
    training_command_accuracy: float
    validation_command_accuracy: float
    validation_loss: float
    validation_forward_mae_mps: float
    validation_turning_mae_rad_s: float
    validation_stop_mae: float


@dataclass(frozen=True, slots=True)
class TaskPolicyTrainingJob:
    job_id: str
    policy_id: str
    state: TaskPolicyTrainingState
    epochs: int
    seed: int
    dataset_sha256: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    checkpoint_sha256: str | None = None
    evidence_sha256: str | None = None
    metrics: TaskPolicyTrainingMetrics | None = None
    error_type: str | None = None

    def as_json_value(self) -> dict[str, object]:
        value = asdict(self)
        value.update(
            {
                "state": self.state.value,
                "training_data_split": "training",
                "robot_component": "high_level_task_policy",
                "promotion_status": "not_evaluated",
            }
        )
        return value


TrainingFunction = Callable[..., None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subprocess_training(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    evidence_path: Path,
    epochs: int,
    seed: int,
    policy_id: str,
    timeout_seconds: float,
) -> None:
    command = (
        sys.executable,
        "-m",
        "ops.policy.train_behavior_clone",
        "--dataset",
        str(dataset_path),
        "--output",
        str(checkpoint_path),
        "--evidence",
        str(evidence_path),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--policy-id",
        policy_id,
    )
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError("task-policy training process failed")


class TaskPolicyTrainingManager:
    """Run at most one fixed-dataset behavior-cloning job at a time."""

    def __init__(
        self,
        *,
        output_root: Path,
        training_function: TrainingFunction = _subprocess_training,
        timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("task-policy training timeout must be positive")
        dataset_path = EXPERT_DATASET_V1.resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError("the fixed task-policy training dataset is unavailable")
        root = output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        self._dataset_path = dataset_path
        self._dataset_sha256 = _sha256_file(dataset_path)
        if self._dataset_sha256 != FIXED_EXPERT_DATASET_SHA256:
            raise RuntimeError("the fixed task-policy training dataset checksum changed")
        self._output_root = root
        self._training_function = training_function
        self._timeout_seconds = timeout_seconds
        self._jobs: dict[str, TaskPolicyTrainingJob] = {}
        self._threads: dict[str, Thread] = {}
        self._active_job_id: str | None = None
        self._lock = RLock()
        self._closed = False
        self._restore_jobs()

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> TaskPolicyTrainingManager:
        values = os.environ if environ is None else environ
        configured = values.get(TRAINING_ROOT_ENV, "").strip()
        root = (
            Path(configured)
            if configured
            else REPOSITORY_ROOT / "artifacts" / "policy" / "training-jobs"
        )
        return cls(output_root=root)

    def start(self, *, epochs: int, seed: int) -> TaskPolicyTrainingJob:
        if not MINIMUM_EPOCHS <= epochs <= MAXIMUM_EPOCHS:
            raise ValueError(
                f"epochs must be between {MINIMUM_EPOCHS} and {MAXIMUM_EPOCHS}"
            )
        if not MINIMUM_SEED <= seed <= MAXIMUM_SEED:
            raise ValueError(f"seed must be between {MINIMUM_SEED} and {MAXIMUM_SEED}")
        with self._lock:
            if self._closed:
                raise RuntimeError("task-policy training manager is closed")
            if self._active_job_id is not None:
                active = self._jobs[self._active_job_id]
                if active.state in {
                    TaskPolicyTrainingState.QUEUED,
                    TaskPolicyTrainingState.RUNNING,
                }:
                    raise TaskPolicyTrainingConflictError(
                        "one task-policy training job is already active"
                    )
            identifier = uuid.uuid4().hex
            job_id = f"task-policy-{identifier}"
            policy_id = f"local-candidate-{identifier}"
            job = TaskPolicyTrainingJob(
                job_id=job_id,
                policy_id=policy_id,
                state=TaskPolicyTrainingState.QUEUED,
                epochs=epochs,
                seed=seed,
                dataset_sha256=self._dataset_sha256,
                created_at=datetime.now(UTC),
            )
            job_root = self._output_root / job_id
            job_root.mkdir(mode=0o700, exist_ok=False)
            self._jobs[job_id] = job
            self._active_job_id = job_id
            try:
                self._persist_job(job)
            except BaseException:
                self._jobs.pop(job_id, None)
                self._active_job_id = None
                job_root.rmdir()
                raise
            thread = Thread(
                target=self._run,
                args=(job_id, job_root),
                name=job_id,
                daemon=False,
            )
            self._threads[job_id] = thread
            thread.start()
            return job

    def get(self, job_id: str) -> TaskPolicyTrainingJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise TaskPolicyTrainingNotFoundError(job_id) from exc

    def list(self) -> tuple[TaskPolicyTrainingJob, ...]:
        with self._lock:
            return tuple(
                sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            )

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            threads = tuple(self._threads.values())
        for thread in threads:
            thread.join()

    def _restore_jobs(self) -> None:
        """Recover job history and quarantine work interrupted by a process restart."""

        for job_root in sorted(self._output_root.glob("task-policy-*")):
            manifest = job_root / JOB_MANIFEST_NAME
            if not job_root.is_dir() or not manifest.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                job = self._job_from_manifest(payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"training job manifest is invalid: {manifest}") from exc
            if job.job_id != job_root.name:
                raise RuntimeError("training job manifest identity does not match its directory")
            if job.state in {
                TaskPolicyTrainingState.QUEUED,
                TaskPolicyTrainingState.RUNNING,
            }:
                job = replace(
                    job,
                    state=TaskPolicyTrainingState.FAILED,
                    completed_at=datetime.now(UTC),
                    error_type="process_restart",
                )
                self._jobs[job.job_id] = job
                self._persist_job(job)
            else:
                self._jobs[job.job_id] = job

    def _persist_job(self, job: TaskPolicyTrainingJob) -> None:
        job_root = self._output_root / job.job_id
        manifest = job_root / JOB_MANIFEST_NAME
        temporary = job_root / f".{JOB_MANIFEST_NAME}.tmp"
        payload = self._manifest_value(job)
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=True, allow_nan=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _manifest_value(job: TaskPolicyTrainingJob) -> dict[str, object]:
        payload = job.as_json_value()
        for key in ("created_at", "started_at", "completed_at"):
            value = payload[key]
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
        if job.metrics is not None:
            payload["metrics"] = asdict(job.metrics)
        return payload

    @staticmethod
    def _job_from_manifest(payload: object) -> TaskPolicyTrainingJob:
        if not isinstance(payload, dict):
            raise ValueError("training job manifest must be an object")
        metrics_payload = payload.get("metrics")
        metrics = None
        if metrics_payload is not None:
            if not isinstance(metrics_payload, dict):
                raise ValueError("training job metrics must be an object")
            metrics = TaskPolicyTrainingMetrics(**metrics_payload)
        return TaskPolicyTrainingJob(
            job_id=str(payload["job_id"]),
            policy_id=str(payload["policy_id"]),
            state=TaskPolicyTrainingState(str(payload["state"])),
            epochs=int(payload["epochs"]),
            seed=int(payload["seed"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            started_at=(
                None
                if payload.get("started_at") is None
                else datetime.fromisoformat(str(payload["started_at"]))
            ),
            completed_at=(
                None
                if payload.get("completed_at") is None
                else datetime.fromisoformat(str(payload["completed_at"]))
            ),
            checkpoint_sha256=(
                None
                if payload.get("checkpoint_sha256") is None
                else str(payload["checkpoint_sha256"])
            ),
            evidence_sha256=(
                None
                if payload.get("evidence_sha256") is None
                else str(payload["evidence_sha256"])
            ),
            metrics=metrics,
            error_type=(None if payload.get("error_type") is None else str(payload["error_type"])),
        )

    def _run(self, job_id: str, job_root: Path) -> None:
        checkpoint_path = job_root / "checkpoint.npz"
        evidence_path = job_root / "training.json"
        with self._lock:
            job = replace(
                self._jobs[job_id],
                state=TaskPolicyTrainingState.RUNNING,
                started_at=datetime.now(UTC),
            )
            self._jobs[job_id] = job
            self._persist_job(job)
        try:
            self._training_function(
                dataset_path=self._dataset_path,
                checkpoint_path=checkpoint_path,
                evidence_path=evidence_path,
                epochs=job.epochs,
                seed=job.seed,
                policy_id=job.policy_id,
                timeout_seconds=self._timeout_seconds,
            )
            metrics, checkpoint_sha256, evidence_sha256 = self._verify_outputs(
                job,
                checkpoint_path,
                evidence_path,
            )
        except Exception as exc:
            with self._lock:
                self._jobs[job_id] = replace(
                    self._jobs[job_id],
                    state=TaskPolicyTrainingState.FAILED,
                    completed_at=datetime.now(UTC),
                    error_type=type(exc).__name__,
                )
                self._persist_job(self._jobs[job_id])
            return
        with self._lock:
            self._jobs[job_id] = replace(
                self._jobs[job_id],
                state=TaskPolicyTrainingState.COMPLETED,
                completed_at=datetime.now(UTC),
                checkpoint_sha256=checkpoint_sha256,
                evidence_sha256=evidence_sha256,
                metrics=metrics,
            )
            self._persist_job(self._jobs[job_id])

    def _verify_outputs(
        self,
        job: TaskPolicyTrainingJob,
        checkpoint_path: Path,
        evidence_path: Path,
    ) -> tuple[TaskPolicyTrainingMetrics, str, str]:
        if not checkpoint_path.is_file() or not evidence_path.is_file():
            raise RuntimeError("task-policy training did not produce both immutable artifacts")
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("task-policy training evidence is invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("task-policy training evidence must be an object")
        if (
            payload.get("policy_id") != job.policy_id
            or payload.get("policy_sha256") != checkpoint_sha256
            or payload.get("dataset_sha256") != self._dataset_sha256
        ):
            raise RuntimeError("task-policy training evidence is detached from its inputs")
        metric_names = tuple(TaskPolicyTrainingMetrics.__dataclass_fields__)
        try:
            metrics = TaskPolicyTrainingMetrics(
                **{name: payload[name] for name in metric_names}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("task-policy training metrics are invalid") from exc
        return metrics, checkpoint_sha256, _sha256_file(evidence_path)


__all__ = [
    "JOB_MANIFEST_NAME",
    "MAXIMUM_EPOCHS",
    "MAXIMUM_SEED",
    "MINIMUM_EPOCHS",
    "MINIMUM_SEED",
    "TaskPolicyTrainingConflictError",
    "TaskPolicyTrainingJob",
    "TaskPolicyTrainingManager",
    "TaskPolicyTrainingMetrics",
    "TaskPolicyTrainingNotFoundError",
    "TaskPolicyTrainingState",
]
