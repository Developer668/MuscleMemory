"""Pinned source, run-plan, artifact, and qualification contracts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PLAYGROUND_REPOSITORY = "https://github.com/google-deepmind/mujoco_playground.git"
PLAYGROUND_TAG = "v0.2.0"
PLAYGROUND_COMMIT = "124a73fa3303f75a62f8fe04d329b829ed0ebdfb"
MENAGERIE_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"

ENVIRONMENT_NAME = "G1JoystickFlatTerrain"
CTRL_DT_SECONDS = 0.01
CONTROLLER_HZ = 100
PHYSICS_HZ = 500
EPISODE_LENGTH = 2000
PHYSICAL_EPISODE_SECONDS = 20.0
TASK_COMMAND_OUTPUTS = 3
GAIT_ACTION_OUTPUTS = 29

PATCH_FILENAME = "mm01_g1_100hz.patch"
PATCH_SHA256 = "dbae77a33cc433ef0e9bf7987faadcd12b418180ea6d79844de88a9d0fb85de0"
JOYSTICK_PATH = Path("mujoco_playground/_src/locomotion/g1/joystick.py")
MENAGERIE_PATH = Path("mujoco_playground/external_deps/mujoco_menagerie")

UNPATCHED_SOURCE_SHA256 = {
    JOYSTICK_PATH.as_posix(): "562118e443266439756270213ee0b52f14d057ab0211d79e1057b9ccc579149f",
    "mujoco_playground/config/locomotion_params.py": (
        "130ddebcddeb5a4403639f3f2f3d08630146e66cbdb2e57f6184dcd08f9a01a8"
    ),
    "learning/train_jax_ppo.py": (
        "c082570f0e72bb9761fb409167c16de881172ead076b353cdc9e636eeb38982b"
    ),
    "uv.lock": "de1ef10180515a4e2baf4116726e49395e01ed85d9a7ae03310fcbe132435c7a",
}
PATCHED_JOYSTICK_SHA256 = "7604c887863c5d2a4a3e7355318139e1a75416e87609085e5e62d5ba74b94da2"


class ContractError(RuntimeError):
    """A pinned input or qualification claim failed closed."""


class RunMode(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    mode: RunMode
    num_timesteps: int
    num_envs: int
    num_eval_envs: int
    num_evals: int
    episode_length: int = EPISODE_LENGTH

    def cli_overrides(self) -> tuple[str, ...]:
        """Return resource-only overrides; rewards and network shape stay upstream."""

        common = (f"--episode_length={self.episode_length}",)
        if self.mode is RunMode.FULL:
            return common
        return (
            *common,
            f"--num_timesteps={self.num_timesteps}",
            f"--num_envs={self.num_envs}",
            f"--num_eval_envs={self.num_eval_envs}",
            f"--num_evals={self.num_evals}",
        )


TRAINING_PLANS = {
    RunMode.SMOKE: TrainingPlan(
        mode=RunMode.SMOKE,
        num_timesteps=1_048_576,
        num_envs=512,
        num_eval_envs=32,
        num_evals=1,
    ),
    RunMode.FULL: TrainingPlan(
        mode=RunMode.FULL,
        num_timesteps=200_000_000,
        num_envs=8192,
        num_eval_envs=128,
        num_evals=20,
    ),
}


@dataclass(frozen=True, slots=True)
class SourceVerification:
    playground_commit: str
    playground_tag_commit: str
    menagerie_commit: str
    patch_sha256: str
    joystick_sha256: str
    patched: bool


@dataclass(frozen=True, slots=True)
class QualificationLimits:
    minimum_forward_progress_metres: float = 1.0
    maximum_forward_cross_track_error_metres: float = 0.25
    maximum_heading_error_degrees: float = 15.0
    maximum_turn_error_degrees: float = 15.0
    maximum_stop_speed_metres_per_second: float = 0.05
    minimum_standstill_duration_seconds: float = 60.0
    maximum_standstill_drift_metres: float = 0.25
    maximum_payload_tray_tilt_degrees: float = 12.0
    maximum_repeat_metric_delta: float = 1e-5


QUALIFICATION_LIMITS = QualificationLimits()


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    controller_hz: int
    ctrl_dt_seconds: float
    physics_hz: int
    task_command_outputs: int
    gait_action_outputs: int
    finite_state: bool
    fall_count: int
    body_collision_count: int
    forward_progress_metres: float
    forward_cross_track_error_metres: float
    forward_heading_error_degrees: float
    left_turn_error_degrees: float
    right_turn_error_degrees: float
    stop_speed_metres_per_second: float
    standstill_duration_seconds: float
    standstill_drift_metres: float
    payload_stop_speed_metres_per_second: float
    maximum_payload_tray_tilt_degrees: float
    payload_package_slipped: bool
    deterministic_repeat_max_metric_delta: float
    robot_checksum: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> QualificationEvidence:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ContractError(
                f"qualification evidence fields differ: missing={missing}, extra={extra}"
            )
        try:
            return cls(**value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ContractError("qualification evidence has invalid value types") from exc


@dataclass(frozen=True, slots=True)
class QualificationResult:
    qualified: bool
    failures: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise ContractError(f"git verification failed for {repository}: {detail}") from exc
    return result.stdout.strip()


def verify_source_checkout(
    checkout: Path,
    patch_path: Path,
    *,
    patched: bool,
) -> SourceVerification:
    """Verify exact commits and bytes before any training process starts."""

    if sha256_file(patch_path) != PATCH_SHA256:
        raise ContractError("controller patch checksum mismatch")
    playground_commit = _git_output(checkout, "rev-parse", "HEAD")
    if playground_commit != PLAYGROUND_COMMIT:
        raise ContractError(
            f"Playground commit mismatch: expected {PLAYGROUND_COMMIT}, got {playground_commit}"
        )
    tag_commit = _git_output(checkout, "rev-parse", f"{PLAYGROUND_TAG}^{{commit}}")
    if tag_commit != PLAYGROUND_COMMIT:
        raise ContractError(f"{PLAYGROUND_TAG} does not resolve to the pinned Playground commit")

    menagerie_root = checkout / MENAGERIE_PATH
    menagerie_commit = _git_output(menagerie_root, "rev-parse", "HEAD")
    if menagerie_commit != MENAGERIE_COMMIT:
        raise ContractError(
            f"Menagerie commit mismatch: expected {MENAGERIE_COMMIT}, got {menagerie_commit}"
        )

    tracked_changes = tuple(
        line
        for line in _git_output(checkout, "diff", "--name-only", "HEAD", "--").splitlines()
        if line
    )
    expected_changes = (JOYSTICK_PATH.as_posix(),) if patched else ()
    if tracked_changes != expected_changes:
        raise ContractError(
            f"unexpected tracked source changes: expected {expected_changes}, got {tracked_changes}"
        )

    expected_hashes = dict(UNPATCHED_SOURCE_SHA256)
    if patched:
        expected_hashes[JOYSTICK_PATH.as_posix()] = PATCHED_JOYSTICK_SHA256
    for relative_path, expected_hash in expected_hashes.items():
        actual_hash = sha256_file(checkout / relative_path)
        if actual_hash != expected_hash:
            raise ContractError(
                f"source checksum mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    joystick_hash = expected_hashes[JOYSTICK_PATH.as_posix()]
    return SourceVerification(
        playground_commit=playground_commit,
        playground_tag_commit=tag_commit,
        menagerie_commit=menagerie_commit,
        patch_sha256=PATCH_SHA256,
        joystick_sha256=joystick_hash,
        patched=patched,
    )


def evaluate_qualification(
    evidence: QualificationEvidence,
    mode: RunMode,
    limits: QualificationLimits = QUALIFICATION_LIMITS,
) -> QualificationResult:
    """Evaluate every controller gate and retain every failure reason."""

    failures: list[str] = []
    numeric_values = (
        evidence.ctrl_dt_seconds,
        evidence.forward_progress_metres,
        evidence.forward_cross_track_error_metres,
        evidence.forward_heading_error_degrees,
        evidence.left_turn_error_degrees,
        evidence.right_turn_error_degrees,
        evidence.stop_speed_metres_per_second,
        evidence.standstill_duration_seconds,
        evidence.standstill_drift_metres,
        evidence.payload_stop_speed_metres_per_second,
        evidence.maximum_payload_tray_tilt_degrees,
        evidence.deterministic_repeat_max_metric_delta,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        failures.append("qualification metrics must all be finite")
    if mode is not RunMode.FULL:
        failures.append("smoke training cannot qualify a controller")
    if evidence.controller_hz != CONTROLLER_HZ or evidence.ctrl_dt_seconds != CTRL_DT_SECONDS:
        failures.append("controller did not execute at exactly 100 Hz")
    if evidence.physics_hz != PHYSICS_HZ:
        failures.append("physics did not execute at exactly 500 Hz")
    if evidence.task_command_outputs != TASK_COMMAND_OUTPUTS:
        failures.append("task command contract changed from three outputs")
    if evidence.gait_action_outputs != GAIT_ACTION_OUTPUTS:
        failures.append("frozen gait action shape changed from 29 joint targets")
    if not evidence.finite_state:
        failures.append("non-finite state occurred")
    if evidence.fall_count != 0:
        failures.append("one or more falls occurred")
    if evidence.body_collision_count != 0:
        failures.append("one or more body collisions occurred")
    if evidence.forward_progress_metres < limits.minimum_forward_progress_metres:
        failures.append("forward progress did not meet its minimum")
    if evidence.forward_cross_track_error_metres > limits.maximum_forward_cross_track_error_metres:
        failures.append("forward cross-track error exceeded its bound")
    if evidence.forward_heading_error_degrees > limits.maximum_heading_error_degrees:
        failures.append("forward heading error exceeded its bound")
    if evidence.left_turn_error_degrees > limits.maximum_turn_error_degrees:
        failures.append("left-turn angular error exceeded its bound")
    if evidence.right_turn_error_degrees > limits.maximum_turn_error_degrees:
        failures.append("right-turn angular error exceeded its bound")
    if evidence.stop_speed_metres_per_second > limits.maximum_stop_speed_metres_per_second:
        failures.append("stop speed exceeded its bound")
    if evidence.standstill_duration_seconds < limits.minimum_standstill_duration_seconds:
        failures.append("standstill trial was shorter than 60 seconds")
    if evidence.standstill_drift_metres > limits.maximum_standstill_drift_metres:
        failures.append("standstill drift exceeded its bound")
    if evidence.payload_stop_speed_metres_per_second > limits.maximum_stop_speed_metres_per_second:
        failures.append("payload stop speed exceeded its bound")
    if (
        evidence.maximum_payload_tray_tilt_degrees
        >= limits.maximum_payload_tray_tilt_degrees
    ):
        failures.append("payload tray tilt reached or exceeded 12 degrees")
    if evidence.payload_package_slipped:
        failures.append("payload package slipped")
    if evidence.deterministic_repeat_max_metric_delta > limits.maximum_repeat_metric_delta:
        failures.append("deterministic repeat tolerance was exceeded")
    if not evidence.robot_checksum:
        failures.append("robot checksum is missing")
    return QualificationResult(qualified=not failures, failures=tuple(failures))


def build_artifact_manifest(
    run_root: Path,
    *,
    mode: RunMode,
    seed: int,
    source: SourceVerification,
    qualification: QualificationResult | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest over every persisted run artifact."""

    manifest_path = run_root / "artifact-manifest.json"
    files = []
    for path in sorted(candidate for candidate in run_root.rglob("*") if candidate.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    qualification = qualification or QualificationResult(
        qualified=False,
        failures=("qualification evidence has not been evaluated",),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "environment": ENVIRONMENT_NAME,
        "mode": mode.value,
        "seed": seed,
        "controller_hz": CONTROLLER_HZ,
        "ctrl_dt_seconds": CTRL_DT_SECONDS,
        "physics_hz": PHYSICS_HZ,
        "episode_length": EPISODE_LENGTH,
        "physical_episode_seconds": PHYSICAL_EPISODE_SECONDS,
        "source": asdict(source),
        "training_plan": asdict(TRAINING_PLANS[mode]),
        "qualification": asdict(qualification),
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_artifact_manifest(run_root: Path, manifest: Mapping[str, object]) -> Path:
    destination = run_root / "artifact-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
