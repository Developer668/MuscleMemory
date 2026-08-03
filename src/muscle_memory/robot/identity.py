"""Fail-closed identity verification for candidate and qualified robot bundles."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from muscle_memory.paths import (
    G1_POLICY_ONNX,
    G1_SCENE_XML,
    MM01_CONTROLLER_ONNX,
    MM01_MANIFEST,
    MM01_ONNX_PARITY_EVIDENCE,
    MM01_QUALIFICATION_EVIDENCE,
    MM01_QUALIFICATION_TRIALS,
    MM01_TRAINING_CONTRACT,
    REPOSITORY_ROOT,
    ROBOT_MANIFEST,
    THIRD_PARTY_ROOT,
)

PLAYGROUND_COMMIT = "124a73fa3303f75a62f8fe04d329b829ed0ebdfb"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"
EXPECTED_CONTROLLER_SHA256 = "db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b"
PHYSICS_HZ = 500
CONTROLLER_SUPERVISOR_HZ = 100
CANDIDATE_CONTROLLER_INFERENCE_HZ = 50
CONTROLLER_INFERENCE_HZ = 100
TASK_POLICY_HZ = 10
MM01_ROBOT_ID = "MM-01"
MM01_CONTROLLER_ID = "gait-controller-v1"
MM01_SENSOR_PROFILE = REPOSITORY_ROOT / "config" / "robot" / "mm01-sensor-profile.json"
MM01_CONTROLLER_SOURCE = REPOSITORY_ROOT / "src" / "muscle_memory" / "simulation" / "controller.py"
MM01_RUNTIME_SOURCE = REPOSITORY_ROOT / "src" / "muscle_memory" / "simulation" / "runtime.py"
MM01_COMMAND_SOURCE = REPOSITORY_ROOT / "src" / "muscle_memory" / "robot" / "command.py"
MM01_SENSOR_SOURCE = REPOSITORY_ROOT / "src" / "muscle_memory" / "simulation" / "sensors.py"
MM01_IDENTITY_SOURCE = Path(__file__).resolve()
MM01_QUALIFICATION_PROGRAM = REPOSITORY_ROOT / "ops" / "controller" / "native_qualify.py"
MM01_QUALIFICATION_CONTRACT = REPOSITORY_ROOT / "ops" / "controller" / "contract.py"
MM01_TRAINING_PATCH = REPOSITORY_ROOT / "ops" / "controller" / "mm01_g1_100hz.patch"
MM01_TRAINING_PATCH_SHA256 = "dbae77a33cc433ef0e9bf7987faadcd12b418180ea6d79844de88a9d0fb85de0"
MM01_PATCHED_JOYSTICK_SHA256 = (
    "7604c887863c5d2a4a3e7355318139e1a75416e87609085e5e62d5ba74b94da2"
)


class CandidateBundleError(RuntimeError):
    """Raised when candidate bytes or immutable metadata do not match."""


class FrozenFile(BaseModel):
    """One byte-for-byte frozen candidate input."""

    model_config = ConfigDict(frozen=True)

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourcePins(BaseModel):
    """Upstream identities for the vendored model and gait policy."""

    model_config = ConfigDict(frozen=True)

    mujoco_playground_tag: str
    mujoco_playground_commit: str
    mujoco_menagerie_commit: str
    mujoco_version: str
    onnxruntime_version: str


class CandidateRates(BaseModel):
    """Actual runtime rates, including the known qualification mismatch."""

    model_config = ConfigDict(frozen=True)

    physics_hz: int
    controller_supervisor_hz: int
    controller_inference_hz: int
    task_policy_hz: int


class CandidateQualification(BaseModel):
    """Qualification state that cannot be inferred from bundle validity."""

    model_config = ConfigDict(frozen=True)

    qualified: bool
    blockers: tuple[str, ...]


class CandidateManifest(BaseModel):
    """Structured identity for candidate bytes and execution metadata."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    candidate_id: str
    source: SourcePins
    rates: CandidateRates
    qualification: CandidateQualification
    files: tuple[FrozenFile, ...]
    aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BundleVerification(BaseModel):
    """Successful verification result suitable for CLI JSON output."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    valid: bool
    qualified: bool
    file_count: int
    aggregate_sha256: str
    blockers: tuple[str, ...]


class ControllerParityEvidence(BaseModel):
    """Measured JAX-to-ONNX output parity for the selected checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    checkpoint: str
    sample_count: int = Field(gt=0)
    maximum_absolute_delta: float = Field(ge=0.0)
    limit: float = Field(gt=0.0)
    passed: bool


class QualifiedControllerEvidence(BaseModel):
    """Native MuJoCo measurements required to freeze a controller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    robot_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    controller_onnx_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_program_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_trials_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QualifiedRobotManifest(BaseModel):
    """Immutable identity for the fully qualified MM-01 runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    robot_id: str
    controller_id: str
    qualified: bool
    source: SourcePins
    rates: CandidateRates
    training_run_id: str
    training_seed: int
    selected_checkpoint: str
    final_checkpoint: str
    later_rejected_checkpoints: tuple[str, ...]
    base_candidate_aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robot_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[FrozenFile, ...]
    aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RobotBundleVerification(BaseModel):
    """Successful fail-closed verification of the qualified MM-01 bundle."""

    model_config = ConfigDict(frozen=True)

    robot_id: str
    controller_id: str
    valid: bool
    qualified: bool
    file_count: int
    robot_checksum: str
    manifest_aggregate_sha256: str
    training_run_id: str
    selected_checkpoint: str
    blockers: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise CandidateBundleError(f"candidate path escapes repository: {path}") from error


def _xml_dependencies(xml_path: Path) -> set[Path]:
    dependencies: set[Path] = {xml_path.resolve()}
    root = ET.parse(xml_path).getroot()
    for element in root.iter():
        dependency = None
        if element.tag == "include" or element.tag == "mesh":
            dependency = element.get("file")
        if dependency is None:
            continue
        resolved = (xml_path.parent / dependency).resolve()
        dependencies.add(resolved)
        if resolved.suffix.lower() == ".xml":
            dependencies.update(_xml_dependencies(resolved))
    return dependencies


def discover_candidate_files(repository_root: Path = REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Discover runtime inputs from XML includes and mesh references."""
    relative_third_party = THIRD_PARTY_ROOT.relative_to(REPOSITORY_ROOT)
    third_party_root = repository_root / relative_third_party
    relative_scene = G1_SCENE_XML.relative_to(REPOSITORY_ROOT)
    relative_policy = G1_POLICY_ONNX.relative_to(REPOSITORY_ROOT)
    scene_path = repository_root / relative_scene
    files = _xml_dependencies(scene_path)
    files.update(
        {
            repository_root / relative_policy,
            third_party_root / "LICENSE",
            third_party_root / "mujoco_menagerie" / "unitree_g1" / "LICENSE",
            third_party_root
            / "mujoco_playground"
            / "experimental"
            / "sim2sim"
            / "play_g1_joystick.py",
        }
    )
    return tuple(sorted(files))


def _aggregate_sha256(manifest: CandidateManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"aggregate_sha256"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def build_candidate_manifest(repository_root: Path = REPOSITORY_ROOT) -> CandidateManifest:
    """Mechanically build the candidate manifest from vendored bytes."""
    files = tuple(
        FrozenFile(
            path=_repository_path(path, repository_root),
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
        )
        for path in discover_candidate_files(repository_root)
    )
    manifest = CandidateManifest(
        schema_version=1,
        candidate_id="mm01-playground-g1-candidate",
        source=SourcePins(
            mujoco_playground_tag="v0.2.0",
            mujoco_playground_commit=PLAYGROUND_COMMIT,
            mujoco_menagerie_commit=MENAGERIE_COMMIT,
            mujoco_version="3.6.0",
            onnxruntime_version="1.28.0",
        ),
        rates=CandidateRates(
            physics_hz=PHYSICS_HZ,
            controller_supervisor_hz=CONTROLLER_SUPERVISOR_HZ,
            controller_inference_hz=CANDIDATE_CONTROLLER_INFERENCE_HZ,
            task_policy_hz=TASK_POLICY_HZ,
        ),
        qualification=CandidateQualification(
            qualified=False,
            blockers=(
                "candidate ONNX gait inference is 50 Hz, not the required 100 Hz",
                "zero-command stopping has not passed qualification",
            ),
        ),
        files=files,
        aggregate_sha256="0" * 64,
    )
    return manifest.model_copy(update={"aggregate_sha256": _aggregate_sha256(manifest)})


def write_candidate_manifest(
    destination: Path = ROBOT_MANIFEST,
    repository_root: Path = REPOSITORY_ROOT,
) -> CandidateManifest:
    """Write a deterministically formatted manifest generated from actual files."""
    manifest = build_candidate_manifest(repository_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def _validate_immutable_metadata(manifest: CandidateManifest) -> None:
    expected = {
        "schema_version": 1,
        "playground_tag": "v0.2.0",
        "playground_commit": PLAYGROUND_COMMIT,
        "menagerie_commit": MENAGERIE_COMMIT,
        "mujoco_version": "3.6.0",
        "onnxruntime_version": "1.28.0",
        "physics_hz": PHYSICS_HZ,
        "controller_supervisor_hz": CONTROLLER_SUPERVISOR_HZ,
        "controller_inference_hz": CANDIDATE_CONTROLLER_INFERENCE_HZ,
        "task_policy_hz": TASK_POLICY_HZ,
    }
    actual = {
        "schema_version": manifest.schema_version,
        "playground_tag": manifest.source.mujoco_playground_tag,
        "playground_commit": manifest.source.mujoco_playground_commit,
        "menagerie_commit": manifest.source.mujoco_menagerie_commit,
        "mujoco_version": manifest.source.mujoco_version,
        "onnxruntime_version": manifest.source.onnxruntime_version,
        "physics_hz": manifest.rates.physics_hz,
        "controller_supervisor_hz": manifest.rates.controller_supervisor_hz,
        "controller_inference_hz": manifest.rates.controller_inference_hz,
        "task_policy_hz": manifest.rates.task_policy_hz,
    }
    if actual != expected:
        raise CandidateBundleError("candidate immutable metadata does not match pinned values")
    if manifest.qualification.qualified:
        raise CandidateBundleError("unqualified candidate manifest cannot claim qualification")


def verify_candidate_bundle(
    manifest_path: Path = ROBOT_MANIFEST,
    repository_root: Path = REPOSITORY_ROOT,
) -> BundleVerification:
    """Verify all candidate inputs and fail closed on any discrepancy."""
    try:
        manifest = CandidateManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CandidateBundleError(f"invalid candidate manifest: {manifest_path}") from error

    _validate_immutable_metadata(manifest)
    if _aggregate_sha256(manifest) != manifest.aggregate_sha256:
        raise CandidateBundleError("candidate manifest aggregate hash mismatch")

    paths = [entry.path for entry in manifest.files]
    if len(paths) != len(set(paths)):
        raise CandidateBundleError("candidate manifest contains duplicate paths")

    expected_paths = {
        _repository_path(path, repository_root)
        for path in discover_candidate_files(repository_root)
    }
    if set(paths) != expected_paths:
        raise CandidateBundleError(
            "candidate manifest file set does not match runtime dependencies"
        )

    for entry in manifest.files:
        path = repository_root / entry.path
        try:
            path.resolve().relative_to(repository_root.resolve())
        except ValueError as error:
            raise CandidateBundleError(
                f"candidate file escapes repository: {entry.path}"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise CandidateBundleError(f"candidate file missing or not regular: {entry.path}")
        if path.stat().st_size != entry.size_bytes:
            raise CandidateBundleError(f"candidate file size mismatch: {entry.path}")
        if _file_sha256(path) != entry.sha256:
            raise CandidateBundleError(f"candidate file digest mismatch: {entry.path}")

    policy_relative = G1_POLICY_ONNX.relative_to(REPOSITORY_ROOT).as_posix()
    policy = next((entry for entry in manifest.files if entry.path == policy_relative), None)
    if policy is None or policy.sha256 != EXPECTED_CONTROLLER_SHA256:
        raise CandidateBundleError("candidate controller is missing or has the wrong digest")

    return BundleVerification(
        candidate_id=manifest.candidate_id,
        valid=True,
        qualified=False,
        file_count=len(manifest.files),
        aggregate_sha256=manifest.aggregate_sha256,
        blockers=manifest.qualification.blockers,
    )


def _rooted(path: Path, repository_root: Path) -> Path:
    """Map a canonical repository path into an alternate repository root."""
    return repository_root / path.resolve().relative_to(REPOSITORY_ROOT.resolve())


def robot_bundle_checksum(
    policy_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    candidate_manifest_path: Path | None = None,
) -> str:
    """Hash every byte that can change frozen low-level robot behavior."""
    candidate = verify_candidate_bundle(
        manifest_path=candidate_manifest_path or _rooted(ROBOT_MANIFEST, repository_root),
        repository_root=repository_root,
    )
    payload = {
        "candidate_robot": candidate.aggregate_sha256,
        "controller_onnx": _file_sha256(policy_path),
        "controller_runtime": _file_sha256(
            _rooted(MM01_CONTROLLER_SOURCE, repository_root)
        ),
        "rate_runtime": _file_sha256(_rooted(MM01_RUNTIME_SOURCE, repository_root)),
        "task_command_runtime": _file_sha256(
            _rooted(MM01_COMMAND_SOURCE, repository_root)
        ),
        "sensor_profile": _file_sha256(_rooted(MM01_SENSOR_PROFILE, repository_root)),
        "sensor_runtime": _file_sha256(_rooted(MM01_SENSOR_SOURCE, repository_root)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(canonical).hexdigest()


def discover_mm01_files(repository_root: Path = REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Return the exact controller, runtime, and evidence files frozen for MM-01."""
    canonical_paths = (
        ROBOT_MANIFEST,
        MM01_CONTROLLER_ONNX,
        MM01_SENSOR_PROFILE,
        MM01_CONTROLLER_SOURCE,
        MM01_RUNTIME_SOURCE,
        MM01_COMMAND_SOURCE,
        MM01_SENSOR_SOURCE,
        MM01_IDENTITY_SOURCE,
        MM01_QUALIFICATION_PROGRAM,
        MM01_QUALIFICATION_CONTRACT,
        MM01_TRAINING_PATCH,
        MM01_ONNX_PARITY_EVIDENCE,
        MM01_QUALIFICATION_EVIDENCE,
        MM01_QUALIFICATION_TRIALS,
        MM01_TRAINING_CONTRACT,
    )
    return tuple(sorted(_rooted(path, repository_root) for path in canonical_paths))


def _qualified_aggregate_sha256(manifest: QualifiedRobotManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"aggregate_sha256"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CandidateBundleError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise CandidateBundleError(f"{label} must be a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateBundleError(message)


def _validate_raw_trials(repository_root: Path) -> dict[str, dict[str, object]]:
    payload = _read_json_object(
        _rooted(MM01_QUALIFICATION_TRIALS, repository_root),
        "qualification trials",
    )
    _require(payload.get("schema_version") == 1, "qualification-trial schema changed")
    trials = payload.get("trials")
    _require(isinstance(trials, dict), "qualification trials are missing")
    assert isinstance(trials, dict)
    durations = {
        "forward": 10.0,
        "forward_repeat": 10.0,
        "left": 4.0,
        "right": 4.0,
        "stop": 15.0,
        "standstill": 60.0,
        "payload": 19.0,
    }
    _require(set(trials) == set(durations), "qualification trial set changed")
    validated: dict[str, dict[str, object]] = {}
    for name, duration in durations.items():
        trial = trials[name]
        _require(isinstance(trial, dict), f"qualification trial {name} is invalid")
        assert isinstance(trial, dict)
        _require(trial.get("duration_seconds") == duration, f"{name} duration changed")
        _require(
            trial.get("physics_steps") == round(duration * PHYSICS_HZ),
            f"{name} physics rate did not measure 500 Hz",
        )
        _require(
            trial.get("controller_inferences") == round(duration * CONTROLLER_INFERENCE_HZ),
            f"{name} controller rate did not measure 100 Hz",
        )
        _require(
            trial.get("controller_supervisor_ticks")
            == round(duration * CONTROLLER_SUPERVISOR_HZ),
            f"{name} supervisor rate did not measure 100 Hz",
        )
        _require(trial.get("finite_state") is True, f"{name} produced non-finite state")
        _require(trial.get("fell") is False, f"{name} recorded a fall")
        _require(trial.get("body_collisions") == 0, f"{name} recorded a body collision")
        validated[name] = trial
    return validated


def _validate_controller_evidence(
    repository_root: Path,
) -> tuple[QualifiedControllerEvidence, ControllerParityEvidence]:
    try:
        evidence = QualifiedControllerEvidence.model_validate_json(
            _rooted(MM01_QUALIFICATION_EVIDENCE, repository_root).read_text(
                encoding="utf-8"
            )
        )
        parity = ControllerParityEvidence.model_validate_json(
            _rooted(MM01_ONNX_PARITY_EVIDENCE, repository_root).read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise CandidateBundleError("controller evidence does not match its schema") from error

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
        parity.maximum_absolute_delta,
        parity.limit,
    )
    _require(all(math.isfinite(value) for value in numeric_values), "evidence is non-finite")
    gates = (
        (evidence.controller_hz == 100, "controller did not qualify at 100 Hz"),
        (evidence.ctrl_dt_seconds == 0.01, "controller period changed"),
        (evidence.physics_hz == 500, "qualification physics rate changed"),
        (evidence.task_command_outputs == 3, "task command shape changed"),
        (evidence.gait_action_outputs == 29, "gait action shape changed"),
        (evidence.finite_state, "qualification produced non-finite state"),
        (evidence.fall_count == 0, "qualification recorded a fall"),
        (evidence.body_collision_count == 0, "qualification recorded a body collision"),
        (evidence.forward_progress_metres >= 1.0, "forward progress gate failed"),
        (
            evidence.forward_cross_track_error_metres <= 0.25,
            "forward cross-track gate failed",
        ),
        (evidence.forward_heading_error_degrees <= 15.0, "heading gate failed"),
        (evidence.left_turn_error_degrees <= 15.0, "left-turn gate failed"),
        (evidence.right_turn_error_degrees <= 15.0, "right-turn gate failed"),
        (evidence.stop_speed_metres_per_second <= 0.05, "stop-speed gate failed"),
        (evidence.standstill_duration_seconds >= 60.0, "standstill duration gate failed"),
        (evidence.standstill_drift_metres <= 0.25, "standstill drift gate failed"),
        (
            evidence.payload_stop_speed_metres_per_second <= 0.05,
            "payload stop-speed gate failed",
        ),
        (
            evidence.maximum_payload_tray_tilt_degrees < 12.0,
            "payload tray-tilt gate failed",
        ),
        (not evidence.payload_package_slipped, "payload package slipped"),
        (
            evidence.deterministic_repeat_max_metric_delta <= 1e-5,
            "repeatability gate failed",
        ),
        (parity.schema_version == 1, "ONNX parity schema changed"),
        (parity.passed, "ONNX parity did not pass"),
        (parity.limit <= 1e-5, "ONNX parity limit was widened"),
        (
            parity.maximum_absolute_delta <= parity.limit,
            "ONNX parity delta exceeds its limit",
        ),
    )
    for passed, message in gates:
        _require(passed, message)

    controller_path = _rooted(MM01_CONTROLLER_ONNX, repository_root)
    trial_path = _rooted(MM01_QUALIFICATION_TRIALS, repository_root)
    program_path = _rooted(MM01_QUALIFICATION_PROGRAM, repository_root)
    _require(
        _file_sha256(controller_path) == evidence.controller_onnx_sha256,
        "qualified controller digest differs from physical evidence",
    )
    _require(
        _file_sha256(trial_path) == evidence.qualification_trials_sha256,
        "qualification trial digest differs from physical evidence",
    )
    _require(
        _file_sha256(program_path) == evidence.qualification_program_sha256,
        "qualification program digest differs from physical evidence",
    )
    _require(
        robot_bundle_checksum(controller_path, repository_root=repository_root)
        == evidence.robot_checksum,
        "qualified robot checksum differs from runtime bytes",
    )
    trials = _validate_raw_trials(repository_root)
    forward_delta = trials["forward"].get("planar_delta_m")
    _require(
        isinstance(forward_delta, list) and len(forward_delta) == 2,
        "forward trial displacement is invalid",
    )
    assert isinstance(forward_delta, list)
    _require(
        math.isclose(float(forward_delta[0]), evidence.forward_progress_metres, abs_tol=1e-12),
        "forward progress differs from raw trials",
    )
    _require(
        math.isclose(
            abs(float(forward_delta[1])),
            evidence.forward_cross_track_error_metres,
            abs_tol=1e-12,
        ),
        "forward cross-track error differs from raw trials",
    )
    raw_metric_bindings = (
        ("stop", "last_second_maximum_speed_mps", evidence.stop_speed_metres_per_second),
        ("standstill", "maximum_drift_m", evidence.standstill_drift_metres),
        (
            "payload",
            "last_second_maximum_speed_mps",
            evidence.payload_stop_speed_metres_per_second,
        ),
        (
            "payload",
            "maximum_tray_tilt_degrees",
            evidence.maximum_payload_tray_tilt_degrees,
        ),
    )
    for trial_name, metric, expected in raw_metric_bindings:
        actual = trials[trial_name].get(metric)
        _require(
            isinstance(actual, int | float)
            and math.isclose(float(actual), expected, abs_tol=1e-12),
            f"{trial_name} {metric} differs from raw trials",
        )
    _require(
        trials["payload"].get("package_slipped") is evidence.payload_package_slipped,
        "payload slip result differs from raw trials",
    )
    return evidence, parity


def _checkpoint_step(path: object, label: str) -> int:
    _require(isinstance(path, str), f"{label} checkpoint path is missing")
    assert isinstance(path, str)
    name = Path(path).name
    _require(name.isdigit(), f"{label} checkpoint path is invalid")
    return int(name)


def _validate_training_contract(
    repository_root: Path,
    evidence: QualifiedControllerEvidence,
    parity: ControllerParityEvidence,
) -> tuple[str, int, str, str, tuple[str, ...]]:
    contract = _read_json_object(
        _rooted(MM01_TRAINING_CONTRACT, repository_root),
        "training contract",
    )
    source = contract.get("source")
    expected_source = {
        "joystick_sha256": MM01_PATCHED_JOYSTICK_SHA256,
        "menagerie_commit": MENAGERIE_COMMIT,
        "patch_sha256": MM01_TRAINING_PATCH_SHA256,
        "patched": True,
        "playground_commit": PLAYGROUND_COMMIT,
        "playground_tag_commit": PLAYGROUND_COMMIT,
    }
    _require(source == expected_source, "training source pins changed")
    _require(
        _file_sha256(_rooted(MM01_TRAINING_PATCH, repository_root))
        == MM01_TRAINING_PATCH_SHA256,
        "training patch digest changed",
    )
    expected_plan = {
        "episode_length": 2000,
        "mode": "full",
        "num_envs": 8192,
        "num_eval_envs": 128,
        "num_evals": 20,
        "num_timesteps": 200_000_000,
    }
    expected_environment = {
        "controller_hz": 100,
        "ctrl_dt_seconds": 0.01,
        "name": "G1JoystickFlatTerrain",
        "physical_episode_seconds": 20.0,
        "physics_hz": 500,
    }
    _require(contract.get("schema_version") == 2, "training contract schema changed")
    _require(contract.get("mode") == "full", "controller was not trained with the full plan")
    _require(
        contract.get("status") == "qualified_checkpoint_selected",
        "training run did not finish checkpoint selection",
    )
    _require(contract.get("training_plan") == expected_plan, "full training plan changed")
    _require(contract.get("environment") == expected_environment, "training environment changed")
    _require(
        contract.get("preserved_upstream_settings")
        == {
            "g1_network_configuration": True,
            "reward_configuration": True,
            "robot_and_sensors": True,
        },
        "upstream controller settings were not preserved",
    )
    restart = contract.get("restart_semantics")
    _require(isinstance(restart, dict), "restart semantics are missing")
    assert isinstance(restart, dict)
    _require(restart.get("optimizer_state_restored") is False, "optimizer state was restored")
    _require(restart.get("policy_warm_start_used") is False, "policy warm start was used")

    run_id = contract.get("run_id")
    seed = contract.get("seed")
    _require(isinstance(run_id, str) and bool(run_id), "training run ID is missing")
    _require(isinstance(seed, int) and seed >= 0, "training seed is invalid")
    assert isinstance(run_id, str)
    assert isinstance(seed, int)
    selected_attempt_id = contract.get("selected_attempt_id")
    attempts = contract.get("attempts")
    _require(isinstance(selected_attempt_id, str), "selected training attempt is missing")
    _require(isinstance(attempts, list), "training attempts are missing")
    assert isinstance(attempts, list)
    attempt = next(
        (
            item
            for item in attempts
            if isinstance(item, dict) and item.get("attempt_id") == selected_attempt_id
        ),
        None,
    )
    _require(isinstance(attempt, dict), "selected training attempt cannot be resolved")
    assert isinstance(attempt, dict)
    _require(attempt.get("status") == "training_complete", "training attempt was incomplete")
    _require(attempt.get("process_return_code") == 0, "training process did not exit cleanly")
    _require(attempt.get("optimizer_state_restored") is False, "attempt restored optimizer state")
    _require(attempt.get("policy_warm_start_used") is False, "attempt used a policy warm start")
    _require(
        attempt.get("restart_kind") == "fresh_complete_plan",
        "training attempt was not one fresh complete plan",
    )

    selection = contract.get("checkpoint_selection")
    _require(isinstance(selection, dict), "checkpoint selection evidence is missing")
    assert isinstance(selection, dict)
    selected_checkpoint = selection.get("selected_checkpoint")
    final_checkpoint = attempt.get("checkpoint")
    selected_step = _checkpoint_step(selected_checkpoint, "selected")
    final_step = _checkpoint_step(final_checkpoint, "final")
    _require(final_step >= 200_000_000, "full training plan did not reach 200M steps")
    _require(selected_step <= final_step, "selected checkpoint is later than the final checkpoint")
    _require(
        contract.get("exported_checkpoint") == selected_checkpoint,
        "exported checkpoint differs from the selected checkpoint",
    )
    _require(
        selection.get("previous_exported_checkpoint") == final_checkpoint,
        "original final checkpoint provenance changed",
    )
    _require(
        selection.get("strategy") == "latest_checkpoint_passing_full_native_qualification",
        "checkpoint selection strategy changed",
    )
    _require(
        selection.get("controller_onnx_sha256") == evidence.controller_onnx_sha256,
        "selected controller digest differs from qualification evidence",
    )
    _require(
        selection.get("qualification_trials_sha256")
        == evidence.qualification_trials_sha256,
        "selected trial digest differs from qualification evidence",
    )
    _require(parity.checkpoint == selected_checkpoint, "parity used a different checkpoint")

    rejections = selection.get("later_rejected_checkpoints")
    _require(
        isinstance(rejections, list) and bool(rejections),
        "later checkpoints were not audited",
    )
    assert isinstance(rejections, list)
    rejected_names: list[str] = []
    for rejection in rejections:
        _require(isinstance(rejection, dict), "later checkpoint rejection is invalid")
        assert isinstance(rejection, dict)
        name = rejection.get("checkpoint")
        rejected_step = _checkpoint_step(name, "rejected")
        _require(rejected_step > selected_step, "rejected checkpoint is not later than selection")
        failures = rejection.get("failures")
        _require(
            isinstance(failures, list)
            and bool(failures)
            and all(isinstance(item, str) and bool(item) for item in failures),
            "later checkpoint rejection has no measured failure",
        )
        digest = rejection.get("controller_onnx_sha256")
        _require(
            isinstance(digest, str) and len(digest) == 64,
            "later checkpoint rejection has no controller digest",
        )
        assert isinstance(name, str)
        rejected_names.append(name)
    _require(
        rejected_names == sorted(rejected_names, key=int),
        "later checkpoint rejections are not ordered",
    )
    _require(Path(str(final_checkpoint)).name in rejected_names, "final checkpoint was not audited")
    assert isinstance(selected_checkpoint, str)
    assert isinstance(final_checkpoint, str)
    return run_id, seed, selected_checkpoint, final_checkpoint, tuple(rejected_names)


def build_mm01_manifest(repository_root: Path = REPOSITORY_ROOT) -> QualifiedRobotManifest:
    """Build a qualified manifest only after independently rechecking all evidence."""
    candidate = verify_candidate_bundle(
        manifest_path=_rooted(ROBOT_MANIFEST, repository_root),
        repository_root=repository_root,
    )
    evidence, parity = _validate_controller_evidence(repository_root)
    run_id, seed, selected, final, rejections = _validate_training_contract(
        repository_root,
        evidence,
        parity,
    )
    files = tuple(
        FrozenFile(
            path=_repository_path(path, repository_root),
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
        )
        for path in discover_mm01_files(repository_root)
    )
    manifest = QualifiedRobotManifest(
        schema_version=1,
        robot_id=MM01_ROBOT_ID,
        controller_id=MM01_CONTROLLER_ID,
        qualified=True,
        source=SourcePins(
            mujoco_playground_tag="v0.2.0",
            mujoco_playground_commit=PLAYGROUND_COMMIT,
            mujoco_menagerie_commit=MENAGERIE_COMMIT,
            mujoco_version="3.6.0",
            onnxruntime_version="1.28.0",
        ),
        rates=CandidateRates(
            physics_hz=PHYSICS_HZ,
            controller_supervisor_hz=CONTROLLER_SUPERVISOR_HZ,
            controller_inference_hz=CONTROLLER_INFERENCE_HZ,
            task_policy_hz=TASK_POLICY_HZ,
        ),
        training_run_id=run_id,
        training_seed=seed,
        selected_checkpoint=selected,
        final_checkpoint=final,
        later_rejected_checkpoints=rejections,
        base_candidate_aggregate_sha256=candidate.aggregate_sha256,
        robot_checksum=evidence.robot_checksum,
        files=files,
        aggregate_sha256="0" * 64,
    )
    return manifest.model_copy(
        update={"aggregate_sha256": _qualified_aggregate_sha256(manifest)}
    )


def write_mm01_manifest(
    destination: Path = MM01_MANIFEST,
    repository_root: Path = REPOSITORY_ROOT,
) -> QualifiedRobotManifest:
    """Write the qualified manifest in deterministic JSON form."""
    manifest = build_mm01_manifest(repository_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def _validate_mm01_metadata(manifest: QualifiedRobotManifest) -> None:
    expected = {
        "schema_version": 1,
        "robot_id": MM01_ROBOT_ID,
        "controller_id": MM01_CONTROLLER_ID,
        "qualified": True,
        "playground_tag": "v0.2.0",
        "playground_commit": PLAYGROUND_COMMIT,
        "menagerie_commit": MENAGERIE_COMMIT,
        "mujoco_version": "3.6.0",
        "onnxruntime_version": "1.28.0",
        "physics_hz": PHYSICS_HZ,
        "controller_supervisor_hz": CONTROLLER_SUPERVISOR_HZ,
        "controller_inference_hz": CONTROLLER_INFERENCE_HZ,
        "task_policy_hz": TASK_POLICY_HZ,
    }
    actual = {
        "schema_version": manifest.schema_version,
        "robot_id": manifest.robot_id,
        "controller_id": manifest.controller_id,
        "qualified": manifest.qualified,
        "playground_tag": manifest.source.mujoco_playground_tag,
        "playground_commit": manifest.source.mujoco_playground_commit,
        "menagerie_commit": manifest.source.mujoco_menagerie_commit,
        "mujoco_version": manifest.source.mujoco_version,
        "onnxruntime_version": manifest.source.onnxruntime_version,
        "physics_hz": manifest.rates.physics_hz,
        "controller_supervisor_hz": manifest.rates.controller_supervisor_hz,
        "controller_inference_hz": manifest.rates.controller_inference_hz,
        "task_policy_hz": manifest.rates.task_policy_hz,
    }
    _require(actual == expected, "qualified MM-01 immutable metadata changed")


def verify_mm01_bundle(
    manifest_path: Path = MM01_MANIFEST,
    repository_root: Path = REPOSITORY_ROOT,
) -> RobotBundleVerification:
    """Verify qualified runtime bytes, training provenance, and physical evidence."""
    try:
        manifest = QualifiedRobotManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise CandidateBundleError(f"invalid qualified robot manifest: {manifest_path}") from error
    _validate_mm01_metadata(manifest)
    _require(
        _qualified_aggregate_sha256(manifest) == manifest.aggregate_sha256,
        "qualified robot manifest aggregate hash mismatch",
    )
    paths = [entry.path for entry in manifest.files]
    _require(len(paths) == len(set(paths)), "qualified robot manifest contains duplicate paths")
    expected_paths = {
        _repository_path(path, repository_root)
        for path in discover_mm01_files(repository_root)
    }
    _require(set(paths) == expected_paths, "qualified robot manifest file set changed")
    for entry in manifest.files:
        path = repository_root / entry.path
        try:
            path.resolve().relative_to(repository_root.resolve())
        except ValueError as error:
            raise CandidateBundleError(
                f"qualified robot file escapes repository: {entry.path}"
            ) from error
        _require(
            not path.is_symlink() and path.is_file(),
            f"qualified robot file missing or not regular: {entry.path}",
        )
        _require(path.stat().st_size == entry.size_bytes, f"file size changed: {entry.path}")
        _require(_file_sha256(path) == entry.sha256, f"file digest changed: {entry.path}")

    candidate = verify_candidate_bundle(
        manifest_path=_rooted(ROBOT_MANIFEST, repository_root),
        repository_root=repository_root,
    )
    _require(
        candidate.aggregate_sha256 == manifest.base_candidate_aggregate_sha256,
        "base candidate identity differs from qualified manifest",
    )
    evidence, parity = _validate_controller_evidence(repository_root)
    run_id, seed, selected, final, rejections = _validate_training_contract(
        repository_root,
        evidence,
        parity,
    )
    _require(manifest.robot_checksum == evidence.robot_checksum, "robot checksum changed")
    _require(manifest.training_run_id == run_id, "training run ID changed")
    _require(manifest.training_seed == seed, "training seed changed")
    _require(manifest.selected_checkpoint == selected, "selected checkpoint changed")
    _require(manifest.final_checkpoint == final, "final checkpoint changed")
    _require(
        manifest.later_rejected_checkpoints == rejections,
        "later checkpoint evidence changed",
    )
    return RobotBundleVerification(
        robot_id=manifest.robot_id,
        controller_id=manifest.controller_id,
        valid=True,
        qualified=True,
        file_count=len(manifest.files),
        robot_checksum=manifest.robot_checksum,
        manifest_aggregate_sha256=manifest.aggregate_sha256,
        training_run_id=manifest.training_run_id,
        selected_checkpoint=manifest.selected_checkpoint,
        blockers=(),
    )
