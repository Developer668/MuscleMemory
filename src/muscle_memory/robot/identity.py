"""Fail-closed identity verification for the unqualified robot candidate."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from muscle_memory.paths import (
    G1_POLICY_ONNX,
    G1_SCENE_XML,
    REPOSITORY_ROOT,
    ROBOT_MANIFEST,
    THIRD_PARTY_ROOT,
)

PLAYGROUND_COMMIT = "124a73fa3303f75a62f8fe04d329b829ed0ebdfb"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"
EXPECTED_CONTROLLER_SHA256 = "db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b"
PHYSICS_HZ = 500
CONTROLLER_SUPERVISOR_HZ = 100
CONTROLLER_INFERENCE_HZ = 50
TASK_POLICY_HZ = 10


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
            controller_inference_hz=CONTROLLER_INFERENCE_HZ,
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
        "controller_inference_hz": CONTROLLER_INFERENCE_HZ,
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
