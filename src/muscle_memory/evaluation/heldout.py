"""Hash-verified held-out worlds with no path-teacher dependency."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from muscle_memory.paths import HELDOUT_WORLDS_BUNDLE, REPOSITORY_ROOT
from muscle_memory.worlds.models import HeldOutWorld
from muscle_memory.worlds.rules import DEFAULT_RULES_PATH

HELDOUT_WORLD_COUNT = 20
HELDOUT_SPLIT_ID = "heldout-v1"
HELDOUT_VALIDATION_CHECKS = (
    "objects_do_not_overlap",
    "start_and_destination_connected",
    "passages_meet_minimum_clearance",
    "colliders_are_approved",
    "baseline_path_exists",
    "physical_parameters_are_safe",
    "robust_expert_path_exists",
    "direct_route_requires_avoidance",
    "physical_expert_delivery_succeeds",
)


class HeldOutBundleError(RuntimeError):
    """Raised when evaluation world bytes or certificates do not verify."""


class FrozenEvaluationModel(BaseModel):
    """Strict immutable base for held-out records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PhysicalWorldQualification(FrozenEvaluationModel):
    """Measured frozen-controller result used only to qualify a world before freezing."""

    success: bool
    time_to_resident_seconds: float = Field(ge=0.0, le=30.0)
    stop_distance_metres: float = Field(ge=0.0, le=0.5)
    facing_error_degrees: float = Field(ge=0.0, le=30.0)
    stopped_speed_metres_per_second: float = Field(ge=0.0, le=0.05)
    falls: int = Field(ge=0, le=0)
    body_collisions: int = Field(ge=0, le=0)
    minimum_obstacle_clearance_metres: float = Field(ge=0.25)
    maximum_tray_tilt_degrees: float = Field(ge=0.0, lt=12.0)
    package_slipped: bool
    human_interventions: int = Field(ge=0, le=0)


class HeldOutValidationCertificate(FrozenEvaluationModel):
    """Content-addressed proof created before the world enters the held-out split."""

    schema_version: int
    robot_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rules_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robust_expert_path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_checks: tuple[str, ...]
    physical_qualification: PhysicalWorldQualification


class HeldOutWorldRecord(FrozenEvaluationModel):
    """One evaluation world and its path-free validation certificate."""

    world: HeldOutWorld
    certificate: HeldOutValidationCertificate


class HeldOutWorldBundle(FrozenEvaluationModel):
    """The complete immutable 20-world evaluation split."""

    schema_version: int
    split_id: str
    generation_version: int
    records: tuple[HeldOutWorldRecord, ...]
    aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ValidatedHeldOutWorld(FrozenEvaluationModel):
    """Runtime envelope accepted by simulation without exposing an expert path."""

    world: HeldOutWorld
    certificate: HeldOutValidationCertificate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_sha256(world: HeldOutWorld) -> str:
    """Hash one held-out world using stable canonical JSON."""
    payload = world.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def heldout_bundle_aggregate(bundle: HeldOutWorldBundle) -> str:
    """Hash bundle metadata and records while excluding the self-hash field."""
    payload = bundle.model_dump(mode="json", exclude={"aggregate_sha256"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_heldout_worlds(
    path: Path = HELDOUT_WORLDS_BUNDLE,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[ValidatedHeldOutWorld, ...]:
    """Load the frozen split and fail closed without importing training code."""
    try:
        bundle = HeldOutWorldBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HeldOutBundleError(f"invalid held-out world bundle: {path}") from error
    if bundle.schema_version != 1 or bundle.split_id != HELDOUT_SPLIT_ID:
        raise HeldOutBundleError("held-out bundle identity changed")
    if len(bundle.records) != HELDOUT_WORLD_COUNT:
        raise HeldOutBundleError("held-out bundle must contain exactly 20 worlds")
    if heldout_bundle_aggregate(bundle) != bundle.aggregate_sha256:
        raise HeldOutBundleError("held-out bundle aggregate hash mismatch")

    identifiers = [record.world.world_id for record in bundle.records]
    seeds = [record.world.seed for record in bundle.records]
    if len(set(identifiers)) != HELDOUT_WORLD_COUNT or len(set(seeds)) != HELDOUT_WORLD_COUNT:
        raise HeldOutBundleError("held-out world IDs and seeds must be unique")
    rules_path = repository_root / DEFAULT_RULES_PATH.relative_to(REPOSITORY_ROOT)
    rules_hash = _sha256_file(rules_path)
    from muscle_memory.robot.identity import verify_mm01_bundle

    robot = verify_mm01_bundle(repository_root=repository_root)
    validated: list[ValidatedHeldOutWorld] = []
    for record in bundle.records:
        certificate = record.certificate
        measurements = certificate.physical_qualification
        numeric_values = tuple(
            value
            for name, value in measurements.model_dump().items()
            if name != "success" and isinstance(value, int | float)
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise HeldOutBundleError("held-out physical measurements must be finite")
        if record.world.generation_version != bundle.generation_version:
            raise HeldOutBundleError("held-out world generation version changed")
        if world_sha256(record.world) != certificate.world_sha256:
            raise HeldOutBundleError(f"held-out world hash mismatch: {record.world.world_id}")
        if certificate.schema_version != 1 or certificate.rules_sha256 != rules_hash:
            raise HeldOutBundleError("held-out validation source changed")
        if certificate.robot_checksum != robot.robot_checksum:
            raise HeldOutBundleError("held-out world was qualified with a different robot")
        if certificate.validation_checks != HELDOUT_VALIDATION_CHECKS:
            raise HeldOutBundleError("held-out validation checks changed")
        if not measurements.success or measurements.package_slipped:
            raise HeldOutBundleError("held-out world lacks passing physical qualification")
        validated.append(
            ValidatedHeldOutWorld(
                world=record.world,
                certificate=certificate,
            )
        )
    return tuple(validated)
