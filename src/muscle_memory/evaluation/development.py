"""Strict development-evidence gate protecting the frozen held-out split."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)

from muscle_memory.coordinator.models import canonical_json
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import STOPPED_SPEED_MPS, PolicyEpisodeResult
from muscle_memory.evaluation.success import SAFE_DELIVERY_CRITERIA, FailureReason
from muscle_memory.paths import REPOSITORY_ROOT
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.policy.network import BehaviorClonedPolicy
from muscle_memory.robot.identity import verify_mm01_bundle

MAX_DEVELOPMENT_EVIDENCE_BYTES = 16 * 1024 * 1024
MINIMUM_DEVELOPMENT_WORLD_COUNT = 12
MINIMUM_DEVELOPMENT_SEED = 600_000_000
DEFAULT_DEVELOPMENT_LOCK_NAME = "lock.json"
DEVELOPMENT_EVALUATION_SCOPE = "generated_disjoint_development"
DEVELOPMENT_PURPOSE = "development_only_not_held_out"
DEVELOPMENT_SEED_SEARCH_RULE = (
    "first validated worlds with an expert path at most 8.2 m whose direct route requires avoidance"
)

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_WORLD_ID_PATTERN = r"^train-v[0-9]+-[0-9a-f]{16}$"
_Hash = Annotated[StrictStr, Field(pattern=_HASH_PATTERN)]
_WorldId = Annotated[StrictStr, Field(pattern=_WORLD_ID_PATTERN)]
_NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]


class DevelopmentEvidenceError(RuntimeError):
    """Development evidence cannot authorize access to held-out worlds."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DevelopmentProvenance(_StrictModel):
    """Typed meaning for generated worlds whose model split remains `training`."""

    world_generation_mechanics: Literal["training_world_generation"]
    evaluation_dataset_membership: Literal["disjoint_development_only"]
    heldout_world_access: Literal["never"]


class StrictPolicyEpisodeResult(_StrictModel):
    """Non-coercing JSON form of one evaluation result."""

    episode_id: _NonEmptyStr
    world_id: _WorldId
    world_seed: StrictInt
    world_split: Literal["training"]
    world_hash: _Hash
    robot_checksum: _Hash
    policy_id: _NonEmptyStr
    policy_hash: _Hash
    success: StrictBool
    failed_reasons: list[_NonEmptyStr]
    time_to_resident_seconds: StrictFloat | None
    simulated_duration_seconds: StrictFloat
    stop_distance_m: StrictFloat
    facing_error_degrees: StrictFloat | None
    stopped_speed_mps: StrictFloat
    falls: StrictInt
    body_collisions: StrictInt
    minimum_obstacle_clearance_m: StrictFloat
    maximum_tray_tilt_degrees: StrictFloat
    package_slipped: StrictBool
    human_interventions: StrictInt
    direct_distance_m: StrictFloat
    path_length_m: StrictFloat
    path_efficiency: StrictFloat
    energy_joules: StrictFloat
    task_policy_updates: StrictInt
    trace: list[object] = Field(max_length=0)


class DevelopmentEvidence(_StrictModel):
    """Exact v2 envelope admitted by the held-out access gate."""

    schema_version: Literal[2]
    purpose: Literal["development_only_not_held_out"]
    evaluation_scope: Literal["generated_disjoint_development"]
    provenance: DevelopmentProvenance
    selection_status: Literal[
        "eligible_for_blinded_heldout_evaluation",
        "rejected_before_heldout",
    ]
    paired_world_count: StrictInt
    seed_search_start: StrictInt
    seed_search_rule: Literal[
        "first validated worlds with an expert path at most 8.2 m "
        "whose direct route requires avoidance"
    ]
    candidate_policy_id: _NonEmptyStr
    candidate_policy_sha256: _Hash
    world_ids: list[_WorldId]
    world_seeds: list[StrictInt]
    baseline_results: list[StrictPolicyEpisodeResult]
    candidate_results: list[StrictPolicyEpisodeResult]
    promotion_preview: dict[StrictStr, object]


class DevelopmentEvidenceLock(_StrictModel):
    """Operator-controlled binding for one immutable development artifact."""

    schema_version: Literal[2]
    policy_id: _NonEmptyStr
    artifact_status: Literal[
        "immutable_eligible_candidate",
        "immutable_rejected_candidate",
    ]
    heldout_access: Literal[
        "eligible_for_blinded_heldout_evaluation",
        "denied_by_development_gate",
    ]
    evaluation_scope: Literal["generated_disjoint_development"]
    provenance: DevelopmentProvenance
    paired_world_count: StrictInt
    robot_checksum: _Hash
    checkpoint_path: _NonEmptyStr
    checkpoint_sha256: _Hash
    training_dataset_path: _NonEmptyStr
    training_dataset_sha256: _Hash
    training_evidence_path: _NonEmptyStr
    training_evidence_sha256: _Hash
    training_evidence_canonical_sha256: _Hash
    development_round1_evidence_path: _NonEmptyStr
    development_round1_evidence_sha256: _Hash
    development_round1_evidence_canonical_sha256: _Hash
    final_development_evidence_path: _NonEmptyStr
    final_development_evidence_sha256: _Hash
    final_development_evidence_canonical_sha256: _Hash


@dataclass(frozen=True, slots=True)
class DevelopmentEvidenceReceipt:
    """Verified result returned before the access decision is enforced."""

    evidence_sha256: str
    evidence_canonical_sha256: str
    checkpoint_sha256: str
    candidate_policy_id: str
    paired_world_count: int
    promotable: bool
    selection_status: str


def verify_development_evidence(
    evidence_path: Path,
    checkpoint_path: Path,
    *,
    lock_path: Path | None = None,
) -> DevelopmentEvidenceReceipt:
    """Verify one locked generated-development comparison without held-out access."""

    selected_lock_path = lock_path or evidence_path.with_name(DEFAULT_DEVELOPMENT_LOCK_NAME)
    lock = _parse_model(
        DevelopmentEvidenceLock,
        _load_unique_json(selected_lock_path, "development evidence lock"),
        "development evidence lock",
    )
    if _resolve_bound_path(lock.final_development_evidence_path) != evidence_path.resolve():
        raise DevelopmentEvidenceError(
            "development evidence path does not equal the path bound by the lock"
        )
    if _resolve_bound_path(lock.checkpoint_path) != checkpoint_path.resolve():
        raise DevelopmentEvidenceError(
            "candidate checkpoint path does not equal the path bound by the lock"
        )

    evidence_sha256 = _sha256_file(evidence_path, "development evidence")
    if evidence_sha256 != lock.final_development_evidence_sha256:
        raise DevelopmentEvidenceError(
            "development evidence does not match the independently locked hash"
        )
    checkpoint_sha256 = _sha256_file(checkpoint_path, "candidate checkpoint")
    if checkpoint_sha256 != lock.checkpoint_sha256:
        raise DevelopmentEvidenceError(
            "candidate checkpoint does not match the independently locked hash"
        )
    _verify_training_lineage(lock, checkpoint_path)

    evidence_payload = _load_unique_json(evidence_path, "development evidence")
    evidence_canonical_sha256 = hashlib.sha256(
        canonical_json(evidence_payload).encode("utf-8")
    ).hexdigest()
    if evidence_canonical_sha256 != lock.final_development_evidence_canonical_sha256:
        raise DevelopmentEvidenceError(
            "development evidence does not match the locked canonical hash"
        )
    evidence = _parse_model(
        DevelopmentEvidence,
        evidence_payload,
        "development evidence",
    )
    try:
        candidate_policy = BehaviorClonedPolicy.load(checkpoint_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DevelopmentEvidenceError(
            "candidate checkpoint could not be independently verified"
        ) from exc
    robot = verify_mm01_bundle()
    if not robot.valid or not robot.qualified:
        raise DevelopmentEvidenceError("qualified MM-01 verification failed")
    if (
        candidate_policy.policy_hash != checkpoint_sha256
        or candidate_policy.policy_hash != evidence.candidate_policy_sha256
        or candidate_policy.policy_hash != lock.checkpoint_sha256
        or candidate_policy.policy_id != evidence.candidate_policy_id
        or candidate_policy.policy_id != lock.policy_id
        or robot.robot_checksum != lock.robot_checksum
    ):
        raise DevelopmentEvidenceError(
            "candidate checkpoint identity does not match the evidence lock"
        )
    if evidence.evaluation_scope != lock.evaluation_scope or evidence.provenance != lock.provenance:
        raise DevelopmentEvidenceError(
            "development evaluation scope or provenance differs from the lock"
        )

    baseline_results = tuple(_episode_result(item) for item in evidence.baseline_results)
    candidate_results = tuple(_episode_result(item) for item in evidence.candidate_results)
    _validate_development_pairs(evidence, lock, baseline_results, candidate_results)
    for result in (*baseline_results, *candidate_results):
        _validate_canonical_outcome(result)
    try:
        decision = evaluate_promotion(baseline_results, candidate_results)
    except ValueError as exc:
        raise DevelopmentEvidenceError(
            "development results are not an exact paired comparison"
        ) from exc
    if canonical_json(evidence.promotion_preview) != canonical_json(asdict(decision)):
        raise DevelopmentEvidenceError(
            "development promotion preview does not equal recomputed measurements"
        )

    expected_selection = (
        "eligible_for_blinded_heldout_evaluation"
        if decision.promotable
        else "rejected_before_heldout"
    )
    expected_access = (
        "eligible_for_blinded_heldout_evaluation"
        if decision.promotable
        else "denied_by_development_gate"
    )
    expected_status = (
        "immutable_eligible_candidate" if decision.promotable else "immutable_rejected_candidate"
    )
    if (
        evidence.selection_status != expected_selection
        or lock.heldout_access != expected_access
        or lock.artifact_status != expected_status
    ):
        raise DevelopmentEvidenceError(
            "development access state does not equal the recomputed promotion preview"
        )
    return DevelopmentEvidenceReceipt(
        evidence_sha256=evidence_sha256,
        evidence_canonical_sha256=evidence_canonical_sha256,
        checkpoint_sha256=checkpoint_sha256,
        candidate_policy_id=candidate_policy.policy_id,
        paired_world_count=len(candidate_results),
        promotable=decision.promotable,
        selection_status=evidence.selection_status,
    )


def assert_development_gate(
    evidence_path: Path,
    checkpoint_path: Path,
    *,
    lock_path: Path | None = None,
) -> None:
    """Permit held-out import only for an exact, locked, passing candidate."""

    receipt = verify_development_evidence(
        evidence_path,
        checkpoint_path,
        lock_path=lock_path,
    )
    if not receipt.promotable:
        raise DevelopmentEvidenceError(
            "candidate failed the development gate; held-out access denied"
        )


def _validate_development_pairs(
    evidence: DevelopmentEvidence,
    lock: DevelopmentEvidenceLock,
    baseline: tuple[PolicyEpisodeResult, ...],
    candidate: tuple[PolicyEpisodeResult, ...],
) -> None:
    count = evidence.paired_world_count
    if count < MINIMUM_DEVELOPMENT_WORLD_COUNT or count != lock.paired_world_count:
        raise DevelopmentEvidenceError(
            "development evidence does not contain the locked minimum paired world count"
        )
    if not (
        len(evidence.world_ids)
        == len(evidence.world_seeds)
        == len(baseline)
        == len(candidate)
        == count
    ):
        raise DevelopmentEvidenceError(
            "development evidence arrays do not match the locked paired world count"
        )
    if len({result.episode_id for result in (*baseline, *candidate)}) != count * 2:
        raise DevelopmentEvidenceError("development episode ids must be unique")
    if (
        len({(item.policy_id, item.policy_hash) for item in baseline}) != 1
        or len({(item.policy_id, item.policy_hash) for item in candidate}) != 1
    ):
        raise DevelopmentEvidenceError("development results cannot mix immutable policy identities")
    baseline_policy = DirectGoalPolicy()
    if (
        baseline[0].policy_id != baseline_policy.policy_id
        or baseline[0].policy_hash != baseline_policy.policy_hash
    ):
        raise DevelopmentEvidenceError(
            "development baseline is not the deterministic direct-goal policy"
        )
    if (
        candidate[0].policy_id != evidence.candidate_policy_id
        or candidate[0].policy_hash != evidence.candidate_policy_sha256
        or any(result.policy_hash != lock.checkpoint_sha256 for result in candidate)
    ):
        raise DevelopmentEvidenceError(
            "development candidate results do not match the locked checkpoint"
        )
    for index, (world_id, world_seed, baseline_result, candidate_result) in enumerate(
        zip(
            evidence.world_ids,
            evidence.world_seeds,
            baseline,
            candidate,
            strict=True,
        )
    ):
        if world_seed < MINIMUM_DEVELOPMENT_SEED:
            raise DevelopmentEvidenceError(
                "development evidence is outside the disjoint seed namespace"
            )
        expected = (
            world_id,
            world_seed,
            baseline_result.world_hash,
            lock.robot_checksum,
            "training",
        )
        baseline_identity = (
            baseline_result.world_id,
            baseline_result.world_seed,
            baseline_result.world_hash,
            baseline_result.robot_checksum,
            baseline_result.world_split,
        )
        candidate_identity = (
            candidate_result.world_id,
            candidate_result.world_seed,
            candidate_result.world_hash,
            candidate_result.robot_checksum,
            candidate_result.world_split,
        )
        if baseline_identity != expected or candidate_identity != expected:
            raise DevelopmentEvidenceError(
                f"development pair {index} does not match the locked world and robot identity"
            )


def _verify_training_lineage(
    lock: DevelopmentEvidenceLock,
    checkpoint_path: Path,
) -> None:
    dataset_path = _resolve_bound_path(lock.training_dataset_path)
    training_evidence_path = _resolve_bound_path(lock.training_evidence_path)
    round1_path = _resolve_bound_path(lock.development_round1_evidence_path)
    if _sha256_file(dataset_path, "training dataset") != lock.training_dataset_sha256:
        raise DevelopmentEvidenceError("training dataset does not match the evidence lock")
    if _sha256_file(training_evidence_path, "training evidence") != lock.training_evidence_sha256:
        raise DevelopmentEvidenceError("training evidence does not match the evidence lock")
    training_payload = _load_unique_json(training_evidence_path, "training evidence")
    if _canonical_payload_sha256(training_payload) != lock.training_evidence_canonical_sha256:
        raise DevelopmentEvidenceError("training evidence does not match the locked canonical hash")
    if (
        _sha256_file(round1_path, "round-one development evidence")
        != lock.development_round1_evidence_sha256
    ):
        raise DevelopmentEvidenceError(
            "round-one development evidence does not match the evidence lock"
        )
    round1_payload = _load_unique_json(round1_path, "round-one development evidence")
    if (
        _canonical_payload_sha256(round1_payload)
        != lock.development_round1_evidence_canonical_sha256
    ):
        raise DevelopmentEvidenceError(
            "round-one development evidence does not match the locked canonical hash"
        )

    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            checkpoint_policy_id = str(checkpoint["policy_id"])
            checkpoint_robot_checksum = str(checkpoint["robot_checksum"])
            checkpoint_dataset_sha256 = str(checkpoint["dataset_sha256"])
        with np.load(dataset_path, allow_pickle=False) as dataset:
            dataset_schema_version = int(dataset["schema_version"])
            dataset_robot_checksum = str(dataset["robot_checksum"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DevelopmentEvidenceError(
            "training dataset or checkpoint lineage metadata is invalid"
        ) from exc

    config = training_payload.get("config")
    if not isinstance(config, dict):
        raise DevelopmentEvidenceError("training evidence config is invalid")
    if (
        type(training_payload.get("schema_version")) is not int
        or training_payload.get("schema_version") != 1
        or type(training_payload.get("dataset_schema_version")) is not int
        or training_payload.get("dataset_schema_version") != dataset_schema_version
        or training_payload.get("dataset_sha256") != lock.training_dataset_sha256
        or training_payload.get("policy_sha256") != lock.checkpoint_sha256
        or training_payload.get("policy_id") != lock.policy_id
        or training_payload.get("robot_checksum") != lock.robot_checksum
        or config.get("policy_id") != lock.policy_id
        or checkpoint_policy_id != lock.policy_id
        or checkpoint_robot_checksum != lock.robot_checksum
        or checkpoint_dataset_sha256 != lock.training_dataset_sha256
        or dataset_robot_checksum != lock.robot_checksum
    ):
        raise DevelopmentEvidenceError(
            "training evidence, dataset, and checkpoint lineage identities differ"
        )


def _validate_canonical_outcome(result: PolicyEpisodeResult) -> None:
    criteria = SAFE_DELIVERY_CRITERIA
    nonnegative = (
        result.simulated_duration_seconds,
        result.stop_distance_m,
        result.stopped_speed_mps,
        result.falls,
        result.body_collisions,
        result.maximum_tray_tilt_degrees,
        result.human_interventions,
        result.direct_distance_m,
        result.path_length_m,
        result.path_efficiency,
        result.energy_joules,
        result.task_policy_updates,
    )
    if any(not math.isfinite(float(value)) or value < 0 for value in nonnegative):
        raise DevelopmentEvidenceError(
            "development result contains an invalid non-negative measurement"
        )
    if result.time_to_resident_seconds is not None and (
        not math.isfinite(result.time_to_resident_seconds)
        or result.time_to_resident_seconds < 0
        or result.time_to_resident_seconds > result.simulated_duration_seconds
    ):
        raise DevelopmentEvidenceError("development result has an invalid measured completion time")
    if (
        result.facing_error_degrees is not None
        and (
            not math.isfinite(result.facing_error_degrees)
            or not 0.0 <= result.facing_error_degrees <= 180.0
        )
    ) or result.path_efficiency > 1.0:
        raise DevelopmentEvidenceError(
            "development result has an invalid measured angle or path efficiency"
        )
    failed: list[str] = []
    if (
        result.time_to_resident_seconds is None
        or result.time_to_resident_seconds > criteria.maximum_time_to_resident_seconds
    ):
        failed.append(FailureReason.DID_NOT_REACH_RESIDENT_IN_TIME.name)
    if criteria.stopped_required and result.stopped_speed_mps > STOPPED_SPEED_MPS:
        failed.append(FailureReason.DID_NOT_STOP.name)
    if result.stop_distance_m > criteria.maximum_stop_distance_metres:
        failed.append(FailureReason.STOPPED_TOO_FAR_AWAY.name)
    if (
        result.facing_error_degrees is None
        or result.facing_error_degrees > criteria.maximum_facing_error_degrees
    ):
        failed.append(FailureReason.NOT_FACING_RESIDENT.name)
    if result.falls > criteria.maximum_falls:
        failed.append(FailureReason.FELL.name)
    if result.body_collisions > criteria.maximum_body_collisions:
        failed.append(FailureReason.BODY_COLLISION.name)
    if result.minimum_obstacle_clearance_m < criteria.minimum_obstacle_clearance_metres:
        failed.append(FailureReason.INSUFFICIENT_OBSTACLE_CLEARANCE.name)
    if result.maximum_tray_tilt_degrees >= criteria.maximum_tray_tilt_degrees_exclusive:
        failed.append(FailureReason.EXCESSIVE_TRAY_TILT.name)
    if result.package_slipped and not criteria.package_slip_allowed:
        failed.append(FailureReason.PACKAGE_SLIPPED.name)
    if result.human_interventions > criteria.maximum_human_interventions:
        failed.append(FailureReason.HUMAN_INTERVENTION.name)
    if result.failed_reasons != tuple(failed) or result.success != (not failed):
        raise DevelopmentEvidenceError(
            "development result success and failed_reasons do not match measurements"
        )


def _episode_result(item: StrictPolicyEpisodeResult) -> PolicyEpisodeResult:
    values = item.model_dump()
    values["failed_reasons"] = tuple(item.failed_reasons)
    values["trace"] = ()
    return PolicyEpisodeResult(**values)


def _parse_model[ModelT: BaseModel](
    model: type[ModelT],
    payload: dict[str, object],
    label: str,
) -> ModelT:
    try:
        return model.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise DevelopmentEvidenceError(f"{label} has an unsupported strict schema") from exc


def _load_unique_json(path: Path, label: str) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DevelopmentEvidenceError(f"{label} is unavailable") from exc
    if size <= 0 or size > MAX_DEVELOPMENT_EVIDENCE_BYTES:
        raise DevelopmentEvidenceError(f"{label} size is invalid")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        canonical_json(decoded)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DevelopmentEvidenceError(
            f"{label} is not a finite JSON object with unique keys"
        ) from exc
    if not isinstance(decoded, dict):
        raise DevelopmentEvidenceError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _sha256_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DevelopmentEvidenceError(f"could not hash {label}") from exc
    return digest.hexdigest()


def _canonical_payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _resolve_bound_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


__all__ = [
    "DEVELOPMENT_EVALUATION_SCOPE",
    "DEVELOPMENT_PURPOSE",
    "DEVELOPMENT_SEED_SEARCH_RULE",
    "DevelopmentEvidenceError",
    "DevelopmentEvidenceReceipt",
    "DevelopmentProvenance",
    "assert_development_gate",
    "verify_development_evidence",
]
