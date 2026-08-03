"""Production admission of immutable paired held-out evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter, ValidationError

from muscle_memory.coordinator import (
    CoordinatorStore,
    EpisodeState,
    HeldOutEvaluationArtifact,
    HeldOutEvaluationEpisodeMetadata,
    HeldOutEvaluationResult,
)
from muscle_memory.coordinator.models import canonical_json, require_hash, sha256_text
from muscle_memory.evaluation.heldout import load_heldout_worlds
from muscle_memory.evaluation.promotion import PromotionDecision, evaluate_promotion
from muscle_memory.evaluation.runner import STOPPED_SPEED_MPS, PolicyEpisodeResult
from muscle_memory.evaluation.success import SAFE_DELIVERY_CRITERIA, FailureReason
from muscle_memory.graph_memory import EvaluatedPolicyVersion
from muscle_memory.paths import HELDOUT_WORLDS_BUNDLE, REPOSITORY_ROOT
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.policy.network import BehaviorClonedPolicy
from muscle_memory.robot.identity import verify_mm01_bundle

MAX_EVALUATION_ARTIFACT_BYTES = 16 * 1024 * 1024
HELD_OUT_WORLD_SET_ID = "heldout-v1"
_RESULTS = TypeAdapter(tuple[PolicyEpisodeResult, ...])
_REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "heldout_bundle_sha256",
    "candidate_checkpoint_sha256",
    "baseline_results",
    "candidate_results",
    "promotion_decision",
}


class HeldOutEvaluationAdmissionError(ValueError):
    """An artifact cannot be proven to be the frozen paired evaluation output."""


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationAdmissionReceipt:
    artifact_hash: str
    baseline_policy_id: str
    candidate_policy_id: str
    paired_world_count: int
    promotable: bool
    stable_alias: str


def canonical_artifact_sha256(path: Path) -> str:
    """Hash the parsed canonical object, independent of JSON whitespace."""

    _payload, encoded = _load_canonical_artifact(path)
    return sha256_text(encoded)


def admit_held_out_evaluation(
    coordinator: CoordinatorStore,
    *,
    artifact_path: Path,
    expected_artifact_hash: str,
    candidate_checkpoint_path: Path,
    evaluated_at: datetime,
    stable_alias: str = "stable",
) -> HeldOutEvaluationAdmissionReceipt:
    """Verify, recompute, and durably admit one exact 20-world paired run."""

    require_hash(expected_artifact_hash, "expected_artifact_hash")
    payload, artifact_json = _load_canonical_artifact(artifact_path)
    artifact_hash = sha256_text(artifact_json)
    if artifact_hash != expected_artifact_hash:
        raise HeldOutEvaluationAdmissionError(
            "held-out artifact does not match the independently configured canonical hash"
        )
    if set(payload) != _REQUIRED_ARTIFACT_FIELDS or payload.get("schema_version") != 1:
        raise HeldOutEvaluationAdmissionError(
            "held-out artifact has unsupported, missing, or unexpected fields"
        )
    if payload.get("heldout_bundle_sha256") != _sha256_file(HELDOUT_WORLDS_BUNDLE):
        raise HeldOutEvaluationAdmissionError(
            "held-out artifact does not reference the frozen world bundle"
        )

    try:
        baseline_results = _RESULTS.validate_python(payload["baseline_results"])
        candidate_results = _RESULTS.validate_python(payload["candidate_results"])
    except ValidationError as exc:
        raise HeldOutEvaluationAdmissionError(
            "held-out artifact contains invalid measured episode results"
        ) from exc
    _validate_frozen_pairs(baseline_results, candidate_results)
    for result in (*baseline_results, *candidate_results):
        _validate_canonical_outcome(result)

    baseline_policy = DirectGoalPolicy()
    if (
        baseline_results[0].policy_id != baseline_policy.policy_id
        or baseline_results[0].policy_hash != baseline_policy.policy_hash
    ):
        raise HeldOutEvaluationAdmissionError(
            "held-out baseline is not the deterministic direct-goal policy"
        )
    try:
        candidate_policy = BehaviorClonedPolicy.load(candidate_checkpoint_path)
    except (OSError, ValueError) as exc:
        raise HeldOutEvaluationAdmissionError(
            "candidate checkpoint could not be independently verified"
        ) from exc
    declared_checkpoint_hash = payload.get("candidate_checkpoint_sha256")
    if (
        declared_checkpoint_hash != _sha256_file(candidate_checkpoint_path)
        or candidate_results[0].policy_id != candidate_policy.policy_id
        or candidate_results[0].policy_hash != candidate_policy.policy_hash
        or declared_checkpoint_hash != candidate_policy.policy_hash
    ):
        raise HeldOutEvaluationAdmissionError(
            "candidate checkpoint identity does not match the held-out artifact"
        )

    decision = evaluate_promotion(baseline_results, candidate_results)
    if payload.get("promotion_decision") != asdict(decision):
        raise HeldOutEvaluationAdmissionError(
            "declared promotion decision does not equal recomputed episode measurements"
        )

    artifact = HeldOutEvaluationArtifact(
        artifact_hash=artifact_hash,
        held_out_world_set_id=HELD_OUT_WORLD_SET_ID,
        artifact_json=artifact_json,
        evaluated_at=evaluated_at,
    )
    coordinator.record_held_out_evaluation_artifact(artifact)
    baseline_checkpoint = _checkpoint_from_summary(
        decision,
        candidate=False,
        evaluation_evidence_hash=artifact_hash,
        evaluated_at=artifact.evaluated_at,
    )
    candidate_checkpoint = _checkpoint_from_summary(
        decision,
        candidate=True,
        evaluation_evidence_hash=artifact_hash,
        evaluated_at=artifact.evaluated_at,
    )
    _register_or_verify_checkpoint(coordinator, baseline_checkpoint)
    coordinator.register_evaluated_checkpoint(candidate_checkpoint)

    for result in (*baseline_results, *candidate_results):
        metadata = HeldOutEvaluationEpisodeMetadata(
            episode_id=result.episode_id,
            robot_checksum=result.robot_checksum,
            world_hash=result.world_hash,
            policy_hash=result.policy_hash,
            held_out_world_set_id=HELD_OUT_WORLD_SET_ID,
            created_at=artifact.evaluated_at,
        )
        coordinator.register_held_out_evaluation_episode(metadata)
        state = coordinator.episode_state(result.episode_id)
        if state is EpisodeState.CREATED:
            coordinator.transition_episode(
                result.episode_id,
                EpisodeState.RUNNING,
                occurred_at=artifact.evaluated_at,
            )
            state = EpisodeState.RUNNING
        terminal = EpisodeState.SUCCEEDED if result.success else EpisodeState.FAILED
        if state is EpisodeState.RUNNING:
            coordinator.transition_episode(
                result.episode_id,
                terminal,
                occurred_at=artifact.evaluated_at,
            )
        elif state is not terminal:
            raise HeldOutEvaluationAdmissionError(
                "existing held-out episode state conflicts with the measured result"
            )
        coordinator.record_held_out_evaluation_result(
            HeldOutEvaluationResult(
                episode_id=result.episode_id,
                evaluation_artifact_hash=artifact_hash,
                result_json=canonical_json(asdict(result)),
            )
        )

    if coordinator.current_policy(stable_alias) is None:
        coordinator.initialize_policy_alias(
            stable_alias,
            baseline_checkpoint.policy_id,
            occurred_at=artifact.evaluated_at,
        )
    return HeldOutEvaluationAdmissionReceipt(
        artifact_hash=artifact_hash,
        baseline_policy_id=baseline_checkpoint.policy_id,
        candidate_policy_id=candidate_checkpoint.policy_id,
        paired_world_count=len(baseline_results),
        promotable=decision.promotable,
        stable_alias=stable_alias,
    )


def admit_held_out_evaluation_from_env(
    coordinator: CoordinatorStore,
    environ: Mapping[str, str],
) -> HeldOutEvaluationAdmissionReceipt | None:
    """Opt-in production startup path; partial configuration fails closed."""

    names = (
        "MM_HELDOUT_EVALUATION_ARTIFACT",
        "MM_HELDOUT_EVALUATION_ARTIFACT_SHA256",
        "MM_HELDOUT_CANDIDATE_CHECKPOINT",
        "MM_HELDOUT_EVALUATED_AT",
    )
    configured = {name: environ.get(name, "").strip() for name in names}
    if not any(configured.values()):
        return None
    missing = tuple(name for name, value in configured.items() if not value)
    if missing:
        raise HeldOutEvaluationAdmissionError(
            f"held-out evaluation admission is missing {', '.join(missing)}"
        )
    try:
        evaluated_at = datetime.fromisoformat(configured["MM_HELDOUT_EVALUATED_AT"])
    except ValueError as exc:
        raise HeldOutEvaluationAdmissionError(
            "MM_HELDOUT_EVALUATED_AT must be an ISO-8601 timestamp"
        ) from exc
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise HeldOutEvaluationAdmissionError(
            "MM_HELDOUT_EVALUATED_AT must include a timezone"
        )
    return admit_held_out_evaluation(
        coordinator,
        artifact_path=_repository_path(configured["MM_HELDOUT_EVALUATION_ARTIFACT"]),
        expected_artifact_hash=configured[
            "MM_HELDOUT_EVALUATION_ARTIFACT_SHA256"
        ],
        candidate_checkpoint_path=_repository_path(
            configured["MM_HELDOUT_CANDIDATE_CHECKPOINT"]
        ),
        evaluated_at=evaluated_at,
        stable_alias=environ.get("MM_STABLE_POLICY_ALIAS", "stable").strip() or "stable",
    )


def _validate_frozen_pairs(
    baseline: tuple[PolicyEpisodeResult, ...],
    candidate: tuple[PolicyEpisodeResult, ...],
) -> None:
    worlds = load_heldout_worlds()
    verification = verify_mm01_bundle()
    if not verification.valid or not verification.qualified:
        raise HeldOutEvaluationAdmissionError("qualified MM-01 verification failed")
    if len(baseline) != 20 or len(candidate) != 20 or len(worlds) != 20:
        raise HeldOutEvaluationAdmissionError(
            "held-out admission requires exactly twenty paired worlds"
        )
    if len({item.episode_id for item in (*baseline, *candidate)}) != 40:
        raise HeldOutEvaluationAdmissionError("held-out episode ids must be unique")
    if (
        len({(item.policy_id, item.policy_hash) for item in baseline}) != 1
        or len({(item.policy_id, item.policy_hash) for item in candidate}) != 1
        or baseline[0].policy_id == candidate[0].policy_id
        or baseline[0].policy_hash == candidate[0].policy_hash
    ):
        raise HeldOutEvaluationAdmissionError(
            "held-out results require two distinct immutable policy identities"
        )
    for index, (world, baseline_result, candidate_result) in enumerate(
        zip(worlds, baseline, candidate, strict=True)
    ):
        world_hash = hashlib.sha256(
            world.world.model_dump_json().encode("utf-8")
        ).hexdigest()
        expected = (
            world.world.world_id,
            world.world.seed,
            world_hash,
            verification.robot_checksum,
            "held_out",
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
            raise HeldOutEvaluationAdmissionError(
                f"held-out pair {index} does not match the frozen world and robot identity"
            )


def _checkpoint_from_summary(
    decision: PromotionDecision,
    *,
    candidate: bool,
    evaluation_evidence_hash: str,
    evaluated_at: datetime,
) -> EvaluatedPolicyVersion:
    summary = decision.candidate if candidate else decision.baseline
    metrics = asdict(summary)
    metrics.update(
        {
            "falls": summary.total_falls,
            "median_clearance_m": summary.median_minimum_clearance_m,
            "path_efficiency_regression_fraction": (
                decision.path_efficiency_regression if candidate else 0.0
            ),
        }
    )
    return EvaluatedPolicyVersion.create(
        policy_id=summary.policy_id,
        checkpoint_hash=summary.policy_hash,
        evaluation_evidence_hash=evaluation_evidence_hash,
        evaluation_split="held_out",
        metrics=metrics,
        evaluated_at=evaluated_at,
    )


def _register_or_verify_checkpoint(
    coordinator: CoordinatorStore,
    checkpoint: EvaluatedPolicyVersion,
) -> None:
    """Keep a checkpoint immutable while allowing it in later comparisons."""

    existing = next(
        (
            item
            for item in coordinator.evaluated_checkpoints()
            if item.policy_id == checkpoint.policy_id
        ),
        None,
    )
    if existing is None:
        coordinator.register_evaluated_checkpoint(checkpoint)
        return
    if (
        existing.checkpoint_hash != checkpoint.checkpoint_hash
        or existing.evaluation_split != checkpoint.evaluation_split
    ):
        raise HeldOutEvaluationAdmissionError(
            "existing baseline checkpoint identity conflicts with the held-out artifact"
        )


def _validate_canonical_outcome(result: PolicyEpisodeResult) -> None:
    """Recompute the terminal decision from the artifact's measured fields."""

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
        raise HeldOutEvaluationAdmissionError(
            "held-out result contains an invalid non-negative measurement"
        )
    if (
        result.time_to_resident_seconds is not None
        and (
            result.time_to_resident_seconds < 0
            or result.time_to_resident_seconds > result.simulated_duration_seconds
        )
    ):
        raise HeldOutEvaluationAdmissionError(
            "held-out result has an invalid measured completion time"
        )
    if (
        result.facing_error_degrees is not None
        and not 0.0 <= result.facing_error_degrees <= 180.0
    ) or result.path_efficiency > 1.0:
        raise HeldOutEvaluationAdmissionError(
            "held-out result has an invalid measured angle or path efficiency"
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
    if (
        result.minimum_obstacle_clearance_m
        < criteria.minimum_obstacle_clearance_metres
    ):
        failed.append(FailureReason.INSUFFICIENT_OBSTACLE_CLEARANCE.name)
    if (
        result.maximum_tray_tilt_degrees
        >= criteria.maximum_tray_tilt_degrees_exclusive
    ):
        failed.append(FailureReason.EXCESSIVE_TRAY_TILT.name)
    if result.package_slipped and not criteria.package_slip_allowed:
        failed.append(FailureReason.PACKAGE_SLIPPED.name)
    if result.human_interventions > criteria.maximum_human_interventions:
        failed.append(FailureReason.HUMAN_INTERVENTION.name)

    expected_reasons = tuple(failed)
    if result.failed_reasons != expected_reasons or result.success != (not failed):
        raise HeldOutEvaluationAdmissionError(
            "held-out result success and failed_reasons do not match canonical measurements"
        )


def _load_canonical_artifact(path: Path) -> tuple[dict[str, object], str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise HeldOutEvaluationAdmissionError("held-out artifact is unavailable") from exc
    if size <= 0 or size > MAX_EVALUATION_ARTIFACT_BYTES:
        raise HeldOutEvaluationAdmissionError("held-out artifact size is invalid")
    try:
        raw = path.read_text(encoding="utf-8")
        decoded = json.loads(raw, object_pairs_hook=_unique_object)
        encoded = canonical_json(decoded)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HeldOutEvaluationAdmissionError(
            "held-out artifact is not a finite JSON object with unique keys"
        ) from exc
    if not isinstance(decoded, dict):
        raise HeldOutEvaluationAdmissionError("held-out artifact must be a JSON object")
    return cast(dict[str, object], decoded), encoded


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HeldOutEvaluationAdmissionError(f"could not hash {path.name}") from exc
    return digest.hexdigest()


def _repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


__all__ = [
    "HeldOutEvaluationAdmissionError",
    "HeldOutEvaluationAdmissionReceipt",
    "admit_held_out_evaluation",
    "admit_held_out_evaluation_from_env",
    "canonical_artifact_sha256",
]
