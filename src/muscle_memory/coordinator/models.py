"""Immutable records for the durable coordinator store."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from muscle_memory.evaluation.promotion import (
    MAXIMUM_PATH_EFFICIENCY_REGRESSION,
    MAXIMUM_V1_COLLISION_RATE,
    MINIMUM_COLLISION_RATE_REDUCTION,
    MINIMUM_SUCCESS_RATE_IMPROVEMENT,
    MINIMUM_V1_MEDIAN_CLEARANCE_M,
    MINIMUM_V1_SUCCESS_RATE,
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class CoordinatorIntegrityError(RuntimeError):
    """An immutable identity was reused for different content."""


class CoordinatorStateError(RuntimeError):
    """A requested lifecycle transition is not allowed."""


class ApprovalRequiredError(CoordinatorStateError):
    """A gated operation does not have a matching approved human decision."""


class EpisodeKind(StrEnum):
    TRAINING = "training"
    HELD_OUT_EVALUATION = "held_out_evaluation"


class EpisodeState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


TERMINAL_EPISODE_STATES = frozenset(
    {EpisodeState.SUCCEEDED, EpisodeState.FAILED, EpisodeState.ABORTED}
)


class WorkflowStepState(StrEnum):
    AWAITING_APPROVAL = "awaiting_approval"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PolicyAction(StrEnum):
    PROMOTE = "promote"
    ROLL_BACK = "roll_back"


def canonical_json(value: object) -> str:
    """Encode a coordinator fact deterministically."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinator payload must be JSON serializable") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("coordinator timestamps must be timezone-aware")
    return value.isoformat()


def require_identifier(value: str, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable non-empty identifier")


def require_hash(value: str, name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class TrainingEpisodeMetadata:
    """Training-facing episode metadata, intentionally without held-out identifiers."""

    episode_id: str
    robot_checksum: str
    world_hash: str
    policy_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_identifier(self.episode_id, "episode_id")
        require_hash(self.robot_checksum, "robot_checksum")
        require_hash(self.world_hash, "world_hash")
        require_hash(self.policy_hash, "policy_hash")
        isoformat_utc(self.created_at)

    @property
    def content_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "episode_id": self.episode_id,
                    "kind": EpisodeKind.TRAINING.value,
                    "robot_checksum": self.robot_checksum,
                    "world_hash": self.world_hash,
                    "policy_hash": self.policy_hash,
                    "created_at": isoformat_utc(self.created_at),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationEpisodeMetadata:
    """Evaluation-only metadata kept behind a separate store API and table."""

    episode_id: str
    robot_checksum: str
    world_hash: str
    policy_hash: str
    held_out_world_set_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_identifier(self.episode_id, "episode_id")
        require_hash(self.robot_checksum, "robot_checksum")
        require_hash(self.world_hash, "world_hash")
        require_hash(self.policy_hash, "policy_hash")
        require_identifier(self.held_out_world_set_id, "held_out_world_set_id")
        isoformat_utc(self.created_at)

    @property
    def content_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "episode_id": self.episode_id,
                    "kind": EpisodeKind.HELD_OUT_EVALUATION.value,
                    "robot_checksum": self.robot_checksum,
                    "world_hash": self.world_hash,
                    "policy_hash": self.policy_hash,
                    "held_out_world_set_id": self.held_out_world_set_id,
                    "created_at": isoformat_utc(self.created_at),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationArtifact:
    """Canonical, independently verified output of one paired held-out run."""

    artifact_hash: str
    held_out_world_set_id: str
    artifact_json: str
    evaluated_at: datetime

    def __post_init__(self) -> None:
        require_hash(self.artifact_hash, "artifact_hash")
        require_identifier(self.held_out_world_set_id, "held_out_world_set_id")
        isoformat_utc(self.evaluated_at)
        try:
            decoded = json.loads(self.artifact_json)
        except json.JSONDecodeError as exc:
            raise ValueError("held-out evaluation artifact is not valid JSON") from exc
        if not isinstance(decoded, dict) or canonical_json(decoded) != self.artifact_json:
            raise ValueError("held-out evaluation artifact must be a canonical JSON object")
        if sha256_text(self.artifact_json) != self.artifact_hash:
            raise ValueError("held-out evaluation artifact hash does not match its content")

    @property
    def content_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "artifact_hash": self.artifact_hash,
                    "held_out_world_set_id": self.held_out_world_set_id,
                    "artifact_json": self.artifact_json,
                    "evaluated_at": isoformat_utc(self.evaluated_at),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationResult:
    """One immutable measured result admitted from a verified evaluation artifact."""

    episode_id: str
    evaluation_artifact_hash: str
    result_json: str

    def __post_init__(self) -> None:
        require_identifier(self.episode_id, "episode_id")
        require_hash(self.evaluation_artifact_hash, "evaluation_artifact_hash")
        try:
            decoded = json.loads(self.result_json)
        except json.JSONDecodeError as exc:
            raise ValueError("held-out evaluation result is not valid JSON") from exc
        if not isinstance(decoded, dict) or canonical_json(decoded) != self.result_json:
            raise ValueError("held-out evaluation result must be a canonical JSON object")
        if decoded.get("episode_id") != self.episode_id:
            raise ValueError("held-out evaluation result identity does not match its payload")

    @property
    def content_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "episode_id": self.episode_id,
                    "evaluation_artifact_hash": self.evaluation_artifact_hash,
                    "result_json": self.result_json,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class ProviderEvidenceReference:
    """Content-addressed pointer proving one provider-side observation."""

    evidence_id: str
    provider: str
    evidence_kind: str
    provider_object_id: str
    artifact_hash: str
    observed_at: datetime

    def __post_init__(self) -> None:
        require_identifier(self.evidence_id, "evidence_id")
        require_identifier(self.evidence_kind, "evidence_kind")
        require_hash(self.artifact_hash, "artifact_hash")
        if not self.provider.strip() or not self.provider_object_id.strip():
            raise ValueError("provider evidence requires provider and provider_object_id")
        isoformat_utc(self.observed_at)

    @property
    def content_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "evidence_id": self.evidence_id,
                    "provider": self.provider,
                    "evidence_kind": self.evidence_kind,
                    "provider_object_id": self.provider_object_id,
                    "artifact_hash": self.artifact_hash,
                    "observed_at": isoformat_utc(self.observed_at),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    episode_id: str
    sequence: int
    state: EpisodeState
    occurred_at: datetime
    details_json: str
    evidence_id: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class WorkflowStepAudit:
    run_id: str
    sequence: int
    step: str
    state: WorkflowStepState
    occurred_at: datetime
    details_json: str
    evidence_id: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class PolicyGateMetrics:
    """Numeric evidence used by the fixed policy-promotion gate."""

    held_out_success_rate: float
    collision_rate: float
    fall_count: int
    median_clearance_m: float
    success_rate_delta: float
    collision_reduction_fraction: float
    path_efficiency_regression_fraction: float

    def __post_init__(self) -> None:
        float_values = (
            self.held_out_success_rate,
            self.collision_rate,
            self.median_clearance_m,
            self.success_rate_delta,
            self.collision_reduction_fraction,
            self.path_efficiency_regression_fraction,
        )
        if not all(math.isfinite(value) for value in float_values):
            raise ValueError("policy gate metrics must be finite")
        if not 0.0 <= self.held_out_success_rate <= 1.0:
            raise ValueError("held_out_success_rate must be between zero and one")
        if not 0.0 <= self.collision_rate <= 1.0:
            raise ValueError("collision_rate must be between zero and one")
        if self.fall_count < 0 or self.median_clearance_m < 0.0:
            raise ValueError("fall_count and median_clearance_m must be non-negative")

    @property
    def passes_promotion_gate(self) -> bool:
        improvement_passed = (
            self.success_rate_delta >= MINIMUM_SUCCESS_RATE_IMPROVEMENT
            or self.collision_reduction_fraction >= MINIMUM_COLLISION_RATE_REDUCTION
        )
        return (
            self.held_out_success_rate >= MINIMUM_V1_SUCCESS_RATE
            and self.fall_count == 0
            and self.collision_rate <= MAXIMUM_V1_COLLISION_RATE
            and self.median_clearance_m >= MINIMUM_V1_MEDIAN_CLEARANCE_M
            and improvement_passed
            and self.path_efficiency_regression_fraction
            <= MAXIMUM_PATH_EFFICIENCY_REGRESSION
        )

    def as_mapping(self) -> Mapping[str, int | float]:
        return {
            "held_out_success_rate": self.held_out_success_rate,
            "collision_rate": self.collision_rate,
            "fall_count": self.fall_count,
            "median_clearance_m": self.median_clearance_m,
            "success_rate_delta": self.success_rate_delta,
            "collision_reduction_fraction": self.collision_reduction_fraction,
            "path_efficiency_regression_fraction": (
                self.path_efficiency_regression_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class NumericPolicyDecision:
    """Immutable measured action; human approval is intentionally not embedded."""

    decision_id: str
    run_id: str
    plan_digest: str
    action: PolicyAction
    alias: str
    from_policy_id: str | None
    target_policy_id: str
    evaluation_evidence_hash: str
    metrics: PolicyGateMetrics
    decided_at: datetime

    def __post_init__(self) -> None:
        require_identifier(self.decision_id, "decision_id")
        require_identifier(self.run_id, "run_id")
        require_hash(self.plan_digest, "plan_digest")
        require_identifier(self.alias, "alias")
        if self.from_policy_id is not None:
            require_identifier(self.from_policy_id, "from_policy_id")
        require_identifier(self.target_policy_id, "target_policy_id")
        require_hash(self.evaluation_evidence_hash, "evaluation_evidence_hash")
        isoformat_utc(self.decided_at)
        if (
            self.action is PolicyAction.PROMOTE
            and self.from_policy_id == self.target_policy_id
        ):
            raise ValueError("policy promotion must move the alias to a new checkpoint")
        if self.action is PolicyAction.PROMOTE and not self.metrics.passes_promotion_gate:
            raise ValueError("a promotion decision must pass the numeric promotion gate")

    @property
    def content_hash(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "decision_id": self.decision_id,
                    "run_id": self.run_id,
                    "plan_digest": self.plan_digest,
                    "action": self.action.value,
                    "alias": self.alias,
                    "from_policy_id": self.from_policy_id,
                    "target_policy_id": self.target_policy_id,
                    "evaluation_evidence_hash": self.evaluation_evidence_hash,
                    "metrics": self.metrics.as_mapping(),
                    "promotion_gate_passed": self.metrics.passes_promotion_gate,
                    "decided_at": isoformat_utc(self.decided_at),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class PolicyAliasEvent:
    alias: str
    sequence: int
    target_policy_id: str
    action: str
    occurred_at: datetime
    numeric_decision_id: str | None
    approval_requirement_id: str | None
    content_hash: str
