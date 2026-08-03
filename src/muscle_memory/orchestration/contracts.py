"""Immutable contracts shared by Guild reasoning and RocketRide execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast


class ContractViolationError(ValueError):
    """An orchestration request violates a structural safety invariant."""


class GuildRole(StrEnum):
    WORLD_AND_PHYSICS = "World and Physics Agent"
    FAILURE_AND_CURRICULUM = "Failure and Curriculum Agent"
    SAFETY_AND_EVALUATION = "Safety and Evaluation Agent"


EXACT_GUILD_ROLES = (
    GuildRole.WORLD_AND_PHYSICS,
    GuildRole.FAILURE_AND_CURRICULUM,
    GuildRole.SAFETY_AND_EVALUATION,
)


@dataclass(frozen=True, slots=True)
class GuildRoster:
    roles: tuple[GuildRole, ...] = EXACT_GUILD_ROLES

    def __post_init__(self) -> None:
        if self.roles != EXACT_GUILD_ROLES:
            raise ContractViolationError(
                "Guild roster must contain the exact three specialist roles"
            )


class PipelineStep(StrEnum):
    VALIDATE_WORLD = "validate_world"
    RUN_EPISODE = "run_episode"
    SUMMARIZE_TELEMETRY = "summarize_telemetry"
    QUERY_GRAPH_MEMORY = "query_graph_memory"
    SELECT_CURRICULUM = "select_curriculum"
    TRAIN_CANDIDATE_POLICY = "train_candidate_policy"
    EVALUATE_CANDIDATE_POLICY = "evaluate_candidate_policy"
    PROMOTE_OR_ROLL_BACK = "promote_or_roll_back"


FIXED_PIPELINE = (
    PipelineStep.VALIDATE_WORLD,
    PipelineStep.RUN_EPISODE,
    PipelineStep.SUMMARIZE_TELEMETRY,
    PipelineStep.QUERY_GRAPH_MEMORY,
    PipelineStep.SELECT_CURRICULUM,
    PipelineStep.TRAIN_CANDIDATE_POLICY,
    PipelineStep.EVALUATE_CANDIDATE_POLICY,
    PipelineStep.PROMOTE_OR_ROLL_BACK,
)


class ApprovalKind(StrEnum):
    UNCERTAIN_PHYSICAL_PROPERTIES = "uncertain_physical_properties"
    REWARD_CHANGE = "reward_change"
    CURRICULUM_CHANGE = "curriculum_change"
    POLICY_PROMOTION = "policy_promotion"
    POLICY_ROLLBACK = "policy_rollback"


class ProviderName(StrEnum):
    GUILD = "guild.ai"
    ROCKETRIDE = "rocketride.ai"


class ProviderMode(StrEnum):
    UNCONFIGURED = "unconfigured"
    LIVE = "live"
    SIMULATION = "simulation"
    CACHED = "cached"


class HealthState(StrEnum):
    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    END_TO_END_VERIFIED = "end_to_end_verified"


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    provider: ProviderName
    mode: ProviderMode
    health: HealthState
    detail: str
    checked_at: datetime

    @classmethod
    def unconfigured(cls, provider: ProviderName, detail: str) -> ProviderStatus:
        return cls(
            provider=provider,
            mode=ProviderMode.UNCONFIGURED,
            health=HealthState.UNCONFIGURED,
            detail=detail,
            checked_at=datetime.now(UTC),
        )


@dataclass(frozen=True, slots=True)
class FallbackRecord:
    provider: ProviderName
    mode: ProviderMode
    reason: str
    source_digest: str

    def __post_init__(self) -> None:
        if self.mode not in (ProviderMode.CACHED, ProviderMode.SIMULATION):
            raise ContractViolationError("fallback mode must be cached or simulation")
        if not self.reason or not self.source_digest:
            raise ContractViolationError("fallbacks require a reason and source digest")


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolationError("orchestration payload must be JSON serializable") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PipelineCommand:
    step: PipelineStep
    payload_json: str
    payload_sha256: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ContractViolationError("command payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ContractViolationError("command payload must be a JSON object")
        if canonical_json(payload) != self.payload_json:
            raise ContractViolationError("command payload is not canonically encoded")
        if sha256_text(self.payload_json) != self.payload_sha256:
            raise ContractViolationError("command payload checksum mismatch")
        self._validate_step_payload(payload)

    @classmethod
    def create(cls, step: PipelineStep, payload: Mapping[str, object]) -> PipelineCommand:
        payload_json = canonical_json(payload)
        return cls(step=step, payload_json=payload_json, payload_sha256=sha256_text(payload_json))

    @property
    def payload(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.payload_json))

    def _validate_step_payload(self, payload: Mapping[str, object]) -> None:
        if self.step is PipelineStep.EVALUATE_CANDIDATE_POLICY:
            allowed = {
                "baseline_policy_id",
                "candidate_policy_id",
                "heldout_world_set_id",
            }
            if set(payload) != allowed:
                raise ContractViolationError(
                    "evaluation accepts only policy ids and the held-out world-set id"
                )
            if not all(isinstance(payload[key], str) and payload[key] for key in allowed):
                raise ContractViolationError("evaluation identifiers must be non-empty strings")
        if self.step is PipelineStep.VALIDATE_WORLD:
            _require_boolean(payload, "uncertain_physical_properties")
        if self.step is PipelineStep.SELECT_CURRICULUM:
            _require_boolean(payload, "curriculum_change_requested")
        if self.step is PipelineStep.TRAIN_CANDIDATE_POLICY:
            _require_boolean(payload, "reward_change_requested")
        if self.step is PipelineStep.PROMOTE_OR_ROLL_BACK:
            action = payload.get("action")
            if action not in {"promote", "roll_back"}:
                raise ContractViolationError("final pipeline action must be promote or roll_back")


def _require_boolean(payload: Mapping[str, object], key: str) -> None:
    if not isinstance(payload.get(key), bool):
        raise ContractViolationError(f"{key} must be an explicit boolean")


@dataclass(frozen=True, slots=True)
class ApprovalRequirement:
    requirement_id: str
    run_id: str
    plan_digest: str
    step: PipelineStep
    kind: ApprovalKind
    summary: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    run_id: str
    commands: tuple[PipelineCommand, ...]
    digest: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ContractViolationError("run_id must not be empty")
        if tuple(command.step for command in self.commands) != FIXED_PIPELINE:
            raise ContractViolationError("RocketRide plan must use the fixed eight-step pipeline")
        expected = self.calculate_digest(self.run_id, self.commands)
        if self.digest != expected:
            raise ContractViolationError("execution-plan digest mismatch")

    @classmethod
    def create(cls, run_id: str, commands: tuple[PipelineCommand, ...]) -> ExecutionPlan:
        return cls(
            run_id=run_id,
            commands=commands,
            digest=cls.calculate_digest(run_id, commands),
        )

    @staticmethod
    def calculate_digest(run_id: str, commands: tuple[PipelineCommand, ...]) -> str:
        envelope = {
            "run_id": run_id,
            "commands": [
                {
                    "step": command.step.value,
                    "payload_sha256": command.payload_sha256,
                }
                for command in commands
            ],
        }
        return sha256_text(canonical_json(envelope))

    @property
    def approval_requirements(self) -> tuple[ApprovalRequirement, ...]:
        requirements: list[ApprovalRequirement] = []
        for command in self.commands:
            payload = command.payload
            kind: ApprovalKind | None = None
            summary = ""
            if (
                command.step is PipelineStep.VALIDATE_WORLD
                and payload["uncertain_physical_properties"]
            ):
                kind = ApprovalKind.UNCERTAIN_PHYSICAL_PROPERTIES
                summary = "Approve uncertain or proposed physical properties"
            elif (
                command.step is PipelineStep.SELECT_CURRICULUM
                and payload["curriculum_change_requested"]
            ):
                kind = ApprovalKind.CURRICULUM_CHANGE
                summary = "Approve the proposed curriculum change"
            elif (
                command.step is PipelineStep.TRAIN_CANDIDATE_POLICY
                and payload["reward_change_requested"]
            ):
                kind = ApprovalKind.REWARD_CHANGE
                summary = "Approve the proposed reward-function change"
            elif command.step is PipelineStep.PROMOTE_OR_ROLL_BACK:
                if payload["action"] == "promote":
                    kind = ApprovalKind.POLICY_PROMOTION
                    summary = "Approve policy promotion"
                else:
                    kind = ApprovalKind.POLICY_ROLLBACK
                    summary = "Approve policy rollback"
            if kind is not None:
                requirement_id = sha256_text(
                    f"{self.digest}:{command.step.value}:{kind.value}"
                )
                requirements.append(
                    ApprovalRequirement(
                        requirement_id=requirement_id,
                        run_id=self.run_id,
                        plan_digest=self.digest,
                        step=command.step,
                        kind=kind,
                        summary=summary,
                    )
                )
        return tuple(requirements)


class ReviewRecommendation(StrEnum):
    PROCEED = "proceed"
    REVISE = "revise"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class GuildReview:
    role: GuildRole
    plan_digest: str
    recommendation: ReviewRecommendation
    summary: str
    requested_approvals: tuple[ApprovalKind, ...] = ()
    provider_session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.plan_digest or not self.summary:
            raise ContractViolationError("Guild reviews require a plan digest and summary")
        if self.provider_session_id is not None and (
            not self.provider_session_id
            or len(self.provider_session_id) > 256
            or any(
                character.isspace() or not character.isprintable()
                for character in self.provider_session_id
            )
        ):
            raise ContractViolationError("Guild provider session id is invalid")


@dataclass(frozen=True, slots=True)
class GuildReviewSet:
    plan_digest: str
    reviews: tuple[GuildReview, ...]
    provider_status: ProviderStatus
    fallback: FallbackRecord | None = None

    def __post_init__(self) -> None:
        if tuple(review.role for review in self.reviews) != EXACT_GUILD_ROLES:
            raise ContractViolationError(
                "review set must contain one review from each Guild role"
            )
        if any(review.plan_digest != self.plan_digest for review in self.reviews):
            raise ContractViolationError("Guild review digest does not match the reviewed plan")
        if (
            self.provider_status.mode is ProviderMode.LIVE
            and self.provider_status.health
            in {HealthState.HEALTHY, HealthState.END_TO_END_VERIFIED}
        ):
            session_ids = tuple(review.provider_session_id for review in self.reviews)
            if any(session_id is None for session_id in session_ids):
                raise ContractViolationError(
                    "healthy live Guild reviews require retained provider session ids"
                )
            if len(set(session_ids)) != len(session_ids):
                raise ContractViolationError(
                    "healthy live Guild reviews require distinct provider session ids"
                )

    @property
    def blocked(self) -> bool:
        return any(review.recommendation is ReviewRecommendation.BLOCK for review in self.reviews)

    @property
    def executable(self) -> bool:
        provider_authorized = (
            self.provider_status.mode is ProviderMode.LIVE
            and self.provider_status.health
            in {HealthState.HEALTHY, HealthState.END_TO_END_VERIFIED}
        ) or (
            self.provider_status.mode is ProviderMode.SIMULATION
            and self.provider_status.health is HealthState.HEALTHY
        ) or (
            self.provider_status.mode is ProviderMode.CACHED
            and self.provider_status.health is HealthState.DEGRADED
            and self.fallback is not None
        )
        return provider_authorized and all(
            review.recommendation is ReviewRecommendation.PROCEED
            for review in self.reviews
        )
