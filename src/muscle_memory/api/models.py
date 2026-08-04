"""Versioned HTTP and live-stream contracts for the Muscle Memory backend."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from muscle_memory.api.redaction import redact_sensitive_mapping, redact_sensitive_text
from muscle_memory.orchestration.evidence import GuildEvidenceBundle

API_VERSION = "v1"
NUMERIC_TELEMETRY_HZ = 20
FRAME_JOIN_KEY = "frame_id"


class ApiModel(BaseModel):
    """Strict public model; provider implementation details stay server-side."""

    model_config = ConfigDict(extra="forbid")


class ProviderOperationalState(StrEnum):
    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    END_TO_END_VERIFIED = "end_to_end_verified"
    SIMULATION = "simulation"
    CACHED = "cached"


class ProviderHealth(ApiModel):
    provider: str = Field(min_length=1, max_length=128)
    state: ProviderOperationalState
    detail: str = Field(min_length=1, max_length=1_000)
    checked_at: AwareDatetime
    evidence_id: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("detail")
    @classmethod
    def redact_detail(cls, value: str) -> str:
        return redact_sensitive_text(value)


class ServiceHealth(ApiModel):
    service: Literal["muscle-memory-api"] = "muscle-memory-api"
    api_version: Literal["v1"] = "v1"
    state: ProviderOperationalState
    providers: tuple[ProviderHealth, ...]
    checked_at: AwareDatetime


MemoryGraphOwner = Literal[
    "system",
    "World & Physics Agent",
    "Failure & Curriculum Agent",
    "Safety & Evaluation Agent",
]


class MemoryGraphNode(ApiModel):
    id: str = Field(min_length=1, max_length=256)
    label: str = Field(min_length=1, max_length=256)
    record_kind: str = Field(min_length=1, max_length=64)
    owner: MemoryGraphOwner
    properties: dict[str, object]


class MemoryGraphEdge(ApiModel):
    id: str = Field(min_length=1, max_length=512)
    source: str = Field(min_length=1, max_length=256)
    target: str = Field(min_length=1, max_length=256)
    relationship: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")


class MemoryGraphSnapshot(ApiModel):
    exposure: Literal["operational_only"] = "operational_only"
    provider: Literal["FalkorDB"] = "FalkorDB"
    provider_state: ProviderOperationalState
    graph_name: str = Field(min_length=1, max_length=128)
    source: Literal["falkordb", "local_cache"]
    provider_checked_at: AwareDatetime
    refreshed_at: AwareDatetime
    fact_count: int = Field(ge=0)
    nodes: tuple[MemoryGraphNode, ...]
    edges: tuple[MemoryGraphEdge, ...]


class EpisodeKind(StrEnum):
    """Only non-held-out episode kinds may cross the operational API."""

    TRAINING = "training"
    DEVELOPMENT_EVALUATION = "development_evaluation"
    DEMO = "demo"


class EpisodeState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class EpisodeSummary(ApiModel):
    episode_id: str = Field(min_length=1, max_length=128)
    kind: EpisodeKind
    state: EpisodeState
    robot_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_id: str = Field(min_length=1, max_length=128)
    world_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: str = Field(min_length=1, max_length=128)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    opened_at: AwareDatetime
    closed_at: AwareDatetime | None = None


class EpisodeList(ApiModel):
    exposure: Literal["operational_only"] = "operational_only"
    items: tuple[EpisodeSummary, ...]
    next_cursor: str | None = None


class EpisodeDetail(ApiModel):
    episode: EpisodeSummary
    telemetry_records: int = Field(ge=0)
    provider_delivery: ProviderOperationalState
    result: dict[str, object] | None = None
    failure_ids: tuple[str, ...] = ()
    correction_ids: tuple[str, ...] = ()


class EpisodeReviewNote(ApiModel):
    """Operator-owned annotation; the referenced episode evidence stays immutable."""

    note_id: str = Field(pattern=r"^note-[0-9a-f]{32}$")
    episode_id: str = Field(min_length=1, max_length=128)
    author_subject: str = Field(min_length=1, max_length=128)
    body: str = Field(min_length=1, max_length=4_000)
    tags: tuple[str, ...] = Field(default=(), max_length=12)
    created_at: AwareDatetime
    updated_at: AwareDatetime
    archived: bool = False


class EpisodeReviewNoteList(ApiModel):
    episode_id: str = Field(min_length=1, max_length=128)
    items: tuple[EpisodeReviewNote, ...]


class EpisodeReviewNoteCreateRequest(ApiModel):
    body: str = Field(min_length=1, max_length=4_000)
    tags: tuple[str, ...] = Field(default=(), max_length=12)

    @field_validator("body")
    @classmethod
    def non_blank_body(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("review note body must not be blank")
        return value


class EpisodeReviewNoteUpdateRequest(ApiModel):
    body: str | None = Field(default=None, min_length=1, max_length=4_000)
    tags: tuple[str, ...] | None = Field(default=None, max_length=12)
    archived: bool | None = None

    @field_validator("body")
    @classmethod
    def non_blank_update_body(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("review note body must not be blank")
        return value

    @model_validator(mode="after")
    def has_update(self) -> Self:
        if self.body is None and self.tags is None and self.archived is None:
            raise ValueError("review note update must change at least one field")
        return self


class SensorReadingView(ApiModel):
    category: str = Field(min_length=1)
    signal_use: Literal["Used by policy", "Logged only", "Simulator ground truth"]
    available: bool
    values: object | None

    @model_validator(mode="after")
    def unavailable_values_are_null(self) -> Self:
        if not self.available and self.values is not None:
            raise ValueError("unavailable sensor values must be null")
        return self


class TelemetryRecordView(ApiModel):
    episode_id: str = Field(min_length=1, max_length=128)
    world_id: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    sim_time_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    event_time: float = Field(ge=0.0, allow_inf_nan=False)
    failure_type: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    frame_id: str | None = Field(default=None, min_length=1, max_length=256)
    frame_join_key: Literal["frame_id"] = "frame_id"
    signal_use: Literal["Used by policy", "Logged only", "Simulator ground truth"]
    sensors: tuple[SensorReadingView, ...] = Field(min_length=8, max_length=8)
    payload: dict[str, object]
    payload_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    delivery: ProviderOperationalState


class TelemetryPage(ApiModel):
    episode_id: str
    cadence_hz: Literal[20] = 20
    records: tuple[TelemetryRecordView, ...]
    next_sequence: int | None = Field(default=None, ge=0)


class ReplayPage(ApiModel):
    episode_id: str
    frame_join_key: Literal["frame_id"] = "frame_id"
    records: tuple[TelemetryRecordView, ...]
    next_sequence: int | None = Field(default=None, ge=0)


class LivePolicyOptionView(ApiModel):
    policy_id: str = Field(min_length=1, max_length=128)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated_episode_count: int = Field(gt=0)
    promotable: bool
    deployment_status: Literal["stable_deployed", "candidate_live_test"] = (
        "candidate_live_test"
    )
    is_default: bool = False


class LiveEpisodeOptionsView(ApiModel):
    enabled: bool
    unavailable_reason: str | None = Field(default=None, max_length=500)
    mode: Literal["training"] = "training"
    catalog_id: str | None = Field(default=None, min_length=1, max_length=128)
    catalog_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seeds: tuple[int, ...]
    policies: tuple[LivePolicyOptionView, ...]
    default_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    video_products: tuple[
        Literal[
            "third_person",
            "left_eye_rgb",
            "right_eye_rgb",
            "stereo_composite",
            "derived_depth",
            "simulator_debug_segmentation",
        ],
        ...,
    ]
    maximum_duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        le=30.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def capability_shape(self) -> Self:
        if self.enabled:
            if (
                self.unavailable_reason is not None
                or self.catalog_id is None
                or self.catalog_sha256 is None
                or not self.seeds
                or not self.policies
                or len(self.video_products) != 6
                or self.maximum_duration_seconds is None
            ):
                raise ValueError("enabled live episodes require complete admitted options")
        elif self.unavailable_reason is None:
            raise ValueError("disabled live episodes require an unavailable reason")
        return self


class LiveEpisodeStartRequest(ApiModel):
    seed: int = Field(ge=0, le=(2**63) - 1)
    policy_id: str = Field(min_length=1, max_length=128)


class LiveEpisodeStatusView(ApiModel):
    episode_id: str = Field(min_length=1, max_length=128)
    phase: Literal["queued", "starting", "running", "cancelling", "closed", "failed"]
    health: Literal["starting", "healthy", "degraded", "terminal", "failed"]
    world_id: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=128)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_promotable: bool
    simulation_time_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    wall_elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    wall_clock_lag_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    telemetry_records: int = Field(ge=0)
    video_frames: int = Field(ge=0)
    dropped_video_frames: int = Field(ge=0)
    last_frame_id: str | None = Field(default=None, min_length=1, max_length=256)
    provider_state: str | None = Field(default=None, max_length=128)
    completion_reason: str | None = Field(default=None, max_length=128)
    success: bool | None = None
    failed_reasons: tuple[str, ...]
    graph_provider_complete: bool | None = None
    telemetry_provider_complete: bool | None = None
    error_type: str | None = Field(default=None, max_length=128)
    detail: str | None = Field(default=None, max_length=500)
    video_streams: dict[str, str]


class TaskPolicyTrainingStartRequest(ApiModel):
    epochs: int = Field(default=180, ge=1, le=200)
    seed: int = Field(default=668, ge=0, le=(2**31) - 1)


class TaskPolicyTrainingMetricsView(ApiModel):
    training_episode_count: int = Field(gt=0)
    validation_episode_count: int = Field(gt=0)
    training_sample_count: int = Field(gt=0)
    validation_sample_count: int = Field(gt=0)
    best_epoch: int = Field(gt=0)
    training_command_accuracy: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    validation_command_accuracy: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    validation_loss: float = Field(ge=0.0, allow_inf_nan=False)
    validation_forward_mae_mps: float = Field(ge=0.0, allow_inf_nan=False)
    validation_turning_mae_rad_s: float = Field(ge=0.0, allow_inf_nan=False)
    validation_stop_mae: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TaskPolicyTrainingJobView(ApiModel):
    job_id: str = Field(pattern=r"^task-policy-[0-9a-f]{32}$")
    policy_id: str = Field(pattern=r"^local-candidate-[0-9a-f]{32}$")
    state: Literal["queued", "running", "completed", "failed"]
    epochs: int = Field(ge=1, le=200)
    seed: int = Field(ge=0, le=(2**31) - 1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_data_split: Literal["training"] = "training"
    robot_component: Literal["high_level_task_policy"] = "high_level_task_policy"
    promotion_status: Literal["not_evaluated"] = "not_evaluated"
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metrics: TaskPolicyTrainingMetricsView | None = None
    error_type: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def terminal_artifacts_match_state(self) -> Self:
        complete = self.state == "completed"
        if complete is (
            self.checkpoint_sha256 is None
            or self.evidence_sha256 is None
            or self.metrics is None
        ):
            raise ValueError("only completed training jobs carry verified artifacts")
        if self.state == "failed" and self.error_type is None:
            raise ValueError("failed training jobs require an error type")
        if self.state != "failed" and self.error_type is not None:
            raise ValueError("only failed training jobs carry an error type")
        return self


class TaskPolicyTrainingJobList(ApiModel):
    items: tuple[TaskPolicyTrainingJobView, ...]


class ApprovalKind(StrEnum):
    UNCERTAIN_PHYSICAL_PROPERTIES = "uncertain_physical_properties"
    REWARD_CHANGE = "reward_change"
    CURRICULUM_CHANGE = "curriculum_change"
    POLICY_PROMOTION = "policy_promotion"
    POLICY_ROLLBACK = "policy_rollback"
    CORRECTION = "correction"


class PendingApproval(ApiModel):
    requirement_id: str = Field(min_length=1, max_length=256)
    kind: ApprovalKind
    summary: str = Field(min_length=1, max_length=1_000)
    blocking: Literal[True] = True
    plan_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    run_id: str | None = Field(default=None, min_length=1, max_length=128)
    created_at: AwareDatetime


class PendingApprovalList(ApiModel):
    items: tuple[PendingApproval, ...]


class HumanVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class DecisionRequest(ApiModel):
    verdict: HumanVerdict
    note: str = Field(default="", max_length=2_000)


class ApprovalDecisionView(ApiModel):
    requirement_id: str
    verdict: HumanVerdict
    human_subject: str
    authentication_method: str
    decided_at: AwareDatetime
    immutable: Literal[True] = True


class WorkflowStep(StrEnum):
    VALIDATE_WORLD = "validate_world"
    RUN_EPISODE = "run_episode"
    SUMMARIZE_TELEMETRY = "summarize_telemetry"
    QUERY_GRAPH_MEMORY = "query_graph_memory"
    SELECT_CURRICULUM = "select_curriculum"
    TRAIN_CANDIDATE_POLICY = "train_candidate_policy"
    EVALUATE_CANDIDATE_POLICY = "evaluate_candidate_policy"
    PROMOTE_OR_ROLL_BACK = "promote_or_roll_back"


FIXED_WORKFLOW = tuple(WorkflowStep)


class WorkflowCommandRequest(ApiModel):
    step: WorkflowStep
    payload: dict[str, object]


class WorkflowReviewRequest(ApiModel):
    run_id: str = Field(min_length=1, max_length=128)
    commands: tuple[WorkflowCommandRequest, ...] = Field(min_length=8, max_length=8)
    evidence: GuildEvidenceBundle

    @model_validator(mode="after")
    def fixed_pipeline_only(self) -> Self:
        if tuple(command.step for command in self.commands) != FIXED_WORKFLOW:
            raise ValueError("workflow commands must use the fixed eight-step pipeline")
        return self


class SpecialistReview(ApiModel):
    role: Literal[
        "World and Physics Agent",
        "Failure and Curriculum Agent",
        "Safety and Evaluation Agent",
    ]
    recommendation: Literal["proceed", "revise", "block"]
    summary: str = Field(min_length=1, max_length=4_000)
    provider_session_id: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("summary")
    @classmethod
    def redact_summary(cls, value: str) -> str:
        return redact_sensitive_text(value)


class WorkflowReview(ApiModel):
    run_id: str
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviews: tuple[SpecialistReview, SpecialistReview, SpecialistReview]
    executable: bool
    provider: ProviderHealth


class WorkflowRunState(StrEnum):
    REVIEWED = "reviewed"
    AWAITING_HUMAN_APPROVAL = "awaiting_human_approval"
    BLOCKED = "blocked"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    CACHED = "cached"


class WorkflowStepResult(ApiModel):
    step: WorkflowStep
    state: Literal["completed", "failed", "blocked"]
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provider_task_receipt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    provider_run_id: str | None = Field(default=None, min_length=1, max_length=128)


class WorkflowRun(ApiModel):
    run_id: str
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: WorkflowRunState
    completed_steps: tuple[WorkflowStepResult, ...]
    blocked_requirement_id: str | None = None
    failure: str | None = None
    provider: ProviderHealth

    @field_validator("failure")
    @classmethod
    def redact_failure(cls, value: str | None) -> str | None:
        return None if value is None else redact_sensitive_text(value)


class CorrectionKind(StrEnum):
    ROUTE = "route"
    KEEP_OUT = "keep_out"


class CorrectionPoint(ApiModel):
    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)


class CorrectionRequest(ApiModel):
    failure_id: str = Field(min_length=1, max_length=128)
    kind: CorrectionKind
    points: tuple[CorrectionPoint, ...] = Field(min_length=2, max_length=1_000)
    description: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def enough_points(self) -> Self:
        minimum = 2 if self.kind is CorrectionKind.ROUTE else 3
        if len(self.points) < minimum:
            raise ValueError(f"{self.kind.value} corrections require at least {minimum} points")
        return self


class CorrectionState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CorrectionView(ApiModel):
    correction_id: str
    episode_id: str
    failure_id: str
    kind: CorrectionKind
    state: CorrectionState
    submitted_by: str
    created_at: AwareDatetime
    graph_delivery: ProviderOperationalState


class PolicyMetrics(ApiModel):
    episode_count: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    collision_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    falls: int = Field(ge=0)
    median_clearance_m: float = Field(ge=0.0, allow_inf_nan=False)
    median_path_efficiency: float = Field(ge=0.0, allow_inf_nan=False)


class PolicySummary(ApiModel):
    policy_id: str
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated: bool
    evaluation_scope: Literal["development", "held_out_aggregate", "none"]
    metrics: PolicyMetrics | None = None
    immutable: bool


class PolicySummaryList(ApiModel):
    items: tuple[PolicySummary, ...]


class PromotionEligibility(ApiModel):
    baseline_policy_id: str
    candidate_policy_id: str
    held_out_episode_count: int = Field(ge=0)
    checks: dict[str, bool]
    numerically_eligible: bool
    human_approval_required: Literal[True] = True
    promotion_applied: bool = False
    evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AssetStatus(ApiModel):
    asset_id: str
    state: Literal["ready", "blocked_approval", "rejected", "unavailable"]
    generation_route: Literal["live_provider", "request_cache", "verified_fallback"]
    rendering_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    collider_source: Literal["deterministic_primitive", "approved_convex"] | None = None
    approval_requirement_id: str | None = None
    providers: tuple[ProviderHealth, ...]
    detail: str

    @field_validator("detail")
    @classmethod
    def redact_detail(cls, value: str) -> str:
        return redact_sensitive_text(value)


class AssetStatusList(ApiModel):
    items: tuple[AssetStatus, ...]


class LiveMessageKind(StrEnum):
    TELEMETRY = "telemetry"
    STATUS = "status"


class LiveStreamMessage(ApiModel):
    schema_version: Literal["muscle-memory.live.v1"] = "muscle-memory.live.v1"
    kind: LiveMessageKind
    episode_id: str
    cadence_hz: Literal[20] = 20
    frame_join_key: Literal["frame_id"] = "frame_id"
    frame_id: str | None = None
    telemetry: TelemetryRecordView | None = None
    status: dict[str, object] | None = None
    emitted_at: AwareDatetime
    dropped_before: int = Field(default=0, ge=0)

    @field_validator("status")
    @classmethod
    def redact_status(
        cls,
        value: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return None if value is None else redact_sensitive_mapping(value)

    @model_validator(mode="after")
    def validate_message_shape(self) -> Self:
        if self.kind is LiveMessageKind.TELEMETRY:
            if self.telemetry is None or self.status is not None:
                raise ValueError("telemetry messages require only telemetry data")
            if self.frame_id != self.telemetry.frame_id:
                raise ValueError("live frame_id must match the telemetry frame_id")
        elif self.status is None or self.telemetry is not None or self.frame_id is not None:
            raise ValueError("status messages cannot carry telemetry or a video join value")
        return self


class ApiError(ApiModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    details: dict[str, object] | None = None

    @field_validator("message")
    @classmethod
    def redact_message(cls, value: str) -> str:
        return redact_sensitive_text(value)

    @field_validator("details")
    @classmethod
    def redact_details(
        cls,
        value: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return None if value is None else redact_sensitive_mapping(value)


class ApiErrorResponse(ApiModel):
    error: ApiError


def utc_now() -> datetime:
    """Small injection-friendly timestamp helper used by API infrastructure."""

    from datetime import UTC

    return datetime.now(UTC)
