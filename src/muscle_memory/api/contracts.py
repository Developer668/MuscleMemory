"""Dependency-injection boundaries used by the HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from muscle_memory.api.models import (
    ApprovalDecisionView,
    AssetStatus,
    CorrectionRequest,
    CorrectionView,
    DecisionRequest,
    EpisodeDetail,
    EpisodeList,
    PendingApprovalList,
    PolicySummaryList,
    PromotionEligibility,
    ReplayPage,
    ServiceHealth,
    TelemetryPage,
    TelemetryRecordView,
    WorkflowReview,
    WorkflowReviewRequest,
    WorkflowRun,
)


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    """Identity established by server-side authentication, never request JSON."""

    subject: str
    authentication_method: str
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.subject.strip() or not self.authentication_method.strip():
            raise ValueError("authenticated principals require an identity and method")


class ApiBackendError(RuntimeError):
    """A safe, typed domain failure suitable for an HTTP response."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class ApiBackend(Protocol):
    """Application-facing operations; implementations retain domain safety gates."""

    def bind_live_publisher(self, publisher: LiveEventPublisher) -> None: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def health(self) -> ServiceHealth: ...

    async def list_episodes(self, *, cursor: str | None, limit: int) -> EpisodeList: ...

    async def episode(self, episode_id: str) -> EpisodeDetail | None: ...

    async def telemetry(
        self,
        episode_id: str,
        *,
        after_sequence: int | None,
        limit: int,
    ) -> TelemetryPage | None: ...

    async def replay(
        self,
        episode_id: str,
        *,
        after_sequence: int | None,
        limit: int,
    ) -> ReplayPage | None: ...

    async def pending_approvals(self) -> PendingApprovalList: ...

    async def submit_approval_decision(
        self,
        requirement_id: str,
        request: DecisionRequest,
        principal: AuthenticatedPrincipal,
    ) -> ApprovalDecisionView: ...

    async def review_workflow(
        self,
        request: WorkflowReviewRequest,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowReview: ...

    async def execute_workflow(
        self,
        run_id: str,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowRun: ...

    async def resume_workflow(
        self,
        run_id: str,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowRun: ...

    async def workflow(self, run_id: str) -> WorkflowRun | None: ...

    async def submit_correction(
        self,
        episode_id: str,
        request: CorrectionRequest,
        principal: AuthenticatedPrincipal,
    ) -> CorrectionView: ...

    async def decide_correction(
        self,
        correction_id: str,
        request: DecisionRequest,
        principal: AuthenticatedPrincipal,
    ) -> CorrectionView: ...

    async def policy_summaries(self) -> PolicySummaryList: ...

    async def promotion_eligibility(
        self,
        *,
        baseline_policy_id: str,
        candidate_policy_id: str,
    ) -> PromotionEligibility: ...

    async def asset_statuses(self) -> tuple[AssetStatus, ...]: ...

    async def asset_status(self, asset_id: str) -> AssetStatus | None: ...


class Authenticator(Protocol):
    @property
    def configured(self) -> bool: ...

    def authenticate(self, token: str) -> AuthenticatedPrincipal | None: ...


class LiveEventPublisher(Protocol):
    """Narrow hook used by episode ingestion to update connected dashboards."""

    async def publish_telemetry(self, telemetry: TelemetryRecordView) -> None: ...

    async def publish_status(
        self,
        episode_id: str,
        status: dict[str, object],
    ) -> None: ...
