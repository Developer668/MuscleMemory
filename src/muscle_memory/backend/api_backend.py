"""Concrete durable implementation of the HTTP API backend contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast

from pydantic import TypeAdapter

from muscle_memory.api.adapters import (
    asset_provider_view,
    orchestration_provider_view,
    pipeline_run_view,
    promotion_eligibility_view,
    reviewed_execution_view,
    telemetry_view,
)
from muscle_memory.api.contracts import (
    ApiBackendError,
    AuthenticatedPrincipal,
    LiveEventPublisher,
)
from muscle_memory.api.models import (
    ApprovalDecisionView,
    ApprovalKind,
    AssetStatus,
    CorrectionRequest,
    CorrectionState,
    CorrectionView,
    DecisionRequest,
    EpisodeDetail,
    EpisodeKind,
    EpisodeList,
    EpisodeState,
    EpisodeSummary,
    PendingApproval,
    PendingApprovalList,
    PolicyMetrics,
    PolicySummary,
    PolicySummaryList,
    PromotionEligibility,
    ProviderOperationalState,
    ReplayPage,
    ServiceHealth,
    TelemetryPage,
    WorkflowReview,
    WorkflowReviewRequest,
    WorkflowRun,
    WorkflowRunState,
)
from muscle_memory.api.models import (
    CorrectionKind as ApiCorrectionKind,
)
from muscle_memory.api.models import (
    HumanVerdict as ApiHumanVerdict,
)
from muscle_memory.backend.approvals import CoordinatorApprovalLedger
from muscle_memory.backend.episode_journal import CoordinatorEpisodeJournal
from muscle_memory.backend.episode_runtime import OperationalEpisodeRuntime
from muscle_memory.backend.evidence_admission import (
    TrustedWorkflowEvidenceAdmitter,
    WorkflowEvidenceAdmissionError,
)
from muscle_memory.backend.policy_decisions import (
    PolicyDecisionEvidenceError,
    admitted_promotion_evidence,
    record_reviewed_numeric_decision,
)
from muscle_memory.backend.providers import ProviderBundle
from muscle_memory.backend.rocketride_callback import FixedStepDispatcher
from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.coordinator.models import CoordinatorIntegrityError, canonical_json
from muscle_memory.episodes import (
    AuthenticatedHuman,
    CorrectionApproval,
    CorrectionPoint,
    CorrectionSubmission,
    EpisodeLifecycleState,
    EpisodeNotFoundError,
)
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import GraphStorage, ProviderState
from muscle_memory.orchestration.approvals import (
    HumanDecision,
    HumanVerdict,
)
from muscle_memory.orchestration.contracts import (
    ExecutionPlan,
    PipelineCommand,
    PipelineStep,
)
from muscle_memory.orchestration.guild import GuildUnavailableError
from muscle_memory.orchestration.rocketride import (
    PipelineRun,
    ReviewBlockedError,
    RocketRideUnavailableError,
)
from muscle_memory.orchestration.service import ReviewedExecution, SponsorOrchestrator
from muscle_memory.telemetry import (
    LaserDataProviderState,
)

_RESULT_ADAPTER = TypeAdapter(PolicyEpisodeResult)
_REVIEW_ADAPTER = TypeAdapter(ReviewedExecution)
_RUN_ADAPTER = TypeAdapter(PipelineRun)


def _canonical_adapter_json[T](adapter: TypeAdapter[T], value: T) -> str:
    return canonical_json(adapter.dump_python(value, mode="json"))


class MuscleMemoryApiBackend:
    """Composition-facing API with no robot-control or path-teacher dependency."""

    def __init__(
        self,
        *,
        coordinator: CoordinatorStore,
        journal: CoordinatorEpisodeJournal,
        episode_runtime: OperationalEpisodeRuntime,
        providers: ProviderBundle,
        approval_ledger: CoordinatorApprovalLedger,
        rocketride_callback: FixedStepDispatcher | None = None,
        stable_policy_alias: str = "stable",
    ) -> None:
        self.coordinator = coordinator
        self.journal = journal
        self.episode_runtime = episode_runtime
        self.providers = providers
        self.approval_ledger = approval_ledger
        self.rocketride_callback = rocketride_callback
        self.stable_policy_alias = stable_policy_alias
        self.orchestrator = SponsorOrchestrator(providers.guild, providers.rocketride)
        self.evidence_admitter = TrustedWorkflowEvidenceAdmitter(
            coordinator=coordinator,
            journal=journal,
            graph_memory=providers.graph_memory,
        )
        self._reviewed: dict[str, ReviewedExecution] = {}
        self._runs: dict[str, PipelineRun] = {}
        self._started = False
        self._closed = False
        self._restore_workflows()

    def _restore_workflows(self) -> None:
        for payload in self.coordinator.workflow_reviews():
            reviewed = _REVIEW_ADAPTER.validate_json(payload)
            plan = self.coordinator.workflow_plan(reviewed.plan.run_id)
            if plan != reviewed.plan:
                raise RuntimeError("durable Guild review is detached from its execution plan")
            self._reviewed[reviewed.plan.run_id] = reviewed
        for payload in self.coordinator.latest_workflow_run_snapshots():
            run = _RUN_ADAPTER.validate_json(payload)
            active_review = self._reviewed.get(run.run_id)
            if active_review is None or active_review.plan.digest != run.plan_digest:
                raise RuntimeError("durable RocketRide run is detached from its Guild review")
            self._runs[run.run_id] = run

    def bind_live_publisher(self, publisher: LiveEventPublisher) -> None:
        self.episode_runtime.bind_live_publisher(publisher)

    def dispatch_rocketride_callback(
        self,
        encoded_envelope: str,
        authorization: str,
    ) -> dict[str, object]:
        if self.rocketride_callback is None:
            raise ApiBackendError(
                503,
                "rocketride_callback_unconfigured",
                "RocketRide callback configuration is unavailable",
            )
        return self.rocketride_callback.dispatch(encoded_envelope, authorization)

    def authenticate_rocketride_callback(self, authorization: str) -> None:
        if self.rocketride_callback is None:
            raise ApiBackendError(
                503,
                "rocketride_callback_unconfigured",
                "RocketRide callback configuration is unavailable",
            )
        self.rocketride_callback.authenticate(authorization)

    async def startup(self) -> None:
        if self._closed:
            raise RuntimeError("backend resources have already been closed")
        if self._started:
            return
        # A health probe also replays any append-only cache prefix to a recovered FalkorDB.
        self.providers.graph_memory.health()
        await self.providers.laserdata.initialize()
        await self.episode_runtime.service.reconcile_provider_state()
        self._started = True

    async def shutdown(self) -> None:
        if self._closed:
            return
        await self.providers.laserdata.close()
        self.coordinator.close()
        self._closed = True
        self._started = False

    async def health(self) -> ServiceHealth:
        provider_health = self.providers.registry.health()
        graph = next(
            provider
            for provider in provider_health.providers
            if provider.provider == "FalkorDB"
        )
        consumers_list = [
            snapshot.public() for snapshot in self.episode_runtime.consumer_snapshots()
        ]
        for index, consumer in enumerate(consumers_list):
            if consumer.provider != "LaserData consumer: post-episode-graph-handoff":
                continue
            if (
                consumer.state is not ProviderOperationalState.CONFIGURED
                and graph.state not in {
                ProviderOperationalState.HEALTHY,
                ProviderOperationalState.END_TO_END_VERIFIED,
                }
            ):
                consumers_list[index] = consumer.model_copy(
                    update={
                        "state": ProviderOperationalState.CACHED,
                        "detail": (
                            "post-episode facts are retained in the append-only graph "
                            "cache; FalkorDB delivery is not confirmed"
                        ),
                    }
                )
        consumers = tuple(consumers_list)
        state = provider_health.state
        if any(consumer.state is ProviderOperationalState.DEGRADED for consumer in consumers):
            state = ProviderOperationalState.DEGRADED
        return ServiceHealth(
            state=state,
            providers=(*provider_health.providers, *consumers),
            checked_at=datetime.now(UTC),
        )

    async def list_episodes(self, *, cursor: str | None, limit: int) -> EpisodeList:
        identities = sorted(self.journal.identities(), key=lambda item: item.opened_at)
        start = 0
        if cursor is not None:
            positions = [
                index + 1
                for index, identity in enumerate(identities)
                if identity.episode_id == cursor
            ]
            if not positions:
                raise ApiBackendError(422, "invalid_cursor", "episode cursor is invalid")
            start = positions[0]
        selected = identities[start : start + limit]
        next_cursor = (
            selected[-1].episode_id
            if selected and start + len(selected) < len(identities)
            else None
        )
        return EpisodeList(
            items=tuple(self._episode_summary(identity.episode_id) for identity in selected),
            next_cursor=next_cursor,
        )

    async def episode(self, episode_id: str) -> EpisodeDetail | None:
        identity = self._identity(episode_id)
        if identity is None:
            return None
        closure = self.episode_runtime.service.closure_for(episode_id)
        records = self.providers.laserdata.spool.records_for(episode_id)
        corrections = tuple(
            item.correction_id
            for item in self.journal.corrections()
            if item.episode_id == episode_id
        )
        provider_delivery = self._episode_delivery(episode_id)
        return EpisodeDetail(
            episode=self._episode_summary(episode_id),
            telemetry_records=len(records),
            provider_delivery=provider_delivery,
            result=(
                None
                if closure is None
                else cast(
                    dict[str, object],
                    _RESULT_ADAPTER.dump_python(closure.result, mode="json"),
                )
            ),
            failure_ids=(
                () if closure is None else tuple(failure.failure_id for failure in closure.failures)
            ),
            correction_ids=corrections,
        )

    async def telemetry(
        self,
        episode_id: str,
        *,
        after_sequence: int | None,
        limit: int,
    ) -> TelemetryPage | None:
        if self._identity(episode_id) is None:
            return None
        records = self._page_records(episode_id, after_sequence, limit)
        all_records = self.providers.laserdata.spool.records_for(episode_id)
        next_sequence = (
            records[-1].sequence + 1
            if records and records[-1].sequence + 1 < len(all_records)
            else None
        )
        return TelemetryPage(
            episode_id=episode_id,
            records=tuple(
                telemetry_view(record, delivery=self._record_delivery(record.sequence, episode_id))
                for record in records
            ),
            next_sequence=next_sequence,
        )

    async def replay(
        self,
        episode_id: str,
        *,
        after_sequence: int | None,
        limit: int,
    ) -> ReplayPage | None:
        if self._identity(episode_id) is None:
            return None
        if (
            self.episode_runtime.service.episode_state(episode_id)
            is not EpisodeLifecycleState.CLOSED
        ):
            raise ApiBackendError(409, "episode_open", "only closed episodes can be replayed")
        page = await self.telemetry(
            episode_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        if page is None:
            return None
        return ReplayPage(
            episode_id=episode_id,
            records=page.records,
            next_sequence=page.next_sequence,
        )

    async def pending_approvals(self) -> PendingApprovalList:
        workflow = tuple(
            PendingApproval(
                requirement_id=requirement.requirement_id,
                kind=ApprovalKind(requirement.kind.value),
                summary=requirement.summary,
                plan_digest=requirement.plan_digest,
                run_id=requirement.run_id,
                created_at=created_at,
            )
            for requirement, created_at in self.coordinator.pending_approval_requirements()
            if self._approval_is_executable(requirement.run_id, requirement.step)
        )
        approved = {item.submission.correction_id for item in self.journal.approvals()}
        rejected = set(self._correction_rejections())
        corrections = tuple(
            PendingApproval(
                requirement_id=correction.correction_id,
                kind=ApprovalKind.CORRECTION,
                summary=correction.description,
                created_at=correction.created_at,
            )
            for correction in self.journal.corrections()
            if correction.correction_id not in approved and correction.correction_id not in rejected
        )
        return PendingApprovalList(items=(*workflow, *corrections))

    async def submit_approval_decision(
        self,
        requirement_id: str,
        request: DecisionRequest,
        principal: AuthenticatedPrincipal,
    ) -> ApprovalDecisionView:
        pending = {
            requirement.requirement_id: requirement
            for requirement, _ in self.coordinator.pending_approval_requirements()
            if self._approval_is_executable(requirement.run_id, requirement.step)
        }
        requirement = pending.get(requirement_id)
        if requirement is None:
            raise ApiBackendError(
                404,
                "approval_requirement_not_found",
                "approval requirement was not found or is already decided",
            )
        decision = HumanDecision.create(
            requirement,
            human_subject=principal.subject,
            verdict=HumanVerdict(request.verdict.value),
            note=request.note,
        )
        self.approval_ledger.record(decision)
        return ApprovalDecisionView(
            requirement_id=requirement_id,
            verdict=ApiHumanVerdict(decision.verdict.value),
            human_subject=decision.human_subject,
            authentication_method=principal.authentication_method,
            decided_at=decision.decided_at,
        )

    async def review_workflow(
        self,
        request: WorkflowReviewRequest,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowReview:
        del principal
        commands = tuple(
            PipelineCommand.create(PipelineStep(command.step.value), command.payload)
            for command in request.commands
        )
        plan = ExecutionPlan.create(request.run_id, commands)
        existing = self.coordinator.workflow_plan(plan.run_id)
        if existing is not None and existing != plan:
            raise ApiBackendError(409, "workflow_immutable", "workflow id is immutable")
        if existing is None:
            self.coordinator.register_workflow(plan, created_at=datetime.now(UTC))
        stored_evidence = self.coordinator.workflow_guild_evidence(plan.run_id)
        if stored_evidence is None:
            try:
                self.evidence_admitter.admit(plan, request.evidence)
                self.coordinator.record_workflow_guild_evidence(plan.run_id, request.evidence)
            except (
                CoordinatorIntegrityError,
                WorkflowEvidenceAdmissionError,
                ValueError,
            ) as exc:
                raise ApiBackendError(
                    422,
                    "workflow_evidence_invalid",
                    "workflow evidence is not backed by matching coordinator artifacts",
                ) from exc
        elif stored_evidence != request.evidence:
            raise ApiBackendError(
                409,
                "workflow_evidence_immutable",
                "workflow evidence is immutable once admitted",
            )
        prior_review = self._reviewed.get(plan.run_id)
        if prior_review is not None:
            if prior_review.guild_reviews.executable:
                self._ensure_numeric_decision(prior_review)
            return reviewed_execution_view(prior_review)
        try:
            reviewed = await self.orchestrator.review(plan)
        except GuildUnavailableError as exc:
            raise ApiBackendError(
                503,
                "guild_unavailable",
                "Guild specialist review is unavailable",
            ) from exc
        if reviewed.guild_reviews.executable:
            try:
                self._ensure_numeric_decision(reviewed)
            except (CoordinatorIntegrityError, PolicyDecisionEvidenceError, ValueError) as exc:
                raise ApiBackendError(
                    422,
                    "workflow_numeric_evidence_invalid",
                    "reviewed policy action does not match admitted paired evidence",
                ) from exc
        self.coordinator.record_workflow_review(
            plan.run_id,
            _canonical_adapter_json(_REVIEW_ADAPTER, reviewed),
        )
        self._reviewed[plan.run_id] = reviewed
        return reviewed_execution_view(reviewed)

    async def execute_workflow(
        self,
        run_id: str,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowRun:
        del principal
        reviewed = self._reviewed.get(run_id)
        if reviewed is None:
            raise ApiBackendError(
                409,
                "workflow_review_required",
                "an active exact-plan Guild review is required before execution",
            )
        try:
            run = await self.orchestrator.execute(reviewed)
        except ReviewBlockedError as exc:
            raise ApiBackendError(409, "workflow_review_blocked", str(exc)) from exc
        except RocketRideUnavailableError as exc:
            raise ApiBackendError(
                503,
                "rocketride_unavailable",
                "RocketRide execution is unavailable",
            ) from exc
        self._runs[run_id] = run
        self.coordinator.record_workflow_run_snapshot(
            run_id,
            _canonical_adapter_json(_RUN_ADAPTER, run),
        )
        return pipeline_run_view(run)

    async def resume_workflow(
        self,
        run_id: str,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowRun:
        del principal
        reviewed = self._reviewed.get(run_id)
        prior = self._runs.get(run_id)
        if reviewed is None or prior is None:
            raise ApiBackendError(
                409,
                "workflow_resume_unavailable",
                "an active incomplete reviewed run is required for resume",
            )
        try:
            run = await self.orchestrator.execute(reviewed, prior)
        except RocketRideUnavailableError as exc:
            raise ApiBackendError(
                503,
                "rocketride_unavailable",
                "RocketRide execution is unavailable",
            ) from exc
        self._runs[run_id] = run
        self.coordinator.record_workflow_run_snapshot(
            run_id,
            _canonical_adapter_json(_RUN_ADAPTER, run),
        )
        return pipeline_run_view(run)

    async def workflow(self, run_id: str) -> WorkflowRun | None:
        run = self._runs.get(run_id)
        if run is not None:
            return pipeline_run_view(run)
        reviewed = self._reviewed.get(run_id)
        if reviewed is None:
            return None
        return WorkflowRun(
            run_id=run_id,
            plan_digest=reviewed.plan.digest,
            state=WorkflowRunState.REVIEWED,
            completed_steps=(),
            provider=orchestration_provider_view(reviewed.guild_reviews.provider_status),
        )

    async def submit_correction(
        self,
        episode_id: str,
        request: CorrectionRequest,
        principal: AuthenticatedPrincipal,
    ) -> CorrectionView:
        points = tuple(CorrectionPoint(point.x_m, point.y_m) for point in request.points)
        if request.kind is ApiCorrectionKind.ROUTE:
            correction = await self.episode_runtime.service.submit_route_correction(
                episode_id=episode_id,
                failure_id=request.failure_id,
                points=points,
                description=request.description,
                submitted_by=principal.subject,
            )
        else:
            correction = await self.episode_runtime.service.submit_keep_out_correction(
                episode_id=episode_id,
                failure_id=request.failure_id,
                polygon=points,
                description=request.description,
                submitted_by=principal.subject,
            )
        return CorrectionView(
            correction_id=correction.correction_id,
            episode_id=correction.episode_id,
            failure_id=correction.failure_id,
            kind=ApiCorrectionKind(correction.kind.value),
            state=CorrectionState.PENDING,
            submitted_by=correction.submitted_by,
            created_at=correction.created_at,
            graph_delivery=self._graph_configuration_state(),
        )

    async def decide_correction(
        self,
        correction_id: str,
        request: DecisionRequest,
        principal: AuthenticatedPrincipal,
    ) -> CorrectionView:
        correction = next(
            (item for item in self.journal.corrections() if item.correction_id == correction_id),
            None,
        )
        if correction is None:
            raise ApiBackendError(404, "correction_not_found", "correction was not found")
        prior_approval = next(
            (
                item
                for item in self.journal.approvals()
                if item.submission.correction_id == correction_id
            ),
            None,
        )
        prior_rejection = self._correction_rejections().get(correction_id)
        if prior_approval is not None:
            if request.verdict is ApiHumanVerdict.REJECT:
                raise ApiBackendError(
                    409,
                    "correction_decision_immutable",
                    "an approved correction cannot be rejected",
                )
            return self._correction_view(
                correction,
                CorrectionState.APPROVED,
                self._approval_graph_delivery(prior_approval),
            )
        if prior_rejection is not None:
            if request.verdict is ApiHumanVerdict.APPROVE:
                raise ApiBackendError(
                    409,
                    "correction_decision_immutable",
                    "a rejected correction cannot be approved",
                )
            return self._correction_view(
                correction,
                CorrectionState.REJECTED,
                self._graph_configuration_state(),
            )
        if request.verdict is ApiHumanVerdict.REJECT:
            rejected_at = datetime.now(UTC)
            self.coordinator.record_training_correction_rejection(
                correction_id,
                canonical_json(
                    {
                        "authentication_method": principal.authentication_method,
                        "correction_id": correction_id,
                        "decided_at": rejected_at.isoformat(),
                        "human_subject": principal.subject,
                        "note": request.note,
                    }
                ),
            )
            state = CorrectionState.REJECTED
            graph_delivery = self._graph_configuration_state()
        else:
            approval = await self.episode_runtime.service.approve_correction(
                correction_id,
                approver=AuthenticatedHuman(
                    subject_id=principal.subject,
                    authentication_method=principal.authentication_method,
                    authenticated=True,
                ),
            )
            state = CorrectionState.APPROVED
            graph_delivery = self._approval_graph_delivery(approval)
        return self._correction_view(correction, state, graph_delivery)

    @staticmethod
    def _correction_view(
        correction: CorrectionSubmission,
        state: CorrectionState,
        graph_delivery: ProviderOperationalState,
    ) -> CorrectionView:
        return CorrectionView(
            correction_id=correction.correction_id,
            episode_id=correction.episode_id,
            failure_id=correction.failure_id,
            kind=ApiCorrectionKind(correction.kind.value),
            state=state,
            submitted_by=correction.submitted_by,
            created_at=correction.created_at,
            graph_delivery=graph_delivery,
        )

    async def policy_summaries(self) -> PolicySummaryList:
        items: list[PolicySummary] = []
        for checkpoint in self.coordinator.evaluated_checkpoints():
            raw = json.loads(checkpoint.metrics_json)
            metrics: PolicyMetrics | None = None
            if isinstance(raw, dict):
                try:
                    metrics = PolicyMetrics(
                        episode_count=int(raw["episode_count"]),
                        success_rate=float(raw["success_rate"]),
                        collision_rate=float(raw["collision_rate"]),
                        falls=int(raw["total_falls"]),
                        median_clearance_m=float(raw["median_minimum_clearance_m"]),
                        median_path_efficiency=float(raw["median_path_efficiency"]),
                    )
                except (KeyError, TypeError, ValueError):
                    metrics = None
            items.append(
                PolicySummary(
                    policy_id=checkpoint.policy_id,
                    policy_hash=checkpoint.checkpoint_hash,
                    evaluated=True,
                    evaluation_scope=(
                        "held_out_aggregate"
                        if checkpoint.evaluation_split == "held_out"
                        else "development"
                    ),
                    metrics=metrics,
                    immutable=True,
                )
            )
        return PolicySummaryList(items=tuple(items))

    async def promotion_eligibility(
        self,
        *,
        baseline_policy_id: str,
        candidate_policy_id: str,
    ) -> PromotionEligibility:
        policies = {item.policy_id: item for item in self.coordinator.evaluated_checkpoints()}
        if baseline_policy_id not in policies or candidate_policy_id not in policies:
            raise ApiBackendError(404, "policy_not_found", "evaluated policy was not found")
        try:
            admitted = admitted_promotion_evidence(
                self.coordinator,
                baseline_policy_id=baseline_policy_id,
                candidate_policy_id=candidate_policy_id,
            )
        except PolicyDecisionEvidenceError as exc:
            raise ApiBackendError(
                409,
                "paired_evaluation_evidence_unavailable",
                "exact admitted paired evaluation evidence is unavailable",
            ) from exc
        return promotion_eligibility_view(
            admitted.decision,
            evidence_hash=admitted.artifact_hash,
        )

    async def asset_statuses(self) -> tuple[AssetStatus, ...]:
        return (self._fallback_asset_status(),)

    async def asset_status(self, asset_id: str) -> AssetStatus | None:
        status = self._fallback_asset_status()
        return status if status.asset_id == asset_id else None

    def _fallback_asset_status(self) -> AssetStatus:
        manifest = self.providers.assets.fallback_manifest
        return AssetStatus(
            asset_id=manifest.bundle_id,
            state="unavailable",
            generation_route="verified_fallback",
            rendering_artifact_hash=manifest.visual_mesh.sha256,
            collider_source=None,
            approval_requirement_id=None,
            providers=tuple(
                asset_provider_view(snapshot)
                for snapshot in self.providers.assets.provider_snapshots
            ),
            detail=(
                "verified rendering-only fallback is cached; world admission remains "
                "unavailable until deterministic physical properties are supplied"
            ),
        )

    def _ensure_numeric_decision(self, reviewed: ReviewedExecution) -> None:
        record_reviewed_numeric_decision(
            self.coordinator,
            reviewed.plan,
            stable_alias=self.stable_policy_alias,
            decided_at=datetime.now(UTC),
        )

    def _approval_is_executable(self, run_id: str, step: PipelineStep) -> bool:
        reviewed = self._reviewed.get(run_id)
        if reviewed is None or not reviewed.guild_reviews.executable:
            return False
        if self.coordinator.workflow_guild_evidence(run_id) is None:
            return False
        if step is PipelineStep.PROMOTE_OR_ROLL_BACK:
            return self.coordinator.numeric_policy_decision_for_run(run_id) is not None
        return True

    def _identity(self, episode_id: str):  # type: ignore[no-untyped-def]
        return next(
            (
                identity
                for identity in self.journal.identities()
                if identity.episode_id == episode_id
            ),
            None,
        )

    def _episode_summary(self, episode_id: str) -> EpisodeSummary:
        identity = self._identity(episode_id)
        if identity is None:
            raise EpisodeNotFoundError(episode_id)
        closure = self.episode_runtime.service.closure_for(episode_id)
        if closure is None:
            state = EpisodeState.RUNNING
        else:
            state = EpisodeState.SUCCEEDED if closure.result.success else EpisodeState.FAILED
        return EpisodeSummary(
            episode_id=episode_id,
            kind=EpisodeKind.TRAINING,
            state=state,
            robot_checksum=identity.robot_checksum,
            world_id=identity.world_id,
            world_hash=identity.world_hash,
            policy_id=identity.policy_id,
            policy_hash=identity.policy_hash,
            opened_at=identity.opened_at,
            closed_at=None if closure is None else closure.closed_at,
        )

    def _page_records(self, episode_id: str, after_sequence: int | None, limit: int):  # type: ignore[no-untyped-def]
        records = self.providers.laserdata.spool.records_for(episode_id)
        start = 0 if after_sequence is None else after_sequence + 1
        return records[start : start + limit]

    def _record_delivery(
        self,
        sequence: int,
        episode_id: str,
    ) -> ProviderOperationalState:
        records = self.providers.laserdata.spool.records_for(episode_id)
        if sequence >= len(records):
            return ProviderOperationalState.DEGRADED
        from muscle_memory.telemetry import LaserDataTelemetryEnvelope

        event_id = LaserDataTelemetryEnvelope.from_domain(records[sequence]).event_id
        spool = self.providers.laserdata.spool
        if spool.verified_position(event_id) is not None:
            return ProviderOperationalState.END_TO_END_VERIFIED
        if spool.accepted_receipt(event_id) is not None:
            return ProviderOperationalState.HEALTHY
        if self.providers.laserdata.health.state is LaserDataProviderState.UNCONFIGURED:
            return ProviderOperationalState.UNCONFIGURED
        return ProviderOperationalState.DEGRADED

    def _episode_delivery(self, episode_id: str) -> ProviderOperationalState:
        records = self.providers.laserdata.spool.records_for(episode_id)
        if not records:
            return ProviderOperationalState.CONFIGURED
        states = {self._record_delivery(record.sequence, episode_id) for record in records}
        if states == {ProviderOperationalState.END_TO_END_VERIFIED}:
            return ProviderOperationalState.END_TO_END_VERIFIED
        if states <= {
            ProviderOperationalState.HEALTHY,
            ProviderOperationalState.END_TO_END_VERIFIED,
        }:
            return ProviderOperationalState.HEALTHY
        if states == {ProviderOperationalState.UNCONFIGURED}:
            return ProviderOperationalState.UNCONFIGURED
        return ProviderOperationalState.DEGRADED

    def _graph_configuration_state(self) -> ProviderOperationalState:
        health = self.providers.graph_memory.health()
        if health.provider_state is ProviderState.END_TO_END_VERIFIED:
            return ProviderOperationalState.END_TO_END_VERIFIED
        if health.provider_state is ProviderState.HEALTHY:
            return ProviderOperationalState.HEALTHY
        if health.provider_state is ProviderState.UNCONFIGURED:
            return ProviderOperationalState.UNCONFIGURED
        if health.provider_state is ProviderState.CONFIGURED:
            return ProviderOperationalState.CONFIGURED
        return ProviderOperationalState.DEGRADED

    @staticmethod
    def _approval_graph_delivery(
        approval: CorrectionApproval,
    ) -> ProviderOperationalState:
        if approval.graph_error_type is not None:
            return ProviderOperationalState.DEGRADED
        receipt = approval.graph_receipt
        if receipt is None:
            return ProviderOperationalState.CACHED
        if receipt.storage is GraphStorage.LOCAL_CACHE:
            return ProviderOperationalState.CACHED
        if receipt.provider_state is ProviderState.END_TO_END_VERIFIED:
            return ProviderOperationalState.END_TO_END_VERIFIED
        if receipt.provider_state is ProviderState.HEALTHY:
            return ProviderOperationalState.HEALTHY
        return ProviderOperationalState.DEGRADED

    def _correction_rejections(self) -> dict[str, dict[str, object]]:
        values: dict[str, dict[str, object]] = {}
        for payload in self.coordinator.training_correction_rejections():
            decoded = json.loads(payload)
            if isinstance(decoded, dict) and isinstance(decoded.get("correction_id"), str):
                values[str(decoded["correction_id"])] = decoded
        return values


__all__ = ["MuscleMemoryApiBackend"]
