"""Safe projections from domain records into public API models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from muscle_memory.api.models import (
    AssetStatus,
    PolicyMetrics,
    PolicySummary,
    PromotionEligibility,
    ProviderHealth,
    ProviderOperationalState,
    SensorReadingView,
    SpecialistReview,
    TelemetryRecordView,
    WorkflowReview,
    WorkflowRun,
    WorkflowRunState,
    WorkflowStep,
    WorkflowStepResult,
    utc_now,
)
from muscle_memory.api.redaction import redact_sensitive_text
from muscle_memory.assets.models import AssetPipelineResult, ProviderSnapshot
from muscle_memory.evaluation.promotion import PolicyEvaluationSummary, PromotionDecision
from muscle_memory.graph_memory.models import GraphMemoryHealth
from muscle_memory.orchestration.contracts import ProviderStatus
from muscle_memory.orchestration.rocketride import PipelineRun
from muscle_memory.orchestration.service import ReviewedExecution
from muscle_memory.telemetry import EpisodeTelemetryRecord, LaserDataHealth


def redact_provider_detail(detail: str) -> str:
    """Keep health useful without reflecting credentials from provider failures."""

    return redact_sensitive_text(detail)


def provider_state(*, state: str, mode: str | None = None) -> ProviderOperationalState:
    """Normalize sponsor-specific states without overstating their readiness."""

    if mode == "simulation":
        return ProviderOperationalState.SIMULATION
    if mode == "cached":
        return ProviderOperationalState.CACHED
    if state == "unconfigured":
        return ProviderOperationalState.UNCONFIGURED
    if state == "configured":
        return ProviderOperationalState.CONFIGURED
    if state == "healthy":
        return ProviderOperationalState.HEALTHY
    if state == "end_to_end_verified":
        return ProviderOperationalState.END_TO_END_VERIFIED
    return ProviderOperationalState.DEGRADED


def orchestration_provider_view(status: ProviderStatus) -> ProviderHealth:
    return ProviderHealth(
        provider=status.provider.value,
        state=provider_state(state=status.health.value, mode=status.mode.value),
        detail=redact_provider_detail(status.detail),
        checked_at=status.checked_at,
    )


def laserdata_provider_view(
    health: LaserDataHealth,
    *,
    checked_at: datetime | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=health.provider,
        state=provider_state(state=health.state.value),
        detail=redact_provider_detail(health.detail),
        checked_at=checked_at or utc_now(),
    )


def graph_provider_view(health: GraphMemoryHealth) -> ProviderHealth:
    return ProviderHealth(
        provider="FalkorDB",
        state=provider_state(state=health.provider_state.value),
        detail=redact_provider_detail(health.detail),
        checked_at=health.checked_at,
    )


def asset_provider_view(
    snapshot: ProviderSnapshot,
    *,
    checked_at: datetime | None = None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=snapshot.provider,
        state=provider_state(state=snapshot.state.value),
        detail=redact_provider_detail(snapshot.detail),
        checked_at=checked_at or utc_now(),
    )


def telemetry_view(
    record: EpisodeTelemetryRecord,
    *,
    delivery: ProviderOperationalState,
) -> TelemetryRecordView:
    """Project all eight labeled categories and preserve the sole frame join key."""

    payload = record.payload
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("public telemetry payloads must be JSON objects with string keys")
    return TelemetryRecordView(
        episode_id=record.episode_id,
        world_id=record.world_id,
        policy_id=record.policy_id,
        sequence=record.sequence,
        sim_time_seconds=record.sim_time_seconds,
        event_time=record.event_time,
        failure_type=record.failure_type,
        frame_id=record.frame_id,
        signal_use=record.signal_use.value,
        sensors=tuple(
            SensorReadingView(
                category=reading.category.value,
                signal_use=reading.signal_use.value,
                available=reading.available,
                values=reading.values,
            )
            for reading in record.sensors.readings
        ),
        payload=payload,
        payload_checksum=record.payload_checksum,
        delivery=delivery,
    )


def reviewed_execution_view(reviewed: ReviewedExecution) -> WorkflowReview:
    review_set = reviewed.guild_reviews
    reviews = tuple(
        SpecialistReview(
            role=review.role.value,
            recommendation=review.recommendation.value,
            summary=review.summary,
        )
        for review in review_set.reviews
    )
    if len(reviews) != 3:
        raise ValueError("a public workflow review requires exactly three specialists")
    return WorkflowReview(
        run_id=reviewed.plan.run_id,
        plan_digest=reviewed.plan.digest,
        reviews=(reviews[0], reviews[1], reviews[2]),
        executable=review_set.executable,
        provider=orchestration_provider_view(review_set.provider_status),
    )


def pipeline_run_view(run: PipelineRun) -> WorkflowRun:
    state = WorkflowRunState(run.state.value)
    return WorkflowRun(
        run_id=run.run_id,
        plan_digest=run.plan_digest,
        state=state,
        completed_steps=tuple(
            WorkflowStepResult(
                step=WorkflowStep(result.step.value),
                state="completed",
                output_sha256=result.output_sha256,
            )
            for result in run.completed_steps
        ),
        blocked_requirement_id=(
            run.blocked_requirement.requirement_id if run.blocked_requirement is not None else None
        ),
        failure=run.failure,
        provider=orchestration_provider_view(run.provider_status),
    )


def policy_summary_view(
    summary: PolicyEvaluationSummary,
    *,
    evaluation_scope: Literal["development", "held_out_aggregate"],
    immutable: bool = True,
) -> PolicySummary:
    return PolicySummary(
        policy_id=summary.policy_id,
        policy_hash=summary.policy_hash,
        evaluated=True,
        evaluation_scope=evaluation_scope,
        metrics=PolicyMetrics(
            episode_count=summary.episode_count,
            success_rate=summary.success_rate,
            collision_rate=summary.collision_rate,
            falls=summary.total_falls,
            median_clearance_m=summary.median_minimum_clearance_m,
            median_path_efficiency=summary.median_path_efficiency,
        ),
        immutable=immutable,
    )


def promotion_eligibility_view(
    decision: PromotionDecision,
    *,
    evidence_hash: str,
) -> PromotionEligibility:
    return PromotionEligibility(
        baseline_policy_id=decision.baseline.policy_id,
        candidate_policy_id=decision.candidate.policy_id,
        held_out_episode_count=decision.candidate.episode_count,
        checks=decision.checks,
        numerically_eligible=decision.promotable,
        evidence_hash=evidence_hash,
    )


def asset_status_view(
    result: AssetPipelineResult,
    *,
    checked_at: datetime | None = None,
) -> AssetStatus:
    world_asset = result.world_asset
    return AssetStatus(
        asset_id=result.manifest.bundle_id,
        state=result.admission_state.value,
        generation_route=result.route.value,
        rendering_artifact_hash=result.manifest.visual_mesh.sha256,
        collider_source=(world_asset.collider.source.value if world_asset else None),
        approval_requirement_id=result.approval_requirement_id,
        providers=tuple(
            asset_provider_view(snapshot, checked_at=checked_at)
            for snapshot in result.provider_snapshots
        ),
        detail=redact_provider_detail(result.detail),
    )


__all__ = [
    "asset_provider_view",
    "asset_status_view",
    "graph_provider_view",
    "laserdata_provider_view",
    "orchestration_provider_view",
    "pipeline_run_view",
    "policy_summary_view",
    "promotion_eligibility_view",
    "provider_state",
    "redact_provider_detail",
    "reviewed_execution_view",
    "telemetry_view",
]
