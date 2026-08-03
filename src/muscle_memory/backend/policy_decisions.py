"""Shared policy-gate derivation from coordinator-admitted paired evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from pydantic import TypeAdapter, ValidationError

from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.coordinator.models import (
    NumericPolicyDecision,
    PolicyAction,
    PolicyGateMetrics,
    canonical_json,
    sha256_text,
)
from muscle_memory.evaluation.promotion import PromotionDecision, evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.orchestration.contracts import ExecutionPlan, PipelineStep

_RESULTS = TypeAdapter(tuple[PolicyEpisodeResult, ...])
_RESULT = TypeAdapter(PolicyEpisodeResult)


class PolicyDecisionEvidenceError(ValueError):
    """Admitted policy evidence is missing or no longer internally consistent."""


@dataclass(frozen=True, slots=True)
class AdmittedPromotionEvidence:
    artifact_hash: str
    decision: PromotionDecision


def admitted_promotion_evidence(
    coordinator: CoordinatorStore,
    *,
    baseline_policy_id: str,
    candidate_policy_id: str,
) -> AdmittedPromotionEvidence:
    """Recompute the gate from one exact coordinator-admitted paired artifact."""

    checkpoints = {
        checkpoint.policy_id: checkpoint
        for checkpoint in coordinator.evaluated_checkpoints()
    }
    try:
        baseline = checkpoints[baseline_policy_id]
        candidate = checkpoints[candidate_policy_id]
    except KeyError as exc:
        raise PolicyDecisionEvidenceError("evaluated policy was not found") from exc
    if baseline.evaluation_split != "held_out" or candidate.evaluation_split != "held_out":
        raise PolicyDecisionEvidenceError("paired evidence must use held-out checkpoints")

    artifact_hash = candidate.evaluation_evidence_hash
    artifact = coordinator.held_out_evaluation_artifact(artifact_hash)
    if artifact is None:
        raise PolicyDecisionEvidenceError("candidate has no admitted paired artifact")
    try:
        payload = json.loads(artifact.artifact_json)
        if not isinstance(payload, dict):
            raise TypeError
        baseline_results = _RESULTS.validate_python(payload["baseline_results"])
        candidate_results = _RESULTS.validate_python(payload["candidate_results"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise PolicyDecisionEvidenceError("admitted paired artifact is invalid") from exc
    if len(baseline_results) != 20 or len(candidate_results) != 20:
        raise PolicyDecisionEvidenceError("paired evidence requires exactly twenty worlds")
    if (
        {item.policy_id for item in baseline_results} != {baseline.policy_id}
        or {item.policy_hash for item in baseline_results} != {baseline.checkpoint_hash}
        or {item.policy_id for item in candidate_results} != {candidate.policy_id}
        or {item.policy_hash for item in candidate_results} != {candidate.checkpoint_hash}
    ):
        raise PolicyDecisionEvidenceError(
            "paired artifact policy identities do not match evaluated checkpoints"
        )

    stored = coordinator.held_out_evaluation_results_for_artifact(artifact_hash)
    stored_by_episode = {item.episode_id: item.result_json for item in stored}
    artifact_results = (*baseline_results, *candidate_results)
    if len(stored_by_episode) != 40 or any(
        stored_by_episode.get(item.episode_id)
        != canonical_json(_RESULT.dump_python(item, mode="json"))
        for item in artifact_results
    ):
        raise PolicyDecisionEvidenceError(
            "durable paired results do not equal the admitted artifact"
        )
    try:
        decision = evaluate_promotion(baseline_results, candidate_results)
    except ValueError as exc:
        raise PolicyDecisionEvidenceError("paired evaluation identities do not match") from exc
    return AdmittedPromotionEvidence(artifact_hash=artifact_hash, decision=decision)


def record_reviewed_numeric_decision(
    coordinator: CoordinatorStore,
    plan: ExecutionPlan,
    *,
    stable_alias: str,
    decided_at: datetime,
) -> NumericPolicyDecision:
    """Record the deterministic action for one executable, evidence-bound plan."""

    existing = coordinator.numeric_policy_decision_for_run(plan.run_id)
    if existing is not None:
        return existing
    bundle = coordinator.workflow_guild_evidence(plan.run_id)
    if bundle is None:
        raise PolicyDecisionEvidenceError("workflow has no admitted Guild evidence")
    evaluation = bundle.evaluation.evaluation_evidence
    admitted = admitted_promotion_evidence(
        coordinator,
        baseline_policy_id=evaluation.baseline.policy_id,
        candidate_policy_id=evaluation.candidate.policy_id,
    )
    decision = admitted.decision
    if (
        evaluation.baseline.evaluation_id != admitted.artifact_hash
        or evaluation.candidate.evaluation_id != admitted.artifact_hash
        or evaluation.baseline.success_rate != decision.baseline.success_rate
        or evaluation.baseline.collision_rate != decision.baseline.collision_rate
        or evaluation.candidate.success_rate != decision.candidate.success_rate
        or evaluation.candidate.collision_rate != decision.candidate.collision_rate
        or evaluation.candidate.falls != decision.candidate.total_falls
        or evaluation.candidate.median_clearance_m
        != decision.candidate.median_minimum_clearance_m
        or evaluation.candidate.path_efficiency_regression_fraction
        != decision.path_efficiency_regression
    ):
        raise PolicyDecisionEvidenceError(
            "Guild evaluation aggregate differs from the admitted paired artifact"
        )
    expected_action = PolicyAction.PROMOTE if decision.promotable else PolicyAction.ROLL_BACK
    final_payload = next(
        command.payload
        for command in plan.commands
        if command.step is PipelineStep.PROMOTE_OR_ROLL_BACK
    )
    if (
        final_payload.get("action") != expected_action.value
        or evaluation.proposed_action != expected_action.value
    ):
        raise PolicyDecisionEvidenceError(
            "reviewed final action differs from the recomputed numeric gate"
        )
    evaluation_artifact_hash = next(
        artifact_hash
        for _evidence_id, kind, artifact_hash in bundle.artifact_hashes()
        if kind == "guild_evaluation_evidence"
    )
    collision_reduction = (
        0.0
        if decision.baseline.collision_rate == 0.0
        else (decision.baseline.collision_rate - decision.candidate.collision_rate)
        / decision.baseline.collision_rate
    )
    metrics = PolicyGateMetrics(
        held_out_success_rate=decision.candidate.success_rate,
        collision_rate=decision.candidate.collision_rate,
        fall_count=decision.candidate.total_falls,
        median_clearance_m=decision.candidate.median_minimum_clearance_m,
        success_rate_delta=decision.success_rate_improvement,
        collision_reduction_fraction=collision_reduction,
        path_efficiency_regression_fraction=decision.path_efficiency_regression,
    )
    target = (
        evaluation.candidate.policy_id
        if expected_action is PolicyAction.PROMOTE
        else evaluation.baseline.policy_id
    )
    current_policy_id = coordinator.current_policy(stable_alias)
    if current_policy_id is None:
        coordinator.initialize_policy_alias(
            stable_alias,
            evaluation.baseline.policy_id,
            occurred_at=decided_at,
        )
        current_policy_id = evaluation.baseline.policy_id
    numeric = NumericPolicyDecision(
        decision_id=sha256_text(f"{plan.digest}:{stable_alias}:numeric-policy-decision"),
        run_id=plan.run_id,
        plan_digest=plan.digest,
        action=expected_action,
        alias=stable_alias,
        from_policy_id=current_policy_id,
        target_policy_id=target,
        evaluation_evidence_hash=evaluation_artifact_hash,
        metrics=metrics,
        decided_at=decided_at,
    )
    return coordinator.record_numeric_policy_decision(numeric)


__all__ = [
    "AdmittedPromotionEvidence",
    "PolicyDecisionEvidenceError",
    "admitted_promotion_evidence",
    "record_reviewed_numeric_decision",
]
