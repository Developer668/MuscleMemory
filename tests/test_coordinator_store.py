from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from muscle_memory.coordinator import (
    ApprovalRequiredError,
    CoordinatorIntegrityError,
    CoordinatorStateError,
    CoordinatorStore,
    EpisodeState,
    HeldOutEvaluationEpisodeMetadata,
    NumericPolicyDecision,
    PolicyAction,
    PolicyGateMetrics,
    ProviderEvidenceReference,
    TrainingEpisodeMetadata,
    WorkflowStepState,
)
from muscle_memory.graph_memory import EvaluatedPolicyVersion
from muscle_memory.orchestration import (
    FIXED_PIPELINE,
    ExecutionPlan,
    HumanDecision,
    HumanVerdict,
    PipelineCommand,
    PipelineStep,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROBOT_HASH = "a" * 64
WORLD_HASH = "b" * 64
POLICY_V0_HASH = "c" * 64
POLICY_V1_HASH = "d" * 64
EVIDENCE_HASH = "e" * 64


def training_episode(episode_id: str = "episode-training-1") -> TrainingEpisodeMetadata:
    return TrainingEpisodeMetadata(
        episode_id=episode_id,
        robot_checksum=ROBOT_HASH,
        world_hash=WORLD_HASH,
        policy_hash=POLICY_V0_HASH,
        created_at=NOW,
    )


def execution_plan(run_id: str, action: PolicyAction = PolicyAction.PROMOTE) -> ExecutionPlan:
    payloads: dict[PipelineStep, dict[str, object]] = {
        PipelineStep.VALIDATE_WORLD: {"uncertain_physical_properties": False},
        PipelineStep.RUN_EPISODE: {},
        PipelineStep.SUMMARIZE_TELEMETRY: {},
        PipelineStep.QUERY_GRAPH_MEMORY: {},
        PipelineStep.SELECT_CURRICULUM: {"curriculum_change_requested": False},
        PipelineStep.TRAIN_CANDIDATE_POLICY: {"reward_change_requested": False},
        PipelineStep.EVALUATE_CANDIDATE_POLICY: {
            "baseline_policy_id": "policy-v0",
            "candidate_policy_id": "policy-v1",
            "heldout_world_set_id": "heldout-v1",
        },
        PipelineStep.PROMOTE_OR_ROLL_BACK: {"action": action.value},
    }
    commands = tuple(PipelineCommand.create(step, payloads[step]) for step in FIXED_PIPELINE)
    return ExecutionPlan.create(run_id, commands)


def evaluated_policy(policy_id: str, checkpoint_hash: str) -> EvaluatedPolicyVersion:
    return EvaluatedPolicyVersion.create(
        policy_id=policy_id,
        checkpoint_hash=checkpoint_hash,
        evaluation_evidence_hash=EVIDENCE_HASH,
        evaluation_split="held_out",
        metrics={"success_rate": 0.9},
        evaluated_at=NOW,
    )


def passing_metrics() -> PolicyGateMetrics:
    return PolicyGateMetrics(
        held_out_success_rate=0.9,
        collision_rate=0.05,
        fall_count=0,
        median_clearance_m=0.31,
        success_rate_delta=0.25,
        collision_reduction_fraction=0.6,
        path_efficiency_regression_fraction=0.03,
    )


def test_episode_registration_is_idempotent_and_training_api_hides_held_out_ids(
    tmp_path: Path,
) -> None:
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    training = training_episode()
    held_out = HeldOutEvaluationEpisodeMetadata(
        episode_id="episode-evaluation-1",
        robot_checksum=ROBOT_HASH,
        world_hash="f" * 64,
        policy_hash=POLICY_V1_HASH,
        held_out_world_set_id="heldout-v1",
        created_at=NOW,
    )

    assert store.register_training_episode(training) == training
    assert store.register_training_episode(training) == training
    assert store.register_held_out_evaluation_episode(held_out) == held_out
    assert store.training_episodes() == (training,)
    assert store.training_episode(held_out.episode_id) is None
    assert store.held_out_evaluation_episode(held_out.episode_id) == held_out

    with pytest.raises(CoordinatorIntegrityError, match="immutable"):
        store.register_training_episode(replace(training, world_hash="1" * 64))
    store.close()


def test_episode_lifecycle_and_provider_evidence_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "coordinator.sqlite3"
    evidence = ProviderEvidenceReference(
        evidence_id="laserdata-episode-proof",
        provider="laserdata",
        evidence_kind="episode_replay",
        provider_object_id="partition-0-offset-41",
        artifact_hash=EVIDENCE_HASH,
        observed_at=NOW,
    )
    with CoordinatorStore(path) as store:
        store.register_training_episode(training_episode())
        assert store.record_provider_evidence(evidence) == evidence
        started = store.transition_episode(
            "episode-training-1",
            EpisodeState.RUNNING,
            occurred_at=NOW + timedelta(seconds=1),
        )
        assert store.transition_episode(
            "episode-training-1",
            EpisodeState.RUNNING,
            occurred_at=NOW + timedelta(seconds=1),
        ) == started
        store.transition_episode(
            "episode-training-1",
            EpisodeState.SUCCEEDED,
            occurred_at=NOW + timedelta(seconds=12),
            details={"telemetry_digest": "2" * 64},
            evidence_id=evidence.evidence_id,
        )

    with CoordinatorStore(path) as reopened:
        assert reopened.episode_state("episode-training-1") is EpisodeState.SUCCEEDED
        history = reopened.episode_history("episode-training-1")
        assert [event.state for event in history] == [
            EpisodeState.CREATED,
            EpisodeState.RUNNING,
            EpisodeState.SUCCEEDED,
        ]
        assert history[-1].evidence_id == evidence.evidence_id
        with pytest.raises(CoordinatorStateError, match="cannot transition"):
            reopened.transition_episode(
                "episode-training-1",
                EpisodeState.FAILED,
                occurred_at=NOW + timedelta(seconds=13),
            )


def test_database_triggers_reject_history_updates_and_deletes(tmp_path: Path) -> None:
    path = tmp_path / "coordinator.sqlite3"
    with CoordinatorStore(path) as store:
        store.register_training_episode(training_episode())

    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE episodes SET world_hash = ? WHERE episode_id = ?",
                ("1" * 64, "episode-training-1"),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM episode_transitions WHERE episode_id = ?",
                ("episode-training-1",),
            )
    finally:
        connection.close()


def test_workflow_audit_blocks_before_human_approval_and_preserves_order(
    tmp_path: Path,
) -> None:
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    plan = execution_plan("run-workflow-1")
    store.register_workflow(plan, created_at=NOW)
    evidence = ProviderEvidenceReference(
        evidence_id="rocketride-step-proof",
        provider="rocketride.ai",
        evidence_kind="step_receipt",
        provider_object_id="task-123",
        artifact_hash=EVIDENCE_HASH,
        observed_at=NOW,
    )
    store.record_provider_evidence(evidence)

    with pytest.raises(CoordinatorStateError, match="prior steps"):
        store.record_workflow_step(
            plan.run_id,
            PipelineStep.RUN_EPISODE,
            WorkflowStepState.STARTED,
            occurred_at=NOW,
        )

    for index, step in enumerate(FIXED_PIPELINE[:-1]):
        started_at = NOW + timedelta(seconds=index * 2)
        store.record_workflow_step(
            plan.run_id,
            step,
            WorkflowStepState.STARTED,
            occurred_at=started_at,
        )
        store.record_workflow_step(
            plan.run_id,
            step,
            WorkflowStepState.COMPLETED,
            occurred_at=started_at + timedelta(seconds=1),
            evidence_id=evidence.evidence_id,
        )

    requirement = plan.approval_requirements[-1]
    store.record_workflow_step(
        plan.run_id,
        PipelineStep.PROMOTE_OR_ROLL_BACK,
        WorkflowStepState.AWAITING_APPROVAL,
        occurred_at=NOW + timedelta(seconds=20),
    )
    with pytest.raises(ApprovalRequiredError, match="human approval"):
        store.record_workflow_step(
            plan.run_id,
            PipelineStep.PROMOTE_OR_ROLL_BACK,
            WorkflowStepState.STARTED,
            occurred_at=NOW + timedelta(seconds=21),
        )

    decision = HumanDecision(
        requirement_id=requirement.requirement_id,
        plan_digest=plan.digest,
        human_subject="operator@example.test",
        verdict=HumanVerdict.APPROVE,
        decided_at=NOW + timedelta(seconds=21),
        note="Measured gate reviewed.",
    )
    assert store.record_human_decision(decision) == decision
    assert store.record_human_decision(decision) == decision
    store.record_workflow_step(
        plan.run_id,
        PipelineStep.PROMOTE_OR_ROLL_BACK,
        WorkflowStepState.STARTED,
        occurred_at=NOW + timedelta(seconds=22),
    )
    store.record_workflow_step(
        plan.run_id,
        PipelineStep.PROMOTE_OR_ROLL_BACK,
        WorkflowStepState.COMPLETED,
        occurred_at=NOW + timedelta(seconds=23),
        evidence_id=evidence.evidence_id,
    )

    history = store.workflow_history(plan.run_id)
    assert history[-1].state is WorkflowStepState.COMPLETED
    assert history[-1].evidence_id == evidence.evidence_id
    store.close()


def test_policy_actions_require_numeric_gate_and_separate_human_approval(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coordinator.sqlite3"
    store = CoordinatorStore(path)
    baseline = evaluated_policy("policy-v0", POLICY_V0_HASH)
    candidate = evaluated_policy("policy-v1", POLICY_V1_HASH)
    store.register_evaluated_checkpoint(baseline)
    store.register_evaluated_checkpoint(candidate)
    store.register_evaluated_checkpoint(candidate)
    store.initialize_policy_alias("stable", baseline.policy_id, occurred_at=NOW)

    with pytest.raises(ValueError, match="numeric promotion gate"):
        NumericPolicyDecision(
            decision_id="decision-failing-promotion",
            run_id="run-missing",
            plan_digest="0" * 64,
            action=PolicyAction.PROMOTE,
            alias="stable",
            from_policy_id=baseline.policy_id,
            target_policy_id=candidate.policy_id,
            evaluation_evidence_hash=EVIDENCE_HASH,
            metrics=replace(passing_metrics(), held_out_success_rate=0.5),
            decided_at=NOW,
        )

    promotion_plan = execution_plan("run-policy-promotion", PolicyAction.PROMOTE)
    store.register_workflow(promotion_plan, created_at=NOW)
    promotion = NumericPolicyDecision(
        decision_id="decision-policy-promotion",
        run_id=promotion_plan.run_id,
        plan_digest=promotion_plan.digest,
        action=PolicyAction.PROMOTE,
        alias="stable",
        from_policy_id=baseline.policy_id,
        target_policy_id=candidate.policy_id,
        evaluation_evidence_hash=EVIDENCE_HASH,
        metrics=passing_metrics(),
        decided_at=NOW + timedelta(seconds=1),
    )
    assert store.record_numeric_policy_decision(promotion) == promotion
    assert store.record_numeric_policy_decision(promotion) == promotion
    promotion_requirement = promotion_plan.approval_requirements[-1]
    with pytest.raises(ApprovalRequiredError, match="human decision"):
        store.apply_policy_action(
            promotion.decision_id,
            promotion_requirement.requirement_id,
            occurred_at=NOW + timedelta(seconds=2),
        )

    store.record_human_decision(
        HumanDecision(
            requirement_id=promotion_requirement.requirement_id,
            plan_digest=promotion_plan.digest,
            human_subject="operator@example.test",
            verdict=HumanVerdict.APPROVE,
            decided_at=NOW + timedelta(seconds=2),
            note="Promote based on the held-out comparison.",
        )
    )
    promoted = store.apply_policy_action(
        promotion.decision_id,
        promotion_requirement.requirement_id,
        occurred_at=NOW + timedelta(seconds=3),
    )
    assert store.current_policy("stable") == candidate.policy_id
    assert store.apply_policy_action(
        promotion.decision_id,
        promotion_requirement.requirement_id,
        occurred_at=NOW + timedelta(seconds=3),
    ) == promoted
    assert store.record_numeric_policy_decision(promotion) == promotion

    rollback_plan = execution_plan("run-policy-rollback", PolicyAction.ROLL_BACK)
    store.register_workflow(rollback_plan, created_at=NOW + timedelta(seconds=4))
    rollback = NumericPolicyDecision(
        decision_id="decision-policy-rollback",
        run_id=rollback_plan.run_id,
        plan_digest=rollback_plan.digest,
        action=PolicyAction.ROLL_BACK,
        alias="stable",
        from_policy_id=candidate.policy_id,
        target_policy_id=baseline.policy_id,
        evaluation_evidence_hash="f" * 64,
        metrics=replace(passing_metrics(), held_out_success_rate=0.4),
        decided_at=NOW + timedelta(seconds=5),
    )
    store.record_numeric_policy_decision(rollback)
    rollback_requirement = rollback_plan.approval_requirements[-1]
    store.record_human_decision(
        HumanDecision(
            requirement_id=rollback_requirement.requirement_id,
            plan_digest=rollback_plan.digest,
            human_subject="operator@example.test",
            verdict=HumanVerdict.APPROVE,
            decided_at=NOW + timedelta(seconds=6),
            note="Roll back after measured regression.",
        )
    )
    store.apply_policy_action(
        rollback.decision_id,
        rollback_requirement.requirement_id,
        occurred_at=NOW + timedelta(seconds=7),
    )
    assert store.current_policy("stable") == baseline.policy_id
    assert [event.action for event in store.policy_alias_history("stable")] == [
        "initialize",
        "promote",
        "roll_back",
    ]
    store.close()

    with CoordinatorStore(path) as reopened:
        assert reopened.current_policy("stable") == baseline.policy_id
        connection = sqlite3.connect(path)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute(
                    "UPDATE evaluated_checkpoints SET checkpoint_hash = ? WHERE policy_id = ?",
                    ("1" * 64, candidate.policy_id),
                )
        finally:
            connection.close()
