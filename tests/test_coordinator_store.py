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
from muscle_memory.orchestration.evidence import GuildEvidenceBundle

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROBOT_HASH = "a" * 64
WORLD_HASH = "b" * 64
POLICY_V0_HASH = "c" * 64
POLICY_V1_HASH = "d" * 64
EVIDENCE_HASH = "e" * 64
POLICY_V0_EVIDENCE_HASH = "f" * 64
POLICY_V1_EVIDENCE_HASH = "1" * 64


def training_episode(episode_id: str = "episode-training-1") -> TrainingEpisodeMetadata:
    return TrainingEpisodeMetadata(
        episode_id=episode_id,
        robot_checksum=ROBOT_HASH,
        world_hash=WORLD_HASH,
        policy_hash=POLICY_V0_HASH,
        created_at=NOW,
    )


def execution_plan(run_id: str, action: PolicyAction = PolicyAction.PROMOTE) -> ExecutionPlan:
    baseline_policy_id = "policy-v0" if action is PolicyAction.PROMOTE else "policy-v1"
    candidate_policy_id = "policy-v1" if action is PolicyAction.PROMOTE else "policy-v0"
    payloads: dict[PipelineStep, dict[str, object]] = {
        PipelineStep.VALIDATE_WORLD: {
            "uncertain_physical_properties": False,
            "world_id": "world-training-1",
        },
        PipelineStep.RUN_EPISODE: {
            "episode_id": "episode-training-1",
            "world_id": "world-training-1",
        },
        PipelineStep.SUMMARIZE_TELEMETRY: {"episode_id": "episode-training-1"},
        PipelineStep.QUERY_GRAPH_MEMORY: {"episode_id": "episode-training-1"},
        PipelineStep.SELECT_CURRICULUM: {
            "curriculum_change_requested": False,
            "episode_id": "episode-training-1",
        },
        PipelineStep.TRAIN_CANDIDATE_POLICY: {
            "reward_change_requested": False,
            "candidate_policy_id": candidate_policy_id,
        },
        PipelineStep.EVALUATE_CANDIDATE_POLICY: {
            "baseline_policy_id": baseline_policy_id,
            "candidate_policy_id": candidate_policy_id,
            "heldout_world_set_id": "heldout-v1",
        },
        PipelineStep.PROMOTE_OR_ROLL_BACK: {
            "action": action.value,
            "candidate_policy_id": candidate_policy_id,
        },
    }
    commands = tuple(PipelineCommand.create(step, payloads[step]) for step in FIXED_PIPELINE)
    return ExecutionPlan.create(run_id, commands)


def evaluated_policy(policy_id: str, checkpoint_hash: str) -> EvaluatedPolicyVersion:
    if policy_id == "policy-v0":
        evidence_hash = POLICY_V0_EVIDENCE_HASH
        metrics = {
            "success_rate": 0.65,
            "collision_rate": 0.125,
            "falls": 0,
            "median_clearance_m": 0.31,
            "path_efficiency_regression_fraction": 0.03,
        }
    else:
        evidence_hash = POLICY_V1_EVIDENCE_HASH
        metrics = {
            "success_rate": 0.9,
            "collision_rate": 0.05,
            "falls": 0,
            "median_clearance_m": 0.31,
            "path_efficiency_regression_fraction": 0.03,
        }
    return EvaluatedPolicyVersion.create(
        policy_id=policy_id,
        checkpoint_hash=checkpoint_hash,
        evaluation_evidence_hash=evidence_hash,
        evaluation_split="held_out",
        metrics=metrics,
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


def rollback_metrics() -> PolicyGateMetrics:
    return PolicyGateMetrics(
        held_out_success_rate=0.65,
        collision_rate=0.125,
        fall_count=0,
        median_clearance_m=0.31,
        success_rate_delta=0.65 - 0.9,
        collision_reduction_fraction=(0.05 - 0.125) / 0.05,
        path_efficiency_regression_fraction=0.03,
    )


def register_workflow_evidence(
    store: CoordinatorStore,
    plan: ExecutionPlan,
    *,
    candidate_checksum_override: str | None = None,
) -> str:
    evaluation_command = plan.commands[FIXED_PIPELINE.index(PipelineStep.EVALUATE_CANDIDATE_POLICY)]
    final_command = plan.commands[FIXED_PIPELINE.index(PipelineStep.PROMOTE_OR_ROLL_BACK)]
    baseline_id = str(evaluation_command.payload["baseline_policy_id"])
    candidate_id = str(evaluation_command.payload["candidate_policy_id"])
    policies = {
        "policy-v0": {
            "checksum": POLICY_V0_HASH,
            "evidence_hash": POLICY_V0_EVIDENCE_HASH,
            "success_rate": 0.65,
            "collision_rate": 0.125,
        },
        "policy-v1": {
            "checksum": POLICY_V1_HASH,
            "evidence_hash": POLICY_V1_EVIDENCE_HASH,
            "success_rate": 0.9,
            "collision_rate": 0.05,
        },
    }
    safe_run_id = plan.run_id.replace("-", ".")
    bundle = GuildEvidenceBundle.model_validate(
        {
            "world": {
                "evidence_id": f"guild.world.{safe_run_id}",
                "world_evidence": {
                    "world_id": "world-training-1",
                    "world_digest": WORLD_HASH,
                    "baseline_path_digest": "2" * 64,
                    "robot_checksum_unchanged": True,
                    "validation": {
                        "no_overlapping_objects": True,
                        "start_destination_connected": True,
                        "passages_meet_minimum_clearance": True,
                        "approved_colliders_only": True,
                        "baseline_path_exists": True,
                        "physical_parameters_within_safe_limits": True,
                    },
                    "obstacles": [
                        {
                            "obstacle_id": "chair-001",
                            "proposal_digest": "3" * 64,
                            "dimensions_m": [0.5, 0.5, 0.9],
                            "mass_kg": 6.5,
                            "friction": 0.7,
                            "property_origin": "catalog_confirmed",
                            "collision_geometry": "primitive",
                            "render_mesh_used_for_collision": False,
                        }
                    ],
                },
            },
            "failure_curriculum": {
                "evidence_id": f"guild.curriculum.{safe_run_id}",
                "failure_curriculum_evidence": {
                    "source_split": "training",
                    "source_policy_id": baseline_id,
                    "graph_query_digest": "4" * 64,
                    "failure_patterns": [
                        {
                            "signature": "clearance-chair",
                            "source_episode_ids": ["episode-011", "episode-024"],
                            "distinct_source_episode_count": 2,
                            "obstacle_categories": ["chair"],
                            "approved_correction_ids": ["correction-009"],
                            "lesson_ids": ["lesson-003"],
                        }
                    ],
                    "curriculum_change_requested": False,
                },
            },
            "evaluation": {
                "evidence_id": f"guild.evaluation.{safe_run_id}",
                "evaluation_evidence": {
                    "heldout_world_set_id": "heldout-v1",
                    "heldout_world_set_digest": "5" * 64,
                    "paired_world_count": 20,
                    "baseline": {
                        "policy_id": baseline_id,
                        "policy_checksum": policies[baseline_id]["checksum"],
                        "evaluation_id": policies[baseline_id]["evidence_hash"],
                        "success_rate": policies[baseline_id]["success_rate"],
                        "collision_rate": policies[baseline_id]["collision_rate"],
                    },
                    "candidate": {
                        "policy_id": candidate_id,
                        "policy_checksum": (
                            candidate_checksum_override
                            or policies[candidate_id]["checksum"]
                        ),
                        "evaluation_id": policies[candidate_id]["evidence_hash"],
                        "success_rate": policies[candidate_id]["success_rate"],
                        "collision_rate": policies[candidate_id]["collision_rate"],
                        "falls": 0,
                        "median_clearance_m": 0.31,
                        "path_efficiency_regression_fraction": 0.03,
                    },
                    "proposed_action": final_command.payload["action"],
                },
            },
        }
    )
    for evidence_id, kind, artifact_hash in bundle.artifact_hashes():
        store.record_provider_evidence(
            ProviderEvidenceReference(
                evidence_id=evidence_id,
                provider="coordinator-domain-validation",
                evidence_kind=kind,
                provider_object_id=evidence_id,
                artifact_hash=artifact_hash,
                observed_at=NOW,
            )
        )
    store.record_workflow_guild_evidence(plan.run_id, bundle)
    return next(
        artifact_hash
        for _evidence_id, kind, artifact_hash in bundle.artifact_hashes()
        if kind == "guild_evaluation_evidence"
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


def test_episode_review_notes_persist_without_mutating_episode_history(tmp_path: Path) -> None:
    path = tmp_path / "coordinator.sqlite3"
    note_id = "note-" + "1" * 32
    with CoordinatorStore(path) as store:
        store.register_training_episode(training_episode())
        note = store.create_episode_review_note(
            note_id=note_id,
            episode_id="episode-training-1",
            author_subject="human-operator",
            body="Inspect the clearance trace before curriculum review.",
            tags=("clearance", "curriculum"),
            created_at=NOW,
        )
        assert store.episode_review_notes("episode-training-1") == (note,)
        updated = store.update_episode_review_note(
            note_id,
            archived=True,
            updated_at=NOW + timedelta(minutes=1),
        )
        assert updated is not None and updated.archived is True
        assert store.episode_review_notes("episode-training-1") == ()

    with CoordinatorStore(path) as reopened:
        archived = reopened.episode_review_notes(
            "episode-training-1",
            include_archived=True,
        )
        assert archived[0].body.startswith("Inspect the clearance")
        assert [event.state for event in reopened.episode_history("episode-training-1")] == [
            EpisodeState.CREATED
        ]


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
    promotion_evidence_hash = register_workflow_evidence(store, promotion_plan)
    promotion = NumericPolicyDecision(
        decision_id="decision-policy-promotion",
        run_id=promotion_plan.run_id,
        plan_digest=promotion_plan.digest,
        action=PolicyAction.PROMOTE,
        alias="stable",
        from_policy_id=baseline.policy_id,
        target_policy_id=candidate.policy_id,
        evaluation_evidence_hash=promotion_evidence_hash,
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
    rollback_evidence_hash = register_workflow_evidence(store, rollback_plan)
    rollback_evaluation = rollback_plan.commands[
        FIXED_PIPELINE.index(PipelineStep.EVALUATE_CANDIDATE_POLICY)
    ].payload
    rollback_baseline_id = str(rollback_evaluation["baseline_policy_id"])
    failed_candidate_id = str(rollback_evaluation["candidate_policy_id"])
    with pytest.raises(
        CoordinatorIntegrityError,
        match="identities do not match trusted evaluation evidence",
    ):
        store.record_numeric_policy_decision(
            NumericPolicyDecision(
                decision_id="decision-policy-rollback-to-failed-candidate",
                run_id=rollback_plan.run_id,
                plan_digest=rollback_plan.digest,
                action=PolicyAction.ROLL_BACK,
                alias="stable",
                from_policy_id=candidate.policy_id,
                target_policy_id=failed_candidate_id,
                evaluation_evidence_hash=rollback_evidence_hash,
                metrics=rollback_metrics(),
                decided_at=NOW + timedelta(seconds=5),
            )
        )
    rollback = NumericPolicyDecision(
        decision_id="decision-policy-rollback",
        run_id=rollback_plan.run_id,
        plan_digest=rollback_plan.digest,
        action=PolicyAction.ROLL_BACK,
        alias="stable",
        from_policy_id=candidate.policy_id,
        target_policy_id=rollback_baseline_id,
        evaluation_evidence_hash=rollback_evidence_hash,
        metrics=rollback_metrics(),
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
    assert store.current_policy("stable") == rollback_baseline_id
    assert [event.action for event in store.policy_alias_history("stable")] == [
        "initialize",
        "promote",
        "roll_back",
    ]
    store.close()

    with CoordinatorStore(path) as reopened:
        assert reopened.current_policy("stable") == rollback_baseline_id
        connection = sqlite3.connect(path)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute(
                    "UPDATE evaluated_checkpoints SET checkpoint_hash = ? WHERE policy_id = ?",
                    ("1" * 64, candidate.policy_id),
                )
        finally:
            connection.close()


def test_numeric_policy_decision_rejects_evidence_and_metric_tampering(
    tmp_path: Path,
) -> None:
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    store.register_evaluated_checkpoint(evaluated_policy("policy-v0", POLICY_V0_HASH))
    store.register_evaluated_checkpoint(evaluated_policy("policy-v1", POLICY_V1_HASH))
    store.initialize_policy_alias("stable", "policy-v0", occurred_at=NOW)
    plan = execution_plan("run-policy-tamper", PolicyAction.PROMOTE)
    store.register_workflow(plan, created_at=NOW)
    evaluation_hash = register_workflow_evidence(store, plan)

    def decision(
        decision_id: str,
        *,
        evidence_hash: str = evaluation_hash,
        metrics: PolicyGateMetrics | None = None,
    ) -> NumericPolicyDecision:
        return NumericPolicyDecision(
            decision_id=decision_id,
            run_id=plan.run_id,
            plan_digest=plan.digest,
            action=PolicyAction.PROMOTE,
            alias="stable",
            from_policy_id="policy-v0",
            target_policy_id="policy-v1",
            evaluation_evidence_hash=evidence_hash,
            metrics=passing_metrics() if metrics is None else metrics,
            decided_at=NOW,
        )

    with pytest.raises(CoordinatorIntegrityError, match="trusted evaluation artifact"):
        store.record_numeric_policy_decision(
            decision("decision-tampered-hash", evidence_hash="9" * 64)
        )
    with pytest.raises(CoordinatorIntegrityError, match="exactly recomputed"):
        store.record_numeric_policy_decision(
            decision(
                "decision-tampered-metrics",
                metrics=replace(passing_metrics(), median_clearance_m=0.32),
            )
        )

    mismatched_plan = execution_plan("run-policy-checkpoint-tamper", PolicyAction.PROMOTE)
    store.register_workflow(mismatched_plan, created_at=NOW + timedelta(seconds=1))
    mismatched_hash = register_workflow_evidence(
        store,
        mismatched_plan,
        candidate_checksum_override="8" * 64,
    )
    mismatched = replace(
        decision("decision-tampered-checkpoint"),
        run_id=mismatched_plan.run_id,
        plan_digest=mismatched_plan.digest,
        evaluation_evidence_hash=mismatched_hash,
    )
    with pytest.raises(CoordinatorIntegrityError, match="evaluated checkpoints"):
        store.record_numeric_policy_decision(mismatched)
    store.close()
