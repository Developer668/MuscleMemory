"""End-to-end sponsor callback composition and durable evidence tests."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from muscle_memory.api import Sha256BearerAuthenticator, create_app
from muscle_memory.backend.policy_decisions import record_reviewed_numeric_decision
from muscle_memory.backend.rocketride_callback import (
    MAX_CALLBACK_BODY_BYTES,
    CallbackContractError,
    FixedStepDispatcher,
)
from muscle_memory.coordinator import (
    EpisodeState,
    HeldOutEvaluationArtifact,
    HeldOutEvaluationEpisodeMetadata,
    HeldOutEvaluationResult,
)
from muscle_memory.coordinator.models import ProviderEvidenceReference, sha256_text
from muscle_memory.episodes import EpisodeIdentity
from muscle_memory.evaluation.promotion import PromotionDecision, evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    EvaluatedPolicyVersion,
    WorldMemoryRecord,
    WorldSplit,
)
from muscle_memory.orchestration.approvals import HumanDecision, HumanVerdict
from muscle_memory.orchestration.contracts import (
    FIXED_PIPELINE,
    ContractViolationError,
    ExecutionPlan,
    GuildReview,
    GuildReviewSet,
    GuildRole,
    HealthState,
    PipelineCommand,
    PipelineStep,
    ProviderMode,
    ProviderName,
    ProviderStatus,
    ReviewRecommendation,
    canonical_json,
)
from muscle_memory.orchestration.evidence import (
    GuildEvidenceBundle,
    validate_evidence_plan_binding,
)
from muscle_memory.orchestration.rocketride import ReviewBlockedError
from muscle_memory.orchestration.service import ReviewedExecution
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.runtime import build_api_backend
from muscle_memory.telemetry import (
    EpisodeTelemetryRecord,
    SensorSnapshot,
    SignalUseLabel,
)

CALLBACK_TOKEN = "callback-runtime-token-0123456789abcdef"


def _plan(run_id: str) -> ExecutionPlan:
    commands = (
        PipelineCommand.create(
            PipelineStep.VALIDATE_WORLD,
            {"uncertain_physical_properties": False, "world_id": "world-001"},
        ),
        PipelineCommand.create(
            PipelineStep.RUN_EPISODE,
            {"episode_id": "episode-001", "world_id": "world-001"},
        ),
        PipelineCommand.create(
            PipelineStep.SUMMARIZE_TELEMETRY,
            {"episode_id": "episode-001"},
        ),
        PipelineCommand.create(
            PipelineStep.QUERY_GRAPH_MEMORY,
            {"episode_id": "episode-001"},
        ),
        PipelineCommand.create(
            PipelineStep.SELECT_CURRICULUM,
            {"curriculum_change_requested": False, "episode_id": "episode-001"},
        ),
        PipelineCommand.create(
            PipelineStep.TRAIN_CANDIDATE_POLICY,
            {"reward_change_requested": False, "candidate_policy_id": "candidate-001"},
        ),
        PipelineCommand.create(
            PipelineStep.EVALUATE_CANDIDATE_POLICY,
            {
                "baseline_policy_id": "baseline-001",
                "candidate_policy_id": "candidate-001",
                "heldout_world_set_id": "heldout-v1",
            },
        ),
        PipelineCommand.create(
            PipelineStep.PROMOTE_OR_ROLL_BACK,
            {"action": "promote", "candidate_policy_id": "candidate-001"},
        ),
    )
    return ExecutionPlan.create(run_id, commands)


def _bundle(
    *,
    artifact_hash: str | None = None,
    decision: PromotionDecision | None = None,
) -> GuildEvidenceBundle:
    baseline_evaluation_id = artifact_hash or "evaluation-baseline-001"
    candidate_evaluation_id = artifact_hash or "evaluation-candidate-001"
    baseline_success_rate = 0.55 if decision is None else decision.baseline.success_rate
    baseline_collision_rate = 0.4 if decision is None else decision.baseline.collision_rate
    candidate_success_rate = 0.85 if decision is None else decision.candidate.success_rate
    candidate_collision_rate = 0.1 if decision is None else decision.candidate.collision_rate
    candidate_falls = 0 if decision is None else decision.candidate.total_falls
    candidate_clearance = (
        0.3 if decision is None else decision.candidate.median_minimum_clearance_m
    )
    path_regression = (
        0.05 if decision is None else decision.path_efficiency_regression
    )
    return GuildEvidenceBundle.model_validate(
        {
            "world": {
                "evidence_id": "evidence.world.1",
                "world_evidence": {
                    "world_id": "world-001",
                    "world_digest": "a" * 64,
                    "baseline_path_digest": "b" * 64,
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
                            "proposal_digest": "c" * 64,
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
                "evidence_id": "evidence.curriculum.1",
                "failure_curriculum_evidence": {
                    "source_split": "training",
                    "source_policy_id": "baseline-001",
                    "graph_query_digest": "d" * 64,
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
                "evidence_id": "evidence.evaluation.1",
                "evaluation_evidence": {
                    "heldout_world_set_id": "heldout-v1",
                    "heldout_world_set_digest": "e" * 64,
                    "paired_world_count": 20,
                    "baseline": {
                        "policy_id": "baseline-001",
                        "policy_checksum": "f" * 64,
                        "evaluation_id": baseline_evaluation_id,
                        "success_rate": baseline_success_rate,
                        "collision_rate": baseline_collision_rate,
                    },
                    "candidate": {
                        "policy_id": "candidate-001",
                        "policy_checksum": "1" * 64,
                        "evaluation_id": candidate_evaluation_id,
                        "success_rate": candidate_success_rate,
                        "collision_rate": candidate_collision_rate,
                        "falls": candidate_falls,
                        "median_clearance_m": candidate_clearance,
                        "path_efficiency_regression_fraction": path_regression,
                    },
                    "proposed_action": "promote",
                },
            },
        }
    )


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "MUSCLE_MEMORY_COORDINATOR_DB_PATH": str(tmp_path / "coordinator.sqlite3"),
        "MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "graph.jsonl"),
        "MUSCLE_MEMORY_TELEMETRY_SPOOL": str(tmp_path / "laser-spool.sqlite3"),
        "MM_ASSET_CACHE_DIR": str(tmp_path / "assets"),
        "MM_ASSET_APPROVAL_LEDGER_DIR": str(tmp_path / "approvals"),
        "ROCKETRIDE_MM_COORDINATOR_URL": "http://127.0.0.1:8000",
        "ROCKETRIDE_MM_COORDINATOR_TOKEN": CALLBACK_TOKEN,
    }


def _register_evidence(
    backend: object,
    plan: ExecutionPlan,
    *,
    bundle: GuildEvidenceBundle | None = None,
) -> None:
    coordinator = backend.coordinator  # type: ignore[attr-defined]
    bundle = bundle or _bundle()
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    for evidence_id, kind, artifact_hash in bundle.artifact_hashes():
        coordinator.record_provider_evidence(
            ProviderEvidenceReference(
                evidence_id=evidence_id,
                provider="coordinator-domain-validation",
                evidence_kind=kind,
                provider_object_id=evidence_id,
                artifact_hash=artifact_hash,
                observed_at=now,
            )
        )
    coordinator.register_workflow(plan, created_at=now)
    coordinator.record_workflow_guild_evidence(plan.run_id, bundle)
    provider = ProviderStatus(
        provider=ProviderName.GUILD,
        mode=ProviderMode.SIMULATION,
        health=HealthState.HEALTHY,
        detail="three synthetic contract reviews completed",
        checked_at=now,
    )
    reviewed = ReviewedExecution(
        plan=plan,
        guild_reviews=GuildReviewSet(
            plan_digest=plan.digest,
            reviews=tuple(
                GuildReview(
                    role=role,
                    plan_digest=plan.digest,
                    recommendation=ReviewRecommendation.PROCEED,
                    summary="synthetic exact-role contract review",
                )
                for role in GuildRole
            ),
            provider_status=provider,
        ),
    )
    coordinator.record_workflow_review(
        plan.run_id,
        canonical_json(
            TypeAdapter(ReviewedExecution).dump_python(reviewed, mode="json")
        ),
    )


def _paired_result(
    *,
    index: int,
    policy_id: str,
    policy_hash: str,
    robot_checksum: str,
    success_count: int,
    collision_start: int,
    collision_count: int,
    path_efficiency: float,
) -> PolicyEpisodeResult:
    world_id = f"synthetic-heldout-world-{index:02d}"
    success = index < success_count
    collision = collision_start <= index < collision_start + collision_count
    return PolicyEpisodeResult(
        episode_id=f"synthetic-{policy_id}-{index:02d}",
        world_id=world_id,
        world_seed=10_000 + index,
        world_split="held_out",
        world_hash=sha256_text(world_id),
        robot_checksum=robot_checksum,
        policy_id=policy_id,
        policy_hash=policy_hash,
        success=success,
        failed_reasons=() if success else ("SAFE_DELIVERY_FAILED",),
        time_to_resident_seconds=20.0 if success else None,
        simulated_duration_seconds=20.0,
        stop_distance_m=0.3 if success else 1.0,
        facing_error_degrees=5.0 if success else None,
        stopped_speed_mps=0.0,
        falls=0,
        body_collisions=int(collision),
        minimum_obstacle_clearance_m=0.3,
        maximum_tray_tilt_degrees=5.0,
        package_slipped=False,
        human_interventions=0,
        direct_distance_m=4.0,
        path_length_m=5.0,
        path_efficiency=path_efficiency,
        energy_joules=20.0,
        task_policy_updates=200,
        trace=(),
    )


def _admit_synthetic_paired_evaluation(
    backend: object,
    plan: ExecutionPlan,
    *,
    robot_checksum: str,
) -> tuple[GuildEvidenceBundle, str]:
    coordinator = backend.coordinator  # type: ignore[attr-defined]
    baseline_results = tuple(
        _paired_result(
            index=index,
            policy_id="baseline-001",
            policy_hash="f" * 64,
            robot_checksum=robot_checksum,
            success_count=11,
            collision_start=11,
            collision_count=8,
            path_efficiency=0.8,
        )
        for index in range(20)
    )
    candidate_results = tuple(
        _paired_result(
            index=index,
            policy_id="candidate-001",
            policy_hash="1" * 64,
            robot_checksum=robot_checksum,
            success_count=17,
            collision_start=17,
            collision_count=2,
            path_efficiency=0.76,
        )
        for index in range(20)
    )
    decision = evaluate_promotion(baseline_results, candidate_results)
    assert decision.promotable is True
    artifact_json = canonical_json(
        {
            "schema_version": 1,
            "heldout_bundle_sha256": "9" * 64,
            "candidate_checkpoint_sha256": "1" * 64,
            "baseline_results": [asdict(item) for item in baseline_results],
            "candidate_results": [asdict(item) for item in candidate_results],
            "promotion_decision": asdict(decision),
        }
    )
    artifact_hash = sha256_text(artifact_json)
    coordinator.record_held_out_evaluation_artifact(
        HeldOutEvaluationArtifact(
            artifact_hash=artifact_hash,
            held_out_world_set_id="heldout-v1",
            artifact_json=artifact_json,
            evaluated_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
        )
    )
    baseline_checkpoint = EvaluatedPolicyVersion.create(
        policy_id="baseline-001",
        checkpoint_hash="f" * 64,
        evaluation_evidence_hash=artifact_hash,
        evaluation_split="held_out",
        metrics={
            "success_rate": decision.baseline.success_rate,
            "collision_rate": decision.baseline.collision_rate,
        },
        evaluated_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    candidate_checkpoint = EvaluatedPolicyVersion.create(
        policy_id="candidate-001",
        checkpoint_hash="1" * 64,
        evaluation_evidence_hash=artifact_hash,
        evaluation_split="held_out",
        metrics={
            "success_rate": decision.candidate.success_rate,
            "collision_rate": decision.candidate.collision_rate,
            "falls": decision.candidate.total_falls,
            "median_clearance_m": decision.candidate.median_minimum_clearance_m,
            "path_efficiency_regression_fraction": decision.path_efficiency_regression,
        },
        evaluated_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    coordinator.register_evaluated_checkpoint(baseline_checkpoint)
    coordinator.register_evaluated_checkpoint(candidate_checkpoint)
    coordinator.initialize_policy_alias(
        "stable",
        baseline_checkpoint.policy_id,
        occurred_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    for result in (*baseline_results, *candidate_results):
        coordinator.register_held_out_evaluation_episode(
            HeldOutEvaluationEpisodeMetadata(
                episode_id=result.episode_id,
                robot_checksum=result.robot_checksum,
                world_hash=result.world_hash,
                policy_hash=result.policy_hash,
                held_out_world_set_id="heldout-v1",
                created_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
            )
        )
        coordinator.transition_episode(
            result.episode_id,
            EpisodeState.RUNNING,
            occurred_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
        )
        coordinator.transition_episode(
            result.episode_id,
            EpisodeState.SUCCEEDED if result.success else EpisodeState.FAILED,
            occurred_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
        )
        coordinator.record_held_out_evaluation_result(
            HeldOutEvaluationResult(
                episode_id=result.episode_id,
                evaluation_artifact_hash=artifact_hash,
                result_json=canonical_json(asdict(result)),
            )
        )
    return _bundle(artifact_hash=artifact_hash, decision=decision), artifact_hash


async def _close_callback_episode(backend: object, *, robot_checksum: str) -> None:
    runtime = backend.episode_runtime  # type: ignore[attr-defined]
    runtime.service._graph_prerequisites = None
    graph = backend.providers.graph_memory  # type: ignore[attr-defined]
    graph.record_world(
        WorldMemoryRecord(
            world_id="world-001",
            world_hash="a" * 64,
            split=WorldSplit.TRAINING,
            seed=7,
            generation_version=1,
            validation_hash="b" * 64,
            validated=True,
            recorded_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
        )
    )
    baseline = next(
        checkpoint
        for checkpoint in backend.coordinator.evaluated_checkpoints()  # type: ignore[attr-defined]
        if checkpoint.policy_id == "baseline-001"
    )
    graph.record_evaluated_policy(baseline)
    identity = EpisodeIdentity(
        episode_id="episode-001",
        robot_checksum=robot_checksum,
        world_id="world-001",
        world_hash="a" * 64,
        world_split=WorldSplit.TRAINING,
        policy_id="baseline-001",
        policy_hash="f" * 64,
        opened_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    await runtime.open_episode(identity)
    await runtime.append_telemetry(
        EpisodeTelemetryRecord.create(
            episode_id=identity.episode_id,
            world_id=identity.world_id,
            policy_id=identity.policy_id,
            sequence=0,
            sim_time_seconds=0.0,
            robot_checksum=identity.robot_checksum,
            policy_hash=identity.policy_hash,
            world_hash=identity.world_hash,
            signal_use=SignalUseLabel.LOGGED_ONLY,
            sensors=SensorSnapshot.all_unavailable(),
            payload={"sample": 0},
            frame_id="frame-episode-001-000000",
        )
    )
    await runtime.close_episode(
        PolicyEpisodeResult(
            episode_id=identity.episode_id,
            world_id=identity.world_id,
            world_seed=7,
            world_split="training",
            world_hash=identity.world_hash,
            robot_checksum=identity.robot_checksum,
            policy_id=identity.policy_id,
            policy_hash=identity.policy_hash,
            success=True,
            failed_reasons=(),
            time_to_resident_seconds=20.0,
            simulated_duration_seconds=20.0,
            stop_distance_m=0.3,
            facing_error_degrees=5.0,
            stopped_speed_mps=0.0,
            falls=0,
            body_collisions=0,
            minimum_obstacle_clearance_m=0.3,
            maximum_tray_tilt_degrees=5.0,
            package_slipped=False,
            human_interventions=0,
            direct_distance_m=4.0,
            path_length_m=5.0,
            path_efficiency=0.8,
            energy_joules=20.0,
            task_policy_updates=200,
            trace=(),
        ),
        closed_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )


def _envelope(
    plan: ExecutionPlan,
    step: PipelineStep,
    *,
    approval_evidence: tuple[dict[str, str], ...] = (),
) -> str:
    command = plan.commands[FIXED_PIPELINE.index(step)]
    payload: dict[str, object] = {
        "contract_version": 1,
        "run_id": plan.run_id,
        "plan_digest": plan.digest,
        "step": step.value,
        "payload": command.payload,
    }
    if approval_evidence:
        payload["approval_evidence"] = list(approval_evidence)
    return canonical_json(payload)


def test_callback_is_authenticated_ordered_and_durable_across_restart(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    plan = _plan("callback-run-001")
    backend = build_api_backend(environment)
    _register_evidence(backend, plan)
    app = create_app(backend=backend, authenticator=Sha256BearerAuthenticator(()))
    encoded = _envelope(plan, PipelineStep.VALIDATE_WORLD)

    with TestClient(app) as client:
        unauthorized = client.post(
            "/webhook/muscle-memory-fixed-step",
            json={"data": encoded},
        )
        first = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
            json={"data": encoded},
        )

    assert unauthorized.status_code == 401
    assert first.status_code == 200
    assert first.json()["output"]["world_valid"] is True

    restarted = build_api_backend(environment)
    restarted_app = create_app(
        backend=restarted,
        authenticator=Sha256BearerAuthenticator(()),
    )
    with TestClient(restarted_app) as client:
        replay = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
            json={"data": encoded},
        )
    assert replay.status_code == 200
    assert replay.json() == first.json()


@pytest.mark.parametrize(
    "session_ids",
    [
        (None, None, None),
        ("shared-session", "shared-session", "shared-session"),
    ],
    ids=("missing", "duplicate"),
)
def test_legacy_live_review_with_untrusted_session_ids_is_quarantined(
    tmp_path: Path,
    session_ids: tuple[str | None, str | None, str | None],
) -> None:
    environment = _environment(tmp_path)
    plan = _plan("legacy-guild-review")
    seeded = build_api_backend(environment)
    seeded.coordinator.register_workflow(
        plan,
        created_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    provider = ProviderStatus(
        provider=ProviderName.GUILD,
        mode=ProviderMode.SIMULATION,
        health=HealthState.HEALTHY,
        detail="legacy provider review completed",
        checked_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    reviewed = ReviewedExecution(
        plan=plan,
        guild_reviews=GuildReviewSet(
            plan_digest=plan.digest,
            reviews=tuple(
                GuildReview(
                    role=role,
                    plan_digest=plan.digest,
                    recommendation=ReviewRecommendation.PROCEED,
                    summary="legacy review with untrusted provider session identity",
                    provider_session_id=session_id,
                )
                for role, session_id in zip(GuildRole, session_ids, strict=True)
            ),
            provider_status=provider,
        ),
    )
    raw = TypeAdapter(ReviewedExecution).dump_python(reviewed, mode="json")
    raw["guild_reviews"]["provider_status"]["mode"] = "live"
    seeded.coordinator.record_workflow_review(plan.run_id, canonical_json(raw))
    asyncio.run(seeded.shutdown())

    restored = build_api_backend(environment)
    legacy = restored._reviewed[plan.run_id]
    assert legacy.guild_reviews.provider_status.health is HealthState.DEGRADED
    assert not legacy.guild_reviews.executable
    assert "execution is quarantined" in (
        legacy.guild_reviews.provider_status.detail
    )
    with pytest.raises(ReviewBlockedError, match="all Guild specialists"):
        asyncio.run(restored.orchestrator.execute(legacy))
    assert '"health":"healthy"' in (
        restored.coordinator.workflow_review(plan.run_id) or ""
    )


def test_callback_authentication_rejects_before_body_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = build_api_backend(_environment(tmp_path))
    app = create_app(backend=backend, authenticator=Sha256BearerAuthenticator(()))

    async def forbidden_body_read(_request: object) -> bytes:
        pytest.fail("unauthenticated callback body was consumed")

    monkeypatch.setattr(
        "muscle_memory.api.app._bounded_callback_body",
        forbidden_body_read,
    )
    with TestClient(app) as client:
        response = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={"Content-Type": "application/json"},
            content=b"not-json",
        )

    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


def test_historical_baseline_checkpoint_does_not_rebind_to_later_artifact() -> None:
    checkpoint = EvaluatedPolicyVersion.create(
        policy_id="baseline-001",
        checkpoint_hash="f" * 64,
        evaluation_evidence_hash="2" * 64,
        evaluation_split="held_out",
        metrics={"success_rate": 0.55, "collision_rate": 0.4},
        evaluated_at=datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
    )
    evidence = _bundle().evaluation.evaluation_evidence.baseline

    FixedStepDispatcher._verify_checkpoint(
        checkpoint,
        evidence,
        require_current_artifact_binding=False,
    )
    with pytest.raises(CallbackContractError, match="does not match trusted evidence"):
        FixedStepDispatcher._verify_checkpoint(checkpoint, evidence)


def test_all_eight_callbacks_resume_across_restarts_and_wait_for_human_gate(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    plan = _plan("callback-all-eight-restart")
    seeded = build_api_backend(environment)
    robot_checksum = verify_mm01_bundle().robot_checksum
    bundle, _artifact_hash = _admit_synthetic_paired_evaluation(
        seeded,
        plan,
        robot_checksum=robot_checksum,
    )
    _register_evidence(seeded, plan, bundle=bundle)
    record_reviewed_numeric_decision(
        seeded.coordinator,
        plan,
        stable_alias="stable",
        decided_at=datetime(2026, 8, 3, 12, 1, tzinfo=UTC),
    )
    asyncio.run(_close_callback_episode(seeded, robot_checksum=robot_checksum))
    asyncio.run(seeded.shutdown())

    first = build_api_backend(environment)
    first_app = create_app(
        backend=first,
        authenticator=Sha256BearerAuthenticator(()),
    )
    with TestClient(first_app) as client:
        for step in FIXED_PIPELINE[:3]:
            response = client.post(
                "/webhook/muscle-memory-fixed-step",
                headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
                json={"data": _envelope(plan, step)},
            )
            assert response.status_code == 200, response.text
            assert response.json()["output"]["operation_execution"]["contract_version"] == 1

    resumed = build_api_backend(environment)
    resumed_app = create_app(
        backend=resumed,
        authenticator=Sha256BearerAuthenticator(()),
    )
    with TestClient(resumed_app) as client:
        for step in FIXED_PIPELINE[3:7]:
            response = client.post(
                "/webhook/muscle-memory-fixed-step",
                headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
                json={"data": _envelope(plan, step)},
            )
            assert response.status_code == 200, response.text
            assert response.json()["output"]["operation_execution"]["contract_version"] == 1
        blocked = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
            json={"data": _envelope(plan, PipelineStep.PROMOTE_OR_ROLL_BACK)},
        )
        assert blocked.status_code == 403
        pending = asyncio.run(resumed.pending_approvals())
        assert tuple(item.run_id for item in pending.items) == (plan.run_id,)

    final_backend = build_api_backend(environment)
    requirement = plan.approval_requirements[-1]
    human_decision = HumanDecision(
        requirement_id=requirement.requirement_id,
        plan_digest=plan.digest,
        human_subject="operator@example.test",
        verdict=HumanVerdict.APPROVE,
        decided_at=datetime(2026, 8, 3, 12, 2, tzinfo=UTC),
        note="Approve the measured held-out promotion.",
    )
    final_backend.approval_ledger.record(human_decision)
    approval_evidence = {
        "requirement_id": requirement.requirement_id,
        "decision_id": human_decision.decision_id,
        "plan_digest": plan.digest,
        "step": requirement.step.value,
        "kind": requirement.kind.value,
        "verdict": human_decision.verdict.value,
        "human_subject": human_decision.human_subject,
        "decided_at": human_decision.decided_at.isoformat(),
    }
    final_app = create_app(
        backend=final_backend,
        authenticator=Sha256BearerAuthenticator(()),
    )
    with TestClient(final_app) as client:
        completed = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
            json={
                "data": _envelope(
                    plan,
                    PipelineStep.PROMOTE_OR_ROLL_BACK,
                    approval_evidence=(approval_evidence,),
                )
            },
        )
        assert completed.status_code == 200, completed.text
        assert completed.json()["output"]["action"] == "promote"
        assert final_backend.coordinator.current_policy("stable") == "candidate-001"


def test_callback_rejects_out_of_order_step(tmp_path: Path) -> None:
    plan = _plan("callback-run-002")
    backend = build_api_backend(_environment(tmp_path))
    _register_evidence(backend, plan)
    app = create_app(backend=backend, authenticator=Sha256BearerAuthenticator(()))

    with TestClient(app) as client:
        response = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={"Authorization": f"Bearer {CALLBACK_TOKEN}"},
            json={"data": _envelope(plan, PipelineStep.SUMMARIZE_TELEMETRY)},
        )

    assert response.status_code == 409
    assert response.json()["error"] == "sequence_violation"


def test_callback_stops_chunked_body_before_unbounded_accumulation(tmp_path: Path) -> None:
    backend = build_api_backend(_environment(tmp_path))
    app = create_app(backend=backend, authenticator=Sha256BearerAuthenticator(()))

    def chunks() -> Iterator[bytes]:
        chunk = b"x" * 8192
        for _ in range(MAX_CALLBACK_BODY_BYTES // len(chunk) + 2):
            yield chunk

    with TestClient(app) as client:
        response = client.post(
            "/webhook/muscle-memory-fixed-step",
            headers={
                "Authorization": f"Bearer {CALLBACK_TOKEN}",
                "Content-Type": "application/json",
            },
            content=chunks(),
        )

    assert response.status_code == 413
    assert response.json() == {"error": "body_too_large"}


@pytest.mark.parametrize(
    ("step", "key", "value"),
    (
        (PipelineStep.RUN_EPISODE, "world_id", "world-002"),
        (PipelineStep.RUN_EPISODE, "episode_id", "episode-002"),
        (PipelineStep.SUMMARIZE_TELEMETRY, "episode_id", "episode-002"),
        (PipelineStep.QUERY_GRAPH_MEMORY, "episode_id", "episode-002"),
        (PipelineStep.SELECT_CURRICULUM, "episode_id", "episode-002"),
        (PipelineStep.TRAIN_CANDIDATE_POLICY, "candidate_policy_id", "candidate-002"),
        (PipelineStep.PROMOTE_OR_ROLL_BACK, "candidate_policy_id", "candidate-002"),
    ),
)
def test_workflow_evidence_rejects_cross_step_identity_changes(
    step: PipelineStep,
    key: str,
    value: str,
) -> None:
    plan = _plan("identity-binding-run")
    commands = list(plan.commands)
    index = FIXED_PIPELINE.index(step)
    changed_payload = {**commands[index].payload, key: value}
    commands[index] = PipelineCommand.create(step, changed_payload)
    changed_plan = ExecutionPlan.create(plan.run_id, tuple(commands))

    with pytest.raises(ContractViolationError):
        validate_evidence_plan_binding(_bundle(), changed_plan)
