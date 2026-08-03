from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from muscle_memory.api import (
    HashedBearerCredential,
    Sha256BearerAuthenticator,
    create_app,
)
from muscle_memory.backend.graph_prerequisites import derive_training_world_artifacts
from muscle_memory.coordinator import (
    EpisodeState,
    HeldOutEvaluationArtifact,
    HeldOutEvaluationEpisodeMetadata,
    HeldOutEvaluationResult,
)
from muscle_memory.coordinator.models import canonical_json, sha256_text
from muscle_memory.episodes import EpisodeIdentity
from muscle_memory.evaluation import load_heldout_worlds
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import EvaluatedPolicyVersion, WorldSplit
from muscle_memory.orchestration import (
    EXACT_GUILD_ROLES,
    FIXED_PIPELINE,
    ExecutionPlan,
    PipelineCommand,
    PipelineStep,
    ReviewRecommendation,
    SimulatedGuildCoordinator,
    SponsorOrchestrator,
)
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.runtime import build_api_backend
from muscle_memory.telemetry import (
    EpisodeTelemetryRecord,
    SensorSnapshot,
    SignalUseLabel,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
BASELINE_HASH = "3" * 64
CANDIDATE_HASH = "4" * 64
TOKEN = "workflow-evidence-admission-token"


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "MUSCLE_MEMORY_COORDINATOR_DB_PATH": str(tmp_path / "coordinator.sqlite3"),
        "MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "graph.jsonl"),
        "MUSCLE_MEMORY_TELEMETRY_SPOOL": str(tmp_path / "laser-spool.sqlite3"),
        "MM_ASSET_CACHE_DIR": str(tmp_path / "assets"),
        "MM_ASSET_APPROVAL_LEDGER_DIR": str(tmp_path / "approvals"),
    }


def _checkpoint(
    policy_id: str,
    checkpoint_hash: str,
    evidence_hash: str,
    *,
    success_rate: float,
    collision_rate: float,
) -> EvaluatedPolicyVersion:
    return EvaluatedPolicyVersion.create(
        policy_id=policy_id,
        checkpoint_hash=checkpoint_hash,
        evaluation_evidence_hash=evidence_hash,
        evaluation_split="held_out",
        metrics={
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "falls": 0,
            "median_clearance_m": 0.31,
            "path_efficiency_regression_fraction": 0.03,
        },
        evaluated_at=NOW,
    )


def _heldout_result(
    *,
    index: int,
    policy_id: str,
    policy_hash: str,
    success_count: int,
    collision_failures: int,
    robot_checksum: str,
) -> PolicyEpisodeResult:
    heldout = load_heldout_worlds()[index]
    world_hash = hashlib.sha256(
        heldout.world.model_dump_json().encode("utf-8")
    ).hexdigest()
    success = index < success_count
    collision = int(not success and index < success_count + collision_failures)
    return PolicyEpisodeResult(
        episode_id=f"heldout-{policy_id}-{index:02d}",
        world_id=heldout.world.world_id,
        world_seed=heldout.world.seed,
        world_split="held_out",
        world_hash=world_hash,
        robot_checksum=robot_checksum,
        policy_id=policy_id,
        policy_hash=policy_hash,
        success=success,
        failed_reasons=() if success else ("SAFE_DELIVERY_FAILED",),
        time_to_resident_seconds=20.0 if success else None,
        simulated_duration_seconds=20.0,
        stop_distance_m=0.3,
        facing_error_degrees=5.0,
        stopped_speed_mps=0.0,
        falls=0,
        body_collisions=collision,
        minimum_obstacle_clearance_m=0.31,
        maximum_tray_tilt_degrees=5.0,
        package_slipped=False,
        human_interventions=0,
        direct_distance_m=4.0,
        path_length_m=5.0,
        path_efficiency=0.8,
        energy_joules=20.0,
        task_policy_updates=1,
        trace=(),
    )


def _plan(
    world_id: str,
    *,
    episode_id: str = "episode-2",
    run_id: str = "fresh-review-1",
) -> ExecutionPlan:
    commands = (
        PipelineCommand.create(
            PipelineStep.VALIDATE_WORLD,
            {"uncertain_physical_properties": False, "world_id": world_id},
        ),
        PipelineCommand.create(
            PipelineStep.RUN_EPISODE,
            {"episode_id": episode_id, "world_id": world_id},
        ),
        PipelineCommand.create(
            PipelineStep.SUMMARIZE_TELEMETRY,
            {"episode_id": episode_id},
        ),
        PipelineCommand.create(
            PipelineStep.QUERY_GRAPH_MEMORY,
            {"episode_id": episode_id},
        ),
        PipelineCommand.create(
            PipelineStep.SELECT_CURRICULUM,
            {"curriculum_change_requested": False, "episode_id": episode_id},
        ),
        PipelineCommand.create(
            PipelineStep.TRAIN_CANDIDATE_POLICY,
            {"reward_change_requested": False, "candidate_policy_id": "candidate-1"},
        ),
        PipelineCommand.create(
            PipelineStep.EVALUATE_CANDIDATE_POLICY,
            {
                "baseline_policy_id": "baseline-1",
                "candidate_policy_id": "candidate-1",
                "heldout_world_set_id": "heldout-v1",
            },
        ),
        PipelineCommand.create(
            PipelineStep.PROMOTE_OR_ROLL_BACK,
            {"action": "promote", "candidate_policy_id": "candidate-1"},
        ),
    )
    assert tuple(command.step for command in commands) == FIXED_PIPELINE
    return ExecutionPlan.create(run_id, commands)


def _identity(
    episode_id: str,
    *,
    robot_checksum: str,
    world_id: str,
    world_hash: str,
    policy_id: str = "baseline-1",
    policy_hash: str = BASELINE_HASH,
) -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_id=episode_id,
        robot_checksum=robot_checksum,
        world_id=world_id,
        world_hash=world_hash,
        world_split=WorldSplit.TRAINING,
        policy_id=policy_id,
        policy_hash=policy_hash,
        opened_at=NOW,
    )


def _telemetry(identity: EpisodeIdentity) -> EpisodeTelemetryRecord:
    return EpisodeTelemetryRecord.create(
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
        failure_type="body_collision",
        frame_id=f"frame-{identity.episode_id}",
    )


def _result(identity: EpisodeIdentity) -> PolicyEpisodeResult:
    return PolicyEpisodeResult(
        episode_id=identity.episode_id,
        world_id=identity.world_id,
        world_seed=42,
        world_split="training",
        world_hash=identity.world_hash,
        robot_checksum=identity.robot_checksum,
        policy_id=identity.policy_id,
        policy_hash=identity.policy_hash,
        success=False,
        failed_reasons=("body_collision",),
        time_to_resident_seconds=None,
        simulated_duration_seconds=0.05,
        stop_distance_m=1.0,
        facing_error_degrees=None,
        stopped_speed_mps=0.0,
        falls=0,
        body_collisions=1,
        minimum_obstacle_clearance_m=-0.01,
        maximum_tray_tilt_degrees=3.0,
        package_slipped=False,
        human_interventions=0,
        direct_distance_m=4.0,
        path_length_m=5.0,
        path_efficiency=0.8,
        energy_joules=20.0,
        task_policy_updates=1,
        trace=(),
    )


def test_fresh_http_review_registers_only_reproduced_provider_evidence(
    tmp_path: Path,
) -> None:
    backend = build_api_backend(_environment(tmp_path))
    robot_checksum = verify_mm01_bundle().robot_checksum
    baseline_results = tuple(
        _heldout_result(
            index=index,
            policy_id="baseline-1",
            policy_hash=BASELINE_HASH,
            success_count=13,
            collision_failures=3,
            robot_checksum=robot_checksum,
        )
        for index in range(20)
    )
    candidate_results = tuple(
        _heldout_result(
            index=index,
            policy_id="candidate-1",
            policy_hash=CANDIDATE_HASH,
            success_count=18,
            collision_failures=1,
            robot_checksum=robot_checksum,
        )
        for index in range(20)
    )
    decision = evaluate_promotion(baseline_results, candidate_results)
    assert decision.promotable is True
    artifact_json = canonical_json(
        {
            "schema_version": 1,
            "heldout_bundle_sha256": "9" * 64,
            "candidate_checkpoint_sha256": CANDIDATE_HASH,
            "baseline_results": [asdict(item) for item in baseline_results],
            "candidate_results": [asdict(item) for item in candidate_results],
            "promotion_decision": asdict(decision),
        }
    )
    artifact_hash = sha256_text(artifact_json)
    backend.coordinator.record_held_out_evaluation_artifact(
        HeldOutEvaluationArtifact(
            artifact_hash=artifact_hash,
            held_out_world_set_id="heldout-v1",
            artifact_json=artifact_json,
            evaluated_at=NOW,
        )
    )
    baseline = _checkpoint(
        "baseline-1",
        BASELINE_HASH,
        artifact_hash,
        success_rate=0.65,
        collision_rate=0.15,
    )
    candidate = _checkpoint(
        "candidate-1",
        CANDIDATE_HASH,
        artifact_hash,
        success_rate=0.9,
        collision_rate=0.05,
    )
    backend.coordinator.register_evaluated_checkpoint(baseline)
    backend.coordinator.register_evaluated_checkpoint(candidate)
    for result in (*baseline_results, *candidate_results):
        episode_id = result.episode_id
        backend.coordinator.register_held_out_evaluation_episode(
            HeldOutEvaluationEpisodeMetadata(
                episode_id=episode_id,
                robot_checksum=robot_checksum,
                world_hash=result.world_hash,
                policy_hash=result.policy_hash,
                held_out_world_set_id="heldout-v1",
                created_at=NOW,
            )
        )
        backend.coordinator.transition_episode(
            episode_id,
            EpisodeState.RUNNING,
            occurred_at=NOW,
        )
        backend.coordinator.transition_episode(
            episode_id,
            EpisodeState.SUCCEEDED if result.success else EpisodeState.FAILED,
            occurred_at=NOW,
        )
        backend.coordinator.record_held_out_evaluation_result(
            HeldOutEvaluationResult(
                episode_id=episode_id,
                evaluation_artifact_hash=artifact_hash,
                result_json=canonical_json(asdict(result)),
            )
        )

    derived = derive_training_world_artifacts(42, recorded_at=NOW)

    async def close_sources() -> None:
        for episode_id, policy_id, policy_hash in (
            ("episode-1", "baseline-1", BASELINE_HASH),
            ("episode-2", "baseline-1", BASELINE_HASH),
            ("episode-wrong-policy", "candidate-1", CANDIDATE_HASH),
        ):
            identity = _identity(
                episode_id,
                robot_checksum=robot_checksum,
                world_id=derived.world.world_id,
                world_hash=derived.world.world_hash,
                policy_id=policy_id,
                policy_hash=policy_hash,
            )
            await backend.episode_runtime.open_episode(identity)
            await backend.episode_runtime.append_telemetry(_telemetry(identity))
            await backend.episode_runtime.close_episode(_result(identity), closed_at=NOW)

    asyncio.run(close_sources())
    mismatched_source_plan = _plan(
        derived.world.world_id,
        episode_id="episode-wrong-policy",
        run_id="mismatched-source-review",
    )
    with pytest.raises(
        ValueError,
        match="source closure does not exactly match the evaluation baseline",
    ):
        backend.evidence_admitter.reproduce(
            mismatched_source_plan,
            world_evidence_id="world.evidence.mismatch.1",
            failure_curriculum_evidence_id="curriculum.evidence.mismatch.1",
            evaluation_evidence_id="evaluation.evidence.mismatch.1",
        )
    plan = _plan(derived.world.world_id)
    bundle = backend.evidence_admitter.reproduce(
        plan,
        world_evidence_id="world.evidence.fresh.1",
        failure_curriculum_evidence_id="curriculum.evidence.fresh.1",
        evaluation_evidence_id="evaluation.evidence.fresh.1",
    )
    assert all(
        backend.coordinator.provider_evidence(evidence_id) is None
        for evidence_id, _kind, _hash in bundle.artifact_hashes()
    )

    backend.orchestrator = SponsorOrchestrator(
        SimulatedGuildCoordinator(
            {
                role: ReviewRecommendation.PROCEED
                for role in EXACT_GUILD_ROLES
            }
        ),
        backend.providers.rocketride,
    )
    credential = HashedBearerCredential.from_plaintext(
        subject="operator@example.test",
        token=TOKEN,
    )
    app = create_app(
        backend=backend,
        authenticator=Sha256BearerAuthenticator((credential,)),
    )
    request = {
        "run_id": plan.run_id,
        "commands": [
            {"step": command.step.value, "payload": command.payload}
            for command in plan.commands
        ],
        "evidence": bundle.model_dump(mode="json"),
    }
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/workflows/review",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=request,
        )
        assert response.status_code == 201, response.text
        assert response.json()["executable"] is True
        assert response.json()["provider"]["state"] == "simulation"
        for evidence_id, kind, artifact_hash in bundle.artifact_hashes():
            reference = backend.coordinator.provider_evidence(evidence_id)
            assert reference is not None
            assert reference.evidence_kind == kind
            assert reference.artifact_hash == artifact_hash
        assert backend.coordinator.workflow_guild_evidence(plan.run_id) == bundle
