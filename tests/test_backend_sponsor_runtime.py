"""End-to-end sponsor callback composition and durable evidence tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from muscle_memory.api import Sha256BearerAuthenticator, create_app
from muscle_memory.backend.rocketride_callback import MAX_CALLBACK_BODY_BYTES
from muscle_memory.coordinator.models import ProviderEvidenceReference
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
from muscle_memory.runtime import build_api_backend
from muscle_memory.orchestration.service import ReviewedExecution

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


def _bundle() -> GuildEvidenceBundle:
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
                        "evaluation_id": "evaluation-baseline-001",
                        "success_rate": 0.55,
                        "collision_rate": 0.4,
                    },
                    "candidate": {
                        "policy_id": "candidate-001",
                        "policy_checksum": "1" * 64,
                        "evaluation_id": "evaluation-candidate-001",
                        "success_rate": 0.85,
                        "collision_rate": 0.1,
                        "falls": 0,
                        "median_clearance_m": 0.3,
                        "path_efficiency_regression_fraction": 0.05,
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


def _register_evidence(backend: object, plan: ExecutionPlan) -> None:
    coordinator = backend.coordinator  # type: ignore[attr-defined]
    bundle = _bundle()
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
        mode=ProviderMode.LIVE,
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


def _envelope(plan: ExecutionPlan, step: PipelineStep) -> str:
    command = plan.commands[FIXED_PIPELINE.index(step)]
    return canonical_json(
        {
            "contract_version": 1,
            "run_id": plan.run_id,
            "plan_digest": plan.digest,
            "step": step.value,
            "payload": command.payload,
        }
    )


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
