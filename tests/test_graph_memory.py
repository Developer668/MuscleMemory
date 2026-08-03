from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from muscle_memory.graph_memory import (
    AppendOnlyGraphCache,
    CorrectionMemoryRecord,
    CurriculumQuery,
    EpisodeMemoryRecord,
    EpisodeOutcome,
    EvaluatedPolicyVersion,
    FailureMemoryRecord,
    FalkorDBSettings,
    FalkorGraphMemory,
    GraphMemoryIntegrityError,
    GraphStorage,
    LessonMemoryRecord,
    ObstacleMemoryRecord,
    ProviderState,
    ResilientGraphMemory,
    WorldMemoryRecord,
    WorldSplit,
    canonical_json,
    settings_from_env,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROBOT_HASH = "a" * 64
POLICY_V0_HASH = "b" * 64
POLICY_V1_HASH = "c" * 64
EVIDENCE_HASH = "d" * 64
VALIDATION_HASH = "e" * 64
TELEMETRY_HASH = "f" * 64


@dataclass
class FakeQueryResult:
    result_set: list[list[object]]


class FakeFalkorGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object] | None, int | None]] = []
        self.returned_hash: str | None = None
        self.curriculum_rows: list[list[object]] = []
        self.fail = False

    def query(
        self,
        query: str,
        params: dict[str, object] | None = None,
        timeout: int | None = None,
    ) -> FakeQueryResult:
        self.calls.append(("write", query, params, timeout))
        if self.fail:
            raise ConnectionError("provider offline")
        assert params is not None
        returned_hash = self.returned_hash or str(params["content_hash"])
        return FakeQueryResult([[returned_hash]])

    def ro_query(
        self,
        query: str,
        params: dict[str, object] | None = None,
        timeout: int | None = None,
    ) -> FakeQueryResult:
        self.calls.append(("read", query, params, timeout))
        if self.fail:
            raise ConnectionError("provider offline")
        if query.strip() == "RETURN 1 AS ok":
            return FakeQueryResult([[1]])
        return FakeQueryResult(self.curriculum_rows)


def make_world(suffix: str, split: WorldSplit = WorldSplit.TRAINING) -> WorldMemoryRecord:
    return WorldMemoryRecord(
        world_id=f"world-{suffix}",
        world_hash=(suffix[0] * 64),
        split=split,
        seed=int(suffix[-1], 16),
        generation_version=1,
        validation_hash=VALIDATION_HASH,
        validated=True,
        recorded_at=NOW,
    )


def make_policy(policy_id: str, checkpoint_hash: str) -> EvaluatedPolicyVersion:
    return EvaluatedPolicyVersion.create(
        policy_id=policy_id,
        checkpoint_hash=checkpoint_hash,
        evaluation_evidence_hash=EVIDENCE_HASH,
        evaluation_split="development",
        metrics={"success_rate": 0.75},
        evaluated_at=NOW,
    )


def record_experience_chain(
    cache: AppendOnlyGraphCache,
    *,
    suffix: str,
    split: WorldSplit,
    signature_hash: str,
) -> None:
    world = make_world(suffix, split)
    obstacle = ObstacleMemoryRecord(
        obstacle_id=f"obstacle-{suffix}",
        obstacle_hash=(suffix[-1] * 64),
        world_id=world.world_id,
        category="laundry_basket",
        collider_kind="box",
        physical_properties_approved=True,
        recorded_at=NOW,
    )
    episode = EpisodeMemoryRecord(
        episode_id=f"episode-{suffix}",
        robot_checksum=ROBOT_HASH,
        world_id=world.world_id,
        world_hash=world.world_hash,
        world_split=split,
        policy_id="policy-v0",
        policy_hash=POLICY_V0_HASH,
        outcome=EpisodeOutcome.FAILED,
        completion_time_seconds=12.0,
        collision_count=1,
        fall_count=0,
        minimum_clearance_m=0.1,
        human_interventions=0,
        telemetry_digest=TELEMETRY_HASH,
        ended_at=NOW,
    )
    failure = FailureMemoryRecord(
        failure_id=f"failure-{suffix}",
        episode_id=episode.episode_id,
        category="clearance_violation",
        obstacle_id=obstacle.obstacle_id,
        severity=0.7,
        summary="Passed too close to an obstacle.",
        detected_at=NOW,
    )
    correction = CorrectionMemoryRecord(
        correction_id=f"correction-{suffix}",
        failure_id=failure.failure_id,
        kind="safer_route",
        description="Use the approved route with additional clearance.",
        approved=True,
        approved_by="operator@example.test",
        approved_at=NOW,
        created_at=NOW,
    )
    lesson = LessonMemoryRecord(
        lesson_id=f"lesson-{suffix}",
        correction_id=correction.correction_id,
        kind="clearance_margin",
        summary="Increase clearance around laundry baskets.",
        signature_hash=signature_hash,
        trained_policy_id="policy-v1",
        created_at=NOW,
    )

    cache.record_world(world)
    cache.record_obstacle(obstacle)
    cache.record_episode(episode)
    cache.record_failure(failure)
    cache.record_correction(correction)
    cache.record_lesson(lesson)


def test_world_gate_and_episode_robot_checksum_fail_closed() -> None:
    with pytest.raises(ValidationError, match="unvalidated worlds"):
        WorldMemoryRecord.model_validate(
            {**make_world("11").model_dump(), "validated": False}
        )


def test_obstacles_require_approved_physics_and_episodes_keep_signed_clearance() -> None:
    world = make_world("11")
    with pytest.raises(ValidationError, match="approved physical properties"):
        ObstacleMemoryRecord(
            obstacle_id="obstacle-11",
            obstacle_hash="1" * 64,
            world_id=world.world_id,
            category="laundry_basket",
            collider_kind="box",
            physical_properties_approved=False,
            recorded_at=NOW,
        )

    episode = EpisodeMemoryRecord(
        episode_id="episode-penetration",
        robot_checksum=ROBOT_HASH,
        world_id=world.world_id,
        world_hash=world.world_hash,
        world_split=world.split,
        policy_id="policy-v0",
        policy_hash=POLICY_V0_HASH,
        outcome=EpisodeOutcome.FAILED,
        completion_time_seconds=1.0,
        collision_count=1,
        fall_count=0,
        minimum_clearance_m=-0.004,
        human_interventions=0,
        telemetry_digest=TELEMETRY_HASH,
        ended_at=NOW,
    )

    assert episode.minimum_clearance_m == -0.004

    world = make_world("11")
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        EpisodeMemoryRecord(
            episode_id="episode-11",
            robot_checksum="not-the-fixed-checksum",
            world_id=world.world_id,
            world_hash=world.world_hash,
            world_split=world.split,
            policy_id="policy-v0",
            policy_hash=POLICY_V0_HASH,
            outcome=EpisodeOutcome.FAILED,
            completion_time_seconds=1.0,
            collision_count=0,
            fall_count=0,
            minimum_clearance_m=0.5,
            human_interventions=0,
            telemetry_digest=TELEMETRY_HASH,
            ended_at=NOW,
        )


def test_remote_writes_use_parameters_and_reject_identity_mutation() -> None:
    graph = FakeFalkorGraph()
    memory = FalkorGraphMemory(graph, graph_name="muscle_memory", query_timeout_ms=750)
    world = make_world("11")

    receipt = memory.record_world(world)

    assert receipt.storage is GraphStorage.FALKORDB
    _, query, params, timeout = graph.calls[-1]
    assert params is not None
    assert params["world_id"] == world.world_id
    assert world.world_id not in query
    assert "$world_id" in query
    assert timeout == 750

    graph.returned_hash = "0" * 64
    with pytest.raises(GraphMemoryIntegrityError, match="different immutable content"):
        memory.record_world(world)


def test_remote_curriculum_query_is_parameterized_and_training_only() -> None:
    graph = FakeFalkorGraph()
    graph.curriculum_rows = [
        [
            "lesson-1",
            "clearance_margin",
            "Increase clearance.",
            "clearance_violation",
            "laundry_basket",
            3,
            ["episode-1", "episode-2", "episode-3"],
            "policy-v1",
        ]
    ]
    memory = FalkorGraphMemory(graph, graph_name="muscle_memory", query_timeout_ms=500)

    result = memory.query_curriculum(
        CurriculumQuery(
            failure_categories=("clearance_violation",),
            obstacle_categories=("laundry_basket",),
            exclude_trained_policy_ids=("policy-v2",),
            limit=4,
        )
    )

    assert result.lessons[0].support_count == 3
    _, query, params, _ = graph.calls[-1]
    assert params is not None
    assert params["training_split"] == "training"
    assert "world_split: $training_split" in query
    assert "clearance_violation" not in query
    assert params["failure_categories"] == ["clearance_violation"]


def test_append_only_cache_reloads_and_excludes_held_out_experience(tmp_path: Path) -> None:
    cache_path = tmp_path / "graph-events.jsonl"
    cache = AppendOnlyGraphCache(cache_path)
    cache.record_evaluated_policy(make_policy("policy-v0", POLICY_V0_HASH))
    cache.record_evaluated_policy(make_policy("policy-v1", POLICY_V1_HASH))
    signature_hash = "1" * 64
    record_experience_chain(
        cache,
        suffix="11",
        split=WorldSplit.TRAINING,
        signature_hash=signature_hash,
    )
    record_experience_chain(
        cache,
        suffix="22",
        split=WorldSplit.TRAINING,
        signature_hash=signature_hash,
    )
    record_experience_chain(
        cache,
        suffix="33",
        split=WorldSplit.HELD_OUT,
        signature_hash=signature_hash,
    )

    reloaded = AppendOnlyGraphCache(cache_path)
    result = reloaded.query_curriculum(CurriculumQuery())

    assert result.storage is GraphStorage.LOCAL_CACHE
    assert result.provider_state is ProviderState.UNCONFIGURED
    assert len(result.lessons) == 1
    assert result.lessons[0].support_count == 2
    assert result.lessons[0].source_episode_ids == ("episode-11", "episode-22")
    excluded = reloaded.query_curriculum(
        CurriculumQuery(exclude_trained_policy_ids=("policy-v1",))
    )
    assert excluded.lessons == ()


def test_append_only_cache_rejects_replacement_under_same_identity(tmp_path: Path) -> None:
    cache = AppendOnlyGraphCache(tmp_path / "events.jsonl")
    world = make_world("11")
    cache.record_world(world)
    changed = WorldMemoryRecord.model_validate({**world.model_dump(), "seed": 99})

    with pytest.raises(GraphMemoryIntegrityError, match="immutable"):
        cache.record_world(changed)


def test_resilient_service_reports_provider_outage_and_local_only_write(
    tmp_path: Path,
) -> None:
    settings = FalkorDBSettings.model_validate(
        {
            "url": "redis://provider.example.test:6379",
            "cache_path": tmp_path / "events.jsonl",
        }
    )
    graph = FakeFalkorGraph()
    graph.fail = True
    remote = FalkorGraphMemory(graph, graph_name=settings.graph_name, query_timeout_ms=500)
    service = ResilientGraphMemory(
        settings=settings,
        cache=AppendOnlyGraphCache(settings.cache_path),
        remote=remote,
    )

    assert service.configuration_health().provider_state is ProviderState.CONFIGURED
    assert service.health().provider_state is ProviderState.UNAVAILABLE
    receipt = service.record_world(make_world("11"))
    assert receipt.storage is GraphStorage.LOCAL_CACHE
    assert receipt.provider_state is ProviderState.UNAVAILABLE
    assert "local cache" in receipt.detail
    assert settings.url is not None
    assert settings.url.get_secret_value() not in receipt.detail


def test_settings_do_not_claim_an_unset_provider_is_configured(tmp_path: Path) -> None:
    settings = settings_from_env(
        {"MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "events.jsonl")}
    )
    service = ResilientGraphMemory(
        settings=settings,
        cache=AppendOnlyGraphCache(settings.cache_path),
        remote=None,
    )

    assert service.configuration_health().provider_state is ProviderState.UNCONFIGURED
    assert service.health().provider_state is ProviderState.UNCONFIGURED


def test_graph_memory_is_absent_from_robot_control_and_evaluation_paths() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    control_paths = (
        repository_root / "src/muscle_memory/policy",
        repository_root / "src/muscle_memory/simulation/runtime.py",
        repository_root / "src/muscle_memory/evaluation",
    )
    for path in control_paths:
        files = path.rglob("*.py") if path.is_dir() else (path,)
        for source_file in files:
            assert "muscle_memory.graph_memory" not in source_file.read_text(encoding="utf-8")


def test_policy_metrics_are_canonical_before_they_are_hashed() -> None:
    policy = make_policy("policy-v0", POLICY_V0_HASH)
    assert policy.metrics_json == canonical_json({"success_rate": 0.75})
