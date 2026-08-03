from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from muscle_memory.backend.episode_journal import CoordinatorEpisodeJournal
from muscle_memory.backend.graph_prerequisites import (
    CoordinatorGraphPrerequisiteResolver,
    derive_training_world_artifacts,
)
from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.episodes import (
    EpisodeClosedError,
    EpisodeIdentity,
    EpisodeLifecycleState,
    EpisodeService,
    GraphPersistenceReport,
)
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    AppendOnlyGraphCache,
    EvaluatedPolicyVersion,
    FalkorDBSettings,
    ResilientGraphMemory,
    WorldSplit,
)
from muscle_memory.telemetry import (
    DurableTelemetrySpool,
    EpisodeTelemetryRecord,
    LaserDataConfig,
    LaserDataTelemetryBackend,
    SensorSnapshot,
    SignalUseLabel,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROBOT_HASH = "a" * 64
POLICY_HASH = "c" * 64
POLICY_EVIDENCE_HASH = "d" * 64
SEED = 42


class FailOnceGraphDeliveryJournal:
    def __init__(self, delegate: CoordinatorEpisodeJournal) -> None:
        self._delegate = delegate
        self.failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def record_graph_delivery(
        self,
        episode_id: str,
        report: GraphPersistenceReport,
    ) -> None:
        del episode_id, report
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected graph-delivery journal failure")
        raise AssertionError("the failed process must not retry graph delivery")


def _checkpoint() -> EvaluatedPolicyVersion:
    return EvaluatedPolicyVersion.create(
        policy_id="policy-1",
        checkpoint_hash=POLICY_HASH,
        evaluation_evidence_hash=POLICY_EVIDENCE_HASH,
        evaluation_split="development",
        metrics={"success_rate": 0.75},
        evaluated_at=NOW,
    )


def _identity(world_id: str, world_hash: str) -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_id="episode-1",
        robot_checksum=ROBOT_HASH,
        world_id=world_id,
        world_hash=world_hash,
        world_split=WorldSplit.TRAINING,
        policy_id="policy-1",
        policy_hash=POLICY_HASH,
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
        frame_id="frame-0",
    )


def _result(identity: EpisodeIdentity) -> PolicyEpisodeResult:
    return PolicyEpisodeResult(
        episode_id=identity.episode_id,
        world_id=identity.world_id,
        world_seed=SEED,
        world_split="training",
        world_hash=identity.world_hash,
        robot_checksum=identity.robot_checksum,
        policy_id=identity.policy_id,
        policy_hash=identity.policy_hash,
        success=True,
        failed_reasons=(),
        time_to_resident_seconds=8.0,
        simulated_duration_seconds=9.0,
        stop_distance_m=0.3,
        facing_error_degrees=4.0,
        stopped_speed_mps=0.0,
        falls=0,
        body_collisions=0,
        minimum_obstacle_clearance_m=0.4,
        maximum_tray_tilt_degrees=3.0,
        package_slipped=False,
        human_interventions=0,
        direct_distance_m=4.0,
        path_length_m=5.0,
        path_efficiency=0.8,
        energy_joules=20.0,
        task_policy_updates=90,
        trace=(),
    )


def _build_service(
    root: Path,
    *,
    fail_delivery_journal: bool = False,
) -> tuple[
    CoordinatorStore,
    DurableTelemetrySpool,
    AppendOnlyGraphCache,
    EpisodeService,
    EpisodeIdentity,
]:
    coordinator = CoordinatorStore(root / "coordinator.sqlite3")
    coordinator.register_evaluated_checkpoint(_checkpoint())
    spool = DurableTelemetrySpool(root / "telemetry.sqlite3")
    telemetry = LaserDataTelemetryBackend(
        LaserDataConfig(connection_string="", spool_path=spool.path),
        spool=spool,
    )
    graph_settings = FalkorDBSettings(cache_path=root / "graph.jsonl")
    graph_cache = AppendOnlyGraphCache(graph_settings.cache_path)
    graph = ResilientGraphMemory(
        settings=graph_settings,
        cache=graph_cache,
        remote=None,
    )
    base_journal = CoordinatorEpisodeJournal(
        coordinator,
        expected_robot_checksum=ROBOT_HASH,
    )
    journal: Any = (
        FailOnceGraphDeliveryJournal(base_journal)
        if fail_delivery_journal
        else base_journal
    )
    derived = derive_training_world_artifacts(SEED, recorded_at=NOW)
    identity = _identity(derived.world.world_id, derived.world.world_hash)
    service = EpisodeService(
        telemetry_backend=telemetry,
        telemetry_store=spool,
        graph_memory=graph,
        journal=journal,
        graph_prerequisites=CoordinatorGraphPrerequisiteResolver(
            coordinator,
            expected_robot_checksum=ROBOT_HASH,
        ),
    )
    return coordinator, spool, graph_cache, service, identity


def test_fresh_episode_registers_validated_graph_parents_before_episode(
    tmp_path: Path,
) -> None:
    coordinator, spool, graph_cache, service, identity = _build_service(tmp_path)

    async def exercise() -> None:
        await service.open_episode(identity)
        await service.append_telemetry(_telemetry(identity))
        closure = await service.close_episode(_result(identity), closed_at=NOW)
        assert closure.graph.complete
        assert closure.graph.expected_records == len(
            derive_training_world_artifacts(SEED, recorded_at=NOW).obstacles
        ) + 3

    asyncio.run(exercise())
    kinds = [event.record_kind for event in graph_cache.events]
    assert kinds[0] == "world"
    assert kinds[-2:] == ["evaluated_policy", "episode"]
    assert all(kind == "obstacle" for kind in kinds[1:-2])
    coordinator.close()
    spool.close()


def test_measured_closure_survives_graph_delivery_journal_failure_and_retries(
    tmp_path: Path,
) -> None:
    coordinator, spool, graph_cache, service, identity = _build_service(
        tmp_path,
        fail_delivery_journal=True,
    )

    async def first_process() -> None:
        await service.open_episode(identity)
        await service.append_telemetry(_telemetry(identity))
        with pytest.raises(RuntimeError, match="injected graph-delivery"):
            await service.close_episode(_result(identity), closed_at=NOW)

    asyncio.run(first_process())
    assert service.episode_state(identity.episode_id) is EpisodeLifecycleState.CLOSED
    assert coordinator.training_episode_closure(identity.episode_id) is not None
    assert coordinator.training_episode_graph_deliveries(identity.episode_id) == ()
    first_events = tuple(graph_cache.events)
    assert first_events[-1].record_kind == "episode"
    coordinator.close()
    spool.close()

    coordinator2, spool2, graph_cache2, restored, identity2 = _build_service(tmp_path)
    closure = restored.closure_for(identity2.episode_id)
    assert closure is not None
    assert closure.graph.complete
    assert len(coordinator2.training_episode_graph_deliveries(identity2.episode_id)) == 1
    assert tuple(graph_cache2.events) == first_events
    with pytest.raises(EpisodeClosedError, match="already closed"):
        asyncio.run(
            restored.close_episode(
                replace(_result(identity2), success=False),
                closed_at=NOW,
            )
        )
    coordinator2.close()
    spool2.close()
