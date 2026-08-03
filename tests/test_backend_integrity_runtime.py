from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
    AuthenticatedHuman,
    CorrectionPoint,
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


class FailOnceReceiptJournal:
    def __init__(self, delegate: CoordinatorEpisodeJournal) -> None:
        self._delegate = delegate
        self.failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def record_receipt(self, receipt: Any) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected telemetry-receipt journal failure")
        self._delegate.record_receipt(receipt)


class FailOnceCorrectionDeliveryJournal:
    def __init__(self, delegate: CoordinatorEpisodeJournal) -> None:
        self._delegate = delegate
        self.failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def record_approval_delivery(self, approval: Any) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected correction-delivery journal failure")
        self._delegate.record_approval_delivery(approval)


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
    fail_receipt_journal: bool = False,
    fail_correction_delivery_journal: bool = False,
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
    journal: Any = base_journal
    if fail_delivery_journal:
        journal = FailOnceGraphDeliveryJournal(base_journal)
    elif fail_receipt_journal:
        journal = FailOnceReceiptJournal(base_journal)
    elif fail_correction_delivery_journal:
        journal = FailOnceCorrectionDeliveryJournal(base_journal)
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


def test_repeated_catalog_seed_reuses_immutable_graph_parents(tmp_path: Path) -> None:
    coordinator, spool, graph_cache, service, first_identity = _build_service(tmp_path)
    second_identity = replace(
        first_identity,
        episode_id="episode-2",
        opened_at=NOW + timedelta(seconds=1),
    )

    async def exercise() -> None:
        await service.open_episode(first_identity)
        await service.append_telemetry(_telemetry(first_identity))
        first = await service.close_episode(_result(first_identity), closed_at=NOW)

        await service.open_episode(second_identity)
        await service.append_telemetry(_telemetry(second_identity))
        second = await service.close_episode(
            _result(second_identity),
            closed_at=NOW + timedelta(seconds=2),
        )
        assert first.graph.complete
        assert second.graph.complete

    asyncio.run(exercise())
    earlier = derive_training_world_artifacts(SEED, recorded_at=NOW)
    later = derive_training_world_artifacts(
        SEED,
        recorded_at=NOW + timedelta(days=1),
    )
    assert earlier.world == later.world
    assert earlier.obstacles == later.obstacles
    kinds = tuple(event.record_kind for event in graph_cache.events)
    assert kinds.count("world") == 1
    assert kinds.count("obstacle") == len(earlier.obstacles)
    assert kinds.count("episode") == 2
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


def test_restart_recovers_receipt_after_durable_telemetry_crash_window(
    tmp_path: Path,
) -> None:
    coordinator, spool, _graph_cache, service, identity = _build_service(
        tmp_path,
        fail_receipt_journal=True,
    )

    async def first_process() -> None:
        await service.open_episode(identity)
        with pytest.raises(RuntimeError, match="telemetry-receipt"):
            await service.append_telemetry(_telemetry(identity))

    asyncio.run(first_process())
    assert len(spool.records_for(identity.episode_id)) == 1
    assert coordinator.training_episode_receipts(identity.episode_id) == ()
    coordinator.close()
    spool.close()

    coordinator2, spool2, _cache2, restored, identity2 = _build_service(tmp_path)
    assert len(coordinator2.training_episode_receipts(identity2.episode_id)) == 1
    closure = asyncio.run(restored.close_episode(_result(identity2), closed_at=NOW))
    assert closure.telemetry.records_without_receipts == 0
    assert closure.telemetry.total_records == 1
    coordinator2.close()
    spool2.close()


def test_restart_recovers_correction_graph_delivery_journal_crash_window(
    tmp_path: Path,
) -> None:
    coordinator, spool, graph_cache, service, identity = _build_service(
        tmp_path,
        fail_correction_delivery_journal=True,
    )

    async def first_process() -> str:
        await service.open_episode(identity)
        await service.append_telemetry(_telemetry(identity))
        failed = replace(
            _result(identity),
            success=False,
            failed_reasons=("BODY_COLLISION",),
            body_collisions=1,
        )
        closure = await service.close_episode(failed, closed_at=NOW)
        correction = await service.submit_route_correction(
            episode_id=identity.episode_id,
            failure_id=closure.failures[0].failure_id,
            points=(CorrectionPoint(0.0, 0.0), CorrectionPoint(1.0, 1.0)),
            description="Keep clear of the collision boundary.",
            submitted_by="operator@example.test",
            created_at=NOW,
        )
        with pytest.raises(RuntimeError, match="correction-delivery"):
            await service.approve_correction(
                correction.correction_id,
                approver=AuthenticatedHuman(
                    subject_id="operator@example.test",
                    authentication_method="bearer",
                    authenticated=True,
                ),
                approved_at=NOW,
            )
        return correction.correction_id

    correction_id = asyncio.run(first_process())
    assert coordinator.training_correction_graph_deliveries() == ()
    first_events = tuple(graph_cache.events)
    assert first_events[-1].record_kind == "correction"
    coordinator.close()
    spool.close()

    coordinator2, spool2, graph_cache2, _restored, _identity2 = _build_service(tmp_path)
    deliveries = coordinator2.training_correction_graph_deliveries()
    assert len(deliveries) == 1
    approval = CoordinatorEpisodeJournal(
        coordinator2,
        expected_robot_checksum=ROBOT_HASH,
    ).approvals()[0]
    assert approval.submission.correction_id == correction_id
    assert approval.graph_receipt is not None
    assert tuple(graph_cache2.events) == first_events
    coordinator2.close()
    spool2.close()
