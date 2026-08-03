from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from muscle_memory.episodes import (
    AuthenticatedHuman,
    CorrectionAlreadyApprovedError,
    CorrectionPoint,
    EpisodeClosedError,
    EpisodeIdentity,
    EpisodeLifecycleState,
    EpisodeService,
    EpisodeServiceError,
    TelemetryCadenceError,
)
from muscle_memory.episodes.training import TrainingCorrectionFeed
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    GraphStorage,
    GraphWriteReceipt,
    ProviderState,
    WorldSplit,
)
from muscle_memory.telemetry import (
    EpisodeTelemetryRecord,
    InMemoryAppendOnlyTelemetrySink,
    LaserDataProviderState,
    LaserDataTelemetryEnvelope,
    OutOfOrderTelemetryError,
    SensorSnapshot,
    SignalUseLabel,
    TelemetryAppendResult,
    TelemetryDelivery,
    TelemetryMutationError,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROBOT_HASH = "a" * 64
WORLD_HASH = "b" * 64
POLICY_HASH = "c" * 64


class FakeTelemetryBackend:
    def __init__(
        self,
        store: InMemoryAppendOnlyTelemetrySink,
        *,
        delivery: TelemetryDelivery = TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE,
    ) -> None:
        self.store = store
        self.delivery = delivery

    async def append(self, record: EpisodeTelemetryRecord) -> TelemetryAppendResult:
        self.store.append(record)
        event_id = LaserDataTelemetryEnvelope.from_domain(record).event_id
        local_only = self.delivery is TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
        return TelemetryAppendResult(
            event_id=event_id,
            delivery=self.delivery,
            provider_state=(
                LaserDataProviderState.UNCONFIGURED
                if local_only
                else LaserDataProviderState.END_TO_END_VERIFIED
            ),
            pending_local_records=1 if local_only else 0,
        )


class FakeGraphMemory:
    def __init__(
        self,
        *,
        storage: GraphStorage = GraphStorage.FALKORDB,
        provider_state: ProviderState = ProviderState.END_TO_END_VERIFIED,
    ) -> None:
        self.storage = storage
        self.provider_state = provider_state
        self.service: EpisodeService | None = None
        self.calls: list[tuple[str, Any, EpisodeLifecycleState | None]] = []

    def _receipt(self, kind: str, record: Any, record_id: str) -> GraphWriteReceipt:
        state = self.service.episode_state("episode-1") if self.service is not None else None
        self.calls.append((kind, record, state))
        return GraphWriteReceipt(
            record_kind=kind,
            record_id=record_id,
            content_hash=record.content_hash,
            storage=self.storage,
            provider_state=self.provider_state,
            mirrored_to_local_cache=True,
            detail="fake graph receipt",
        )

    def record_episode(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("episode", record, record.episode_id)

    def record_failure(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("failure", record, record.failure_id)

    def record_correction(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("correction", record, record.correction_id)

    def record_lesson(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("lesson", record, record.lesson_id)


def identity(split: WorldSplit = WorldSplit.TRAINING) -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_id="episode-1",
        robot_checksum=ROBOT_HASH,
        world_id="world-1",
        world_hash=WORLD_HASH,
        world_split=split,
        policy_id="policy-1",
        policy_hash=POLICY_HASH,
        opened_at=NOW,
    )


def telemetry(
    sequence: int,
    *,
    payload: object | None = None,
    sim_time_seconds: float | None = None,
) -> EpisodeTelemetryRecord:
    return EpisodeTelemetryRecord.create(
        episode_id="episode-1",
        world_id="world-1",
        policy_id="policy-1",
        sequence=sequence,
        sim_time_seconds=sequence / 20 if sim_time_seconds is None else sim_time_seconds,
        robot_checksum=ROBOT_HASH,
        policy_hash=POLICY_HASH,
        world_hash=WORLD_HASH,
        signal_use=SignalUseLabel.LOGGED_ONLY,
        sensors=SensorSnapshot.all_unavailable(),
        payload={"sample": sequence} if payload is None else payload,
        frame_id=f"video-frame-{sequence:04d}",
    )


def result(
    *,
    success: bool = True,
    failed_reasons: tuple[str, ...] = (),
    minimum_clearance_m: float = 0.4,
    world_split: str = "training",
) -> PolicyEpisodeResult:
    return PolicyEpisodeResult(
        episode_id="episode-1",
        world_id="world-1",
        world_seed=42,
        world_split=world_split,
        world_hash=WORLD_HASH,
        robot_checksum=ROBOT_HASH,
        policy_id="policy-1",
        policy_hash=POLICY_HASH,
        success=success,
        failed_reasons=failed_reasons,
        time_to_resident_seconds=8.0 if success else None,
        simulated_duration_seconds=9.0,
        stop_distance_m=0.3,
        facing_error_degrees=4.0,
        stopped_speed_mps=0.0,
        falls=0,
        body_collisions=0 if success else 1,
        minimum_obstacle_clearance_m=minimum_clearance_m,
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


def service(
    *,
    delivery: TelemetryDelivery = TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE,
    graph_storage: GraphStorage = GraphStorage.FALKORDB,
    graph_state: ProviderState = ProviderState.END_TO_END_VERIFIED,
) -> tuple[EpisodeService, FakeGraphMemory]:
    store = InMemoryAppendOnlyTelemetrySink()
    graph = FakeGraphMemory(storage=graph_storage, provider_state=graph_state)
    episode_service = EpisodeService(
        telemetry_backend=FakeTelemetryBackend(store, delivery=delivery),
        telemetry_store=store,
        graph_memory=graph,  # type: ignore[arg-type]
    )
    graph.service = episode_service
    return episode_service, graph


def test_rejects_out_of_order_mutated_and_off_cadence_telemetry() -> None:
    episode_service, _ = service()

    async def exercise() -> None:
        await episode_service.open_episode(identity())
        await episode_service.append_telemetry(telemetry(0))
        with pytest.raises(OutOfOrderTelemetryError, match="expected sequence 1"):
            await episode_service.append_telemetry(telemetry(2))
        with pytest.raises(TelemetryMutationError, match="immutable"):
            await episode_service.append_telemetry(telemetry(0, payload={"changed": True}))
        with pytest.raises(TelemetryCadenceError, match=r"0\.050000s"):
            await episode_service.append_telemetry(telemetry(1, sim_time_seconds=0.06))

    asyncio.run(exercise())


def test_closes_exactly_once_and_rejects_post_close_appends() -> None:
    episode_service, _ = service()

    async def exercise() -> None:
        await episode_service.open_episode(identity())
        await episode_service.append_telemetry(telemetry(0))
        closure = await episode_service.close_episode(result(), closed_at=NOW)
        assert closure.telemetry.provider_complete
        with pytest.raises(EpisodeClosedError, match="already closed"):
            await episode_service.close_episode(result(), closed_at=NOW)
        with pytest.raises(EpisodeClosedError, match="already closed"):
            await episode_service.append_telemetry(telemetry(1))

    asyncio.run(exercise())


def test_graph_writes_happen_only_after_close_and_preserve_negative_clearance() -> None:
    episode_service, graph = service()

    async def exercise() -> None:
        await episode_service.open_episode(identity())
        await episode_service.append_telemetry(telemetry(0))
        closure = await episode_service.close_episode(
            result(
                success=False,
                failed_reasons=("BODY_COLLISION", "INSUFFICIENT_OBSTACLE_CLEARANCE"),
                minimum_clearance_m=-0.04,
            ),
            closed_at=NOW,
        )

        assert closure.graph.complete
        assert graph.calls
        assert all(call[2] is EpisodeLifecycleState.CLOSED for call in graph.calls)
        episode_record = graph.calls[0][1]
        assert episode_record.minimum_clearance_m == -0.04
        assert [call[0] for call in graph.calls] == ["episode", "failure", "failure"]

    asyncio.run(exercise())


def test_replay_is_ordered_and_frame_id_is_the_only_video_join_key() -> None:
    episode_service, _ = service()

    async def exercise() -> None:
        await episode_service.open_episode(identity())
        await episode_service.append_telemetry(telemetry(0))
        await episode_service.append_telemetry(telemetry(1))
        await episode_service.close_episode(result(), closed_at=NOW)

        replay = await episode_service.replay_episode("episode-1")
        assert [item.record.sequence for item in replay] == [0, 1]
        assert [item.video_join for item in replay] == [
            ("frame_id", "video-frame-0000"),
            ("frame_id", "video-frame-0001"),
        ]
        assert not hasattr(replay[0], "video_timestamp")

    asyncio.run(exercise())


def test_delivery_reports_do_not_claim_provider_success_for_local_fallbacks() -> None:
    episode_service, _ = service(
        delivery=TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY,
        graph_storage=GraphStorage.LOCAL_CACHE,
        graph_state=ProviderState.UNCONFIGURED,
    )

    async def exercise() -> None:
        await episode_service.open_episode(identity())
        receipt = await episode_service.append_telemetry(telemetry(0))
        closure = await episode_service.close_episode(result(), closed_at=NOW)

        assert receipt.delivery is TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
        assert closure.telemetry.partial_delivery
        assert not closure.telemetry.provider_complete
        assert closure.graph.complete
        assert closure.graph.partial_provider_delivery
        assert not closure.graph.provider_complete

    asyncio.run(exercise())


def test_corrections_remain_pending_until_authenticated_human_approval() -> None:
    episode_service, graph = service()

    async def exercise() -> None:
        await episode_service.open_episode(identity())
        await episode_service.append_telemetry(telemetry(0))
        closure = await episode_service.close_episode(
            result(
                success=False,
                failed_reasons=("INSUFFICIENT_OBSTACLE_CLEARANCE",),
                minimum_clearance_m=0.1,
            ),
            closed_at=NOW,
        )
        failure_id = closure.failures[0].failure_id
        correction = await episode_service.submit_route_correction(
            episode_id="episode-1",
            failure_id=failure_id,
            points=(CorrectionPoint(0.0, 0.0), CorrectionPoint(1.0, 1.0)),
            description="Keep the tray farther from the basket.",
            submitted_by="user-7",
            created_at=NOW,
        )
        duplicate = await episode_service.submit_route_correction(
            episode_id="episode-1",
            failure_id=failure_id,
            points=(CorrectionPoint(0.0, 0.0), CorrectionPoint(1.0, 1.0)),
            description="Keep the tray farther from the basket.",
            submitted_by="user-7",
            created_at=NOW.replace(minute=1),
        )
        keep_out = await episode_service.submit_keep_out_correction(
            episode_id="episode-1",
            failure_id=failure_id,
            polygon=(
                CorrectionPoint(0.0, 0.0),
                CorrectionPoint(1.0, 0.0),
                CorrectionPoint(1.0, 1.0),
            ),
            description="Keep out of the basket clearance region.",
            submitted_by="user-7",
            created_at=NOW,
        )
        feed = TrainingCorrectionFeed(episode_service)

        assert correction.correction_id == duplicate.correction_id
        assert keep_out.correction_id != correction.correction_id
        assert feed.approved(episode_id="episode-1") == ()
        with pytest.raises(PermissionError, match="authenticated human"):
            await episode_service.approve_correction(
                correction.correction_id,
                approver=AuthenticatedHuman("user-8", "session", False),
                approved_at=NOW,
            )
        approval = await episode_service.approve_correction(
            correction.correction_id,
            approver=AuthenticatedHuman("user-8", "webauthn", True),
            approved_at=NOW,
        )

        assert approval.graph_receipt is not None
        assert feed.approved(episode_id="episode-1")[0].correction_id == correction.correction_id
        correction_call, lesson_call = graph.calls[-2:]
        assert correction_call[0] == "correction"
        assert lesson_call[0] == "lesson"
        assert lesson_call[1].correction_id == correction.correction_id
        assert lesson_call[1].kind == "route_correction"
        assert lesson_call[1].summary == (
            "Approved route correction for insufficient obstacle clearance."
        )
        with pytest.raises(CorrectionAlreadyApprovedError, match="already been approved"):
            await episode_service.approve_correction(
                correction.correction_id,
                approver=AuthenticatedHuman("user-8", "webauthn", True),
                approved_at=NOW,
            )

    asyncio.run(exercise())


def test_held_out_episodes_cannot_produce_training_corrections() -> None:
    episode_service, _ = service()

    async def exercise() -> None:
        await episode_service.open_episode(identity(WorldSplit.HELD_OUT))
        await episode_service.append_telemetry(telemetry(0))
        closure = await episode_service.close_episode(
            result(
                success=False,
                failed_reasons=("INSUFFICIENT_OBSTACLE_CLEARANCE",),
                minimum_clearance_m=0.1,
                world_split="held_out",
            ),
            closed_at=NOW,
        )
        with pytest.raises(EpisodeServiceError, match="held-out"):
            await episode_service.submit_route_correction(
                episode_id="episode-1",
                failure_id=closure.failures[0].failure_id,
                points=(CorrectionPoint(0.0, 0.0), CorrectionPoint(1.0, 1.0)),
                description="This must never enter training.",
                submitted_by="user-7",
                created_at=NOW,
            )

    asyncio.run(exercise())


def test_correction_import_firewall_excludes_evaluation_and_control_paths() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "muscle_memory"
    protected = (
        root / "evaluation",
        root / "policy",
        root / "robot",
        root / "simulation",
    )
    violations: list[str] = []
    for directory in protected:
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "muscle_memory.episodes" in source or "episodes.training" in source:
                violations.append(str(path.relative_to(root)))

    assert violations == []
