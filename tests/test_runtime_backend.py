from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from muscle_memory.api.models import ProviderOperationalState
from muscle_memory.backend.api_backend import MuscleMemoryApiBackend
from muscle_memory.backend.config import BackendConfig
from muscle_memory.backend.episode_journal import CoordinatorEpisodeJournal
from muscle_memory.backend.episode_runtime import OperationalEpisodeRuntime
from muscle_memory.backend.providers import (
    ProviderDeployment,
    ProviderEvidence,
    ProviderRegistry,
    build_provider_bundle,
)
from muscle_memory.coordinator import (
    CoordinatorStateError,
    CoordinatorStore,
    TrainingEpisodeMetadata,
)
from muscle_memory.coordinator import EpisodeState as CoordinatorEpisodeState
from muscle_memory.coordinator.models import canonical_json
from muscle_memory.episodes import (
    AuthenticatedHuman,
    CorrectionPoint,
    EpisodeClosedError,
    EpisodeIdentity,
    EpisodeLifecycleState,
    EpisodeService,
)
from muscle_memory.episodes.training import TrainingCorrectionFeed
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    GraphStorage,
    GraphWriteReceipt,
    ProviderState,
    WorldSplit,
)
from muscle_memory.runtime import create_api_backend
from muscle_memory.telemetry import (
    DurableTelemetrySpool,
    EpisodeTelemetryRecord,
    LaserDataConfig,
    LaserDataConnectionInfo,
    LaserDataTelemetryBackend,
    LaserDataTelemetryEnvelope,
    SensorSnapshot,
    SignalUseLabel,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
ROBOT_HASH = "a" * 64
WORLD_HASH = "b" * 64
POLICY_HASH = "c" * 64


class FakeGraphMemory:
    def _receipt(self, kind: str, record: Any, record_id: str) -> GraphWriteReceipt:
        return GraphWriteReceipt(
            record_kind=kind,
            record_id=record_id,
            content_hash=record.content_hash,
            storage=GraphStorage.LOCAL_CACHE,
            provider_state=ProviderState.UNCONFIGURED,
            mirrored_to_local_cache=True,
            detail="durable test graph cache",
        )

    def record_episode(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("episode", record, record.episode_id)

    def record_failure(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("failure", record, record.failure_id)

    def record_correction(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("correction", record, record.correction_id)


class CapturingPublisher:
    def __init__(self) -> None:
        self.telemetry: list[object] = []
        self.statuses: list[tuple[str, dict[str, object]]] = []

    async def publish_telemetry(self, telemetry: object) -> None:
        self.telemetry.append(telemetry)

    async def publish_status(
        self,
        episode_id: str,
        status: dict[str, object],
    ) -> None:
        self.statuses.append((episode_id, status))


class FakeLaserTransport:
    def __init__(self) -> None:
        self.events: dict[str, LaserDataTelemetryEnvelope] = {}
        self.closed = False

    async def initialize(self) -> LaserDataConnectionInfo:
        return LaserDataConnectionInfo(capabilities=("publish", "replay"))

    async def publish(self, envelope: LaserDataTelemetryEnvelope) -> None:
        self.events[envelope.event_id] = envelope

    async def find_event(self, event_id: str) -> str | None:
        return "0:0" if event_id in self.events else None

    async def close(self) -> None:
        self.closed = True


def identity(*, checksum: str = ROBOT_HASH, split: WorldSplit = WorldSplit.TRAINING):
    return EpisodeIdentity(
        episode_id="episode-1",
        robot_checksum=checksum,
        world_id="world-1",
        world_hash=WORLD_HASH,
        world_split=split,
        policy_id="policy-1",
        policy_hash=POLICY_HASH,
        opened_at=NOW,
    )


def telemetry(sequence: int, *, failure_type: str | None = None):
    return EpisodeTelemetryRecord.create(
        episode_id="episode-1",
        world_id="world-1",
        policy_id="policy-1",
        sequence=sequence,
        sim_time_seconds=sequence / 20,
        robot_checksum=ROBOT_HASH,
        policy_hash=POLICY_HASH,
        world_hash=WORLD_HASH,
        signal_use=SignalUseLabel.LOGGED_ONLY,
        sensors=SensorSnapshot.all_unavailable(),
        payload={"sample": sequence},
        failure_type=failure_type,
        frame_id=f"frame-{sequence}",
    )


def failed_result() -> PolicyEpisodeResult:
    return PolicyEpisodeResult(
        episode_id="episode-1",
        world_id="world-1",
        world_seed=42,
        world_split="training",
        world_hash=WORLD_HASH,
        robot_checksum=ROBOT_HASH,
        policy_id="policy-1",
        policy_hash=POLICY_HASH,
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


def _service(
    root: Path,
) -> tuple[
    CoordinatorStore,
    DurableTelemetrySpool,
    CoordinatorEpisodeJournal,
    EpisodeService,
]:
    coordinator = CoordinatorStore(root / "coordinator.sqlite3")
    spool = DurableTelemetrySpool(root / "telemetry.sqlite3")
    backend = LaserDataTelemetryBackend(
        LaserDataConfig(connection_string="", spool_path=spool.path),
        spool=spool,
    )
    journal = CoordinatorEpisodeJournal(
        coordinator,
        expected_robot_checksum=ROBOT_HASH,
    )
    service = EpisodeService(
        telemetry_backend=backend,
        telemetry_store=spool,
        graph_memory=FakeGraphMemory(),  # type: ignore[arg-type]
        journal=journal,
    )
    return coordinator, spool, journal, service


def test_episode_closure_approval_and_training_feed_survive_restart(
    tmp_path: Path,
) -> None:
    coordinator, spool, _, service = _service(tmp_path)

    async def first_process() -> str:
        await service.open_episode(identity())
        await service.append_telemetry(telemetry(0, failure_type="body_collision"))
        closure = await service.close_episode(failed_result(), closed_at=NOW)
        correction = await service.submit_route_correction(
            episode_id="episode-1",
            failure_id=closure.failures[0].failure_id,
            points=(CorrectionPoint(0.0, 0.0), CorrectionPoint(1.0, 1.0)),
            description="use the wider route",
            submitted_by="operator-1",
            created_at=NOW,
        )
        await service.approve_correction(
            correction.correction_id,
            approver=AuthenticatedHuman("operator-2", "bearer", True),
            approved_at=NOW,
        )
        return correction.correction_id

    correction_id = asyncio.run(first_process())
    coordinator.close()
    spool.close()

    coordinator2, spool2, _, restored = _service(tmp_path)
    closure = restored.closure_for("episode-1")
    assert closure is not None
    assert closure.telemetry_digest
    assert closure.result.minimum_obstacle_clearance_m == -0.01
    assert asyncio.run(restored.replay_episode("episode-1"))[0].video_join == (
        "frame_id",
        "frame-0",
    )
    approved = TrainingCorrectionFeed(restored).approved(episode_id="episode-1")
    assert tuple(item.correction_id for item in approved) == (correction_id,)
    coordinator2.close()
    spool2.close()


def test_aborted_episode_is_terminal_and_survives_restart(tmp_path: Path) -> None:
    coordinator, spool, _, service = _service(tmp_path)

    async def first_process() -> None:
        await service.open_episode(identity())
        await service.append_telemetry(telemetry(0))
        abort = await service.abort_episode(
            "episode-1",
            error_type="RendererFailure",
            aborted_at=NOW,
        )
        assert abort.error_type == "RendererFailure"

    asyncio.run(first_process())
    assert service.episode_state("episode-1") is EpisodeLifecycleState.ABORTED
    assert coordinator.episode_state("episode-1") is CoordinatorEpisodeState.ABORTED
    assert [transition.state for transition in coordinator.episode_history("episode-1")] == [
        CoordinatorEpisodeState.CREATED,
        CoordinatorEpisodeState.RUNNING,
        CoordinatorEpisodeState.ABORTED,
    ]
    coordinator.close()
    spool.close()

    coordinator2, spool2, journal2, restored = _service(tmp_path)
    assert restored.episode_state("episode-1") is EpisodeLifecycleState.ABORTED
    abort = restored.abort_for("episode-1")
    assert abort is not None
    assert abort.error_type == "RendererFailure"
    assert abort.aborted_at == NOW
    api_backend = object.__new__(MuscleMemoryApiBackend)
    api_backend.journal = journal2
    api_backend.episode_runtime = SimpleNamespace(service=restored)
    summary = api_backend._episode_summary("episode-1")
    assert summary.state.value == "aborted"
    assert summary.closed_at == NOW
    with pytest.raises(EpisodeClosedError, match="was aborted"):
        asyncio.run(restored.append_telemetry(telemetry(1)))
    with pytest.raises(EpisodeClosedError, match="was aborted"):
        asyncio.run(restored.close_episode(failed_result(), closed_at=NOW))
    coordinator2.close()
    spool2.close()


def test_operational_journal_rejects_held_out_and_wrong_robot_state(tmp_path: Path) -> None:
    coordinator = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    journal = CoordinatorEpisodeJournal(
        coordinator,
        expected_robot_checksum=ROBOT_HASH,
    )
    with pytest.raises(CoordinatorStateError, match="training episodes only"):
        journal.record_identity(identity(split=WorldSplit.HELD_OUT))

    wrong = identity(checksum="d" * 64)
    coordinator.register_training_episode(
        TrainingEpisodeMetadata(
            episode_id=wrong.episode_id,
            robot_checksum=wrong.robot_checksum,
            world_hash=wrong.world_hash,
            policy_hash=wrong.policy_hash,
            created_at=wrong.opened_at,
        )
    )
    from pydantic import TypeAdapter

    coordinator.record_training_episode_session(
        wrong.episode_id,
        canonical_json(TypeAdapter(EpisodeIdentity).dump_python(wrong, mode="json")),
    )
    coordinator.close()
    reopened = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    with pytest.raises(CoordinatorStateError, match="qualified MM-01"):
        CoordinatorEpisodeJournal(reopened, expected_robot_checksum=ROBOT_HASH)
    reopened.close()


def test_unconfigured_provider_registry_is_honest_and_secrets_are_redacted(
    tmp_path: Path,
) -> None:
    secret = "super-secret-provider-value"
    config = BackendConfig.from_env(
        {
            "MUSCLE_MEMORY_COORDINATOR_DB_PATH": str(tmp_path / "coordinator.sqlite3"),
            "MUSCLE_MEMORY_TELEMETRY_SPOOL": str(tmp_path / "telemetry.sqlite3"),
            "MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "graph.jsonl"),
            "MM_ASSET_CACHE_DIR": str(tmp_path / "assets"),
            "MM_ASSET_APPROVAL_LEDGER_DIR": str(tmp_path / "approvals"),
            "MM_ASSET_REFERENCE_API_KEY": secret,
            "MM_ASSET_TRELLIS_API_KEY": secret,
        }
    )
    bundle = build_provider_bundle(config)
    health = bundle.registry.health()
    rendered = health.model_dump_json()
    assert health.state is ProviderOperationalState.UNCONFIGURED
    assert secret not in repr(config)
    assert secret not in repr(bundle)
    assert secret not in rendered
    asyncio.run(bundle.laserdata.close())


def test_operational_consumers_do_not_claim_health_before_evidence(
    tmp_path: Path,
) -> None:
    coordinator, spool, _journal, service = _service(tmp_path)
    runtime = OperationalEpisodeRuntime(service, expected_robot_checksum=ROBOT_HASH)
    runtime.bind_live_publisher(CapturingPublisher())  # type: ignore[arg-type]

    snapshots = {item.consumer: item for item in runtime.consumer_snapshots()}
    assert set(snapshots) == {
        "dashboard-and-timeline",
        "replay",
        "safety-and-failure-summary",
        "post-episode-graph-handoff",
        "training-and-evaluation-evidence",
    }
    assert all(
        item.state is ProviderOperationalState.CONFIGURED
        for item in snapshots.values()
    )
    assert all("no " in item.detail.lower() for item in snapshots.values())
    coordinator.close()
    spool.close()


def test_injected_laser_transport_proves_publish_replay_and_live_fanout(
    tmp_path: Path,
) -> None:
    fake = FakeLaserTransport()
    spool = DurableTelemetrySpool(tmp_path / "telemetry.sqlite3")
    laser = LaserDataTelemetryBackend(
        LaserDataConfig(
            connection_string="iggy:secret@127.0.0.1:8090",
            spool_path=spool.path,
        ),
        spool=spool,
        transport_factory=lambda _config: fake,
    )
    coordinator = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    journal = CoordinatorEpisodeJournal(
        coordinator,
        expected_robot_checksum=ROBOT_HASH,
    )
    service = EpisodeService(
        telemetry_backend=laser,
        telemetry_store=spool,
        graph_memory=FakeGraphMemory(),  # type: ignore[arg-type]
        journal=journal,
    )
    runtime = OperationalEpisodeRuntime(service, expected_robot_checksum=ROBOT_HASH)
    publisher = CapturingPublisher()
    runtime.bind_live_publisher(publisher)  # type: ignore[arg-type]

    async def exercise() -> str:
        await laser.initialize()
        await runtime.open_episode(identity())
        receipt = await runtime.append_telemetry(telemetry(0, failure_type="body_collision"))
        await laser.verify_event(receipt.event_id)
        return receipt.event_id

    event_id = asyncio.run(exercise())
    assert event_id in fake.events
    assert len(publisher.telemetry) == 1
    assert publisher.statuses[0][1]["numeric_telemetry_hz"] == 20

    base = BackendConfig.from_env(
        {
            "MUSCLE_MEMORY_COORDINATOR_DB_PATH": str(tmp_path / "unused.sqlite3"),
            "MUSCLE_MEMORY_TELEMETRY_SPOOL": str(tmp_path / "unused-telemetry.sqlite3"),
            "MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "graph.jsonl"),
            "MM_ASSET_CACHE_DIR": str(tmp_path / "assets"),
            "MM_ASSET_APPROVAL_LEDGER_DIR": str(tmp_path / "approvals"),
        }
    )
    other = build_provider_bundle(base)
    registry = ProviderRegistry(
        laserdata=laser,
        graph_memory=other.graph_memory,
        guild=other.guild,
        rocketride=other.rocketride,
        assets=other.assets,
        deployments={"LaserData": ProviderDeployment.SELF_HOSTED},
        evidence=(
            ProviderEvidence(
                provider="LaserData",
                evidence_id=f"laser-event-{event_id}",
                deployment=ProviderDeployment.SELF_HOSTED,
                operation="publish-and-replay",
                observed_at=NOW,
            ),
        ),
    )
    laser_snapshot = next(item for item in registry.snapshots() if item.provider == "LaserData")
    assert laser_snapshot.state is ProviderOperationalState.END_TO_END_VERIFIED
    assert laser_snapshot.deployment is ProviderDeployment.SELF_HOSTED
    assert "secret" not in laser_snapshot.detail
    asyncio.run(laser.close())
    asyncio.run(other.laserdata.close())
    coordinator.close()


def test_factory_path_is_zero_argument() -> None:
    assert tuple(inspect.signature(create_api_backend).parameters) == ()
