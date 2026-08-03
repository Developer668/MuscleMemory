from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from muscle_memory.telemetry import (
    DurableTelemetrySpool,
    EpisodeTelemetryRecord,
    LaserDataConfig,
    LaserDataConnectionInfo,
    LaserDataProviderState,
    LaserDataTelemetryBackend,
    LaserDataTelemetryEnvelope,
    NumericTelemetryCadence,
    OfficialLaserDataTransport,
    OutOfOrderTelemetryError,
    SensorCategory,
    SensorReading,
    SensorSnapshot,
    SignalUseLabel,
    TelemetryDelivery,
)


def make_record(
    sequence: int,
    *,
    episode_id: str = "episode-laserdata-001",
    sim_time_seconds: float | None = None,
    frame_id: str | None = None,
) -> EpisodeTelemetryRecord:
    joined_frame_id = frame_id or f"{episode_id}:{sequence:08d}"
    snapshot = SensorSnapshot.all_unavailable()
    snapshot = replace(
        snapshot,
        stereo_vision_and_depth=SensorReading.available_reading(
            SensorCategory.STEREO_VISION_AND_DEPTH,
            SignalUseLabel.USED_BY_POLICY,
            {
                "frame_id": joined_frame_id,
                "derived_depth_sectors_m": [1.0] * 48,
            },
        ),
    )
    return EpisodeTelemetryRecord.create(
        episode_id=episode_id,
        world_id="world-laserdata-001",
        policy_id="policy-laserdata-v1",
        sequence=sequence,
        sim_time_seconds=(sequence / 20 if sim_time_seconds is None else sim_time_seconds),
        robot_checksum="a" * 64,
        policy_hash="b" * 64,
        world_hash="c" * 64,
        signal_use=SignalUseLabel.LOGGED_ONLY,
        sensors=snapshot,
        payload={"event": "numeric_telemetry", "sample_rate_hz": 20},
        failure_type=None,
        frame_id=joined_frame_id,
    )


class FakeLaserDataTransport:
    def __init__(
        self,
        *,
        initialize_error: Exception | None = None,
        publish_error: Exception | None = None,
    ) -> None:
        self.initialize_error = initialize_error
        self.publish_error = publish_error
        self.published: list[LaserDataTelemetryEnvelope] = []
        self.positions: dict[str, str] = {}
        self.closed = False

    async def initialize(self) -> LaserDataConnectionInfo:
        if self.initialize_error is not None:
            raise self.initialize_error
        return LaserDataConnectionInfo(capabilities=("streams", "replay"))

    async def publish(self, envelope: LaserDataTelemetryEnvelope) -> None:
        if self.publish_error is not None:
            raise self.publish_error
        self.published.append(envelope)
        self.positions[envelope.event_id] = f"0:{len(self.published) - 1}"

    async def find_event(self, event_id: str) -> str | None:
        return self.positions.get(event_id)

    async def close(self) -> None:
        self.closed = True


class FakeSdkTypedRecord:
    def __init__(self, value: object, position: str) -> None:
        self.value = value
        self.position = position


class FakeSdkReader:
    def __init__(self, records: list[LaserDataTelemetryEnvelope]) -> None:
        self._records = records
        self._index = 0

    async def next(self) -> FakeSdkTypedRecord | None:
        if self._index >= len(self._records):
            return None
        index = self._index
        self._index += 1
        return FakeSdkTypedRecord(self._records[index], f"1:{index}")


class FakeSdkPublishRequest:
    def __init__(
        self,
        topic: FakeSdkTopic,
        envelope: LaserDataTelemetryEnvelope,
    ) -> None:
        self.topic = topic
        self.envelope = envelope
        self.indexes: dict[str, str] = {}
        self.headers: dict[str, str] = {}
        self.key: str | None = None
        self.inline = False

    def index(self, key: str, value: str) -> FakeSdkPublishRequest:
        self.indexes[key] = value
        return self

    def header(self, key: str, value: str) -> FakeSdkPublishRequest:
        self.headers[key] = value
        return self

    def inline_payload(self) -> FakeSdkPublishRequest:
        self.inline = True
        return self

    def partition_key(self, key: str) -> FakeSdkPublishRequest:
        self.key = key
        return self

    async def send(self) -> object:
        self.topic.sent_requests.append(self)
        self.topic.records_on_log.append(self.envelope)
        return object()


class FakeSdkTopic:
    def __init__(self) -> None:
        self.ensured_partitions: int | None = None
        self.sent_requests: list[FakeSdkPublishRequest] = []
        self.records_on_log: list[LaserDataTelemetryEnvelope] = []
        self.reader_offsets: tuple[int, ...] | None = None

    async def ensure(self, partitions: int) -> object:
        self.ensured_partitions = partitions
        return object()

    def publish(self, body: object) -> FakeSdkPublishRequest:
        return FakeSdkPublishRequest(
            self,
            cast(LaserDataTelemetryEnvelope, body),
        )

    def records(
        self,
        _reader_name: str,
        *,
        from_offsets: tuple[int, ...],
    ) -> FakeSdkReader:
        self.reader_offsets = from_offsets
        return FakeSdkReader(self.records_on_log)


class FakeSdkClient:
    def __init__(self) -> None:
        self.topic_instance = FakeSdkTopic()
        self.topic_name: str | None = None
        self.topic_class: type[object] | None = None
        self.closed = False

    def topic(self, name: str, *, cls: type[object]) -> FakeSdkTopic:
        self.topic_name = name
        self.topic_class = cls
        return self.topic_instance

    async def capabilities(self) -> object:
        return {"streams": True, "replay": True}

    async def __aexit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> object:
        self.closed = True
        return object()


class FakeSdkLaser:
    client = FakeSdkClient()
    connection: str | None = None
    stream: str | None = None

    @classmethod
    async def connect(cls, connection_string: str, *, stream: str) -> FakeSdkClient:
        cls.connection = connection_string
        cls.stream = stream
        return cls.client


def config(path: Path, *, connection: str = "iggy://user:secret@example:8090") -> LaserDataConfig:
    return LaserDataConfig(connection_string=connection, spool_path=path)


def test_numeric_telemetry_cadence_is_exactly_20_hz() -> None:
    cadence = NumericTelemetryCadence(physics_hz=500)

    assert cadence.interval_steps == 25
    assert [step for step in range(101) if cadence.is_due(step)] == [0, 25, 50, 75, 100]
    assert cadence.sequence_for_step(100) == 4
    with pytest.raises(ValueError, match="not on"):
        cadence.sequence_for_step(99)


def test_wire_envelope_round_trips_and_preserves_the_single_frame_join_key() -> None:
    record = make_record(0)

    envelope = LaserDataTelemetryEnvelope.from_domain(record)
    replayed = LaserDataTelemetryEnvelope.model_validate_json(envelope.canonical_json()).to_domain()

    assert replayed == record
    assert envelope.schema_version == "muscle-memory.episode-event.v2"
    assert envelope.record.world_id == "world-laserdata-001"
    assert envelope.record.policy_id == "policy-laserdata-v1"
    assert envelope.record.event_time == record.sim_time_seconds
    assert envelope.record.failure_type is None
    assert envelope.frame_join_key == "frame_id"
    assert envelope.record.frame_id == record.frame_id
    assert envelope.record.sensors[0].model_dump()["category"] == (
        SensorCategory.STEREO_VISION_AND_DEPTH
    )


def test_wire_envelope_rejects_a_different_stereo_frame_id() -> None:
    record = make_record(0)
    stereo = SensorReading.available_reading(
        SensorCategory.STEREO_VISION_AND_DEPTH,
        SignalUseLabel.USED_BY_POLICY,
        {"frame_id": "different-frame", "derived_depth_sectors_m": [1.0] * 48},
    )

    with pytest.raises(ValueError, match="share exactly one frame_id"):
        LaserDataTelemetryEnvelope.from_domain(
            replace(record, sensors=replace(record.sensors, stereo_vision_and_depth=stereo))
        )


def test_durable_spool_survives_reopen_and_rejects_out_of_order_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.sqlite3"
    first = make_record(0)
    with DurableTelemetrySpool(path) as spool:
        spool.append(first)
        with pytest.raises(OutOfOrderTelemetryError, match="expected sequence 1"):
            spool.append(make_record(2))

    with DurableTelemetrySpool(path) as reopened:
        assert reopened.records_for(first.episode_id) == (first,)
        assert reopened.pending_count == 1


def test_sqlite_triggers_block_record_updates_and_deletes(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    with DurableTelemetrySpool(path) as spool:
        spool.append(make_record(0))
        connection = sqlite3.connect(path)
        try:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute("UPDATE telemetry_records SET sequence = 1 WHERE sequence = 0")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute("DELETE FROM telemetry_records")
        finally:
            connection.close()


def test_unconfigured_backend_uses_only_the_honestly_labeled_durable_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.sqlite3"
    backend = LaserDataTelemetryBackend(config(path, connection=""))

    initial_health = asyncio.run(backend.initialize())
    result = asyncio.run(backend.append(make_record(0)))

    assert initial_health.state is LaserDataProviderState.UNCONFIGURED
    assert not initial_health.provider_writes_active
    assert result.delivery is TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
    assert result.pending_local_records == 1
    asyncio.run(backend.close())

    with DurableTelemetrySpool(path) as reopened:
        assert len(reopened.records_for("episode-laserdata-001")) == 1


def test_healthy_provider_write_and_exact_replay_reach_verified_state(
    tmp_path: Path,
) -> None:
    fake = FakeLaserDataTransport()
    backend = LaserDataTelemetryBackend(
        config(tmp_path / "telemetry.sqlite3"),
        transport_factory=lambda _config: fake,
    )

    health = asyncio.run(backend.initialize())
    result = asyncio.run(backend.append(make_record(0)))
    position = asyncio.run(backend.verify_event(result.event_id))

    assert health.state is LaserDataProviderState.HEALTHY
    assert result.delivery is TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE
    assert result.pending_local_records == 0
    assert len(fake.published) == 1
    assert fake.published[0].record.episode_id == "episode-laserdata-001"
    assert position == "0:0"
    assert backend.health.state is LaserDataProviderState.END_TO_END_VERIFIED
    assert backend.spool.verified_position(result.event_id) == "0:0"
    asyncio.run(backend.close())


def test_official_transport_uses_the_sdk_topic_publish_and_replay_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSdkClient()
    FakeSdkLaser.client = client
    module = SimpleNamespace(Laser=FakeSdkLaser)
    monkeypatch.setattr(
        OfficialLaserDataTransport,
        "_load_sdk",
        staticmethod(lambda: module),
    )
    provider_config = config(tmp_path / "telemetry.sqlite3")
    transport = OfficialLaserDataTransport(provider_config)
    envelope = LaserDataTelemetryEnvelope.from_domain(make_record(0))

    async def exercise() -> tuple[LaserDataConnectionInfo, str | None]:
        connection = await transport.initialize()
        await transport.publish(envelope)
        position = await transport.find_event(envelope.event_id)
        await transport.close()
        return connection, position

    connection, position = asyncio.run(exercise())

    request = client.topic_instance.sent_requests[0]
    assert connection.capabilities == ("replay", "streams")
    assert FakeSdkLaser.stream == provider_config.stream
    assert client.topic_name == provider_config.topic
    assert client.topic_class is LaserDataTelemetryEnvelope
    assert client.topic_instance.ensured_partitions == provider_config.partitions
    assert request.key == envelope.record.episode_id
    assert request.indexes == {
        "episode_id": envelope.record.episode_id,
        "event_time": "0.000000000",
        "event_id": envelope.event_id,
        "failure_type": "none",
        "policy_id": envelope.record.policy_id,
        "sequence": "0",
        "world_id": envelope.record.world_id,
    }
    assert request.headers == {"frame_id": envelope.record.frame_id}
    assert request.inline
    assert client.topic_instance.reader_offsets == (0, 0, 0, 0)
    assert position == "1:0"
    assert client.closed


def test_failure_events_publish_typed_search_indexes(tmp_path: Path) -> None:
    client = FakeSdkClient()
    FakeSdkLaser.client = client
    transport = OfficialLaserDataTransport(config(tmp_path / "telemetry.sqlite3"))
    failure_record = replace(
        make_record(1, sim_time_seconds=0.05),
        failure_type="body_collision",
    )
    envelope = LaserDataTelemetryEnvelope.from_domain(failure_record)

    async def exercise() -> None:
        await transport.initialize()
        await transport.publish(envelope)
        await transport.close()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            OfficialLaserDataTransport,
            "_load_sdk",
            staticmethod(lambda: SimpleNamespace(Laser=FakeSdkLaser)),
        )
        asyncio.run(exercise())

    assert client.topic_instance.sent_requests[0].indexes["failure_type"] == ("body_collision")
    assert client.topic_instance.sent_requests[0].indexes["event_time"] == ("0.050000000")


def test_provider_failure_is_degraded_without_leaking_credentials(
    tmp_path: Path,
) -> None:
    connection_string = "iggy://admin:do-not-log@example:8090"
    fake = FakeLaserDataTransport(publish_error=RuntimeError(f"failed at {connection_string}"))
    backend = LaserDataTelemetryBackend(
        config(tmp_path / "telemetry.sqlite3", connection=connection_string),
        transport_factory=lambda _config: fake,
    )

    asyncio.run(backend.initialize())
    result = asyncio.run(backend.append(make_record(0)))

    assert result.delivery is TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
    assert result.provider_state is LaserDataProviderState.DEGRADED
    assert result.pending_local_records == 1
    assert backend.health.last_error_type == "RuntimeError"
    assert connection_string not in backend.health.detail
    assert "do-not-log" not in repr(backend.config)
    assert fake.closed
    asyncio.run(backend.close())


def test_reinitialize_flushes_pending_records_in_episode_order(tmp_path: Path) -> None:
    failing = FakeLaserDataTransport(publish_error=RuntimeError("offline"))
    recovered = FakeLaserDataTransport()
    transports = iter((failing, recovered))

    def factory(_config: LaserDataConfig) -> FakeLaserDataTransport:
        return next(transports)

    backend = LaserDataTelemetryBackend(
        config(tmp_path / "telemetry.sqlite3"),
        transport_factory=factory,
    )

    asyncio.run(backend.initialize())
    first = asyncio.run(backend.append(make_record(0)))
    second = asyncio.run(backend.append(make_record(1)))
    recovered_health = asyncio.run(backend.initialize())

    assert first.pending_local_records == 1
    assert second.pending_local_records == 2
    assert [item.record.sequence for item in recovered.published] == [0, 1]
    assert recovered_health.state is LaserDataProviderState.HEALTHY
    assert recovered_health.pending_local_records == 0
    asyncio.run(backend.close())


def test_startup_recovers_publish_receipt_without_duplicating_provider_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.sqlite3"
    record = make_record(0)
    envelope = LaserDataTelemetryEnvelope.from_domain(record)
    spool = DurableTelemetrySpool(path)
    spool.append(record)
    provider = FakeLaserDataTransport()
    provider.published.append(envelope)
    provider.positions[envelope.event_id] = "2:41"
    backend = LaserDataTelemetryBackend(
        config(path),
        spool=spool,
        transport_factory=lambda _config: provider,
    )

    health = asyncio.run(backend.initialize())

    assert provider.published == [envelope]
    assert health.state is LaserDataProviderState.END_TO_END_VERIFIED
    assert health.pending_local_records == 0
    assert spool.verified_position(envelope.event_id) == "2:41"
    asyncio.run(backend.close())


def test_provider_verification_position_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    with DurableTelemetrySpool(path) as spool:
        record = make_record(0)
        envelope = LaserDataTelemetryEnvelope.from_domain(record)
        spool.append(record)
        spool.mark_provider_accepted(
            envelope.event_id,
            provider="LaserData",
            accepted_at_utc="2026-08-03T12:00:00Z",
        )
        spool.mark_provider_verified(
            envelope.event_id,
            provider_position="0:1",
            verified_at_utc="2026-08-03T12:00:01Z",
        )
        with pytest.raises(ValueError, match="cannot move"):
            spool.mark_provider_verified(
                envelope.event_id,
                provider_position="0:2",
                verified_at_utc="2026-08-03T12:00:02Z",
            )


def test_live_event_consumers_are_unreachable_from_policy_control_and_evaluation() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "muscle_memory"
    protected = (
        root / "policy",
        root / "robot",
        root / "simulation" / "controller.py",
        root / "simulation" / "runtime.py",
        root / "evaluation",
    )
    forbidden = (
        "telemetry.laserdata",
        "LaserDataTelemetryBackend",
        "LiveTelemetryHub",
        "api.streaming",
    )
    violations: list[str] = []
    for path in protected:
        files = path.rglob("*.py") if path.is_dir() else (path,)
        for source_file in files:
            source = source_file.read_text(encoding="utf-8")
            if any(symbol in source for symbol in forbidden):
                violations.append(str(source_file.relative_to(root)))
    assert violations == []


def test_startup_probe_failure_never_claims_provider_health(tmp_path: Path) -> None:
    fake = FakeLaserDataTransport(initialize_error=ConnectionError("not reachable"))
    backend = LaserDataTelemetryBackend(
        config(tmp_path / "telemetry.sqlite3"),
        transport_factory=lambda _config: fake,
    )

    health = asyncio.run(backend.initialize())

    assert health.state is LaserDataProviderState.DEGRADED
    assert not health.provider_writes_active
    assert health.active_store is TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
    assert health.last_error_type == "ConnectionError"
    assert fake.closed
    asyncio.run(backend.close())
