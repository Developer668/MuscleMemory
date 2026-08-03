"""LaserData provider adapter with an honestly labeled durable fallback."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from muscle_memory.paths import REPOSITORY_ROOT
from muscle_memory.telemetry.durable import DurableTelemetrySpool
from muscle_memory.telemetry.models import EpisodeTelemetryRecord
from muscle_memory.telemetry.wire import LaserDataTelemetryEnvelope

LASERDATA_PROVIDER_NAME = "LaserData"
# The first stable release uses Iggy's VSR wire protocol, while the supported
# self-hosted Apache Iggy image remains on the classic protocol.
LASERDATA_SDK_REQUIREMENT = "laser-sdk==0.0.1"
NUMERIC_TELEMETRY_HZ = 20
DEFAULT_LASERDATA_STREAM = "muscle-memory"
DEFAULT_LASERDATA_TOPIC = "episode-events-v2"
DEFAULT_LASERDATA_PARTITIONS = 4
DEFAULT_LASERDATA_TIMEOUT_SECONDS = 5.0
DEFAULT_DURABLE_SPOOL_PATH = REPOSITORY_ROOT / "artifacts" / "telemetry" / "laserdata-spool.sqlite3"


class LaserDataProviderState(StrEnum):
    """Truthful readiness levels; local fallback never counts as provider health."""

    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    END_TO_END_VERIFIED = "end_to_end_verified"


class TelemetryDelivery(StrEnum):
    """Where a telemetry append is durably present."""

    LASERDATA_AND_DURABLE_CACHE = "laserdata_and_durable_cache"
    DURABLE_LOCAL_CACHE_ONLY = "durable_local_cache_only"


class LaserDataDependencyError(RuntimeError):
    """The official provider SDK is not installed in the runtime image."""


@dataclass(frozen=True, slots=True)
class NumericTelemetryCadence:
    """Exact 20 Hz sampling schedule over an integer physics clock."""

    physics_hz: int = 500
    sample_hz: int = NUMERIC_TELEMETRY_HZ

    def __post_init__(self) -> None:
        if self.physics_hz <= 0 or self.sample_hz <= 0:
            raise ValueError("telemetry and physics rates must be positive")
        if self.physics_hz % self.sample_hz:
            raise ValueError("numeric telemetry must divide the physics clock exactly")

    @property
    def interval_steps(self) -> int:
        return self.physics_hz // self.sample_hz

    def is_due(self, physics_step: int) -> bool:
        if physics_step < 0:
            raise ValueError("physics step must be non-negative")
        return physics_step % self.interval_steps == 0

    def sequence_for_step(self, physics_step: int) -> int:
        if not self.is_due(physics_step):
            raise ValueError("physics step is not on the numeric telemetry clock")
        return physics_step // self.interval_steps


@dataclass(frozen=True, slots=True)
class LaserDataConfig:
    """Provider settings loaded without ever rendering embedded credentials."""

    connection_string: str = field(repr=False)
    stream: str = DEFAULT_LASERDATA_STREAM
    topic: str = DEFAULT_LASERDATA_TOPIC
    partitions: int = DEFAULT_LASERDATA_PARTITIONS
    timeout_seconds: float = DEFAULT_LASERDATA_TIMEOUT_SECONDS
    spool_path: Path = DEFAULT_DURABLE_SPOOL_PATH

    def __post_init__(self) -> None:
        if not self.stream.strip() or not self.topic.strip():
            raise ValueError("LaserData stream and topic must not be empty")
        if not 1 <= self.partitions <= 64:
            raise ValueError("LaserData partitions must be between 1 and 64")
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
            or self.timeout_seconds > 60.0
        ):
            raise ValueError("LaserData timeout must be within (0, 60] seconds")

    @property
    def configured(self) -> bool:
        return bool(self.connection_string.strip())

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LaserDataConfig:
        source = os.environ if environ is None else environ
        raw_partitions = source.get("LASERDATA_PARTITIONS", str(DEFAULT_LASERDATA_PARTITIONS))
        raw_timeout = source.get(
            "LASERDATA_TIMEOUT_SECONDS", str(DEFAULT_LASERDATA_TIMEOUT_SECONDS)
        )
        raw_spool_path = source.get(
            "MUSCLE_MEMORY_TELEMETRY_SPOOL", str(DEFAULT_DURABLE_SPOOL_PATH)
        )
        try:
            partitions = int(raw_partitions)
        except ValueError as error:
            raise ValueError("LASERDATA_PARTITIONS must be an integer") from error
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as error:
            raise ValueError("LASERDATA_TIMEOUT_SECONDS must be numeric") from error
        return cls(
            connection_string=source.get("LASERDATA_CONNECTION_STRING", "").strip(),
            stream=source.get("LASERDATA_STREAM", DEFAULT_LASERDATA_STREAM).strip(),
            topic=source.get("LASERDATA_TOPIC", DEFAULT_LASERDATA_TOPIC).strip(),
            partitions=partitions,
            timeout_seconds=timeout_seconds,
            spool_path=Path(raw_spool_path).expanduser(),
        )


@dataclass(frozen=True, slots=True)
class LaserDataConnectionInfo:
    capabilities: tuple[str, ...]


class LaserDataTransport(Protocol):
    async def initialize(self) -> LaserDataConnectionInfo: ...

    async def publish(self, envelope: LaserDataTelemetryEnvelope) -> None: ...

    async def find_event(self, event_id: str) -> str | None: ...

    async def close(self) -> None: ...


class _LaserPublishRequest(Protocol):
    def index(self, key: str, value: str) -> _LaserPublishRequest: ...

    def header(self, key: str, value: str) -> _LaserPublishRequest: ...

    def inline_payload(self) -> _LaserPublishRequest: ...

    def partition_key(self, key: str) -> _LaserPublishRequest: ...

    def send(self) -> Awaitable[object]: ...


class _LaserTypedRecord(Protocol):
    @property
    def value(self) -> object: ...

    @property
    def position(self) -> str: ...


class _LaserTypedRecords(Protocol):
    def next(self) -> Awaitable[_LaserTypedRecord | None]: ...


class _LaserTopic(Protocol):
    def ensure(self, partitions: int) -> Awaitable[object]: ...

    def publish(self, body: object) -> _LaserPublishRequest: ...

    def records(
        self,
        reader_name: str,
        *,
        from_offsets: Sequence[int],
    ) -> _LaserTypedRecords: ...


class _LaserClient(Protocol):
    def topic(self, name: str, *, cls: type[object]) -> _LaserTopic: ...

    def capabilities(self) -> Awaitable[object]: ...

    def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> Awaitable[object]: ...


class _LaserFactory(Protocol):
    def connect(self, connection_string: str, *, stream: str) -> Awaitable[_LaserClient]: ...


class OfficialLaserDataTransport:
    """Production transport backed by the official LaserData Python SDK."""

    def __init__(self, config: LaserDataConfig) -> None:
        if not config.configured:
            raise ValueError("official LaserData transport requires provider configuration")
        self._config = config
        self._client: _LaserClient | None = None
        self._topic: _LaserTopic | None = None

    @staticmethod
    def _load_sdk() -> ModuleType:
        try:
            return importlib.import_module("laser_sdk")
        except ModuleNotFoundError as error:
            raise LaserDataDependencyError(
                f"production LaserData transport requires {LASERDATA_SDK_REQUIREMENT}"
            ) from error

    async def initialize(self) -> LaserDataConnectionInfo:
        module = self._load_sdk()
        laser_factory = cast(_LaserFactory, module.Laser)
        client = await asyncio.wait_for(
            laser_factory.connect(
                self._config.connection_string,
                stream=self._config.stream,
            ),
            timeout=self._config.timeout_seconds,
        )
        try:
            topic = client.topic(self._config.topic, cls=LaserDataTelemetryEnvelope)
            await asyncio.wait_for(
                topic.ensure(self._config.partitions),
                timeout=self._config.timeout_seconds,
            )
            raw_capabilities = await asyncio.wait_for(
                client.capabilities(),
                timeout=self._config.timeout_seconds,
            )
        except BaseException:
            await client.__aexit__(None, None, None)
            raise
        capabilities = self._capability_names(raw_capabilities)
        self._client = client
        self._topic = topic
        return LaserDataConnectionInfo(capabilities=capabilities)

    @staticmethod
    def _capability_names(raw_capabilities: object) -> tuple[str, ...]:
        if isinstance(raw_capabilities, Mapping):
            return tuple(sorted(str(key) for key in raw_capabilities))
        if isinstance(raw_capabilities, Sequence) and not isinstance(
            raw_capabilities, (str, bytes, bytearray)
        ):
            return tuple(sorted(str(value) for value in raw_capabilities))
        return (type(raw_capabilities).__name__,)

    def _connected_topic(self) -> _LaserTopic:
        if self._topic is None:
            raise RuntimeError("LaserData transport has not initialized")
        return self._topic

    async def publish(self, envelope: LaserDataTelemetryEnvelope) -> None:
        topic = self._connected_topic()
        request = (
            topic.publish(envelope)
            .partition_key(envelope.record.episode_id)
            .index("episode_id", envelope.record.episode_id)
            .index("world_id", envelope.record.world_id)
            .index("policy_id", envelope.record.policy_id)
            .index("failure_type", envelope.record.failure_type or "none")
            .index("event_time", f"{envelope.record.event_time:.9f}")
            .index("sequence", str(envelope.record.sequence))
            .index("event_id", envelope.event_id)
            .header("frame_id", envelope.record.frame_id or "")
            .inline_payload()
        )
        await asyncio.wait_for(
            request.send(),
            timeout=self._config.timeout_seconds,
        )

    async def find_event(self, event_id: str) -> str | None:
        topic = self._connected_topic()
        reader = topic.records(
            f"muscle-memory-verify-{event_id[:16]}",
            from_offsets=tuple(0 for _ in range(self._config.partitions)),
        )

        async def scan() -> str | None:
            while (provider_record := await reader.next()) is not None:
                envelope = LaserDataTelemetryEnvelope.model_validate(provider_record.value)
                if envelope.event_id == event_id:
                    return provider_record.position
            return None

        return await asyncio.wait_for(scan(), timeout=self._config.timeout_seconds)

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._topic = None
        if client is not None:
            await asyncio.wait_for(
                client.__aexit__(None, None, None),
                timeout=self._config.timeout_seconds,
            )


TransportFactory = Callable[[LaserDataConfig], LaserDataTransport]


@dataclass(frozen=True, slots=True)
class LaserDataHealth:
    provider: str
    state: LaserDataProviderState
    configured: bool
    provider_writes_active: bool
    active_store: TelemetryDelivery
    pending_local_records: int
    capabilities: tuple[str, ...]
    detail: str
    last_error_type: str | None


@dataclass(frozen=True, slots=True)
class TelemetryAppendResult:
    event_id: str
    delivery: TelemetryDelivery
    provider_state: LaserDataProviderState
    pending_local_records: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class LaserDataTelemetryBackend:
    """Write-through provider backend with an immutable local outbox.

    Every event lands in the durable spool first. A successful LaserData append
    adds a separate provider receipt. Provider failures leave the original event
    pending and return ``durable_local_cache_only`` instead of claiming success.
    """

    def __init__(
        self,
        config: LaserDataConfig,
        *,
        spool: DurableTelemetrySpool | None = None,
        transport_factory: TransportFactory = OfficialLaserDataTransport,
    ) -> None:
        self.config = config
        self.spool = spool or DurableTelemetrySpool(config.spool_path)
        self._transport_factory = transport_factory
        self._transport: LaserDataTransport | None = None
        self._capabilities: tuple[str, ...] = ()
        self._last_error_type: str | None = None
        self._lock = asyncio.Lock()
        self._state = (
            LaserDataProviderState.CONFIGURED
            if config.configured
            else LaserDataProviderState.UNCONFIGURED
        )
        self._detail = (
            "LaserData is configured but has not passed its startup probe"
            if config.configured
            else "LASERDATA_CONNECTION_STRING is unset; durable local cache only"
        )

    @property
    def health(self) -> LaserDataHealth:
        provider_active = self._transport is not None and self._state in {
            LaserDataProviderState.HEALTHY,
            LaserDataProviderState.END_TO_END_VERIFIED,
        }
        return LaserDataHealth(
            provider=LASERDATA_PROVIDER_NAME,
            state=self._state,
            configured=self.config.configured,
            provider_writes_active=provider_active,
            active_store=(
                TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE
                if provider_active
                else TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
            ),
            pending_local_records=self.spool.pending_count,
            capabilities=self._capabilities,
            detail=self._detail,
            last_error_type=self._last_error_type,
        )

    async def initialize(self) -> LaserDataHealth:
        async with self._lock:
            if not self.config.configured:
                return self.health
            if self._transport is not None:
                await self._close_transport_locked()
            transport = self._transport_factory(self.config)
            try:
                connection = await transport.initialize()
            except Exception as error:
                self._mark_degraded("initialization", error)
                with contextlib.suppress(Exception):
                    await transport.close()
                return self.health
            self._transport = transport
            self._capabilities = connection.capabilities
            self._last_error_type = None
            self._state = LaserDataProviderState.HEALTHY
            self._detail = "provider connection and telemetry topic probe succeeded"
            await self._flush_pending_locked()
            return self.health

    async def append(self, record: EpisodeTelemetryRecord) -> TelemetryAppendResult:
        envelope = LaserDataTelemetryEnvelope.from_domain(record)
        async with self._lock:
            self.spool.append(record)
            delivery = TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
            if self._provider_active:
                accepted = await self._publish_locked(envelope)
                if accepted:
                    delivery = TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE
            return TelemetryAppendResult(
                event_id=envelope.event_id,
                delivery=delivery,
                provider_state=self._state,
                pending_local_records=self.spool.pending_count,
            )

    def recover_append_result(self, record: EpisodeTelemetryRecord) -> TelemetryAppendResult:
        """Rebuild a truthful receipt after the spool/journal crash window."""

        records = self.spool.records_for(record.episode_id)
        if record.sequence >= len(records) or records[record.sequence] != record:
            raise ValueError("cannot recover a receipt for non-durable telemetry")
        envelope = LaserDataTelemetryEnvelope.from_domain(record)
        accepted = self.spool.accepted_receipt(envelope.event_id)
        if accepted is None:
            delivery = TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY
            state = self._state
        else:
            delivery = TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE
            state = (
                LaserDataProviderState.END_TO_END_VERIFIED
                if self.spool.verified_position(envelope.event_id) is not None
                else LaserDataProviderState.HEALTHY
            )
        return TelemetryAppendResult(
            event_id=envelope.event_id,
            delivery=delivery,
            provider_state=state,
            pending_local_records=self.spool.pending_count,
        )

    @property
    def _provider_active(self) -> bool:
        return self._transport is not None and self._state in {
            LaserDataProviderState.HEALTHY,
            LaserDataProviderState.END_TO_END_VERIFIED,
        }

    async def _publish_locked(self, envelope: LaserDataTelemetryEnvelope) -> bool:
        transport = self._transport
        if transport is None:
            return False
        try:
            await transport.publish(envelope)
        except Exception as error:
            self._mark_degraded("publish", error)
            await self._close_transport_locked()
            return False
        self.spool.mark_provider_accepted(
            envelope.event_id,
            provider=LASERDATA_PROVIDER_NAME,
            accepted_at_utc=_utc_now(),
        )
        return True

    async def flush_pending(self) -> int:
        async with self._lock:
            return await self._flush_pending_locked()

    async def _flush_pending_locked(self) -> int:
        flushed = 0
        for envelope in self.spool.pending_envelopes():
            if not self._provider_active:
                break
            recovered = await self._recover_pending_locked(envelope)
            if recovered is None:
                break
            if not recovered and not await self._publish_locked(envelope):
                break
            flushed += 1
        if flushed:
            self._detail = f"provider healthy; flushed {flushed} durable local record(s)"
        return flushed

    async def _recover_pending_locked(
        self,
        envelope: LaserDataTelemetryEnvelope,
    ) -> bool | None:
        """Recover a receipt after a publish/crash window without duplicating the event."""

        transport = self._transport
        if transport is None:
            return None
        try:
            position = await transport.find_event(envelope.event_id)
        except Exception as error:
            self._mark_degraded("pending-event replay check", error)
            await self._close_transport_locked()
            return None
        if position is None:
            return False
        observed_at = _utc_now()
        self.spool.mark_provider_accepted(
            envelope.event_id,
            provider=LASERDATA_PROVIDER_NAME,
            accepted_at_utc=observed_at,
        )
        self.spool.mark_provider_verified(
            envelope.event_id,
            provider_position=position,
            verified_at_utc=observed_at,
        )
        self._state = LaserDataProviderState.END_TO_END_VERIFIED
        self._detail = "provider replay recovered an exact pending event without republishing"
        return True

    async def verify_event(self, event_id: str) -> str | None:
        async with self._lock:
            if self.spool.accepted_receipt(event_id) is None:
                raise ValueError("event has not been accepted by LaserData")
            transport = self._transport
            if not self._provider_active or transport is None:
                return None
            try:
                position = await transport.find_event(event_id)
            except Exception as error:
                self._mark_degraded("replay verification", error)
                await self._close_transport_locked()
                return None
            if position is None:
                self._state = LaserDataProviderState.HEALTHY
                self._detail = "provider healthy; appended event was not found during replay"
                return None
            durable_position = self.spool.mark_provider_verified(
                event_id,
                provider_position=position,
                verified_at_utc=_utc_now(),
            )
            self._state = LaserDataProviderState.END_TO_END_VERIFIED
            self._detail = "provider append and exact event replay both succeeded"
            return durable_position

    def _mark_degraded(self, operation: str, error: Exception) -> None:
        error_type = type(error).__name__
        self._state = LaserDataProviderState.DEGRADED
        self._last_error_type = error_type
        self._detail = f"LaserData {operation} failed ({error_type}); durable local cache only"

    async def _close_transport_locked(self) -> None:
        transport = self._transport
        self._transport = None
        self._capabilities = ()
        if transport is not None:
            with contextlib.suppress(Exception):
                await transport.close()

    async def close(self) -> None:
        async with self._lock:
            await self._close_transport_locked()
            self.spool.close()
