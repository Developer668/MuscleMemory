"""Append-only telemetry storage interfaces and a deterministic local sink."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Protocol

from muscle_memory.telemetry.models import EpisodeTelemetryRecord


class TelemetryAppendError(ValueError):
    """Base class for rejected append operations."""


class DuplicateTelemetryRecordError(TelemetryAppendError):
    """The identical event was already appended."""


class TelemetryMutationError(TelemetryAppendError):
    """An append attempted to replace an existing sequence."""


class OutOfOrderTelemetryError(TelemetryAppendError):
    """An append did not continue the episode's ordered stream."""


class AppendOnlyTelemetrySink(Protocol):
    def append(self, record: EpisodeTelemetryRecord) -> None: ...

    def records_for(self, episode_id: str) -> tuple[EpisodeTelemetryRecord, ...]: ...


class InMemoryAppendOnlyTelemetrySink:
    """Thread-safe local sink used by tests and offline development."""

    def __init__(self) -> None:
        self._records: dict[str, list[EpisodeTelemetryRecord]] = defaultdict(list)
        self._lock = RLock()

    def append(self, record: EpisodeTelemetryRecord) -> None:
        record.verify_integrity()
        with self._lock:
            episode_records = self._records[record.episode_id]
            if record.sequence < len(episode_records):
                existing = episode_records[record.sequence]
                if existing == record:
                    raise DuplicateTelemetryRecordError(
                        f"episode {record.episode_id!r} sequence {record.sequence} already exists"
                    )
                raise TelemetryMutationError(
                    f"episode {record.episode_id!r} sequence {record.sequence} is immutable"
                )

            expected_sequence = len(episode_records)
            if record.sequence != expected_sequence:
                raise OutOfOrderTelemetryError(
                    f"expected sequence {expected_sequence}, received {record.sequence}"
                )
            if episode_records and record.sim_time_seconds < episode_records[-1].sim_time_seconds:
                raise OutOfOrderTelemetryError("simulation time cannot move backwards")

            episode_records.append(record)

    def records_for(self, episode_id: str) -> tuple[EpisodeTelemetryRecord, ...]:
        with self._lock:
            return tuple(self._records.get(episode_id, ()))
