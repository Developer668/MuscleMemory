"""Durable append-only local spool for LaserData-bound telemetry."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from muscle_memory.telemetry.models import EpisodeTelemetryRecord
from muscle_memory.telemetry.sink import (
    DuplicateTelemetryRecordError,
    OutOfOrderTelemetryError,
    TelemetryMutationError,
)
from muscle_memory.telemetry.wire import LaserDataTelemetryEnvelope


@dataclass(frozen=True, slots=True)
class ProviderDeliveryReceipt:
    """An append-only acknowledgement that a provider accepted one event."""

    event_id: str
    provider: str
    accepted_at_utc: str


class DurableTelemetrySpool:
    """SQLite-backed immutable ledger and provider-delivery outbox.

    Database triggers reject updates and deletes, including writes attempted
    outside this Python API. Provider acknowledgements and replay verifications
    are separate append-only facts, so delivery never mutates an episode record.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._configure()

    def _configure(self) -> None:
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS telemetry_records (
                    episode_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    sim_time_seconds REAL NOT NULL CHECK (sim_time_seconds >= 0),
                    event_id TEXT NOT NULL UNIQUE,
                    envelope_json TEXT NOT NULL,
                    PRIMARY KEY (episode_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS provider_delivery_receipts (
                    event_id TEXT PRIMARY KEY
                        REFERENCES telemetry_records(event_id),
                    provider TEXT NOT NULL,
                    accepted_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_verifications (
                    event_id TEXT PRIMARY KEY
                        REFERENCES provider_delivery_receipts(event_id),
                    provider_position TEXT NOT NULL,
                    verified_at_utc TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS telemetry_records_no_update
                BEFORE UPDATE ON telemetry_records
                BEGIN
                    SELECT RAISE(ABORT, 'telemetry records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS telemetry_records_no_delete
                BEFORE DELETE ON telemetry_records
                BEGIN
                    SELECT RAISE(ABORT, 'telemetry records are append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS delivery_receipts_no_update
                BEFORE UPDATE ON provider_delivery_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'provider receipts are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS delivery_receipts_no_delete
                BEFORE DELETE ON provider_delivery_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'provider receipts are append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS provider_verifications_no_update
                BEFORE UPDATE ON provider_verifications
                BEGIN
                    SELECT RAISE(ABORT, 'provider verifications are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS provider_verifications_no_delete
                BEFORE DELETE ON provider_verifications
                BEGIN
                    SELECT RAISE(ABORT, 'provider verifications are append-only');
                END;
                """
            )

    def append(self, record: EpisodeTelemetryRecord) -> None:
        envelope = LaserDataTelemetryEnvelope.from_domain(record)
        envelope_json = envelope.canonical_json()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    """
                    SELECT envelope_json
                    FROM telemetry_records
                    WHERE episode_id = ? AND sequence = ?
                    """,
                    (record.episode_id, record.sequence),
                ).fetchone()
                if existing is not None:
                    if str(existing["envelope_json"]) == envelope_json:
                        raise DuplicateTelemetryRecordError(
                            f"episode {record.episode_id!r} sequence "
                            f"{record.sequence} already exists"
                        )
                    raise TelemetryMutationError(
                        f"episode {record.episode_id!r} sequence {record.sequence} is immutable"
                    )

                latest = self._connection.execute(
                    """
                    SELECT sequence, sim_time_seconds
                    FROM telemetry_records
                    WHERE episode_id = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                    (record.episode_id,),
                ).fetchone()
                expected_sequence = 0 if latest is None else int(latest["sequence"]) + 1
                if record.sequence != expected_sequence:
                    raise OutOfOrderTelemetryError(
                        f"expected sequence {expected_sequence}, received {record.sequence}"
                    )
                if latest is not None and record.sim_time_seconds < float(
                    latest["sim_time_seconds"]
                ):
                    raise OutOfOrderTelemetryError("simulation time cannot move backwards")

                self._connection.execute(
                    """
                    INSERT INTO telemetry_records (
                        episode_id,
                        sequence,
                        sim_time_seconds,
                        event_id,
                        envelope_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.episode_id,
                        record.sequence,
                        record.sim_time_seconds,
                        envelope.event_id,
                        envelope_json,
                    ),
                )
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def records_for(self, episode_id: str) -> tuple[EpisodeTelemetryRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT envelope_json
                FROM telemetry_records
                WHERE episode_id = ?
                ORDER BY sequence
                """,
                (episode_id,),
            ).fetchall()
        return tuple(
            LaserDataTelemetryEnvelope.model_validate_json(str(row["envelope_json"])).to_domain()
            for row in rows
        )

    def pending_envelopes(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[LaserDataTelemetryEnvelope, ...]:
        if limit is not None and limit < 1:
            raise ValueError("pending envelope limit must be positive")
        statement = """
            SELECT records.envelope_json
            FROM telemetry_records AS records
            LEFT JOIN provider_delivery_receipts AS receipts
                ON receipts.event_id = records.event_id
            WHERE receipts.event_id IS NULL
            ORDER BY records.rowid
        """
        parameters: tuple[int, ...] = ()
        if limit is not None:
            statement += " LIMIT ?"
            parameters = (limit,)
        with self._lock:
            rows = self._connection.execute(statement, parameters).fetchall()
        return tuple(
            LaserDataTelemetryEnvelope.model_validate_json(str(row["envelope_json"]))
            for row in rows
        )

    def mark_provider_accepted(
        self,
        event_id: str,
        *,
        provider: str,
        accepted_at_utc: str,
    ) -> ProviderDeliveryReceipt:
        receipt = ProviderDeliveryReceipt(
            event_id=event_id,
            provider=provider,
            accepted_at_utc=accepted_at_utc,
        )
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT provider, accepted_at_utc
                FROM provider_delivery_receipts
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO provider_delivery_receipts (
                        event_id,
                        provider,
                        accepted_at_utc
                    ) VALUES (?, ?, ?)
                    """,
                    (event_id, provider, accepted_at_utc),
                )
            else:
                if str(existing["provider"]) != provider:
                    raise TelemetryMutationError(
                        "provider acceptance cannot be rebound to another provider"
                    )
                receipt = ProviderDeliveryReceipt(
                    event_id=event_id,
                    provider=str(existing["provider"]),
                    accepted_at_utc=str(existing["accepted_at_utc"]),
                )
        return receipt

    def mark_provider_verified(
        self,
        event_id: str,
        *,
        provider_position: str,
        verified_at_utc: str,
    ) -> str:
        if not provider_position:
            raise ValueError("provider position must not be empty")
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT provider_position
                FROM provider_verifications
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO provider_verifications (
                        event_id,
                        provider_position,
                        verified_at_utc
                    ) VALUES (?, ?, ?)
                    """,
                    (event_id, provider_position, verified_at_utc),
                )
                return provider_position
            existing_position = str(existing["provider_position"])
            if existing_position != provider_position:
                raise TelemetryMutationError("provider verification cannot move an immutable event")
            return existing_position

    def accepted_receipt(self, event_id: str) -> ProviderDeliveryReceipt | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT provider, accepted_at_utc
                FROM provider_delivery_receipts
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return ProviderDeliveryReceipt(
            event_id=event_id,
            provider=str(row["provider"]),
            accepted_at_utc=str(row["accepted_at_utc"]),
        )

    def verified_position(self, event_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT provider_position
                FROM provider_verifications
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return None if row is None else str(row["provider_position"])

    @property
    def pending_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS pending_count
                FROM telemetry_records AS records
                LEFT JOIN provider_delivery_receipts AS receipts
                    ON receipts.event_id = records.event_id
                WHERE receipts.event_id IS NULL
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to count pending telemetry")
        return int(row["pending_count"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> DurableTelemetrySpool:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
