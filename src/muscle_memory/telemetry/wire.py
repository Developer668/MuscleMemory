"""Canonical LaserData wire records for immutable episode telemetry."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from muscle_memory.telemetry.models import (
    EpisodeTelemetryRecord,
    SensorCategory,
    SensorReading,
    SensorSnapshot,
    SignalUseLabel,
)

LASERDATA_TELEMETRY_SCHEMA = "muscle-memory.episode-event.v2"
FRAME_JOIN_KEY = "frame_id"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class WireSensorReading(BaseModel):
    """One exact sensor-reading representation on the provider wire."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: SensorCategory
    signal_use: SignalUseLabel
    available: bool
    values_json: str

    def to_domain(self) -> SensorReading:
        return SensorReading(
            category=self.category,
            signal_use=self.signal_use,
            available=self.available,
            values_json=self.values_json,
        )


class WireEpisodeTelemetryRecord(BaseModel):
    """Lossless provider representation of an episode telemetry record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    sim_time_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    event_time: float = Field(ge=0.0, allow_inf_nan=False)
    robot_checksum: str = Field(min_length=1)
    policy_hash: str = Field(min_length=1)
    world_hash: str = Field(min_length=1)
    signal_use: SignalUseLabel
    sensors: tuple[WireSensorReading, ...] = Field(min_length=8, max_length=8)
    payload_json: str
    payload_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_type: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    frame_id: str | None = None

    @model_validator(mode="after")
    def validate_sensor_rail_and_join_key(self) -> Self:
        categories = tuple(sensor.category for sensor in self.sensors)
        if categories != tuple(SensorCategory):
            raise ValueError("wire telemetry must contain the ordered eight-category sensor rail")
        stereo = self.sensors[0]
        if stereo.available:
            values = json.loads(stereo.values_json)
            stereo_frame_id = values.get(FRAME_JOIN_KEY) if isinstance(values, dict) else None
            if (
                self.frame_id is None
                or not isinstance(stereo_frame_id, str)
                or not stereo_frame_id
                or stereo_frame_id != self.frame_id
            ):
                raise ValueError("stereo telemetry and the record must share exactly one frame_id")
        return self

    @classmethod
    def from_domain(cls, record: EpisodeTelemetryRecord) -> WireEpisodeTelemetryRecord:
        record.verify_integrity()
        return cls(
            episode_id=record.episode_id,
            world_id=record.world_id,
            policy_id=record.policy_id,
            sequence=record.sequence,
            sim_time_seconds=record.sim_time_seconds,
            event_time=record.event_time,
            robot_checksum=record.robot_checksum,
            policy_hash=record.policy_hash,
            world_hash=record.world_hash,
            signal_use=record.signal_use,
            sensors=tuple(
                WireSensorReading(
                    category=reading.category,
                    signal_use=reading.signal_use,
                    available=reading.available,
                    values_json=reading.values_json,
                )
                for reading in record.sensors.readings
            ),
            payload_json=record.payload_json,
            payload_checksum=record.payload_checksum,
            failure_type=record.failure_type,
            frame_id=record.frame_id,
        )

    def to_domain(self) -> EpisodeTelemetryRecord:
        readings = tuple(sensor.to_domain() for sensor in self.sensors)
        snapshot = SensorSnapshot(*readings)
        return EpisodeTelemetryRecord(
            episode_id=self.episode_id,
            world_id=self.world_id,
            policy_id=self.policy_id,
            sequence=self.sequence,
            sim_time_seconds=self.sim_time_seconds,
            event_time=self.event_time,
            robot_checksum=self.robot_checksum,
            policy_hash=self.policy_hash,
            world_hash=self.world_hash,
            signal_use=self.signal_use,
            sensors=snapshot,
            payload_json=self.payload_json,
            payload_checksum=self.payload_checksum,
            failure_type=self.failure_type,
            frame_id=self.frame_id,
        )

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


class LaserDataTelemetryEnvelope(BaseModel):
    """Self-verifying event sent to and replayed from LaserData."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = LASERDATA_TELEMETRY_SCHEMA
    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_join_key: str = FRAME_JOIN_KEY
    record: WireEpisodeTelemetryRecord

    @model_validator(mode="after")
    def verify_event_id(self) -> Self:
        # LaserData replay is evidence only when the provider event remains bound to
        # the exact canonical telemetry bytes that produced this content address.
        expected = hashlib.sha256(self.record.canonical_json().encode("utf-8")).hexdigest()
        if self.event_id != expected:
            raise ValueError("LaserData event_id does not match the canonical telemetry record")
        if self.schema_version != LASERDATA_TELEMETRY_SCHEMA:
            raise ValueError("unsupported LaserData telemetry schema")
        if self.frame_join_key != FRAME_JOIN_KEY:
            raise ValueError("frame_id must remain the only provider/video join key")
        return self

    @classmethod
    def from_domain(cls, record: EpisodeTelemetryRecord) -> LaserDataTelemetryEnvelope:
        wire_record = WireEpisodeTelemetryRecord.from_domain(record)
        event_id = hashlib.sha256(wire_record.canonical_json().encode("utf-8")).hexdigest()
        return cls(event_id=event_id, record=wire_record)

    def to_domain(self) -> EpisodeTelemetryRecord:
        return self.record.to_domain()

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))
