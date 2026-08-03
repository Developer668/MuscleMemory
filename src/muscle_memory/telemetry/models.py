"""Immutable domain records for append-only episode telemetry."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any


class SignalUseLabel(StrEnum):
    """Required disclosure of how a signal participates in the system."""

    USED_BY_POLICY = "Used by policy"
    LOGGED_ONLY = "Logged only"
    SIMULATOR_GROUND_TRUTH = "Simulator ground truth"


class SensorCategory(StrEnum):
    """The complete sensor rail exposed by every telemetry snapshot."""

    STEREO_VISION_AND_DEPTH = "Stereo vision and depth"
    LINKWISE_IMUS = "Linkwise IMUs"
    JOINT_POSITION_AND_EFFORT = "Joint position and effort"
    FOOT_CONTACTS = "Foot contacts"
    WRIST_FORCE_AND_TRAY_BALANCE = "Wrist force and tray balance"
    HAND_PRESSURE_AND_SLIP = "Hand pressure and slip"
    MICROPHONE_ACTIVITY = "Microphone activity"
    BATTERY_AND_ENERGY = "Battery and energy"


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("telemetry payload must be JSON serializable") from exc


def _payload_checksum(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One labeled sensor category, including explicit unavailable states."""

    category: SensorCategory
    signal_use: SignalUseLabel
    available: bool
    values_json: str

    def __post_init__(self) -> None:
        try:
            decoded = json.loads(self.values_json)
        except json.JSONDecodeError as exc:
            raise ValueError("sensor values_json must contain valid JSON") from exc
        if _canonical_json(decoded) != self.values_json:
            raise ValueError("sensor values_json must use canonical JSON encoding")
        if not self.available and decoded is not None:
            raise ValueError("unavailable sensor readings must carry a null value")

    @classmethod
    def available_reading(
        cls,
        category: SensorCategory,
        signal_use: SignalUseLabel,
        values: object,
    ) -> SensorReading:
        return cls(
            category=category,
            signal_use=signal_use,
            available=True,
            values_json=_canonical_json(values),
        )

    @classmethod
    def unavailable(
        cls,
        category: SensorCategory,
        signal_use: SignalUseLabel = SignalUseLabel.LOGGED_ONLY,
    ) -> SensorReading:
        return cls(
            category=category,
            signal_use=signal_use,
            available=False,
            values_json="null",
        )

    @property
    def values(self) -> Any:
        """Return a detached decoded value; the stored record remains immutable."""

        return json.loads(self.values_json)


@dataclass(frozen=True, slots=True)
class SensorSnapshot:
    """A complete sensor rail; no category can be omitted."""

    stereo_vision_and_depth: SensorReading
    linkwise_imus: SensorReading
    joint_position_and_effort: SensorReading
    foot_contacts: SensorReading
    wrist_force_and_tray_balance: SensorReading
    hand_pressure_and_slip: SensorReading
    microphone_activity: SensorReading
    battery_and_energy: SensorReading

    _EXPECTED_CATEGORIES = (
        SensorCategory.STEREO_VISION_AND_DEPTH,
        SensorCategory.LINKWISE_IMUS,
        SensorCategory.JOINT_POSITION_AND_EFFORT,
        SensorCategory.FOOT_CONTACTS,
        SensorCategory.WRIST_FORCE_AND_TRAY_BALANCE,
        SensorCategory.HAND_PRESSURE_AND_SLIP,
        SensorCategory.MICROPHONE_ACTIVITY,
        SensorCategory.BATTERY_AND_ENERGY,
    )

    def __post_init__(self) -> None:
        for sensor_field, expected_category in zip(
            fields(self), self._EXPECTED_CATEGORIES, strict=True
        ):
            reading = getattr(self, sensor_field.name)
            if reading.category is not expected_category:
                raise ValueError(
                    f"{sensor_field.name} must contain {expected_category.value!r}, "
                    f"not {reading.category.value!r}"
                )

    @classmethod
    def all_unavailable(
        cls,
        signal_use: SignalUseLabel = SignalUseLabel.LOGGED_ONLY,
    ) -> SensorSnapshot:
        return cls(
            *(
                SensorReading.unavailable(category, signal_use)
                for category in cls._EXPECTED_CATEGORIES
            )
        )

    @property
    def readings(self) -> tuple[SensorReading, ...]:
        return tuple(getattr(self, sensor_field.name) for sensor_field in fields(self))


@dataclass(frozen=True, slots=True)
class EpisodeTelemetryRecord:
    """A checksummed event in an episode's immutable telemetry stream.

    ``frame_id`` is the only video join key in the record contract.
    """

    episode_id: str
    sequence: int
    sim_time_seconds: float
    robot_checksum: str
    policy_hash: str
    world_hash: str
    signal_use: SignalUseLabel
    sensors: SensorSnapshot
    payload_json: str
    payload_checksum: str
    frame_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "robot_checksum", "policy_hash", "world_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not math.isfinite(self.sim_time_seconds) or self.sim_time_seconds < 0:
            raise ValueError("sim_time_seconds must be finite and non-negative")
        if self.frame_id == "":
            raise ValueError("frame_id must be non-empty when present")
        self.verify_integrity()

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        sequence: int,
        sim_time_seconds: float,
        robot_checksum: str,
        policy_hash: str,
        world_hash: str,
        signal_use: SignalUseLabel,
        sensors: SensorSnapshot,
        payload: object,
        frame_id: str | None = None,
    ) -> EpisodeTelemetryRecord:
        payload_json = _canonical_json(payload)
        return cls(
            episode_id=episode_id,
            sequence=sequence,
            sim_time_seconds=sim_time_seconds,
            robot_checksum=robot_checksum,
            policy_hash=policy_hash,
            world_hash=world_hash,
            signal_use=signal_use,
            sensors=sensors,
            payload_json=payload_json,
            payload_checksum=_payload_checksum(payload_json),
            frame_id=frame_id,
        )

    @property
    def payload(self) -> Any:
        """Return a detached decoded payload; the stored record remains immutable."""

        return json.loads(self.payload_json)

    def verify_integrity(self) -> None:
        try:
            decoded = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("payload_json must contain valid JSON") from exc
        if _canonical_json(decoded) != self.payload_json:
            raise ValueError("payload_json must use canonical JSON encoding")
        expected = _payload_checksum(self.payload_json)
        if self.payload_checksum != expected:
            raise ValueError("payload checksum does not match payload_json")
