from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from muscle_memory.telemetry import (
    DuplicateTelemetryRecordError,
    EpisodeTelemetryRecord,
    InMemoryAppendOnlyTelemetrySink,
    OutOfOrderTelemetryError,
    SensorCategory,
    SensorReading,
    SensorSnapshot,
    SignalUseLabel,
    TelemetryMutationError,
)


def make_record(
    sequence: int,
    *,
    sim_time_seconds: float | None = None,
    payload: object | None = None,
) -> EpisodeTelemetryRecord:
    return EpisodeTelemetryRecord.create(
        episode_id="episode-001",
        world_id="world-seed-17",
        policy_id="policy-v1",
        sequence=sequence,
        sim_time_seconds=float(sequence) if sim_time_seconds is None else sim_time_seconds,
        robot_checksum="a" * 64,
        policy_hash="b" * 64,
        world_hash="c" * 64,
        signal_use=SignalUseLabel.LOGGED_ONLY,
        sensors=SensorSnapshot.all_unavailable(),
        payload={"event": "tick"} if payload is None else payload,
        frame_id=f"frame-{sequence}",
    )


def test_signal_use_labels_are_exact() -> None:
    assert {label.value for label in SignalUseLabel} == {
        "Used by policy",
        "Logged only",
        "Simulator ground truth",
    }


def test_snapshot_always_represents_all_eight_sensor_categories() -> None:
    snapshot = SensorSnapshot.all_unavailable()

    assert tuple(reading.category for reading in snapshot.readings) == tuple(SensorCategory)
    assert len(snapshot.readings) == 8
    assert all(not reading.available for reading in snapshot.readings)


def test_snapshot_rejects_a_category_in_the_wrong_slot() -> None:
    snapshot = SensorSnapshot.all_unavailable()
    wrong_reading = SensorReading.unavailable(SensorCategory.FOOT_CONTACTS)

    with pytest.raises(ValueError, match="stereo_vision_and_depth"):
        replace(snapshot, stereo_vision_and_depth=wrong_reading)


def test_record_is_immutable_and_payload_is_detached() -> None:
    record = make_record(0, payload={"nested": {"value": 1}})

    with pytest.raises(FrozenInstanceError):
        record.sequence = 2  # type: ignore[misc]

    decoded = record.payload
    decoded["nested"]["value"] = 99
    assert record.payload == {"nested": {"value": 1}}


def test_record_rejects_a_forged_payload_checksum() -> None:
    record = make_record(0)

    with pytest.raises(ValueError, match="checksum"):
        replace(record, payload_checksum="0" * 64)


def test_sink_accepts_only_contiguous_append_order() -> None:
    sink = InMemoryAppendOnlyTelemetrySink()
    first = make_record(0)
    second = make_record(1)

    sink.append(first)
    sink.append(second)

    assert sink.records_for("episode-001") == (first, second)

    with pytest.raises(OutOfOrderTelemetryError, match="expected sequence 2"):
        sink.append(make_record(3))


def test_sink_distinguishes_duplicate_from_mutation() -> None:
    sink = InMemoryAppendOnlyTelemetrySink()
    first = make_record(0)
    sink.append(first)

    with pytest.raises(DuplicateTelemetryRecordError):
        sink.append(first)
    with pytest.raises(TelemetryMutationError):
        sink.append(make_record(0, payload={"event": "changed"}))


def test_sink_rejects_simulation_time_regression() -> None:
    sink = InMemoryAppendOnlyTelemetrySink()
    sink.append(make_record(0, sim_time_seconds=2.0))

    with pytest.raises(OutOfOrderTelemetryError, match="cannot move backwards"):
        sink.append(make_record(1, sim_time_seconds=1.0))
