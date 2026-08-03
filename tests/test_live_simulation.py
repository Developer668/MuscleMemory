"""Rate, isolation, and real-MuJoCo checks for the live operator path."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from muscle_memory.episodes.service import EpisodeService
from muscle_memory.graph_memory import (
    GraphStorage,
    GraphWriteReceipt,
    ProviderState,
)
from muscle_memory.live import (
    BoundedVideoService,
    EncodedVideoProduct,
    EvaluatedPolicySelection,
    LiveEpisodeConfig,
    LiveEpisodeManager,
    LiveEpisodePhase,
    VideoFrameMetadata,
    VideoFrameSet,
    VideoProduct,
)
from muscle_memory.paths import POLICY_V1_CHECKPOINT, REPOSITORY_ROOT
from muscle_memory.telemetry import (
    InMemoryAppendOnlyTelemetrySink,
    LaserDataProviderState,
    LaserDataTelemetryEnvelope,
    TelemetryAppendResult,
    TelemetryDelivery,
)
from muscle_memory.worlds.generation import generate_training_world

POLICY_V1_HELDOUT_EVIDENCE = (
    REPOSITORY_ROOT / "evidence" / "policy" / "delivery-v1" / "heldout-evaluation.json"
)


class _TelemetryBackend:
    def __init__(self, store: InMemoryAppendOnlyTelemetrySink) -> None:
        self._store = store

    async def append(self, record: Any) -> TelemetryAppendResult:
        self._store.append(record)
        return TelemetryAppendResult(
            event_id=LaserDataTelemetryEnvelope.from_domain(record).event_id,
            delivery=TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE,
            provider_state=LaserDataProviderState.END_TO_END_VERIFIED,
            pending_local_records=0,
        )


class _GraphMemory:
    def _receipt(self, kind: str, record: Any, record_id: str) -> GraphWriteReceipt:
        return GraphWriteReceipt(
            record_kind=kind,
            record_id=record_id,
            content_hash=record.content_hash,
            storage=GraphStorage.FALKORDB,
            provider_state=ProviderState.END_TO_END_VERIFIED,
            mirrored_to_local_cache=True,
            detail="test provider receipt",
        )

    def record_episode(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("episode", record, record.episode_id)

    def record_failure(self, record: Any) -> GraphWriteReceipt:
        return self._receipt("failure", record, record.failure_id)


def _lifecycle() -> tuple[EpisodeService, InMemoryAppendOnlyTelemetrySink]:
    store = InMemoryAppendOnlyTelemetrySink()
    return (
        EpisodeService(
            telemetry_backend=_TelemetryBackend(store),
            telemetry_store=store,
            graph_memory=_GraphMemory(),  # type: ignore[arg-type]
        ),
        store,
    )


def _video_frame(index: int) -> VideoFrameSet:
    products: list[EncodedVideoProduct] = []
    for product_index, product in enumerate(VideoProduct):
        data = b"jpeg" + bytes((index, product_index))
        products.append(
            EncodedVideoProduct(
                product=product,
                mime_type="image/jpeg",
                width=8,
                height=6,
                sha256=hashlib.sha256(data).hexdigest(),
                data=data,
            )
        )
    metadata = VideoFrameMetadata(
        frame_id=f"episode-video:video:{index:08d}",
        frame_index=index,
        scheduled_time_seconds=index / 30,
        captured_time_seconds=index / 30,
        telemetry_sequence=(index + 1) // 2,
        products=tuple(item.metadata() for item in products),
    )
    return VideoFrameSet(metadata=metadata, products=tuple(products))


def test_video_service_is_bounded_and_serves_named_mjpeg_products() -> None:
    video = BoundedVideoService(maximum_frame_sets=2, maximum_bytes=1_000)
    video.start_episode("episode-video")
    for index in range(3):
        video.append("episode-video", _video_frame(index))
    video.finish_episode("episode-video")

    stats = video.stats("episode-video")
    assert stats.buffered_frames == 2
    assert stats.appended_frames == 3
    assert stats.dropped_frames == 1
    assert video.frame("episode-video", 0) is None
    assert video.latest("episode-video") is not None
    chunks = list(video.iter_mjpeg("episode-video", VideoProduct.LEFT_EYE_RGB))
    assert len(chunks) == 2
    assert all(b"Content-Type: image/jpeg" in chunk for chunk in chunks)
    assert all(b"X-Frame-Id: episode-video:video:" in chunk for chunk in chunks)


def test_live_import_cannot_reach_path_teacher_or_training_expert() -> None:
    audit = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import muscle_memory.live.runner",
                    "assert 'muscle_memory.worlds.generation.pathfinding' not in sys.modules",
                    "assert 'muscle_memory.training.expert' not in sys.modules",
                )
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert audit.returncode == 0, audit.stderr


def test_policy_selection_is_bound_to_real_heldout_evidence(tmp_path: Path) -> None:
    selection = EvaluatedPolicySelection.load(
        checkpoint_path=POLICY_V1_CHECKPOINT,
        evaluation_path=POLICY_V1_HELDOUT_EVIDENCE,
    )
    assert selection.evaluated_episode_count == 20
    assert not selection.promotable

    evidence = json.loads(POLICY_V1_HELDOUT_EVIDENCE.read_text(encoding="utf-8"))
    evidence["candidate_results"][0]["policy_hash"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match its held-out evidence"):
        EvaluatedPolicySelection.load(
            checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluation_path=changed,
        )


def test_live_manager_rejects_raw_training_world() -> None:
    lifecycle, _ = _lifecycle()
    manager = LiveEpisodeManager(lifecycle=lifecycle)
    selection = EvaluatedPolicySelection.load(
        checkpoint_path=POLICY_V1_CHECKPOINT,
        evaluation_path=POLICY_V1_HELDOUT_EVIDENCE,
    )
    validated = generate_training_world(7)
    try:
        with pytest.raises(TypeError, match="validation-gated training world"):
            manager.start_episode(
                episode_id="raw-world",
                world=validated.world,  # type: ignore[arg-type]
                selection=selection,
            )
    finally:
        manager.shutdown()


def test_real_mujoco_live_episode_has_exact_rates_and_frame_joins() -> None:
    lifecycle, store = _lifecycle()
    video = BoundedVideoService(maximum_frame_sets=8, maximum_bytes=8 << 20)
    manager = LiveEpisodeManager(lifecycle=lifecycle, video=video)
    selection = EvaluatedPolicySelection.load(
        checkpoint_path=POLICY_V1_CHECKPOINT,
        evaluation_path=POLICY_V1_HELDOUT_EVIDENCE,
    )
    validated = generate_training_world(7)
    try:
        manager.start_episode(
            episode_id="live-real-01",
            world=validated,
            selection=selection,
            config=LiveEpisodeConfig(
                maximum_duration_seconds=0.1,
                render_width=64,
                render_height=48,
                jpeg_quality=75,
                realtime=False,
            ),
        )
        status = manager.wait("live-real-01", timeout=90)

        assert status.phase is LiveEpisodePhase.CLOSED, status.detail
        assert status.telemetry_records == 3
        assert status.video_frames == 4
        assert status.completion_reason == "time_limit"
        assert status.graph_provider_complete is True
        assert status.telemetry_provider_complete is True
        records = store.records_for("live-real-01")
        assert [record.sequence for record in records] == [0, 1, 2]
        assert [record.sim_time_seconds for record in records] == [0.0, 0.05, 0.1]
        assert all(record.event_time == record.sim_time_seconds for record in records)
        envelopes = tuple(LaserDataTelemetryEnvelope.from_domain(record) for record in records)
        assert all(envelope.record.episode_id == "live-real-01" for envelope in envelopes)
        assert all(envelope.record.world_id == validated.world.world_id for envelope in envelopes)
        assert all(
            envelope.record.policy_id == selection.policy.policy_id for envelope in envelopes
        )
        assert all(envelope.record.event_time in {0.0, 0.05, 0.1} for envelope in envelopes)

        joined_metadata = [
            frame
            for record in records
            for frame in record.payload["video_frames"]
        ]
        assert [frame["frame_index"] for frame in joined_metadata] == [0, 1, 2, 3]
        assert [frame["telemetry_sequence"] for frame in joined_metadata] == [0, 1, 2, 2]
        assert all(frame["transport"] == "direct_video_service" for frame in joined_metadata)
        assert all(len(frame["products"]) == 6 for frame in joined_metadata)
        assert all(
            "data" not in product
            for frame in joined_metadata
            for product in frame["products"]
        )
        assert all(record.frame_id for record in records)
        assert all(
            record.sensors.stereo_vision_and_depth.values["frame_id"] == record.frame_id
            for record in records
        )

        required_payload_fields = {
            "imu",
            "joint_effort",
            "contacts",
            "tray_state",
            "tactile_slip",
            "battery",
            "policy_action",
            "collisions",
            "reward_progress",
            "completion",
            "video_frames",
        }
        assert required_payload_fields <= records[0].payload.keys()
        assert records[-1].payload["completion"] == {
            "completed": True,
            "reason": "time_limit",
        }
        assert records[0].payload["completion"]["completed"] is False

        latest = manager.latest_video_frame("live-real-01")
        assert latest is not None
        assert latest.metadata.frame_index == 3
        assert {item.product for item in latest.products} == set(VideoProduct)
        for item in latest.products:
            decoded = cv2.imdecode(np.frombuffer(item.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None
            assert decoded.size > 0
        assert len(list(manager.iter_mjpeg("live-real-01", VideoProduct.THIRD_PERSON))) == 4
    finally:
        manager.shutdown()


def test_cancellation_closes_a_measured_episode_on_a_20hz_tick() -> None:
    lifecycle, store = _lifecycle()
    manager = LiveEpisodeManager(lifecycle=lifecycle)
    selection = EvaluatedPolicySelection.load(
        checkpoint_path=POLICY_V1_CHECKPOINT,
        evaluation_path=POLICY_V1_HELDOUT_EVIDENCE,
    )
    try:
        manager.start_episode(
            episode_id="live-cancel-01",
            world=generate_training_world(9),
            selection=selection,
            config=LiveEpisodeConfig(
                maximum_duration_seconds=1.0,
                render_width=64,
                render_height=48,
                realtime=True,
            ),
        )
        manager.cancel("live-cancel-01")
        status = manager.wait("live-cancel-01", timeout=90)
        assert status.phase is LiveEpisodePhase.CLOSED, status.detail
        assert status.completion_reason == "cancelled"
        records = store.records_for("live-cancel-01")
        assert records
        assert records[-1].payload["completion"] == {
            "completed": True,
            "reason": "cancelled",
        }
        assert all(
            record.sim_time_seconds == pytest.approx(record.sequence / 20)
            for record in records
        )
    finally:
        manager.shutdown()
