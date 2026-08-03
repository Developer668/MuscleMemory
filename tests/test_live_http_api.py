from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from muscle_memory.api import (
    HashedBearerCredential,
    Sha256BearerAuthenticator,
    create_app,
)
from muscle_memory.live.controller import (
    LiveEpisodeNotFoundError,
    LiveEpisodeOptions,
    LivePolicyOption,
)
from muscle_memory.live.models import (
    EncodedVideoProduct,
    LiveEpisodeHealth,
    LiveEpisodePhase,
    LiveEpisodeStatus,
    VideoFrameMetadata,
    VideoFrameSet,
    VideoProduct,
)

POLICY_HASH = "c" * 64


class _Backend:
    def __init__(self) -> None:
        self.publisher = None
        self.started = 0
        self.stopped = 0

    def bind_live_publisher(self, publisher: object) -> None:
        self.publisher = publisher

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1


def _status(phase: LiveEpisodePhase = LiveEpisodePhase.QUEUED) -> LiveEpisodeStatus:
    return LiveEpisodeStatus(
        episode_id="live-0000000000000007-abc123def456",
        phase=phase,
        health=(
            LiveEpisodeHealth.STARTING
            if phase is LiveEpisodePhase.QUEUED
            else LiveEpisodeHealth.HEALTHY
        ),
        world_id="train-v1-0000000000000007",
        policy_id="delivery-v1-bc",
        policy_hash=POLICY_HASH,
        policy_promotable=False,
        detail="bounded test worker",
    )


def _frame() -> VideoFrameSet:
    products: list[EncodedVideoProduct] = []
    for index, product in enumerate(VideoProduct):
        data = b"jpeg-frame-" + bytes((index,))
        products.append(
            EncodedVideoProduct(
                product=product,
                mime_type="image/jpeg",
                width=64,
                height=48,
                sha256=hashlib.sha256(data).hexdigest(),
                data=data,
            )
        )
    episode_id = _status().episode_id
    metadata = VideoFrameMetadata(
        frame_id=f"{episode_id}:video:00000000",
        frame_index=0,
        scheduled_time_seconds=0.0,
        captured_time_seconds=0.0,
        telemetry_sequence=0,
        products=tuple(
            product.metadata(episode_id=episode_id, frame_index=0)
            for product in products
        ),
    )
    return VideoFrameSet(metadata=metadata, products=tuple(products))


class _LiveControl:
    def __init__(self) -> None:
        self.started: tuple[int, str] | None = None
        self.cancelled: str | None = None
        self.shutdowns = 0
        self.frame = _frame()

    def options(self) -> LiveEpisodeOptions:
        return LiveEpisodeOptions(
            catalog_id="live-training-v1",
            catalog_sha256="a" * 64,
            seeds=(7, 9),
            policies=(
                LivePolicyOption(
                    policy_id="delivery-v1-bc",
                    policy_hash=POLICY_HASH,
                    evaluated_episode_count=20,
                    promotable=False,
                ),
            ),
            video_products=tuple(product.value for product in VideoProduct),
            maximum_duration_seconds=30.0,
        )

    def start(self, *, seed: int, policy_id: str) -> LiveEpisodeStatus:
        self.started = (seed, policy_id)
        return _status()

    def status(self, episode_id: str) -> LiveEpisodeStatus:
        if episode_id != _status().episode_id:
            raise LiveEpisodeNotFoundError("not found")
        return _status(LiveEpisodePhase.RUNNING)

    def cancel(self, episode_id: str) -> LiveEpisodeStatus:
        self.status(episode_id)
        self.cancelled = episode_id
        return _status(LiveEpisodePhase.CANCELLING)

    def iter_mjpeg(
        self,
        episode_id: str,
        product: VideoProduct,
        *,
        after_frame_index: int = -1,
    ):  # type: ignore[no-untyped-def]
        self.status(episode_id)
        del product, after_frame_index
        return iter((b"--mm01-frame\r\nContent-Type: image/jpeg\r\n\r\njpeg\r\n",))

    def video_frame(self, episode_id: str, frame_index: int) -> VideoFrameSet | None:
        self.status(episode_id)
        return self.frame if frame_index == 0 else None

    def shutdown(self) -> None:
        self.shutdowns += 1


def _client(control: _LiveControl | None) -> tuple[TestClient, _Backend]:
    backend = _Backend()
    credential = HashedBearerCredential.from_plaintext(
        subject="operator",
        token="test-token",
    )
    app = create_app(
        backend=backend,  # type: ignore[arg-type]
        authenticator=Sha256BearerAuthenticator((credential,)),
        live_episodes=control,
    )
    return TestClient(app), backend


def test_live_options_and_mutations_are_admission_bound_and_authenticated() -> None:
    control = _LiveControl()
    client, backend = _client(control)
    with client:
        options = client.get("/api/v1/live/options")
        denied = client.post(
            "/api/v1/live/episodes",
            json={"seed": 7, "policy_id": "delivery-v1-bc"},
        )
        accepted = client.post(
            "/api/v1/live/episodes",
            headers={"Authorization": "Bearer test-token"},
            json={"seed": 7, "policy_id": "delivery-v1-bc"},
        )
        episode_id = accepted.json()["episode_id"]
        running = client.get(f"/api/v1/live/episodes/{episode_id}")
        cancelled = client.post(
            f"/api/v1/live/episodes/{episode_id}/cancel",
            headers={"Authorization": "Bearer test-token"},
        )

    assert options.status_code == 200
    assert options.json()["enabled"] is True
    assert options.json()["seeds"] == [7, 9]
    assert options.json()["policies"][0]["evaluated_episode_count"] == 20
    assert len(options.json()["video_products"]) == 6
    assert denied.status_code == 401
    assert accepted.status_code == 202
    assert accepted.json()["policy_promotable"] is False
    assert set(accepted.json()["video_streams"]) == {
        product.value for product in VideoProduct
    }
    assert control.started == (7, "delivery-v1-bc")
    assert running.json()["phase"] == "running"
    assert cancelled.status_code == 202
    assert cancelled.json()["phase"] == "cancelling"
    assert control.cancelled == episode_id
    assert control.shutdowns == 1
    assert backend.started == backend.stopped == 1


def test_all_six_direct_video_products_and_exact_frame_join_are_served() -> None:
    control = _LiveControl()
    client, _backend = _client(control)
    episode_id = _status().episode_id
    with client:
        for product in VideoProduct:
            stream = client.get(
                f"/api/v1/episodes/{episode_id}/video/{product.value}.mjpeg"
            )
            assert stream.status_code == 200
            assert stream.headers["x-frame-join-key"] == "frame_id"
            assert "multipart/x-mixed-replace" in stream.headers["content-type"]

        frame = client.get(
            f"/api/v1/episodes/{episode_id}/video/left_eye_rgb/frames/0"
        )
        missing = client.get(
            f"/api/v1/episodes/{episode_id}/video/left_eye_rgb/frames/99"
        )

    assert frame.status_code == 200
    assert frame.headers["x-frame-id"] == f"{episode_id}:video:00000000"
    assert frame.headers["x-frame-join-key"] == "frame_id"
    assert frame.content == control.frame.product(VideoProduct.LEFT_EYE_RGB).data
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "video_frame_not_found"


def test_unconfigured_live_runtime_reports_an_honest_empty_capability() -> None:
    client, _backend = _client(None)
    with client:
        options = client.get("/api/v1/live/options")
        start = client.post(
            "/api/v1/live/episodes",
            headers={"Authorization": "Bearer test-token"},
            json={"seed": 7, "policy_id": "delivery-v1-bc"},
        )

    assert options.status_code == 200
    assert options.json()["enabled"] is False
    assert options.json()["seeds"] == []
    assert options.json()["policies"] == []
    assert options.json()["unavailable_reason"]
    assert start.status_code == 503
    assert start.json()["error"]["code"] == "live_runtime_unconfigured"
