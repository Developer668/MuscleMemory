from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from muscle_memory.api import (
    HashedBearerCredential,
    LiveTelemetryHub,
    Sha256BearerAuthenticator,
    create_app,
)
from muscle_memory.api.adapters import redact_provider_detail
from muscle_memory.api.contracts import AuthenticatedPrincipal, LiveEventPublisher
from muscle_memory.api.models import (
    ApprovalDecisionView,
    AssetStatus,
    CorrectionRequest,
    CorrectionState,
    CorrectionView,
    DecisionRequest,
    EpisodeDetail,
    EpisodeKind,
    EpisodeList,
    EpisodeState,
    EpisodeSummary,
    LiveMessageKind,
    LiveStreamMessage,
    PendingApprovalList,
    PolicySummaryList,
    PromotionEligibility,
    ProviderHealth,
    ProviderOperationalState,
    ReplayPage,
    SensorReadingView,
    ServiceHealth,
    TelemetryPage,
    TelemetryRecordView,
    WorkflowReview,
    WorkflowReviewRequest,
    WorkflowRun,
    utc_now,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def _provider(
    state: ProviderOperationalState = ProviderOperationalState.HEALTHY,
) -> ProviderHealth:
    return ProviderHealth(
        provider="LaserData",
        state=state,
        detail="provider state from injected test service",
        checked_at=NOW,
    )


def _episode() -> EpisodeSummary:
    return EpisodeSummary(
        episode_id="episode-1",
        kind=EpisodeKind.TRAINING,
        state=EpisodeState.SUCCEEDED,
        robot_checksum=HASH_A,
        world_id="world-training-1",
        world_hash=HASH_B,
        policy_id="policy-v1",
        policy_hash=HASH_C,
        opened_at=NOW,
        closed_at=NOW,
    )


def _telemetry(sequence: int = 0) -> TelemetryRecordView:
    sensors = tuple(
        SensorReadingView(
            category=f"sensor-{index}",
            signal_use="Logged only",
            available=False,
            values=None,
        )
        for index in range(8)
    )
    return TelemetryRecordView(
        episode_id="episode-1",
        world_id="world-training-1",
        policy_id="policy-v1",
        sequence=sequence,
        sim_time_seconds=sequence / 20,
        event_time=sequence / 20,
        failure_type=None,
        frame_id=f"frame-{sequence}",
        signal_use="Logged only",
        sensors=sensors,
        payload={"action": "stop"},
        payload_checksum=HASH_A,
        delivery=ProviderOperationalState.END_TO_END_VERIFIED,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.last_principal: AuthenticatedPrincipal | None = None
        self.live_publisher: LiveEventPublisher | None = None

    def bind_live_publisher(self, publisher: LiveEventPublisher) -> None:
        self.live_publisher = publisher

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1

    async def health(self) -> ServiceHealth:
        return ServiceHealth(
            state=ProviderOperationalState.HEALTHY,
            providers=(_provider(),),
            checked_at=NOW,
        )

    async def list_episodes(self, *, cursor: str | None, limit: int) -> EpisodeList:
        del cursor, limit
        return EpisodeList(items=(_episode(),))

    async def episode(self, episode_id: str) -> EpisodeDetail | None:
        if episode_id != "episode-1":
            return None
        return EpisodeDetail(
            episode=_episode(),
            telemetry_records=1,
            provider_delivery=ProviderOperationalState.END_TO_END_VERIFIED,
        )

    async def telemetry(
        self,
        episode_id: str,
        *,
        after_sequence: int | None,
        limit: int,
    ) -> TelemetryPage | None:
        del after_sequence, limit
        if episode_id != "episode-1":
            return None
        return TelemetryPage(episode_id=episode_id, records=(_telemetry(),))

    async def replay(
        self,
        episode_id: str,
        *,
        after_sequence: int | None,
        limit: int,
    ) -> ReplayPage | None:
        del after_sequence, limit
        if episode_id != "episode-1":
            return None
        return ReplayPage(episode_id=episode_id, records=(_telemetry(),))

    async def pending_approvals(self) -> PendingApprovalList:
        return PendingApprovalList(items=())

    async def submit_approval_decision(
        self,
        requirement_id: str,
        request: DecisionRequest,
        principal: AuthenticatedPrincipal,
    ) -> ApprovalDecisionView:
        self.last_principal = principal
        return ApprovalDecisionView(
            requirement_id=requirement_id,
            verdict=request.verdict,
            human_subject=principal.subject,
            authentication_method=principal.authentication_method,
            decided_at=NOW,
        )

    async def review_workflow(
        self,
        request: WorkflowReviewRequest,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowReview:
        del request
        self.last_principal = principal
        raise AssertionError("not needed by focused route tests")

    async def execute_workflow(
        self,
        run_id: str,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowRun:
        del run_id
        self.last_principal = principal
        raise AssertionError("not needed by focused route tests")

    async def resume_workflow(
        self,
        run_id: str,
        principal: AuthenticatedPrincipal,
    ) -> WorkflowRun:
        del run_id
        self.last_principal = principal
        raise AssertionError("not needed by focused route tests")

    async def workflow(self, run_id: str) -> WorkflowRun | None:
        del run_id
        return None

    async def submit_correction(
        self,
        episode_id: str,
        request: CorrectionRequest,
        principal: AuthenticatedPrincipal,
    ) -> CorrectionView:
        self.last_principal = principal
        return CorrectionView(
            correction_id="correction-1",
            episode_id=episode_id,
            failure_id=request.failure_id,
            kind=request.kind,
            state=CorrectionState.PENDING,
            submitted_by=principal.subject,
            created_at=NOW,
            graph_delivery=ProviderOperationalState.CONFIGURED,
        )

    async def decide_correction(
        self,
        correction_id: str,
        request: DecisionRequest,
        principal: AuthenticatedPrincipal,
    ) -> CorrectionView:
        del correction_id, request
        self.last_principal = principal
        raise AssertionError("not needed by focused route tests")

    async def policy_summaries(self) -> PolicySummaryList:
        return PolicySummaryList(items=())

    async def promotion_eligibility(
        self,
        *,
        baseline_policy_id: str,
        candidate_policy_id: str,
    ) -> PromotionEligibility:
        return PromotionEligibility(
            baseline_policy_id=baseline_policy_id,
            candidate_policy_id=candidate_policy_id,
            held_out_episode_count=20,
            checks={"zero_falls": True},
            numerically_eligible=True,
            evidence_hash=HASH_A,
        )

    async def asset_statuses(self) -> tuple[AssetStatus, ...]:
        return ()

    async def asset_status(self, asset_id: str) -> AssetStatus | None:
        del asset_id
        return None


def _client(
    backend: FakeBackend,
    *,
    hub: LiveTelemetryHub | None = None,
) -> TestClient:
    credential = HashedBearerCredential.from_plaintext(
        subject="human-operator",
        token="test-token",
    )
    app = create_app(
        backend=backend,
        authenticator=Sha256BearerAuthenticator((credential,)),
        live_hub=hub,
    )
    return TestClient(app)


def test_health_episode_telemetry_and_replay_are_typed_and_operational_only() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        health = client.get("/api/v1/health")
        episodes = client.get("/api/v1/episodes")
        telemetry = client.get("/api/v1/episodes/episode-1/telemetry")
        replay = client.get("/api/v1/episodes/episode-1/replay")

    assert health.status_code == 200
    assert health.json()["providers"][0]["state"] == "healthy"
    assert episodes.json()["exposure"] == "operational_only"
    assert episodes.json()["items"][0]["kind"] == "training"
    assert "held_out" not in json.dumps(episodes.json())
    assert telemetry.json()["cadence_hz"] == 20
    assert telemetry.json()["records"][0]["frame_join_key"] == "frame_id"
    assert replay.json()["frame_join_key"] == "frame_id"
    assert backend.started == 1
    assert backend.stopped == 1
    assert backend.live_publisher is not None


def test_built_operator_console_is_served_on_root_and_about(
    tmp_path,  # type: ignore[no-untyped-def]
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><div id="root">operator-console</div>',
        encoding="utf-8",
    )
    monkeypatch.setenv("MM_FRONTEND_DIST", str(dist))

    with _client(FakeBackend()) as client:
        root = client.get("/")
        about = client.get("/about")

    assert root.status_code == 200
    assert about.status_code == 200
    assert "operator-console" in root.text
    assert root.headers["cache-control"] == "no-cache"


def test_mutations_fail_closed_and_authenticated_identity_is_not_body_controlled() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        denied = client.post(
            "/api/v1/approvals/requirement-1/decision",
            json={"verdict": "approve", "note": "reviewed"},
        )
        accepted = client.post(
            "/api/v1/approvals/requirement-1/decision",
            headers={"Authorization": "Bearer test-token"},
            json={"verdict": "approve", "note": "reviewed"},
        )

    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "authentication_required"
    assert denied.headers["www-authenticate"] == "Bearer"
    assert accepted.status_code == 200
    assert accepted.json()["human_subject"] == "human-operator"
    assert backend.last_principal is not None
    assert backend.last_principal.subject == "human-operator"


def test_unconfigured_authentication_rejects_every_mutation() -> None:
    backend = FakeBackend()
    app = create_app(
        backend=backend,
        authenticator=Sha256BearerAuthenticator(()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/approvals/requirement-1/decision",
            headers={"Authorization": "Bearer any-value"},
            json={"verdict": "approve"},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_unconfigured"


def test_authenticator_rejects_one_token_assigned_to_multiple_subjects() -> None:
    first = HashedBearerCredential.from_plaintext(
        subject="operator-1",
        token="shared-secret",
    )
    second = HashedBearerCredential.from_plaintext(
        subject="operator-2",
        token="shared-secret",
    )

    with pytest.raises(ValueError, match="token digests must be unique"):
        Sha256BearerAuthenticator((first, second))


def test_not_found_and_validation_errors_share_the_error_contract() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        missing = client.get("/api/v1/episodes/missing")
        invalid = client.post(
            "/api/v1/episodes/episode-1/corrections",
            headers={"Authorization": "Bearer test-token"},
            json={
                "failure_id": "failure-1",
                "kind": "keep_out",
                "points": [{"x_m": 0, "y_m": 0}, {"x_m": 1, "y_m": 1}],
                "description": "keep clear",
            },
        )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "episode_not_found"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "request_validation_failed"
    assert set(invalid.json()["error"]) == {"code", "message", "request_id", "details"}


def test_openapi_has_versioned_routes_security_and_no_credential_schema() -> None:
    backend = FakeBackend()
    app = create_app(
        backend=backend,
        authenticator=Sha256BearerAuthenticator(()),
    )
    schema = app.openapi()
    paths = schema["paths"]
    required = {
        "/api/v1/health",
        "/api/v1/episodes",
        "/api/v1/episodes/{episode_id}",
        "/api/v1/episodes/{episode_id}/telemetry",
        "/api/v1/episodes/{episode_id}/replay",
        "/api/v1/approvals/pending",
        "/api/v1/approvals/{requirement_id}/decision",
        "/api/v1/workflows/review",
        "/api/v1/workflows/{run_id}/execute",
        "/api/v1/workflows/{run_id}/resume",
        "/api/v1/episodes/{episode_id}/corrections",
        "/api/v1/policies/promotion-eligibility",
        "/api/v1/assets",
    }
    assert required.issubset(paths)
    assert paths["/api/v1/approvals/{requirement_id}/decision"]["post"]["security"]
    rendered = json.dumps(schema, sort_keys=True)
    assert "token_sha256" not in rendered
    assert "api_key" not in rendered.lower()


def test_live_hub_is_bounded_and_reports_dropped_stale_messages() -> None:
    async def exercise() -> tuple[int, int, LiveMessageKind]:
        hub = LiveTelemetryHub(queue_size=2)
        async with hub.subscribe("episode-1") as subscription:
            await hub.publish_telemetry(_telemetry(0))
            await hub.publish_telemetry(_telemetry(1))
            await hub.publish_telemetry(_telemetry(2))
            first = await subscription.receive()
            second = await subscription.receive()
            assert first is not None
            assert second is not None
            await hub.close()
            assert second.telemetry is not None
            return second.telemetry.sequence, second.dropped_before, second.kind

    sequence, dropped, kind = asyncio.run(exercise())
    assert sequence == 2
    assert dropped == 1
    assert kind is LiveMessageKind.TELEMETRY


def test_live_message_allows_frame_id_as_the_only_video_join_key() -> None:
    record = _telemetry()
    message = LiveStreamMessage(
        kind=LiveMessageKind.TELEMETRY,
        episode_id=record.episode_id,
        frame_id=record.frame_id,
        telemetry=record,
        emitted_at=utc_now(),
    )
    encoded = message.model_dump(mode="json")
    assert encoded["frame_join_key"] == "frame_id"
    assert "timestamp_join_key" not in encoded


def test_websocket_delivers_live_telemetry_with_the_frame_join_contract() -> None:
    backend = FakeBackend()
    hub = LiveTelemetryHub(queue_size=2)
    with (
        _client(backend, hub=hub) as client,
        client.websocket_connect("/api/v1/episodes/episode-1/live") as websocket,
    ):
        assert client.portal is not None
        client.portal.call(hub.publish_telemetry, _telemetry())
        message = websocket.receive_json()

    assert message["kind"] == "telemetry"
    assert message["cadence_hz"] == 20
    assert message["frame_join_key"] == "frame_id"
    assert message["frame_id"] == "frame-0"


def test_provider_detail_projection_redacts_common_secret_shapes() -> None:
    detail = (
        "redis://operator:private-value@provider.example/0 "
        "cloud-user:cloud-private@starter.laserdata.cloud:8090 "
        "Authorization=private-header Bearer private-token"
    )
    projected = redact_provider_detail(detail)
    assert "private-value" not in projected
    assert "private-header" not in projected
    assert "private-token" not in projected
    assert "cloud-private" not in projected
    assert projected.count("[redacted]") == 4
