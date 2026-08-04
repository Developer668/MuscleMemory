from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from muscle_memory.api import (
    EPISODE_WRITE_SCOPE,
    HashedBearerCredential,
    LiveTelemetryHub,
    Sha256BearerAuthenticator,
    create_app,
)
from muscle_memory.api.adapters import redact_provider_detail
from muscle_memory.api.app import MAX_REQUEST_BODY_BYTES
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
    EpisodeReviewNote,
    EpisodeReviewNoteCreateRequest,
    EpisodeReviewNoteList,
    EpisodeReviewNoteUpdateRequest,
    EpisodeState,
    EpisodeSummary,
    LiveMessageKind,
    LiveStreamMessage,
    MemoryGraphEdge,
    MemoryGraphNode,
    MemoryGraphSnapshot,
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
from muscle_memory.api.streaming import LiveSubscriberLimitError
from muscle_memory.training.jobs import TaskPolicyTrainingManager

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
        self.notes: list[EpisodeReviewNote] = []

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

    async def memory_graph(self) -> MemoryGraphSnapshot:
        return MemoryGraphSnapshot(
            provider_state=ProviderOperationalState.CACHED,
            graph_name="local-cache",
            source="local_cache",
            provider_checked_at=NOW,
            refreshed_at=NOW,
            fact_count=1,
            nodes=(
                MemoryGraphNode(
                    id="agent:world-physics",
                    label="World & Physics Agent",
                    record_kind="runtime_agent",
                    owner="World & Physics Agent",
                    properties={"fact_count": 1},
                ),
                MemoryGraphNode(
                    id="fact:world:world-training-1",
                    label="world-training-1",
                    record_kind="world",
                    owner="World & Physics Agent",
                    properties={"validated": True},
                ),
            ),
            edges=(
                MemoryGraphEdge(
                    id="agent:world-physics:OWNS:fact:world:world-training-1",
                    source="agent:world-physics",
                    target="fact:world:world-training-1",
                    relationship="OWNS",
                ),
            ),
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

    async def list_episode_notes(
        self,
        episode_id: str,
        *,
        include_archived: bool,
    ) -> EpisodeReviewNoteList | None:
        if episode_id != "episode-1":
            return None
        return EpisodeReviewNoteList(
            episode_id=episode_id,
            items=tuple(note for note in self.notes if include_archived or not note.archived),
        )

    async def create_episode_note(
        self,
        episode_id: str,
        request: EpisodeReviewNoteCreateRequest,
        principal: AuthenticatedPrincipal,
    ) -> EpisodeReviewNote:
        self.last_principal = principal
        note = EpisodeReviewNote(
            note_id="note-" + "1" * 32,
            episode_id=episode_id,
            author_subject=principal.subject,
            body=request.body,
            tags=request.tags,
            created_at=NOW,
            updated_at=NOW,
        )
        self.notes.append(note)
        return note

    async def update_episode_note(
        self,
        episode_id: str,
        note_id: str,
        request: EpisodeReviewNoteUpdateRequest,
        principal: AuthenticatedPrincipal,
    ) -> EpisodeReviewNote | None:
        self.last_principal = principal
        for index, note in enumerate(self.notes):
            if note.episode_id != episode_id or note.note_id != note_id:
                continue
            updated = EpisodeReviewNote(
                note_id=note.note_id,
                episode_id=note.episode_id,
                author_subject=note.author_subject,
                body=note.body if request.body is None else request.body,
                tags=note.tags if request.tags is None else request.tags,
                created_at=note.created_at,
                updated_at=NOW,
                archived=note.archived if request.archived is None else request.archived,
            )
            self.notes[index] = updated
            return updated
        return None

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
    training_jobs: TaskPolicyTrainingManager | None = None,
) -> TestClient:
    credential = HashedBearerCredential.from_plaintext(
        subject="human-operator",
        token="test-token",
    )
    app = create_app(
        backend=backend,
        authenticator=Sha256BearerAuthenticator((credential,)),
        live_hub=hub,
        training_jobs=training_jobs,
    )
    return TestClient(app)


def test_health_episode_telemetry_and_replay_are_typed_and_operational_only() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        health = client.get("/api/v1/health")
        episodes = client.get("/api/v1/episodes")
        telemetry = client.get("/api/v1/episodes/episode-1/telemetry")
        replay = client.get("/api/v1/episodes/episode-1/replay")
        memory_graph = client.get("/api/v1/memory/graph")

    assert health.status_code == 200
    assert health.json()["providers"][0]["state"] == "healthy"
    assert episodes.json()["exposure"] == "operational_only"
    assert episodes.json()["items"][0]["kind"] == "training"
    assert "held_out" not in json.dumps(episodes.json())
    assert telemetry.json()["cadence_hz"] == 20
    assert telemetry.json()["records"][0]["frame_join_key"] == "frame_id"
    assert replay.json()["frame_join_key"] == "frame_id"
    assert memory_graph.status_code == 200
    assert memory_graph.json()["source"] == "local_cache"
    assert memory_graph.json()["fact_count"] == 1
    assert "held_out" not in json.dumps(memory_graph.json())
    assert backend.started == 1
    assert backend.stopped == 1
    assert backend.live_publisher is not None


def test_episode_review_notes_are_scoped_authenticated_and_reversible() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        initial = client.get("/api/v1/episodes/episode-1/notes")
        unauthorized = client.post(
            "/api/v1/episodes/episode-1/notes",
            json={"body": "Should not be accepted"},
        )
        created = client.post(
            "/api/v1/episodes/episode-1/notes",
            json={"body": "Check the clearance trace", "tags": ["clearance", "review"]},
            headers={"Authorization": "Bearer test-token"},
        )
        visible = client.get("/api/v1/episodes/episode-1/notes")
        archived = client.patch(
            f"/api/v1/episodes/episode-1/notes/{created.json()['note_id']}",
            json={"archived": True},
            headers={"Authorization": "Bearer test-token"},
        )
        after_archive = client.get("/api/v1/episodes/episode-1/notes")
        including_archived = client.get(
            "/api/v1/episodes/episode-1/notes?include_archived=true"
        )
        wrong_episode = client.patch(
            "/api/v1/episodes/other/notes/" + created.json()["note_id"],
            json={"archived": True},
            headers={"Authorization": "Bearer test-token"},
        )

    assert initial.status_code == 200
    assert initial.json()["items"] == []
    assert unauthorized.status_code == 401
    assert created.status_code == 201
    assert created.json()["author_subject"] == "human-operator"
    assert visible.json()["items"][0]["tags"] == ["clearance", "review"]
    assert archived.status_code == 200
    assert after_archive.json()["items"] == []
    assert including_archived.json()["items"][0]["archived"] is True
    assert wrong_episode.status_code == 404


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


def test_request_body_limit_runs_before_model_parsing_and_authentication() -> None:
    backend = FakeBackend()

    def oversized_chunks():
        chunk = b"x" * 8192
        for _ in range(MAX_REQUEST_BODY_BYTES // len(chunk) + 2):
            yield chunk

    with _client(backend) as client:
        response = client.post(
            "/api/v1/workflows/review",
            headers={"Content-Type": "application/json"},
            content=oversized_chunks(),
        )

    assert response.status_code == 413
    assert response.json() == {"error": "body_too_large"}


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
        "/api/v1/memory/graph",
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
        "/api/v1/training/jobs",
        "/api/v1/training/jobs/{job_id}",
    }
    assert required.issubset(paths)
    assert paths["/api/v1/approvals/{requirement_id}/decision"]["post"]["security"]
    rendered = json.dumps(schema, sort_keys=True)
    assert "token_sha256" not in rendered
    assert "api_key" not in rendered.lower()


def test_task_policy_training_requires_scope_and_reports_unpromoted_state(
    tmp_path: Path,
) -> None:
    def fail_training(**_kwargs: object) -> None:
        raise RuntimeError("private failure detail")

    manager = TaskPolicyTrainingManager(
        output_root=tmp_path,
        training_function=fail_training,
    )
    backend = FakeBackend()
    with _client(backend, training_jobs=manager) as client:
        denied = client.post("/api/v1/training/jobs", json={"epochs": 1, "seed": 9})
        accepted = client.post(
            "/api/v1/training/jobs",
            headers={"Authorization": "Bearer test-token"},
            json={"epochs": 1, "seed": 9},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        for _ in range(100):
            status_response = client.get(f"/api/v1/training/jobs/{job_id}")
            if status_response.json()["state"] == "failed":
                break
            time.sleep(0.01)
        listed = client.get("/api/v1/training/jobs")

    assert denied.status_code == 401
    assert status_response.status_code == 200
    assert status_response.json()["state"] == "failed"
    assert status_response.json()["error_type"] == "RuntimeError"
    assert "private failure detail" not in json.dumps(status_response.json())
    assert status_response.json()["promotion_status"] == "not_evaluated"
    assert status_response.json()["training_data_split"] == "training"
    assert listed.json()["items"][0]["job_id"] == job_id


def test_task_policy_training_rejects_credentials_without_training_scope(
    tmp_path: Path,
) -> None:
    credential = HashedBearerCredential.from_plaintext(
        subject="episode-only",
        token="episode-token",
        scopes=frozenset({EPISODE_WRITE_SCOPE}),
    )
    manager = TaskPolicyTrainingManager(output_root=tmp_path)
    app = create_app(
        backend=FakeBackend(),
        authenticator=Sha256BearerAuthenticator((credential,)),
        training_jobs=manager,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers={"Authorization": "Bearer episode-token"},
            json={"epochs": 1, "seed": 9},
        )

    assert response.status_code == 403
    assert response.json()["error"]["details"]["required_scope"] == "training:write"


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


def test_live_hub_enforces_global_and_per_episode_subscriber_limits() -> None:
    async def exercise() -> None:
        hub = LiveTelemetryHub(
            maximum_subscribers=2,
            maximum_subscribers_per_episode=1,
        )
        async with hub.subscribe("episode-1"):
            with pytest.raises(LiveSubscriberLimitError, match="capacity"):
                async with hub.subscribe("episode-1"):
                    pass
            async with hub.subscribe("episode-2"):
                with pytest.raises(LiveSubscriberLimitError, match="capacity"):
                    async with hub.subscribe("episode-3"):
                        pass
        await hub.close()

    asyncio.run(exercise())


def test_live_hub_bridges_worker_thread_events_to_the_api_loop() -> None:
    async def exercise() -> tuple[int, int]:
        hub = LiveTelemetryHub(queue_size=2)
        hub.bind_running_loop()
        publisher_thread = 0
        async with hub.subscribe("episode-1") as subscription:
            def publish_from_worker() -> None:
                nonlocal publisher_thread
                publisher_thread = threading.get_ident()
                asyncio.run(hub.publish_telemetry(_telemetry(7)))

            await asyncio.to_thread(publish_from_worker)
            message = await asyncio.wait_for(subscription.receive(), timeout=1.0)
            assert message is not None and message.telemetry is not None
            api_thread = threading.get_ident()
            await hub.close()
            return message.telemetry.sequence, int(publisher_thread != api_thread)

    sequence, crossed_threads = asyncio.run(exercise())
    assert sequence == 7
    assert crossed_threads == 1


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
