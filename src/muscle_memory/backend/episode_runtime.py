"""Operational LaserData fanout around the controller-independent episode service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from muscle_memory.api.adapters import telemetry_view
from muscle_memory.api.contracts import LiveEventPublisher
from muscle_memory.api.models import ProviderHealth, ProviderOperationalState
from muscle_memory.episodes import (
    EpisodeAppendReceipt,
    EpisodeClosure,
    EpisodeIdentity,
    EpisodeService,
)
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.telemetry import (
    EpisodeTelemetryRecord,
    LaserDataProviderState,
    TelemetryDelivery,
)

NUMERIC_TELEMETRY_HZ = 20
VIDEO_METADATA_JOIN_KEY = "frame_id"


@dataclass(frozen=True, slots=True)
class RuntimeConsumerSnapshot:
    consumer: str
    state: ProviderOperationalState
    detail: str

    def public(self) -> ProviderHealth:
        return ProviderHealth(
            provider=f"LaserData consumer: {self.consumer}",
            state=self.state,
            detail=self.detail,
            checked_at=datetime.now(UTC),
        )


class OperationalEpisodeRuntime:
    """Fan durable telemetry to operational consumers, never to robot control."""

    def __init__(
        self,
        service: EpisodeService,
        *,
        expected_robot_checksum: str,
    ) -> None:
        self.service = service
        self._expected_robot_checksum = expected_robot_checksum
        self._publisher: LiveEventPublisher | None = None
        self._last_telemetry_consumer_error: str | None = None
        self._last_status_consumer_error: str | None = None
        self._telemetry_dashboard_deliveries = 0
        self._status_dashboard_deliveries = 0
        self._durable_event_count = 0
        self._last_closure: EpisodeClosure | None = None

    def bind_live_publisher(self, publisher: LiveEventPublisher) -> None:
        self._publisher = publisher

    async def open_episode(self, identity: EpisodeIdentity) -> EpisodeIdentity:
        if identity.robot_checksum != self._expected_robot_checksum:
            raise ValueError("episode robot checksum does not match the qualified MM-01 bundle")
        opened = await self.service.open_episode(identity)
        await self._publish_status(
            identity.episode_id,
            {
                "state": "running",
                "numeric_telemetry_hz": NUMERIC_TELEMETRY_HZ,
                "video_metadata_join_key": VIDEO_METADATA_JOIN_KEY,
            },
        )
        return opened

    async def append_telemetry(
        self,
        record: EpisodeTelemetryRecord,
    ) -> EpisodeAppendReceipt:
        """Publish only after the exact event is durably accepted by the backend."""

        receipt = await self.service.append_telemetry(record)
        self._durable_event_count += 1
        publisher = self._publisher
        if publisher is not None:
            try:
                await publisher.publish_telemetry(
                    telemetry_view(
                        record,
                        delivery=self._delivery_state(receipt),
                    )
                )
                self._last_telemetry_consumer_error = None
                self._telemetry_dashboard_deliveries += 1
            except Exception as exc:
                self._last_telemetry_consumer_error = type(exc).__name__
        return receipt

    async def close_episode(
        self,
        result: PolicyEpisodeResult,
        *,
        closed_at: datetime | None = None,
    ) -> EpisodeClosure:
        closure = await self.service.close_episode(result, closed_at=closed_at)
        self._last_closure = closure
        await self._publish_status(
            result.episode_id,
            {
                "state": "succeeded" if result.success else "failed",
                "failure_ids": [failure.failure_id for failure in closure.failures],
                "graph_provider_complete": closure.graph.provider_complete,
                "telemetry_provider_complete": closure.telemetry.provider_complete,
                "telemetry_digest": closure.telemetry_digest,
            },
        )
        return closure

    def consumer_snapshots(self) -> tuple[RuntimeConsumerSnapshot, ...]:
        live_state = (
            ProviderOperationalState.DEGRADED
            if (
                self._last_telemetry_consumer_error is not None
                or self._last_status_consumer_error is not None
            )
            else (
                ProviderOperationalState.HEALTHY
                if self._telemetry_dashboard_deliveries > 0
                else ProviderOperationalState.CONFIGURED
            )
        )
        live_detail = (
            "bounded telemetry fanout failed on the latest durable event"
            if self._last_telemetry_consumer_error is not None
            else (
                "bounded status fanout failed; telemetry evidence remains independent"
                if self._last_status_consumer_error is not None
                else (
                    "live telemetry fanout has completed successfully; "
                    f"status deliveries={self._status_dashboard_deliveries}"
                    if self._telemetry_dashboard_deliveries > 0
                    else (
                        "API publisher is bound; no telemetry fanout evidence exists yet"
                        if self._publisher is not None
                        else "API publisher is not bound yet; no fanout evidence exists"
                    )
                )
            )
        )
        closure = self._last_closure
        replay_ready = closure is not None and self._durable_event_count > 0
        replay_state = (
            ProviderOperationalState.HEALTHY
            if replay_ready
            else ProviderOperationalState.CONFIGURED
        )
        closed_state = (
            ProviderOperationalState.HEALTHY
            if closure is not None
            else ProviderOperationalState.CONFIGURED
        )
        graph_state = ProviderOperationalState.CONFIGURED
        graph_detail = "no closed episode graph-handoff evidence exists yet"
        if closure is not None:
            if closure.graph.provider_complete:
                graph_state = ProviderOperationalState.END_TO_END_VERIFIED
                graph_detail = "the latest closed episode was stored in configured graph memory"
            elif closure.graph.complete:
                graph_state = ProviderOperationalState.DEGRADED
                graph_detail = (
                    "the latest graph handoff completed only through non-provider storage"
                )
            else:
                graph_state = ProviderOperationalState.DEGRADED
                graph_detail = "the latest closed episode graph handoff is incomplete"
        return (
            RuntimeConsumerSnapshot("dashboard-and-timeline", live_state, live_detail),
            RuntimeConsumerSnapshot(
                "replay",
                replay_state,
                (
                    "closed durable telemetry exists for append-only replay; "
                    "frame_id is the sole video join"
                    if replay_ready
                    else "replay is configured but no closed durable episode exists yet"
                ),
            ),
            RuntimeConsumerSnapshot(
                "safety-and-failure-summary",
                closed_state,
                (
                    "a deterministic measured closure produced safety and failure facts"
                    if closure is not None
                    else "deterministic failure detection is configured; no closure exists yet"
                ),
            ),
            RuntimeConsumerSnapshot(
                "post-episode-graph-handoff",
                graph_state,
                graph_detail,
            ),
            RuntimeConsumerSnapshot(
                "training-and-evaluation-evidence",
                closed_state,
                (
                    "an immutable closure and telemetry digest provide audit evidence"
                    if closure is not None
                    else "evidence consumption is configured; no immutable closure exists yet"
                ),
            ),
        )

    async def _publish_status(
        self,
        episode_id: str,
        status: dict[str, object],
    ) -> None:
        publisher = self._publisher
        if publisher is None:
            return
        try:
            await publisher.publish_status(episode_id, status)
            self._last_status_consumer_error = None
            self._status_dashboard_deliveries += 1
        except Exception as exc:
            self._last_status_consumer_error = type(exc).__name__

    @staticmethod
    def _delivery_state(receipt: EpisodeAppendReceipt) -> ProviderOperationalState:
        if receipt.delivery is TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE:
            return (
                ProviderOperationalState.END_TO_END_VERIFIED
                if receipt.provider_state is LaserDataProviderState.END_TO_END_VERIFIED
                else ProviderOperationalState.HEALTHY
            )
        if receipt.provider_state is LaserDataProviderState.UNCONFIGURED:
            return ProviderOperationalState.UNCONFIGURED
        return ProviderOperationalState.DEGRADED


__all__ = [
    "NUMERIC_TELEMETRY_HZ",
    "VIDEO_METADATA_JOIN_KEY",
    "OperationalEpisodeRuntime",
    "RuntimeConsumerSnapshot",
]
