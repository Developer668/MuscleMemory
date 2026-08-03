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
        self._last_consumer_error: str | None = None

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
        publisher = self._publisher
        if publisher is not None:
            try:
                await publisher.publish_telemetry(
                    telemetry_view(
                        record,
                        delivery=self._delivery_state(receipt),
                    )
                )
                self._last_consumer_error = None
            except Exception as exc:
                self._last_consumer_error = type(exc).__name__
        return receipt

    async def close_episode(
        self,
        result: PolicyEpisodeResult,
        *,
        closed_at: datetime | None = None,
    ) -> EpisodeClosure:
        closure = await self.service.close_episode(result, closed_at=closed_at)
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
            if self._last_consumer_error is not None
            else (
                ProviderOperationalState.HEALTHY
                if self._publisher is not None
                else ProviderOperationalState.CONFIGURED
            )
        )
        live_detail = (
            "bounded API fanout failed on the latest durable event"
            if self._last_consumer_error is not None
            else (
                "durable 20 Hz events fan out to the dashboard and timeline"
                if self._publisher is not None
                else "API publisher is not bound yet"
            )
        )
        return (
            RuntimeConsumerSnapshot("dashboard-and-timeline", live_state, live_detail),
            RuntimeConsumerSnapshot(
                "replay",
                ProviderOperationalState.HEALTHY,
                "append-only replay reads the durable event stream at 20 Hz; "
                "frame_id is the sole video join",
            ),
            RuntimeConsumerSnapshot(
                "safety-and-failure-summary",
                ProviderOperationalState.HEALTHY,
                "deterministic measured episode results produce safety and failure facts",
            ),
            RuntimeConsumerSnapshot(
                "post-episode-graph-handoff",
                ProviderOperationalState.HEALTHY,
                "closed episode, failure, and correction facts are handed to graph "
                "memory after control ends",
            ),
            RuntimeConsumerSnapshot(
                "training-and-evaluation-evidence",
                ProviderOperationalState.HEALTHY,
                "immutable closures and telemetry digests back training and evaluation evidence",
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
            self._last_consumer_error = None
        except Exception as exc:
            self._last_consumer_error = type(exc).__name__

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
