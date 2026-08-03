"""Bounded fan-out for 20 Hz telemetry and episode status updates."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from muscle_memory.api.models import (
    LiveMessageKind,
    LiveStreamMessage,
    TelemetryRecordView,
    utc_now,
)

DEFAULT_SUBSCRIBER_QUEUE_SIZE = 40
_CLOSED = object()


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[LiveStreamMessage | object]
    dropped: int = 0


class LiveSubscription:
    """One bounded consumer owned by a WebSocket connection."""

    def __init__(self, subscriber: _Subscriber) -> None:
        self._subscriber = subscriber

    async def receive(self) -> LiveStreamMessage | None:
        item = await self._subscriber.queue.get()
        if item is _CLOSED:
            return None
        if not isinstance(item, LiveStreamMessage):
            raise RuntimeError("live stream queue contained an invalid item")
        return item


class LiveTelemetryHub:
    """Drop oldest stale data per slow consumer instead of blocking ingestion."""

    def __init__(self, *, queue_size: int = DEFAULT_SUBSCRIBER_QUEUE_SIZE) -> None:
        if queue_size < 1:
            raise ValueError("live subscriber queue_size must be positive")
        self._queue_size = queue_size
        self._subscribers: dict[str, dict[int, _Subscriber]] = {}
        self._ids = itertools.count()
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def queue_size(self) -> int:
        return self._queue_size

    @asynccontextmanager
    async def subscribe(self, episode_id: str) -> AsyncIterator[LiveSubscription]:
        if not episode_id.strip():
            raise ValueError("episode_id must not be blank")
        subscriber = _Subscriber(queue=asyncio.Queue(maxsize=self._queue_size))
        subscription_id = next(self._ids)
        async with self._lock:
            if self._closed:
                raise RuntimeError("live telemetry hub is closed")
            self._subscribers.setdefault(episode_id, {})[subscription_id] = subscriber
        try:
            yield LiveSubscription(subscriber)
        finally:
            async with self._lock:
                episode_subscribers = self._subscribers.get(episode_id)
                if episode_subscribers is not None:
                    episode_subscribers.pop(subscription_id, None)
                    if not episode_subscribers:
                        self._subscribers.pop(episode_id, None)

    async def publish_telemetry(self, telemetry: TelemetryRecordView) -> None:
        await self.publish(
            LiveStreamMessage(
                kind=LiveMessageKind.TELEMETRY,
                episode_id=telemetry.episode_id,
                frame_id=telemetry.frame_id,
                telemetry=telemetry,
                emitted_at=utc_now(),
            )
        )

    async def publish_status(
        self,
        episode_id: str,
        status: dict[str, object],
    ) -> None:
        await self.publish(
            LiveStreamMessage(
                kind=LiveMessageKind.STATUS,
                episode_id=episode_id,
                status=status,
                emitted_at=utc_now(),
            )
        )

    async def publish(self, message: LiveStreamMessage) -> None:
        async with self._lock:
            if self._closed:
                return
            subscribers = tuple(self._subscribers.get(message.episode_id, {}).values())
            for subscriber in subscribers:
                self._enqueue_latest(subscriber, message)

    @staticmethod
    def _enqueue_latest(subscriber: _Subscriber, message: LiveStreamMessage) -> None:
        if subscriber.queue.full():
            stale = subscriber.queue.get_nowait()
            subscriber.dropped += 1
            if isinstance(stale, LiveStreamMessage):
                subscriber.dropped += stale.dropped_before
        outgoing = message.model_copy(
            update={"dropped_before": message.dropped_before + subscriber.dropped}
        )
        subscriber.dropped = 0
        subscriber.queue.put_nowait(outgoing)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = tuple(
                subscriber
                for episode_subscribers in self._subscribers.values()
                for subscriber in episode_subscribers.values()
            )
            self._subscribers.clear()
            for subscriber in subscribers:
                while subscriber.queue.full():
                    subscriber.queue.get_nowait()
                subscriber.queue.put_nowait(_CLOSED)


__all__ = ["LiveSubscription", "LiveTelemetryHub"]

