"""Bounded fan-out for 20 Hz telemetry and episode status updates."""

from __future__ import annotations

import asyncio
import itertools
import threading
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
DEFAULT_MAXIMUM_SUBSCRIBERS = 128
DEFAULT_MAXIMUM_SUBSCRIBERS_PER_EPISODE = 16
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


class LiveSubscriberLimitError(RuntimeError):
    pass


class LiveTelemetryHub:
    """Drop oldest stale data per slow consumer instead of blocking ingestion."""

    def __init__(
        self,
        *,
        queue_size: int = DEFAULT_SUBSCRIBER_QUEUE_SIZE,
        maximum_subscribers: int = DEFAULT_MAXIMUM_SUBSCRIBERS,
        maximum_subscribers_per_episode: int = DEFAULT_MAXIMUM_SUBSCRIBERS_PER_EPISODE,
    ) -> None:
        if queue_size < 1:
            raise ValueError("live subscriber queue_size must be positive")
        if maximum_subscribers < 1 or maximum_subscribers_per_episode < 1:
            raise ValueError("live subscriber limits must be positive")
        if maximum_subscribers_per_episode > maximum_subscribers:
            raise ValueError("per-episode subscriber limit exceeds the global limit")
        self._queue_size = queue_size
        self._maximum_subscribers = maximum_subscribers
        self._maximum_subscribers_per_episode = maximum_subscribers_per_episode
        self._subscribers: dict[str, dict[int, _Subscriber]] = {}
        self._ids = itertools.count()
        self._lock = asyncio.Lock()
        self._loop_guard = threading.Lock()
        self._owning_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    @property
    def queue_size(self) -> int:
        return self._queue_size

    def bind_running_loop(self) -> None:
        """Pin all subscriber state to the FastAPI event loop."""

        running = asyncio.get_running_loop()
        with self._loop_guard:
            if self._owning_loop is None:
                self._owning_loop = running
            elif self._owning_loop is not running:
                raise RuntimeError("live telemetry hub is bound to another event loop")

    async def _on_owning_loop(self, message: LiveStreamMessage) -> None:
        running = asyncio.get_running_loop()
        with self._loop_guard:
            if self._owning_loop is None:
                self._owning_loop = running
            owning = self._owning_loop
        if owning is running:
            await self._publish_on_owning_loop(message)
            return
        if owning.is_closed():
            raise RuntimeError("live telemetry hub event loop is closed")
        future = asyncio.run_coroutine_threadsafe(
            self._publish_on_owning_loop(message),
            owning,
        )
        await asyncio.wrap_future(future)

    @asynccontextmanager
    async def subscribe(self, episode_id: str) -> AsyncIterator[LiveSubscription]:
        if not episode_id.strip():
            raise ValueError("episode_id must not be blank")
        self.bind_running_loop()
        subscriber = _Subscriber(queue=asyncio.Queue(maxsize=self._queue_size))
        subscription_id = next(self._ids)
        async with self._lock:
            if self._closed:
                raise RuntimeError("live telemetry hub is closed")
            episode_subscribers = self._subscribers.setdefault(episode_id, {})
            total_subscribers = sum(len(items) for items in self._subscribers.values())
            if (
                total_subscribers >= self._maximum_subscribers
                or len(episode_subscribers) >= self._maximum_subscribers_per_episode
            ):
                if not episode_subscribers:
                    self._subscribers.pop(episode_id, None)
                raise LiveSubscriberLimitError("live telemetry subscriber capacity reached")
            episode_subscribers[subscription_id] = subscriber
        try:
            yield LiveSubscription(subscriber)
        finally:
            async with self._lock:
                active_subscribers = self._subscribers.get(episode_id)
                if active_subscribers is not None:
                    active_subscribers.pop(subscription_id, None)
                    if not active_subscribers:
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
        await self._on_owning_loop(message)

    async def _publish_on_owning_loop(self, message: LiveStreamMessage) -> None:
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
        self.bind_running_loop()
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


__all__ = ["LiveSubscriberLimitError", "LiveSubscription", "LiveTelemetryHub"]
