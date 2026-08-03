"""Bounded direct-video storage with frame and MJPEG access surfaces."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from muscle_memory.live.models import VideoFrameSet, VideoProduct

MJPEG_BOUNDARY = "mm01-frame"


@dataclass(frozen=True, slots=True)
class VideoBufferStats:
    episode_id: str
    buffered_frames: int
    buffered_bytes: int
    appended_frames: int
    dropped_frames: int
    closed: bool


@dataclass(slots=True)
class _EpisodeBuffer:
    frames: deque[VideoFrameSet]
    byte_length: int = 0
    appended_frames: int = 0
    dropped_frames: int = 0
    closed: bool = False


class BoundedVideoService:
    """Keep video bytes outside telemetry in a strictly bounded process buffer."""

    def __init__(self, *, maximum_frame_sets: int = 180, maximum_bytes: int = 64 << 20) -> None:
        if maximum_frame_sets <= 0 or maximum_bytes <= 0:
            raise ValueError("video buffer limits must be positive")
        self._maximum_frame_sets = maximum_frame_sets
        self._maximum_bytes = maximum_bytes
        self._episodes: dict[str, _EpisodeBuffer] = {}
        self._condition = threading.Condition(threading.RLock())

    def start_episode(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        with self._condition:
            if episode_id in self._episodes:
                raise ValueError(f"video episode {episode_id!r} already exists")
            self._episodes[episode_id] = _EpisodeBuffer(frames=deque())
            self._condition.notify_all()

    def append(self, episode_id: str, frame: VideoFrameSet) -> VideoBufferStats:
        with self._condition:
            state = self._state(episode_id)
            if state.closed:
                raise RuntimeError("cannot append video after episode close")
            if state.frames and frame.metadata.frame_index <= state.frames[-1].metadata.frame_index:
                raise ValueError("video frame indexes must be strictly increasing")
            if frame.byte_length > self._maximum_bytes:
                raise ValueError("one video frame set exceeds the configured byte limit")
            while state.frames and (
                len(state.frames) >= self._maximum_frame_sets
                or state.byte_length + frame.byte_length > self._maximum_bytes
            ):
                discarded = state.frames.popleft()
                state.byte_length -= discarded.byte_length
                state.dropped_frames += 1
            state.frames.append(frame)
            state.byte_length += frame.byte_length
            state.appended_frames += 1
            stats = self._stats(episode_id, state)
            self._condition.notify_all()
            return stats

    def finish_episode(self, episode_id: str) -> VideoBufferStats:
        with self._condition:
            state = self._state(episode_id)
            state.closed = True
            stats = self._stats(episode_id, state)
            self._condition.notify_all()
            return stats

    def stats(self, episode_id: str) -> VideoBufferStats:
        with self._condition:
            state = self._state(episode_id)
            return self._stats(episode_id, state)

    def latest(self, episode_id: str) -> VideoFrameSet | None:
        with self._condition:
            frames = self._state(episode_id).frames
            return frames[-1] if frames else None

    def frame(self, episode_id: str, frame_index: int) -> VideoFrameSet | None:
        with self._condition:
            for frame in self._state(episode_id).frames:
                if frame.metadata.frame_index == frame_index:
                    return frame
            return None

    def iter_mjpeg(
        self,
        episode_id: str,
        product: VideoProduct,
        *,
        after_frame_index: int = -1,
        wait_timeout_seconds: float = 1.0,
    ) -> Iterator[bytes]:
        """Yield a bounded live MJPEG stream and terminate when the episode closes."""
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive")
        cursor = after_frame_index
        while True:
            with self._condition:
                state = self._state(episode_id)
                available = tuple(
                    frame for frame in state.frames if frame.metadata.frame_index > cursor
                )
                closed = state.closed
                if not available and not closed:
                    self._condition.wait(wait_timeout_seconds)
                    continue
            if not available:
                return
            for frame_set in available:
                encoded = frame_set.product(product)
                cursor = frame_set.metadata.frame_index
                yield (
                    f"--{MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(encoded.data)}\r\n"
                    f"X-Frame-Id: {frame_set.metadata.frame_id}\r\n\r\n"
                ).encode("ascii") + encoded.data + b"\r\n"

    def _state(self, episode_id: str) -> _EpisodeBuffer:
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise KeyError(f"video episode {episode_id!r} does not exist") from exc

    @staticmethod
    def _stats(episode_id: str, state: _EpisodeBuffer) -> VideoBufferStats:
        return VideoBufferStats(
            episode_id=episode_id,
            buffered_frames=len(state.frames),
            buffered_bytes=state.byte_length,
            appended_frames=state.appended_frames,
            dropped_frames=state.dropped_frames,
            closed=state.closed,
        )


__all__ = ["MJPEG_BOUNDARY", "BoundedVideoService", "VideoBufferStats"]
