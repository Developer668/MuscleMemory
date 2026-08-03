"""Threaded live-episode supervision with bounded concurrency and cancellation."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace

from muscle_memory.live.models import (
    EvaluatedPolicySelection,
    LiveEpisodeConfig,
    LiveEpisodeHealth,
    LiveEpisodePhase,
    LiveEpisodeStatus,
    ValidatedTrainingWorldEnvelope,
    VideoFrameSet,
    VideoProduct,
    require_validated_training_world,
)
from muscle_memory.live.runner import (
    LiveEpisodeLifecycle,
    LiveEpisodeRunner,
    LiveRunProgress,
)
from muscle_memory.live.video import BoundedVideoService


class LiveEpisodeManager:
    """API-ready supervisor for real simulator workers and direct frame access."""

    def __init__(
        self,
        *,
        lifecycle: LiveEpisodeLifecycle,
        video: BoundedVideoService | None = None,
        maximum_concurrent_episodes: int = 1,
        maximum_retained_episodes: int = 4,
    ) -> None:
        if maximum_concurrent_episodes <= 0:
            raise ValueError("maximum_concurrent_episodes must be positive")
        if maximum_retained_episodes <= 0:
            raise ValueError("maximum_retained_episodes must be positive")
        self._lifecycle = lifecycle
        self._video = video or BoundedVideoService()
        self._maximum_concurrent_episodes = maximum_concurrent_episodes
        self._maximum_retained_episodes = maximum_retained_episodes
        self._executor = ThreadPoolExecutor(
            max_workers=maximum_concurrent_episodes,
            thread_name_prefix="mm01-live",
        )
        self._statuses: dict[str, LiveEpisodeStatus] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[LiveEpisodeStatus]] = {}
        self._lock = threading.RLock()
        self._shutting_down = False

    @property
    def video(self) -> BoundedVideoService:
        return self._video

    def start_episode(
        self,
        *,
        episode_id: str,
        world: ValidatedTrainingWorldEnvelope,
        selection: EvaluatedPolicySelection,
        config: LiveEpisodeConfig | None = None,
    ) -> LiveEpisodeStatus:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        training_world = require_validated_training_world(world)
        active_phases = {
            LiveEpisodePhase.QUEUED,
            LiveEpisodePhase.STARTING,
            LiveEpisodePhase.RUNNING,
            LiveEpisodePhase.CANCELLING,
        }
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("live episode manager is shutting down")
            self._prune_terminal_episodes_locked()
            if episode_id in self._statuses:
                raise ValueError(f"live episode {episode_id!r} already exists")
            active = sum(status.phase in active_phases for status in self._statuses.values())
            if active >= self._maximum_concurrent_episodes:
                raise RuntimeError("live episode concurrency limit reached")
            status = LiveEpisodeStatus(
                episode_id=episode_id,
                phase=LiveEpisodePhase.QUEUED,
                health=LiveEpisodeHealth.STARTING,
                world_id=training_world.world_id,
                policy_id=selection.policy.policy_id,
                policy_hash=selection.policy.policy_hash,
                policy_promotable=selection.promotable,
                detail="waiting for the real MuJoCo worker",
            )
            cancellation = threading.Event()
            self._video.start_episode(episode_id)
            self._statuses[episode_id] = status
            self._cancellations[episode_id] = cancellation
            try:
                future = self._executor.submit(
                    self._run_worker,
                    episode_id,
                    world,
                    selection,
                    config or LiveEpisodeConfig(),
                    cancellation,
                )
            except BaseException:
                self._video.finish_episode(episode_id)
                del self._statuses[episode_id]
                del self._cancellations[episode_id]
                raise
            self._futures[episode_id] = future
            return status

    def status(self, episode_id: str) -> LiveEpisodeStatus:
        with self._lock:
            try:
                return self._statuses[episode_id]
            except KeyError as exc:
                raise KeyError(f"live episode {episode_id!r} does not exist") from exc

    def wait(self, episode_id: str, *, timeout: float | None = None) -> LiveEpisodeStatus:
        with self._lock:
            try:
                future = self._futures[episode_id]
            except KeyError as exc:
                raise KeyError(f"live episode {episode_id!r} does not exist") from exc
        result = future.result(timeout=timeout)
        with self._lock:
            self._prune_terminal_episodes_locked()
        return result

    def cancel(self, episode_id: str) -> LiveEpisodeStatus:
        with self._lock:
            status = self.status(episode_id)
            if status.phase in {LiveEpisodePhase.CLOSED, LiveEpisodePhase.FAILED}:
                return status
            self._cancellations[episode_id].set()
            updated = replace(
                status,
                phase=LiveEpisodePhase.CANCELLING,
                detail="cancellation requested; closing on the next 20 Hz tick",
            )
            self._statuses[episode_id] = updated
            return updated

    def latest_video_frame(self, episode_id: str) -> VideoFrameSet | None:
        return self._video.latest(episode_id)

    def video_frame(self, episode_id: str, frame_index: int) -> VideoFrameSet | None:
        return self._video.frame(episode_id, frame_index)

    def iter_mjpeg(
        self,
        episode_id: str,
        product: VideoProduct,
        *,
        after_frame_index: int = -1,
    ) -> Iterator[bytes]:
        return self._video.iter_mjpeg(
            episode_id,
            product,
            after_frame_index=after_frame_index,
        )

    def shutdown(self, *, cancel_running: bool = True, wait: bool = True) -> None:
        with self._lock:
            self._shutting_down = True
            if cancel_running:
                for episode_id, cancellation in self._cancellations.items():
                    if self._statuses[episode_id].phase not in {
                        LiveEpisodePhase.CLOSED,
                        LiveEpisodePhase.FAILED,
                    }:
                        cancellation.set()
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_worker(
        self,
        episode_id: str,
        world: ValidatedTrainingWorldEnvelope,
        selection: EvaluatedPolicySelection,
        config: LiveEpisodeConfig,
        cancellation: threading.Event,
    ) -> LiveEpisodeStatus:
        self._update(
            episode_id,
            lambda current: replace(
                current,
                phase=LiveEpisodePhase.STARTING,
                detail="compiling the validated world around the frozen MM-01 bundle",
            ),
        )
        runner = LiveEpisodeRunner(
            lifecycle=self._lifecycle,
            video=self._video,
            config=config,
        )

        def on_progress(progress: LiveRunProgress) -> None:
            with self._lock:
                current = self._statuses[episode_id]
                phase = (
                    LiveEpisodePhase.CANCELLING
                    if cancellation.is_set()
                    else LiveEpisodePhase.RUNNING
                )
                self._statuses[episode_id] = replace(
                    current,
                    phase=phase,
                    health=progress.health,
                    simulation_time_seconds=progress.simulation_time_seconds,
                    wall_elapsed_seconds=progress.wall_elapsed_seconds,
                    wall_clock_lag_seconds=progress.wall_clock_lag_seconds,
                    telemetry_records=progress.telemetry_records,
                    video_frames=progress.video_frames,
                    dropped_video_frames=progress.dropped_video_frames,
                    last_frame_id=progress.last_frame_id,
                    provider_state=progress.provider_state,
                    detail="streaming real MuJoCo telemetry and direct video",
                )

        try:
            completed = runner.run(
                episode_id=episode_id,
                world=world,
                selection=selection,
                cancel_requested=cancellation.is_set,
                on_progress=on_progress,
            )
        except BaseException as exc:
            error_type = type(exc).__name__
            with suppress(KeyError):
                self._video.finish_episode(episode_id)
            return self._update(
                episode_id,
                lambda current: replace(
                    current,
                    phase=LiveEpisodePhase.FAILED,
                    health=LiveEpisodeHealth.FAILED,
                    error_type=error_type,
                    detail=f"real simulator worker failed ({error_type})",
                ),
            )
        return self._update(
            episode_id,
            lambda current: replace(
                current,
                phase=LiveEpisodePhase.CLOSED,
                health=LiveEpisodeHealth.TERMINAL,
                simulation_time_seconds=completed.progress.simulation_time_seconds,
                wall_elapsed_seconds=completed.progress.wall_elapsed_seconds,
                wall_clock_lag_seconds=completed.progress.wall_clock_lag_seconds,
                telemetry_records=completed.progress.telemetry_records,
                video_frames=completed.progress.video_frames,
                dropped_video_frames=completed.progress.dropped_video_frames,
                last_frame_id=completed.progress.last_frame_id,
                provider_state=completed.progress.provider_state,
                completion_reason=completed.completion_reason,
                success=completed.result.success,
                failed_reasons=completed.result.failed_reasons,
                graph_provider_complete=completed.closure.graph.provider_complete,
                telemetry_provider_complete=completed.closure.telemetry.provider_complete,
                detail="measured episode closed and handed to post-episode graph memory",
            ),
        )

    def _update(
        self,
        episode_id: str,
        update: Callable[[LiveEpisodeStatus], LiveEpisodeStatus],
    ) -> LiveEpisodeStatus:
        with self._lock:
            current = self._statuses[episode_id]
            updated = update(current)
            self._statuses[episode_id] = updated
            if updated.phase in {LiveEpisodePhase.CLOSED, LiveEpisodePhase.FAILED}:
                self._prune_terminal_episodes_locked()
            return updated

    def _prune_terminal_episodes_locked(self) -> None:
        terminal = tuple(
            episode_id
            for episode_id, status in self._statuses.items()
            if status.phase in {LiveEpisodePhase.CLOSED, LiveEpisodePhase.FAILED}
        )
        excess = len(terminal) - self._maximum_retained_episodes
        if excess <= 0:
            return
        for episode_id in terminal[:excess]:
            future = self._futures.get(episode_id)
            if future is not None and not future.done():
                continue
            self._video.discard_episode(episode_id)
            self._statuses.pop(episode_id, None)
            self._cancellations.pop(episode_id, None)
            self._futures.pop(episode_id, None)


__all__ = ["LiveEpisodeManager"]
