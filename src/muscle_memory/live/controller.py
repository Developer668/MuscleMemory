"""Admission-aware control surface for production live episodes."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from muscle_memory.live.catalog import LiveWorldCatalog
from muscle_memory.live.manager import LiveEpisodeManager
from muscle_memory.live.models import (
    EvaluatedPolicySelection,
    LiveEpisodeConfig,
    LiveEpisodeStatus,
    VideoFrameSet,
    VideoProduct,
)


class LiveEpisodeControlError(RuntimeError):
    """Base error for a bounded operator request."""


class LiveEpisodeSelectionError(LiveEpisodeControlError):
    """The requested seed or policy is not in the admitted runtime catalog."""


class LiveEpisodeNotFoundError(LiveEpisodeControlError):
    """The requested live episode does not exist in this process."""


class LiveEpisodeConflictError(LiveEpisodeControlError):
    """A live worker cannot be started because its bounded capacity is occupied."""


@dataclass(frozen=True, slots=True)
class LivePolicyOption:
    policy_id: str
    policy_hash: str
    evaluated_episode_count: int
    promotable: bool


@dataclass(frozen=True, slots=True)
class LiveEpisodeOptions:
    catalog_id: str
    catalog_sha256: str
    seeds: tuple[int, ...]
    policies: tuple[LivePolicyOption, ...]
    video_products: tuple[str, ...]
    maximum_duration_seconds: float
    mode: str = "training"


class LiveEpisodeController:
    """Select only pinned worlds and evaluation-bound task policies."""

    def __init__(
        self,
        *,
        manager: LiveEpisodeManager,
        worlds: LiveWorldCatalog,
        policies: tuple[EvaluatedPolicySelection, ...],
        config: LiveEpisodeConfig | None = None,
    ) -> None:
        if not policies:
            raise ValueError("live episode control requires an evaluated policy")
        selections = {selection.policy.policy_id: selection for selection in policies}
        if len(selections) != len(policies):
            raise ValueError("live episode policy identities must be unique")
        self._manager = manager
        self._worlds = worlds
        self._policies = selections
        self._config = config or LiveEpisodeConfig()

    def options(self) -> LiveEpisodeOptions:
        policies = tuple(
            LivePolicyOption(
                policy_id=selection.policy.policy_id,
                policy_hash=selection.policy.policy_hash,
                evaluated_episode_count=selection.evaluated_episode_count,
                promotable=selection.promotable,
            )
            for selection in sorted(
                self._policies.values(),
                key=lambda item: item.policy.policy_id,
            )
        )
        return LiveEpisodeOptions(
            catalog_id=self._worlds.catalog_id,
            catalog_sha256=self._worlds.artifact_sha256,
            seeds=self._worlds.seeds,
            policies=policies,
            video_products=tuple(product.value for product in VideoProduct),
            maximum_duration_seconds=self._config.maximum_duration_seconds,
        )

    def start(self, *, seed: int, policy_id: str) -> LiveEpisodeStatus:
        try:
            world = self._worlds.world_for_seed(seed)
        except KeyError as exc:
            raise LiveEpisodeSelectionError(
                "seed is not in the immutable live training-world catalog"
            ) from exc
        try:
            selection = self._policies[policy_id]
        except KeyError as exc:
            raise LiveEpisodeSelectionError(
                "policy is not an admitted immutable evaluated checkpoint"
            ) from exc
        selection.verify_integrity()
        episode_id = f"live-{seed:016x}-{uuid.uuid4().hex[:12]}"
        try:
            return self._manager.start_episode(
                episode_id=episode_id,
                world=world,
                selection=selection,
                config=self._config,
            )
        except RuntimeError as exc:
            raise LiveEpisodeConflictError(
                "the bounded live simulator worker is already occupied"
            ) from exc

    def status(self, episode_id: str) -> LiveEpisodeStatus:
        try:
            return self._manager.status(episode_id)
        except KeyError as exc:
            raise LiveEpisodeNotFoundError("live episode was not found") from exc

    def cancel(self, episode_id: str) -> LiveEpisodeStatus:
        try:
            return self._manager.cancel(episode_id)
        except KeyError as exc:
            raise LiveEpisodeNotFoundError("live episode was not found") from exc

    def iter_mjpeg(
        self,
        episode_id: str,
        product: VideoProduct,
        *,
        after_frame_index: int = -1,
    ) -> Iterator[bytes]:
        self.status(episode_id)
        return self._manager.iter_mjpeg(
            episode_id,
            product,
            after_frame_index=after_frame_index,
        )

    def video_frame(
        self,
        episode_id: str,
        frame_index: int,
    ) -> VideoFrameSet | None:
        self.status(episode_id)
        return self._manager.video_frame(episode_id, frame_index)

    def shutdown(self) -> None:
        self._manager.shutdown(cancel_running=True, wait=True)


__all__ = [
    "LiveEpisodeConflictError",
    "LiveEpisodeControlError",
    "LiveEpisodeController",
    "LiveEpisodeNotFoundError",
    "LiveEpisodeOptions",
    "LiveEpisodeSelectionError",
    "LivePolicyOption",
]
