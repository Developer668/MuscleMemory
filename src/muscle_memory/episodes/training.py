"""Training-only access to authenticated human corrections."""

from __future__ import annotations

from muscle_memory.episodes.models import TrainingCorrection
from muscle_memory.episodes.service import EpisodeService


class TrainingCorrectionFeed:
    """Expose approved demonstrations to dataset builders, never runtime control."""

    def __init__(self, service: EpisodeService) -> None:
        self._service = service

    def approved(self, *, episode_id: str | None = None) -> tuple[TrainingCorrection, ...]:
        return self._service._approved_corrections_for_training(episode_id)


__all__ = ["TrainingCorrectionFeed"]
