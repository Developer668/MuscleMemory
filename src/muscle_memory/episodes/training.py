"""Training-only access to authenticated human corrections."""

from __future__ import annotations

from muscle_memory.episodes.models import TrainingCorrection
from muscle_memory.episodes.service import EpisodeService
from muscle_memory.graph_memory import EvaluatedPolicyVersion, GraphWriteReceipt


class TrainingCorrectionFeed:
    """Expose approved demonstrations to dataset builders, never runtime control."""

    def __init__(self, service: EpisodeService) -> None:
        self._service = service

    def approved(self, *, episode_id: str | None = None) -> tuple[TrainingCorrection, ...]:
        return self._service._approved_corrections_for_training(episode_id)

    def record_policy_lineage(
        self,
        *,
        policy: EvaluatedPolicyVersion,
        lesson_ids: tuple[str, ...],
        evidence_hash: str,
    ) -> tuple[GraphWriteReceipt, ...]:
        return self._service._record_training_lineage(
            policy=policy,
            lesson_ids=lesson_ids,
            evidence_hash=evidence_hash,
        )


__all__ = ["TrainingCorrectionFeed"]
