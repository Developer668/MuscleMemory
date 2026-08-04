"""Graph-memory interface kept outside simulation and policy control paths."""

from __future__ import annotations

from typing import Protocol

from muscle_memory.graph_memory.models import (
    CorrectionMemoryRecord,
    CurriculumQuery,
    CurriculumResult,
    EpisodeMemoryRecord,
    EvaluatedPolicyVersion,
    FailureMemoryRecord,
    GraphMemoryHealth,
    GraphWriteReceipt,
    LessonMemoryRecord,
    ObstacleMemoryRecord,
    PolicyComparisonRecord,
    PolicyEvaluationRecord,
    PolicyTrainingRecord,
    WorldMemoryRecord,
)


class GraphMemoryError(RuntimeError):
    """Base error for explicit experience storage."""


class GraphMemoryIntegrityError(GraphMemoryError):
    """An immutable identity was missing or mapped to different content."""

    def __init__(
        self,
        message: str,
        *,
        record_kind: str | None = None,
        record_id: str | None = None,
        expected_hash: str | None = None,
        actual_hash: str | None = None,
    ) -> None:
        super().__init__(message)
        self.record_kind = record_kind
        self.record_id = record_id
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class GraphProviderUnavailableError(GraphMemoryError):
    """The configured FalkorDB provider could not complete an operation."""


class GraphMemory(Protocol):
    """Post-episode memory API; implementations never participate in robot control."""

    def health(self) -> GraphMemoryHealth: ...

    def record_world(self, record: WorldMemoryRecord) -> GraphWriteReceipt: ...

    def record_obstacle(self, record: ObstacleMemoryRecord) -> GraphWriteReceipt: ...

    def record_evaluated_policy(
        self, record: EvaluatedPolicyVersion
    ) -> GraphWriteReceipt: ...

    def record_episode(self, record: EpisodeMemoryRecord) -> GraphWriteReceipt: ...

    def record_failure(self, record: FailureMemoryRecord) -> GraphWriteReceipt: ...

    def record_correction(self, record: CorrectionMemoryRecord) -> GraphWriteReceipt: ...

    def record_lesson(self, record: LessonMemoryRecord) -> GraphWriteReceipt: ...

    def record_policy_training(self, record: PolicyTrainingRecord) -> GraphWriteReceipt: ...

    def record_policy_evaluation(self, record: PolicyEvaluationRecord) -> GraphWriteReceipt: ...

    def record_outperformance(
        self, record: PolicyComparisonRecord
    ) -> GraphWriteReceipt: ...

    def query_curriculum(self, query: CurriculumQuery) -> CurriculumResult: ...
