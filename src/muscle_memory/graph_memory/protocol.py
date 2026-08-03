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
    WorldMemoryRecord,
)


class GraphMemoryError(RuntimeError):
    """Base error for explicit experience storage."""


class GraphMemoryIntegrityError(GraphMemoryError):
    """An immutable identity was missing or mapped to different content."""


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

    def record_outperformance(
        self, record: PolicyComparisonRecord
    ) -> GraphWriteReceipt: ...

    def query_curriculum(self, query: CurriculumQuery) -> CurriculumResult: ...
