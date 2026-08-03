"""Parameterized FalkorDB adapter for post-episode explicit experience."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from muscle_memory.graph_memory.models import (
    CorrectionMemoryRecord,
    CurriculumLesson,
    CurriculumQuery,
    CurriculumResult,
    EpisodeMemoryRecord,
    EvaluatedPolicyVersion,
    FailureMemoryRecord,
    GraphMemoryHealth,
    GraphStorage,
    GraphWriteReceipt,
    LessonMemoryRecord,
    ObstacleMemoryRecord,
    PolicyComparisonRecord,
    PolicyEvaluationRecord,
    PolicyTrainingRecord,
    ProviderState,
    WorldMemoryRecord,
)
from muscle_memory.graph_memory.protocol import (
    GraphMemoryError,
    GraphMemoryIntegrityError,
    GraphProviderUnavailableError,
)


class FalkorQueryResult(Protocol):
    @property
    def result_set(self) -> Sequence[Sequence[object]]: ...


class FalkorGraph(Protocol):
    def query(
        self,
        query: str,
        params: dict[str, object] | None = None,
        timeout: int | None = None,
    ) -> FalkorQueryResult: ...

    def ro_query(
        self,
        query: str,
        params: dict[str, object] | None = None,
        timeout: int | None = None,
    ) -> FalkorQueryResult: ...


_HEALTH_QUERY = "RETURN 1 AS ok"

# FalkorDB stores explicit experience as immutable, connected facts; MERGE plus
# content hashes prevents a familiar graph identity from being silently rewritten.
_RECORD_WORLD = """
MERGE (world:World {world_id: $world_id})
ON CREATE SET
  world.content_hash = $content_hash,
  world.world_hash = $world_hash,
  world.split = $split,
  world.seed = $seed,
  world.generation_version = $generation_version,
  world.validation_hash = $validation_hash,
  world.validated = true,
  world.recorded_at = $recorded_at
RETURN world.content_hash
"""

_RECORD_OBSTACLE = """
MATCH (world:World {world_id: $world_id})
MERGE (obstacle:Obstacle {obstacle_id: $obstacle_id})
ON CREATE SET
  obstacle.content_hash = $content_hash,
  obstacle.obstacle_hash = $obstacle_hash,
  obstacle.category = $category,
  obstacle.collider_kind = $collider_kind,
  obstacle.physical_properties_approved = $physical_properties_approved,
  obstacle.recorded_at = $recorded_at
WITH world, obstacle
WHERE obstacle.content_hash = $content_hash
MERGE (world)-[:CONTAINS]->(obstacle)
RETURN obstacle.content_hash
"""

_RECORD_POLICY = """
MERGE (policy:PolicyVersion {policy_id: $policy_id})
ON CREATE SET
  policy.content_hash = $content_hash,
  policy.checkpoint_hash = $checkpoint_hash,
  policy.evaluation_evidence_hash = $evaluation_evidence_hash,
  policy.evaluation_split = $evaluation_split,
  policy.metrics_json = $metrics_json,
  policy.evaluated = true,
  policy.evaluated_at = $evaluated_at
RETURN policy.content_hash
"""

_RECORD_EPISODE = """
MATCH (world:World {world_id: $world_id})
MATCH (policy:PolicyVersion {policy_id: $policy_id, evaluated: true})
WHERE world.world_hash = $world_hash
  AND world.split = $world_split
  AND policy.checkpoint_hash = $policy_hash
MERGE (episode:Episode {episode_id: $episode_id})
ON CREATE SET
  episode.content_hash = $content_hash,
  episode.robot_checksum = $robot_checksum,
  episode.world_split = $world_split,
  episode.outcome = $outcome,
  episode.completion_time_seconds = $completion_time_seconds,
  episode.collision_count = $collision_count,
  episode.fall_count = $fall_count,
  episode.minimum_clearance_m = $minimum_clearance_m,
  episode.human_interventions = $human_interventions,
  episode.telemetry_digest = $telemetry_digest,
  episode.ended_at = $ended_at
WITH world, policy, episode
WHERE episode.content_hash = $content_hash
MERGE (episode)-[:RAN_IN]->(world)
MERGE (episode)-[:USED]->(policy)
RETURN episode.content_hash
"""

_RECORD_FAILURE = """
MATCH (episode:Episode {episode_id: $episode_id})
MERGE (failure:Failure {failure_id: $failure_id})
ON CREATE SET
  failure.content_hash = $content_hash,
  failure.category = $category,
  failure.severity = $severity,
  failure.summary = $summary,
  failure.detected_at = $detected_at
WITH episode, failure
WHERE failure.content_hash = $content_hash
MERGE (episode)-[:OBSERVED_FAILURE]->(failure)
RETURN failure.content_hash
"""

_RECORD_FAILURE_NEAR_OBSTACLE = """
MATCH (episode:Episode {episode_id: $episode_id})-[:RAN_IN]->(world:World)
MATCH (world)-[:CONTAINS]->(obstacle:Obstacle {obstacle_id: $obstacle_id})
MERGE (failure:Failure {failure_id: $failure_id})
ON CREATE SET
  failure.content_hash = $content_hash,
  failure.category = $category,
  failure.severity = $severity,
  failure.summary = $summary,
  failure.detected_at = $detected_at
WITH episode, obstacle, failure
WHERE failure.content_hash = $content_hash
MERGE (episode)-[:OBSERVED_FAILURE]->(failure)
MERGE (failure)-[:NEAR]->(obstacle)
MERGE (episode)-[:FAILED_NEAR]->(obstacle)
RETURN failure.content_hash
"""

_RECORD_CORRECTION = """
MATCH (failure:Failure {failure_id: $failure_id})
MERGE (correction:Correction {correction_id: $correction_id})
ON CREATE SET
  correction.content_hash = $content_hash,
  correction.kind = $kind,
  correction.description = $description,
  correction.approved = $approved,
  correction.approved_by = $approved_by,
  correction.approved_at = $approved_at,
  correction.created_at = $created_at
WITH failure, correction
WHERE correction.content_hash = $content_hash
MERGE (correction)-[:CORRECTS]->(failure)
RETURN correction.content_hash
"""

_RECORD_LESSON = """
MATCH (correction:Correction {correction_id: $correction_id})
MERGE (lesson:Lesson {lesson_id: $lesson_id})
ON CREATE SET
  lesson.content_hash = $content_hash,
  lesson.kind = $kind,
  lesson.summary = $summary,
  lesson.signature_hash = $signature_hash,
  lesson.created_at = $created_at
WITH correction, lesson
WHERE lesson.content_hash = $content_hash
MERGE (correction)-[:PRODUCED]->(lesson)
RETURN lesson.content_hash
"""

_RECORD_TRAINED_LESSON = """
MATCH (correction:Correction {correction_id: $correction_id})
MATCH (policy:PolicyVersion {policy_id: $trained_policy_id, evaluated: true})
MERGE (lesson:Lesson {lesson_id: $lesson_id})
ON CREATE SET
  lesson.content_hash = $content_hash,
  lesson.kind = $kind,
  lesson.summary = $summary,
  lesson.signature_hash = $signature_hash,
  lesson.created_at = $created_at
WITH correction, policy, lesson
WHERE lesson.content_hash = $content_hash
MERGE (correction)-[:PRODUCED]->(lesson)
MERGE (lesson)-[:TRAINED_INTO]->(policy)
RETURN lesson.content_hash
"""

_RECORD_POLICY_TRAINING = """
MATCH (lesson:Lesson {lesson_id: $lesson_id})
MATCH (policy:PolicyVersion {policy_id: $policy_id, evaluated: true})
MERGE (lesson)-[training:TRAINED_INTO {evidence_hash: $evidence_hash}]->(policy)
ON CREATE SET
  training.content_hash = $content_hash,
  training.trained_at = $trained_at
RETURN training.content_hash
"""

_RECORD_OUTPERFORMANCE = """
MATCH (candidate:PolicyVersion {policy_id: $candidate_policy_id, evaluated: true})
MATCH (baseline:PolicyVersion {policy_id: $baseline_policy_id, evaluated: true})
WHERE candidate.evaluation_split = $held_out_split
  AND baseline.evaluation_split = $held_out_split
MERGE (candidate)-[comparison:OUTPERFORMED {evidence_hash: $evidence_hash}]->(baseline)
ON CREATE SET
  comparison.content_hash = $content_hash,
  comparison.success_rate_delta = $success_rate_delta,
  comparison.collision_rate_delta = $collision_rate_delta,
  comparison.measured_at = $measured_at
RETURN comparison.content_hash
"""

_RECORD_POLICY_EVALUATION = """
MATCH (candidate:PolicyVersion {policy_id: $candidate_policy_id, evaluated: true})
MATCH (baseline:PolicyVersion {policy_id: $baseline_policy_id, evaluated: true})
WHERE candidate.evaluation_split = $held_out_split
  AND baseline.evaluation_split = $held_out_split
MERGE (candidate)-[evaluation:EVALUATED_AGAINST {evidence_hash: $evidence_hash}]->(baseline)
ON CREATE SET
  evaluation.content_hash = $content_hash,
  evaluation.action = $action,
  evaluation.success_rate_delta = $success_rate_delta,
  evaluation.collision_rate_delta = $collision_rate_delta,
  evaluation.measured_at = $measured_at
RETURN evaluation.content_hash
"""

# FalkorDB's graph traversal turns related failures, corrections, and lessons into
# curriculum evidence without putting graph lookup in the robot's control loop.
_CURRICULUM_QUERY = """
MATCH (source_policy:PolicyVersion)<-[:USED]-(episode:Episode {world_split: $training_split})
MATCH (episode)-[:OBSERVED_FAILURE]->(failure:Failure)
MATCH (correction:Correction {approved: true})-[:CORRECTS]->(failure)
MATCH (correction)-[:PRODUCED]->(lesson:Lesson)
OPTIONAL MATCH (failure)-[:NEAR]->(obstacle:Obstacle)<-[:CONTAINS]-(:World)<-[:RAN_IN]-(episode)
OPTIONAL MATCH (lesson)-[:TRAINED_INTO]->(trained_policy:PolicyVersion)
WHERE (size($failure_categories) = 0 OR failure.category IN $failure_categories)
  AND (size($obstacle_categories) = 0 OR obstacle.category IN $obstacle_categories)
  AND (
    trained_policy IS NULL
    OR NOT trained_policy.policy_id IN $exclude_trained_policy_ids
  )
WITH lesson.signature_hash AS lesson_signature,
  lesson.kind AS lesson_kind,
  lesson.summary AS lesson_summary,
  failure.category AS failure_category,
  obstacle.category AS obstacle_category,
  trained_policy,
  min(lesson.lesson_id) AS lesson_id,
  count(DISTINCT episode) AS support_count,
  collect(DISTINCT episode.episode_id) AS source_episode_ids
RETURN
  lesson_id,
  lesson_kind,
  lesson_summary,
  failure_category,
  obstacle_category,
  support_count,
  source_episode_ids,
  trained_policy.policy_id
ORDER BY support_count DESC, lesson_id ASC
LIMIT $limit
"""


class FalkorGraphMemory:
    """Real FalkorDB graph adapter using only parameterized Cypher values."""

    def __init__(self, graph: FalkorGraph, *, graph_name: str, query_timeout_ms: int) -> None:
        self._graph = graph
        self._graph_name = graph_name
        self._query_timeout_ms = query_timeout_ms

    def health(self) -> GraphMemoryHealth:
        bootstrapped = False
        try:
            try:
                result = self._graph.ro_query(_HEALTH_QUERY, timeout=self._query_timeout_ms)
            except Exception:
                # FalkorDB has no read-only key until the first graph query initializes it.
                bootstrap = self._graph.query(_HEALTH_QUERY, timeout=self._query_timeout_ms)
                self._validate_health_result(bootstrap)
                bootstrapped = True
                result = self._graph.ro_query(_HEALTH_QUERY, timeout=self._query_timeout_ms)
            self._validate_health_result(result)
        except Exception as exc:
            return GraphMemoryHealth(
                provider_state=ProviderState.UNAVAILABLE,
                graph_name=self._graph_name,
                detail=f"provider health check failed: {type(exc).__name__}",
                checked_at=datetime.now(UTC),
            )
        return GraphMemoryHealth(
            provider_state=ProviderState.HEALTHY,
            graph_name=self._graph_name,
            detail=(
                "provider initialized a fresh graph and accepted a read-only graph query"
                if bootstrapped
                else "provider accepted a read-only graph query"
            ),
            checked_at=datetime.now(UTC),
        )

    @staticmethod
    def _validate_health_result(result: FalkorQueryResult) -> None:
        rows = result.result_set
        if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] != 1:
            raise GraphMemoryIntegrityError("FalkorDB health query returned an invalid result")

    def record_world(self, record: WorldMemoryRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "world_id": record.world_id,
            "content_hash": record.content_hash,
            "world_hash": record.world_hash,
            "split": record.split.value,
            "seed": record.seed,
            "generation_version": record.generation_version,
            "validation_hash": record.validation_hash,
            "recorded_at": record.recorded_at.isoformat(),
        }
        self._write_and_verify(_RECORD_WORLD, params, record.content_hash, "world")
        return self._receipt("world", record.world_id, record.content_hash)

    def record_obstacle(self, record: ObstacleMemoryRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "obstacle_id": record.obstacle_id,
            "content_hash": record.content_hash,
            "obstacle_hash": record.obstacle_hash,
            "world_id": record.world_id,
            "category": record.category,
            "collider_kind": record.collider_kind,
            "physical_properties_approved": record.physical_properties_approved,
            "recorded_at": record.recorded_at.isoformat(),
        }
        self._write_and_verify(_RECORD_OBSTACLE, params, record.content_hash, "obstacle")
        return self._receipt("obstacle", record.obstacle_id, record.content_hash)

    def record_evaluated_policy(self, record: EvaluatedPolicyVersion) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "policy_id": record.policy_id,
            "content_hash": record.content_hash,
            "checkpoint_hash": record.checkpoint_hash,
            "evaluation_evidence_hash": record.evaluation_evidence_hash,
            "evaluation_split": record.evaluation_split,
            "metrics_json": record.metrics_json,
            "evaluated_at": record.evaluated_at.isoformat(),
        }
        self._write_and_verify(_RECORD_POLICY, params, record.content_hash, "policy version")
        return self._receipt("evaluated_policy", record.policy_id, record.content_hash)

    def record_episode(self, record: EpisodeMemoryRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "episode_id": record.episode_id,
            "content_hash": record.content_hash,
            "robot_checksum": record.robot_checksum,
            "world_id": record.world_id,
            "world_hash": record.world_hash,
            "world_split": record.world_split.value,
            "policy_id": record.policy_id,
            "policy_hash": record.policy_hash,
            "outcome": record.outcome.value,
            "completion_time_seconds": record.completion_time_seconds,
            "collision_count": record.collision_count,
            "fall_count": record.fall_count,
            "minimum_clearance_m": record.minimum_clearance_m,
            "human_interventions": record.human_interventions,
            "telemetry_digest": record.telemetry_digest,
            "ended_at": record.ended_at.isoformat(),
        }
        self._write_and_verify(_RECORD_EPISODE, params, record.content_hash, "episode")
        return self._receipt("episode", record.episode_id, record.content_hash)

    def record_failure(self, record: FailureMemoryRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "failure_id": record.failure_id,
            "episode_id": record.episode_id,
            "content_hash": record.content_hash,
            "category": record.category,
            "severity": record.severity,
            "summary": record.summary,
            "detected_at": record.detected_at.isoformat(),
        }
        query = _RECORD_FAILURE
        if record.obstacle_id is not None:
            params["obstacle_id"] = record.obstacle_id
            query = _RECORD_FAILURE_NEAR_OBSTACLE
        self._write_and_verify(query, params, record.content_hash, "failure")
        return self._receipt("failure", record.failure_id, record.content_hash)

    def record_correction(self, record: CorrectionMemoryRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "correction_id": record.correction_id,
            "failure_id": record.failure_id,
            "content_hash": record.content_hash,
            "kind": record.kind,
            "description": record.description,
            "approved": record.approved,
            "approved_by": record.approved_by,
            "approved_at": (
                record.approved_at.isoformat() if record.approved_at is not None else None
            ),
            "created_at": record.created_at.isoformat(),
        }
        self._write_and_verify(_RECORD_CORRECTION, params, record.content_hash, "correction")
        return self._receipt("correction", record.correction_id, record.content_hash)

    def record_lesson(self, record: LessonMemoryRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "lesson_id": record.lesson_id,
            "correction_id": record.correction_id,
            "content_hash": record.content_hash,
            "kind": record.kind,
            "summary": record.summary,
            "signature_hash": record.signature_hash,
            "trained_policy_id": record.trained_policy_id,
            "created_at": record.created_at.isoformat(),
        }
        query = _RECORD_LESSON if record.trained_policy_id is None else _RECORD_TRAINED_LESSON
        self._write_and_verify(query, params, record.content_hash, "lesson")
        return self._receipt("lesson", record.lesson_id, record.content_hash)

    def record_policy_training(self, record: PolicyTrainingRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "lesson_id": record.lesson_id,
            "policy_id": record.policy_id,
            "evidence_hash": record.evidence_hash,
            "content_hash": record.content_hash,
            "trained_at": record.trained_at.isoformat(),
        }
        self._write_and_verify(
            _RECORD_POLICY_TRAINING,
            params,
            record.content_hash,
            "policy training lineage",
        )
        record_id = f"{record.lesson_id}:{record.policy_id}:{record.evidence_hash}"
        return self._receipt("policy_training", record_id, record.content_hash)

    def record_policy_evaluation(self, record: PolicyEvaluationRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "candidate_policy_id": record.candidate_policy_id,
            "baseline_policy_id": record.baseline_policy_id,
            "evidence_hash": record.evidence_hash,
            "held_out_split": "held_out",
            "content_hash": record.content_hash,
            "action": record.action,
            "success_rate_delta": record.success_rate_delta,
            "collision_rate_delta": record.collision_rate_delta,
            "measured_at": record.measured_at.isoformat(),
        }
        self._write_and_verify(
            _RECORD_POLICY_EVALUATION,
            params,
            record.content_hash,
            "policy evaluation",
        )
        record_id = (
            f"{record.candidate_policy_id}:{record.baseline_policy_id}:{record.evidence_hash}"
        )
        return self._receipt("policy_evaluation", record_id, record.content_hash)

    def record_outperformance(self, record: PolicyComparisonRecord) -> GraphWriteReceipt:
        params: dict[str, object] = {
            "candidate_policy_id": record.candidate_policy_id,
            "baseline_policy_id": record.baseline_policy_id,
            "evidence_hash": record.evidence_hash,
            "held_out_split": "held_out",
            "content_hash": record.content_hash,
            "success_rate_delta": record.success_rate_delta,
            "collision_rate_delta": record.collision_rate_delta,
            "measured_at": record.measured_at.isoformat(),
        }
        self._write_and_verify(
            _RECORD_OUTPERFORMANCE,
            params,
            record.content_hash,
            "policy comparison",
        )
        record_id = (
            f"{record.candidate_policy_id}:{record.baseline_policy_id}:{record.evidence_hash}"
        )
        return self._receipt("outperformance", record_id, record.content_hash)

    def query_curriculum(self, query: CurriculumQuery) -> CurriculumResult:
        params: dict[str, object] = {
            "training_split": "training",
            "failure_categories": list(query.failure_categories),
            "obstacle_categories": list(query.obstacle_categories),
            "exclude_trained_policy_ids": list(query.exclude_trained_policy_ids),
            "limit": query.limit,
        }
        try:
            # FalkorDB curriculum selection is deliberately read-only: agents may
            # learn from the graph here, but cannot mutate evidence while selecting it.
            result = self._graph.ro_query(
                _CURRICULUM_QUERY,
                params=params,
                timeout=self._query_timeout_ms,
            )
        except Exception as exc:
            raise GraphProviderUnavailableError(
                f"FalkorDB curriculum query failed: {type(exc).__name__}"
            ) from exc

        lessons = tuple(self._parse_curriculum_row(row) for row in result.result_set)
        return CurriculumResult(
            lessons=lessons,
            storage=GraphStorage.FALKORDB,
            provider_state=ProviderState.HEALTHY,
            detail="curriculum traversal completed in FalkorDB",
        )

    def _write_and_verify(
        self,
        query: str,
        params: dict[str, object],
        expected_hash: str,
        record_kind: str,
    ) -> None:
        try:
            # FalkorDB returns the stored content hash so the adapter can distinguish
            # an idempotent write from an identity collision.
            result = self._graph.query(
                query,
                params=params,
                timeout=self._query_timeout_ms,
            )
        except GraphMemoryError:
            raise
        except Exception as exc:
            raise GraphProviderUnavailableError(
                f"FalkorDB {record_kind} write failed: {type(exc).__name__}"
            ) from exc

        rows = result.result_set
        if len(rows) != 1 or len(rows[0]) != 1:
            raise GraphMemoryIntegrityError(
                f"FalkorDB rejected {record_kind} references or returned no immutable identity"
            )
        actual_hash = rows[0][0]
        if actual_hash != expected_hash:
            raise GraphMemoryIntegrityError(
                f"{record_kind} identity already maps to different immutable content"
            )

    def _receipt(self, record_kind: str, record_id: str, content_hash: str) -> GraphWriteReceipt:
        return GraphWriteReceipt(
            record_kind=record_kind,
            record_id=record_id,
            content_hash=content_hash,
            storage=GraphStorage.FALKORDB,
            provider_state=ProviderState.HEALTHY,
            mirrored_to_local_cache=False,
            detail="stored in configured FalkorDB graph",
        )

    @staticmethod
    def _parse_curriculum_row(row: Sequence[object]) -> CurriculumLesson:
        if len(row) != 8:
            raise GraphMemoryIntegrityError("FalkorDB curriculum row has an invalid shape")
        lesson_id, kind, summary, failure_category, obstacle_category = row[:5]
        support_count, source_episode_ids, trained_policy_id = row[5:]
        if (
            not isinstance(lesson_id, str)
            or not isinstance(kind, str)
            or not isinstance(summary, str)
            or not isinstance(failure_category, str)
        ):
            raise GraphMemoryIntegrityError("FalkorDB curriculum row has invalid text values")
        if obstacle_category is not None and not isinstance(obstacle_category, str):
            raise GraphMemoryIntegrityError("FalkorDB obstacle category has an invalid value")
        if trained_policy_id is not None and not isinstance(trained_policy_id, str):
            raise GraphMemoryIntegrityError("FalkorDB trained policy has an invalid value")
        if not isinstance(support_count, int) or support_count < 1:
            raise GraphMemoryIntegrityError("FalkorDB support count has an invalid value")
        if not isinstance(source_episode_ids, (list, tuple)) or not all(
            isinstance(value, str) for value in source_episode_ids
        ):
            raise GraphMemoryIntegrityError("FalkorDB source episode list has an invalid value")
        return CurriculumLesson(
            lesson_id=lesson_id,
            lesson_kind=kind,
            summary=summary,
            failure_category=failure_category,
            obstacle_category=obstacle_category,
            support_count=support_count,
            source_episode_ids=tuple(sorted(source_episode_ids)),
            trained_policy_id=trained_policy_id,
        )
