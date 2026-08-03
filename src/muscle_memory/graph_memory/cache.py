"""Durable append-only fallback for graph facts when FalkorDB is unavailable."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import Field

from muscle_memory.graph_memory.models import (
    ContentAddressedRecord,
    CorrectionMemoryRecord,
    CurriculumLesson,
    CurriculumQuery,
    CurriculumResult,
    EpisodeMemoryRecord,
    EvaluatedPolicyVersion,
    FailureMemoryRecord,
    FrozenGraphModel,
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
    WorldSplit,
    canonical_json,
)
from muscle_memory.graph_memory.protocol import GraphMemory, GraphMemoryIntegrityError

type GraphRecord = (
    WorldMemoryRecord
    | ObstacleMemoryRecord
    | EvaluatedPolicyVersion
    | EpisodeMemoryRecord
    | FailureMemoryRecord
    | CorrectionMemoryRecord
    | LessonMemoryRecord
    | PolicyTrainingRecord
    | PolicyEvaluationRecord
    | PolicyComparisonRecord
)


class CachedGraphEvent(FrozenGraphModel):
    """One checksummed fact in the local append-only event log."""

    schema_version: Literal[1] = 1
    record_kind: str
    record_id: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, object]


_RECORD_TYPES: dict[str, type[ContentAddressedRecord]] = {
    "world": WorldMemoryRecord,
    "obstacle": ObstacleMemoryRecord,
    "evaluated_policy": EvaluatedPolicyVersion,
    "episode": EpisodeMemoryRecord,
    "failure": FailureMemoryRecord,
    "correction": CorrectionMemoryRecord,
    "lesson": LessonMemoryRecord,
    "policy_training": PolicyTrainingRecord,
    "policy_evaluation": PolicyEvaluationRecord,
    "outperformance": PolicyComparisonRecord,
}


class AppendOnlyGraphCache:
    """JSONL graph cache that never disguises itself as a live FalkorDB provider."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[tuple[str, str], GraphRecord] = {}
        self._events: list[CachedGraphEvent] = []
        self._lock = RLock()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def events(self) -> tuple[CachedGraphEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def operational_events(self) -> tuple[CachedGraphEvent, ...]:
        """Return a dependency-closed view that can never expose held-out worlds."""

        with self._lock:
            events = tuple(self._events)

        training_world_ids = {
            str(event.payload["world_id"])
            for event in events
            if event.record_kind == "world" and event.payload.get("split") == WorldSplit.TRAINING
        }
        episode_ids = {
            str(event.payload["episode_id"])
            for event in events
            if event.record_kind == "episode"
            and event.payload.get("world_split") == WorldSplit.TRAINING
            and event.payload.get("world_id") in training_world_ids
        }
        failure_ids = {
            str(event.payload["failure_id"])
            for event in events
            if event.record_kind == "failure" and event.payload.get("episode_id") in episode_ids
        }
        correction_ids = {
            str(event.payload["correction_id"])
            for event in events
            if event.record_kind == "correction" and event.payload.get("failure_id") in failure_ids
        }

        selected: list[CachedGraphEvent] = []
        for event in events:
            payload = event.payload
            include = (
                event.record_kind in {"world", "obstacle"}
                and payload.get("world_id") in training_world_ids
            )
            include = include or (
                event.record_kind == "episode" and payload.get("episode_id") in episode_ids
            )
            include = include or (
                event.record_kind == "failure" and payload.get("failure_id") in failure_ids
            )
            include = include or (
                event.record_kind in {"correction", "lesson"}
                and payload.get("correction_id") in correction_ids
            )
            include = include or event.record_kind in {"evaluated_policy", "outperformance"}
            if include:
                selected.append(event)
        return tuple(selected)

    def health(self) -> GraphMemoryHealth:
        return GraphMemoryHealth(
            provider_state=ProviderState.UNCONFIGURED,
            graph_name="local-cache",
            detail=f"append-only fallback available at {self._path}",
            checked_at=datetime.now(UTC),
        )

    def record_world(self, record: WorldMemoryRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_obstacle(self, record: ObstacleMemoryRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_evaluated_policy(self, record: EvaluatedPolicyVersion) -> GraphWriteReceipt:
        return self._record(record)

    def record_episode(self, record: EpisodeMemoryRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_failure(self, record: FailureMemoryRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_correction(self, record: CorrectionMemoryRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_lesson(self, record: LessonMemoryRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_policy_training(self, record: PolicyTrainingRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_policy_evaluation(self, record: PolicyEvaluationRecord) -> GraphWriteReceipt:
        return self._record(record)

    def record_outperformance(self, record: PolicyComparisonRecord) -> GraphWriteReceipt:
        return self._record(record)

    def query_curriculum(self, query: CurriculumQuery) -> CurriculumResult:
        with self._lock:
            episodes = self._by_type(EpisodeMemoryRecord)
            failures = self._by_type(FailureMemoryRecord)
            corrections = self._by_type(CorrectionMemoryRecord)
            lessons = self._by_type(LessonMemoryRecord)
            policy_training = self._by_type(PolicyTrainingRecord)
            obstacles = self._by_type(ObstacleMemoryRecord)

        grouped_episode_ids: dict[tuple[str, str, str, str, str | None, str | None], set[str]] = (
            defaultdict(set)
        )
        lesson_ids: dict[tuple[str, str, str, str, str | None, str | None], set[str]] = defaultdict(
            set
        )

        failures_by_id = {record.failure_id: record for record in failures}
        episodes_by_id = {record.episode_id: record for record in episodes}
        obstacles_by_id = {record.obstacle_id: record for record in obstacles}
        corrections_by_id = {record.correction_id: record for record in corrections}
        trained_policies_by_lesson: dict[str, set[str]] = defaultdict(set)
        for lineage in policy_training:
            trained_policies_by_lesson[lineage.lesson_id].add(lineage.policy_id)

        for lesson in lessons:
            correction = corrections_by_id.get(lesson.correction_id)
            if correction is None or not correction.approved:
                continue
            failure = failures_by_id.get(correction.failure_id)
            if failure is None:
                continue
            episode = episodes_by_id.get(failure.episode_id)
            if episode is None or episode.world_split is not WorldSplit.TRAINING:
                continue
            obstacle = (
                obstacles_by_id.get(failure.obstacle_id)
                if failure.obstacle_id is not None
                else None
            )
            obstacle_category = obstacle.category if obstacle is not None else None
            if query.failure_categories and failure.category not in query.failure_categories:
                continue
            if query.obstacle_categories and obstacle_category not in query.obstacle_categories:
                continue
            trained_policy_ids = trained_policies_by_lesson[lesson.lesson_id]
            if lesson.trained_policy_id is not None:
                trained_policy_ids.add(lesson.trained_policy_id)
            lineage_targets: tuple[str | None, ...] = (
                tuple(sorted(trained_policy_ids)) if trained_policy_ids else (None,)
            )
            for trained_policy_id in lineage_targets:
                if trained_policy_id in query.exclude_trained_policy_ids:
                    continue
                key = (
                    lesson.signature_hash,
                    lesson.kind,
                    lesson.summary,
                    failure.category,
                    obstacle_category,
                    trained_policy_id,
                )
                grouped_episode_ids[key].add(episode.episode_id)
                lesson_ids[key].add(lesson.lesson_id)

        candidates = [
            CurriculumLesson(
                lesson_id=min(lesson_ids[key]),
                lesson_kind=key[1],
                summary=key[2],
                failure_category=key[3],
                obstacle_category=key[4],
                support_count=len(source_ids),
                source_episode_ids=tuple(sorted(source_ids)),
                trained_policy_id=key[5],
            )
            for key, source_ids in grouped_episode_ids.items()
        ]
        candidates.sort(key=lambda candidate: (-candidate.support_count, candidate.lesson_id))
        return CurriculumResult(
            lessons=tuple(candidates[: query.limit]),
            storage=GraphStorage.LOCAL_CACHE,
            provider_state=ProviderState.UNCONFIGURED,
            detail="curriculum traversal used append-only local cache, not FalkorDB",
        )

    def replay_to(self, target: GraphMemory) -> int:
        """Replay immutable events in dependency order to a recovered provider."""

        replayed, _cursor = self.replay_to_with_cursor(target)
        return replayed

    def replay_to_with_cursor(self, target: GraphMemory) -> tuple[int, int]:
        """Replay one atomic cache snapshot and return its exact event cursor."""

        with self._lock:
            records = tuple(self._records.values())
            event_cursor = len(self._events)
        for record in records:
            if isinstance(record, WorldMemoryRecord):
                target.record_world(record)
            elif isinstance(record, ObstacleMemoryRecord):
                target.record_obstacle(record)
            elif isinstance(record, EvaluatedPolicyVersion):
                target.record_evaluated_policy(record)
            elif isinstance(record, EpisodeMemoryRecord):
                target.record_episode(record)
            elif isinstance(record, FailureMemoryRecord):
                target.record_failure(record)
            elif isinstance(record, CorrectionMemoryRecord):
                target.record_correction(record)
            elif isinstance(record, LessonMemoryRecord):
                target.record_lesson(record)
            elif isinstance(record, PolicyTrainingRecord):
                target.record_policy_training(record)
            elif isinstance(record, PolicyEvaluationRecord):
                target.record_policy_evaluation(record)
            else:
                target.record_outperformance(record)
        return len(records), event_cursor

    def _record(self, record: GraphRecord) -> GraphWriteReceipt:
        record_kind, record_id = self._identity(record)
        key = (record_kind, record_id)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                if existing.content_hash != record.content_hash:
                    raise GraphMemoryIntegrityError(
                        f"{record_kind} {record_id!r} is immutable in the local graph cache"
                    )
                return self._receipt(record_kind, record_id, record.content_hash, "already cached")

            self._validate_references(record)
            event = CachedGraphEvent(
                record_kind=record_kind,
                record_id=record_id,
                content_hash=record.content_hash,
                payload=record.model_dump(mode="json"),
            )
            self._append(event)
            self._records[key] = record
            self._events.append(event)
        return self._receipt(record_kind, record_id, record.content_hash, "cached locally")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise GraphMemoryIntegrityError("could not read local graph cache") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise GraphMemoryIntegrityError(
                    f"local graph cache contains a blank event at line {line_number}"
                )
            try:
                event = CachedGraphEvent.model_validate_json(line)
                record_type = _RECORD_TYPES[event.record_kind]
                parsed = record_type.model_validate(event.payload)
            except (KeyError, ValueError) as exc:
                raise GraphMemoryIntegrityError(
                    f"local graph cache event {line_number} is invalid"
                ) from exc
            record = self._as_graph_record(parsed)
            kind, record_id = self._identity(record)
            if kind != event.record_kind or record_id != event.record_id:
                raise GraphMemoryIntegrityError(
                    f"local graph cache event {line_number} has a mismatched identity"
                )
            if record.content_hash != event.content_hash:
                raise GraphMemoryIntegrityError(
                    f"local graph cache event {line_number} has a mismatched checksum"
                )
            key = (kind, record_id)
            existing = self._records.get(key)
            if existing is not None and existing.content_hash != record.content_hash:
                raise GraphMemoryIntegrityError(
                    f"local graph cache event {line_number} mutates an earlier fact"
                )
            if existing is None:
                self._validate_references(record)
                self._records[key] = record
            self._events.append(event)

    def _append(self, event: CachedGraphEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = canonical_json(event.model_dump(mode="json"))
        try:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise GraphMemoryIntegrityError("could not append to local graph cache") from exc

    def _validate_references(self, record: GraphRecord) -> None:
        if isinstance(record, ObstacleMemoryRecord):
            self._require("world", record.world_id)
        elif isinstance(record, EpisodeMemoryRecord):
            world = self._require("world", record.world_id)
            policy = self._require("evaluated_policy", record.policy_id)
            if not isinstance(world, WorldMemoryRecord) or not isinstance(
                policy, EvaluatedPolicyVersion
            ):
                raise GraphMemoryIntegrityError("episode references have invalid record types")
            if world.world_hash != record.world_hash or world.split is not record.world_split:
                raise GraphMemoryIntegrityError(
                    "episode world identity does not match graph memory"
                )
            if policy.checkpoint_hash != record.policy_hash:
                raise GraphMemoryIntegrityError("episode policy hash does not match graph memory")
        elif isinstance(record, FailureMemoryRecord):
            episode = self._require("episode", record.episode_id)
            if record.obstacle_id is not None:
                obstacle = self._require("obstacle", record.obstacle_id)
                if not isinstance(episode, EpisodeMemoryRecord) or not isinstance(
                    obstacle, ObstacleMemoryRecord
                ):
                    raise GraphMemoryIntegrityError("failure references have invalid record types")
                if obstacle.world_id != episode.world_id:
                    raise GraphMemoryIntegrityError(
                        "failure obstacle does not belong to the episode world"
                    )
        elif isinstance(record, CorrectionMemoryRecord):
            self._require("failure", record.failure_id)
        elif isinstance(record, LessonMemoryRecord):
            self._require("correction", record.correction_id)
            if record.trained_policy_id is not None:
                self._require("evaluated_policy", record.trained_policy_id)
        elif isinstance(record, PolicyTrainingRecord):
            self._require("lesson", record.lesson_id)
            self._require("evaluated_policy", record.policy_id)
        elif isinstance(record, (PolicyEvaluationRecord, PolicyComparisonRecord)):
            candidate = self._require("evaluated_policy", record.candidate_policy_id)
            baseline = self._require("evaluated_policy", record.baseline_policy_id)
            if not isinstance(candidate, EvaluatedPolicyVersion) or not isinstance(
                baseline, EvaluatedPolicyVersion
            ):
                raise GraphMemoryIntegrityError(
                    "policy comparison references have invalid record types"
                )
            if candidate.evaluation_split != "held_out" or baseline.evaluation_split != "held_out":
                raise GraphMemoryIntegrityError(
                    "outperformance claims require held-out evaluation evidence"
                )

    def _require(self, record_kind: str, record_id: str) -> GraphRecord:
        try:
            return self._records[(record_kind, record_id)]
        except KeyError as exc:
            raise GraphMemoryIntegrityError(
                f"local graph cache is missing {record_kind} {record_id!r}"
            ) from exc

    def _by_type[T: GraphRecord](self, record_type: type[T]) -> tuple[T, ...]:
        return tuple(record for record in self._records.values() if isinstance(record, record_type))

    @staticmethod
    def _as_graph_record(record: ContentAddressedRecord) -> GraphRecord:
        if not isinstance(
            record,
            (
                WorldMemoryRecord,
                ObstacleMemoryRecord,
                EvaluatedPolicyVersion,
                EpisodeMemoryRecord,
                FailureMemoryRecord,
                CorrectionMemoryRecord,
                LessonMemoryRecord,
                PolicyTrainingRecord,
                PolicyEvaluationRecord,
                PolicyComparisonRecord,
            ),
        ):
            raise GraphMemoryIntegrityError("local graph cache contains an unknown record type")
        return record

    @staticmethod
    def _identity(record: GraphRecord) -> tuple[str, str]:
        if isinstance(record, WorldMemoryRecord):
            return "world", record.world_id
        if isinstance(record, ObstacleMemoryRecord):
            return "obstacle", record.obstacle_id
        if isinstance(record, EvaluatedPolicyVersion):
            return "evaluated_policy", record.policy_id
        if isinstance(record, EpisodeMemoryRecord):
            return "episode", record.episode_id
        if isinstance(record, FailureMemoryRecord):
            return "failure", record.failure_id
        if isinstance(record, CorrectionMemoryRecord):
            return "correction", record.correction_id
        if isinstance(record, LessonMemoryRecord):
            return "lesson", record.lesson_id
        if isinstance(record, PolicyTrainingRecord):
            training_id = f"{record.lesson_id}:{record.policy_id}:{record.evidence_hash}"
            return "policy_training", training_id
        if isinstance(record, PolicyEvaluationRecord):
            evaluation_id = (
                f"{record.candidate_policy_id}:{record.baseline_policy_id}:{record.evidence_hash}"
            )
            return "policy_evaluation", evaluation_id
        comparison_id = (
            f"{record.candidate_policy_id}:{record.baseline_policy_id}:{record.evidence_hash}"
        )
        return "outperformance", comparison_id

    @staticmethod
    def _receipt(
        record_kind: str,
        record_id: str,
        content_hash: str,
        detail: str,
    ) -> GraphWriteReceipt:
        return GraphWriteReceipt(
            record_kind=record_kind,
            record_id=record_id,
            content_hash=content_hash,
            storage=GraphStorage.LOCAL_CACHE,
            provider_state=ProviderState.UNCONFIGURED,
            mirrored_to_local_cache=True,
            detail=f"{detail}; this is not FalkorDB delivery proof",
        )
