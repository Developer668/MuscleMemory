"""Immutable records shared by FalkorDB and the append-only graph cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    SecretStr,
    field_validator,
    model_validator,
)

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


def canonical_json(value: object) -> str:
    """Encode graph payloads deterministically before hashing or persistence."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("graph-memory payload must be JSON serializable") from exc


class FrozenGraphModel(BaseModel):
    """Strict immutable base for every graph-memory boundary object."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ContentAddressedRecord(FrozenGraphModel):
    """A fact whose serialized value cannot be replaced under the same identity."""

    @property
    def content_hash(self) -> str:
        payload = canonical_json(self.model_dump(mode="json"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WorldSplit(StrEnum):
    TRAINING = "training"
    HELD_OUT = "held_out"


class EpisodeOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class ProviderState(StrEnum):
    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    END_TO_END_VERIFIED = "end_to_end_verified"


class GraphStorage(StrEnum):
    FALKORDB = "falkordb"
    LOCAL_CACHE = "local_cache"


class FalkorDBSettings(FrozenGraphModel):
    """Connection settings loaded without ever serializing the credential-bearing URL."""

    url: SecretStr | None = None
    graph_name: str = Field(default="muscle_memory", pattern=r"^[A-Za-z0-9:_-]{1,128}$")
    query_timeout_ms: int = Field(default=2_000, ge=50, le=30_000)
    cache_path: Path = Path(".cache/muscle-memory/falkordb-events.jsonl")

    @field_validator("url", mode="before")
    @classmethod
    def normalize_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def supported_url(self) -> Self:
        if self.url is None:
            return self
        url = self.url.get_secret_value()
        if not url.startswith(("redis://", "rediss://")):
            raise ValueError("FalkorDB URL must use redis:// or rediss://")
        return self


class WorldMemoryRecord(ContentAddressedRecord):
    world_id: str = Field(pattern=_ID_PATTERN)
    world_hash: str = Field(pattern=_HASH_PATTERN)
    split: WorldSplit
    seed: int = Field(ge=0, le=(2**63) - 1)
    generation_version: int = Field(ge=1)
    validation_hash: str = Field(pattern=_HASH_PATTERN)
    validated: bool
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def require_validation_gate(self) -> Self:
        if not self.validated:
            raise ValueError("unvalidated worlds cannot enter graph memory")
        return self


class ObstacleMemoryRecord(ContentAddressedRecord):
    obstacle_id: str = Field(pattern=_ID_PATTERN)
    obstacle_hash: str = Field(pattern=_HASH_PATTERN)
    world_id: str = Field(pattern=_ID_PATTERN)
    category: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    collider_kind: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]*$")
    physical_properties_approved: bool
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def require_approved_physical_properties(self) -> Self:
        if not self.physical_properties_approved:
            raise ValueError(
                "obstacles linked to validated worlds require approved physical properties"
            )
        return self


class EvaluatedPolicyVersion(ContentAddressedRecord):
    policy_id: str = Field(pattern=_ID_PATTERN)
    checkpoint_hash: str = Field(pattern=_HASH_PATTERN)
    evaluation_evidence_hash: str = Field(pattern=_HASH_PATTERN)
    evaluation_split: str = Field(pattern=r"^(development|held_out)$")
    metrics_json: str
    evaluated_at: AwareDatetime

    @field_validator("metrics_json")
    @classmethod
    def canonical_metrics(cls, value: str) -> str:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("metrics_json must contain valid JSON") from exc
        if canonical_json(decoded) != value:
            raise ValueError("metrics_json must use canonical JSON encoding")
        return value

    @classmethod
    def create(
        cls,
        *,
        policy_id: str,
        checkpoint_hash: str,
        evaluation_evidence_hash: str,
        evaluation_split: str,
        metrics: object,
        evaluated_at: datetime,
    ) -> Self:
        return cls(
            policy_id=policy_id,
            checkpoint_hash=checkpoint_hash,
            evaluation_evidence_hash=evaluation_evidence_hash,
            evaluation_split=evaluation_split,
            metrics_json=canonical_json(metrics),
            evaluated_at=evaluated_at,
        )


class EpisodeMemoryRecord(ContentAddressedRecord):
    episode_id: str = Field(pattern=_ID_PATTERN)
    robot_checksum: str = Field(pattern=_HASH_PATTERN)
    world_id: str = Field(pattern=_ID_PATTERN)
    world_hash: str = Field(pattern=_HASH_PATTERN)
    world_split: WorldSplit
    policy_id: str = Field(pattern=_ID_PATTERN)
    policy_hash: str = Field(pattern=_HASH_PATTERN)
    outcome: EpisodeOutcome
    completion_time_seconds: FiniteFloat = Field(ge=0.0)
    collision_count: int = Field(ge=0)
    fall_count: int = Field(ge=0)
    # MuJoCo reports signed geom distance, so collisions can produce negative penetration.
    minimum_clearance_m: FiniteFloat
    human_interventions: int = Field(ge=0)
    telemetry_digest: str = Field(pattern=_HASH_PATTERN)
    ended_at: AwareDatetime


class FailureMemoryRecord(ContentAddressedRecord):
    failure_id: str = Field(pattern=_ID_PATTERN)
    episode_id: str = Field(pattern=_ID_PATTERN)
    category: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    obstacle_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    severity: FiniteFloat = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1_000)
    detected_at: AwareDatetime


class CorrectionMemoryRecord(ContentAddressedRecord):
    correction_id: str = Field(pattern=_ID_PATTERN)
    failure_id: str = Field(pattern=_ID_PATTERN)
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=4_000)
    approved: bool = False
    approved_by: str | None = Field(default=None, min_length=1, max_length=256)
    approved_at: AwareDatetime | None = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def approval_is_complete(self) -> Self:
        has_approval_fields = self.approved_by is not None or self.approved_at is not None
        if self.approved and (self.approved_by is None or self.approved_at is None):
            raise ValueError("approved corrections require approved_by and approved_at")
        if not self.approved and has_approval_fields:
            raise ValueError("unapproved corrections cannot carry approval metadata")
        return self


class LessonMemoryRecord(ContentAddressedRecord):
    lesson_id: str = Field(pattern=_ID_PATTERN)
    correction_id: str = Field(pattern=_ID_PATTERN)
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    summary: str = Field(min_length=1, max_length=2_000)
    signature_hash: str = Field(pattern=_HASH_PATTERN)
    trained_policy_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    created_at: AwareDatetime


class PolicyComparisonRecord(ContentAddressedRecord):
    candidate_policy_id: str = Field(pattern=_ID_PATTERN)
    baseline_policy_id: str = Field(pattern=_ID_PATTERN)
    evidence_hash: str = Field(pattern=_HASH_PATTERN)
    success_rate_delta: FiniteFloat
    collision_rate_delta: FiniteFloat
    measured_at: AwareDatetime

    @model_validator(mode="after")
    def distinct_policies(self) -> Self:
        if self.candidate_policy_id == self.baseline_policy_id:
            raise ValueError("policy comparison requires two distinct policy versions")
        return self


class CurriculumQuery(FrozenGraphModel):
    failure_categories: tuple[str, ...] = ()
    obstacle_categories: tuple[str, ...] = ()
    exclude_trained_policy_ids: tuple[str, ...] = ()
    limit: int = Field(default=10, ge=1, le=100)


class CurriculumLesson(FrozenGraphModel):
    lesson_id: str
    lesson_kind: str
    summary: str
    failure_category: str
    obstacle_category: str | None
    support_count: int = Field(ge=1)
    source_episode_ids: tuple[str, ...]
    trained_policy_id: str | None


class GraphMemoryHealth(FrozenGraphModel):
    provider_state: ProviderState
    graph_name: str
    detail: str
    checked_at: AwareDatetime


class GraphWriteReceipt(FrozenGraphModel):
    record_kind: str
    record_id: str
    content_hash: str = Field(pattern=_HASH_PATTERN)
    storage: GraphStorage
    provider_state: ProviderState
    mirrored_to_local_cache: bool
    detail: str


class CurriculumResult(FrozenGraphModel):
    lessons: tuple[CurriculumLesson, ...]
    storage: GraphStorage
    provider_state: ProviderState
    detail: str
