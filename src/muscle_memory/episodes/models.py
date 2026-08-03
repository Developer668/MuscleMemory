"""Immutable episode-lifecycle, replay, and correction records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    FailureMemoryRecord,
    GraphStorage,
    GraphWriteReceipt,
    ProviderState,
    WorldSplit,
)
from muscle_memory.telemetry import (
    EpisodeTelemetryRecord,
    LaserDataProviderState,
    TelemetryDelivery,
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def canonical_json(value: object) -> str:
    """Return deterministic JSON used by content-addressed episode records."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("episode content must be JSON serializable") from exc


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash(value: str, name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_identifier(value: str, name: str) -> None:
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is not a valid identifier")


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


class EpisodeLifecycleState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    """Permanent identity fixed before the first telemetry append."""

    episode_id: str
    robot_checksum: str
    world_id: str
    world_hash: str
    world_split: WorldSplit
    policy_id: str
    policy_hash: str
    opened_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.world_id, "world_id")
        _require_identifier(self.policy_id, "policy_id")
        _require_hash(self.robot_checksum, "robot_checksum")
        _require_hash(self.world_hash, "world_hash")
        _require_hash(self.policy_hash, "policy_hash")
        _require_aware(self.opened_at, "opened_at")


@dataclass(frozen=True, slots=True)
class EpisodeAppendReceipt:
    """Truthful provider result for one accepted episode event."""

    episode_id: str
    sequence: int
    event_id: str
    delivery: TelemetryDelivery
    provider_state: LaserDataProviderState
    pending_local_records: int

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_hash(self.event_id, "event_id")
        if self.sequence < 0 or self.pending_local_records < 0:
            raise ValueError("receipt counts must be non-negative")


@dataclass(frozen=True, slots=True)
class TelemetryDeliveryReport:
    total_records: int
    provider_confirmed_records: int
    local_only_records: int
    records_without_receipts: int
    pending_local_records: int
    provider_states: tuple[LaserDataProviderState, ...]

    @property
    def provider_complete(self) -> bool:
        return (
            self.total_records > 0
            and self.provider_confirmed_records == self.total_records
            and self.records_without_receipts == 0
            and self.pending_local_records == 0
        )

    @property
    def partial_delivery(self) -> bool:
        return not self.provider_complete


@dataclass(frozen=True, slots=True)
class GraphPersistenceReport:
    """Post-close graph receipts without conflating fallback with provider storage."""

    expected_records: int
    receipts: tuple[GraphWriteReceipt, ...]
    error_type: str | None = None
    detail: str | None = None

    @property
    def complete(self) -> bool:
        return len(self.receipts) == self.expected_records and self.error_type is None

    @property
    def provider_complete(self) -> bool:
        return self.complete and all(
            receipt.storage is GraphStorage.FALKORDB
            and receipt.provider_state
            in {ProviderState.HEALTHY, ProviderState.END_TO_END_VERIFIED}
            for receipt in self.receipts
        )

    @property
    def partial_provider_delivery(self) -> bool:
        return not self.provider_complete


@dataclass(frozen=True, slots=True)
class EpisodeClosure:
    """The one immutable close result for an episode."""

    identity: EpisodeIdentity
    result: PolicyEpisodeResult
    telemetry_digest: str
    telemetry: TelemetryDeliveryReport
    failures: tuple[FailureMemoryRecord, ...]
    graph: GraphPersistenceReport
    closed_at: datetime

    def __post_init__(self) -> None:
        _require_hash(self.telemetry_digest, "telemetry_digest")
        _require_aware(self.closed_at, "closed_at")


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """One ordered replay event; video can join only through ``frame_id``."""

    record: EpisodeTelemetryRecord
    frame_id: str | None

    def __post_init__(self) -> None:
        if self.frame_id != self.record.frame_id:
            raise ValueError("replay frame_id must be copied from the telemetry record")

    @property
    def video_join(self) -> tuple[str, str] | None:
        if self.frame_id is None:
            return None
        return ("frame_id", self.frame_id)


class CorrectionKind(StrEnum):
    ROUTE = "route"
    KEEP_OUT = "keep_out"


@dataclass(frozen=True, slots=True)
class CorrectionPoint:
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("correction coordinates must be finite")

    def as_json(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}


@dataclass(frozen=True, slots=True)
class CorrectionSubmission:
    """Content-addressed correction, pending independently of approval."""

    correction_id: str
    episode_id: str
    failure_id: str
    kind: CorrectionKind
    points: tuple[CorrectionPoint, ...]
    geometry_json: str
    description: str
    submitted_by: str
    created_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.correction_id, "correction_id")
        if not self.correction_id.startswith("correction-"):
            raise ValueError("correction_id must contain its content digest")
        _require_hash(self.correction_id.removeprefix("correction-"), "correction content")
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.failure_id, "failure_id")
        _require_aware(self.created_at, "created_at")
        if not self.description.strip() or not self.submitted_by.strip():
            raise ValueError("correction description and submitter must not be empty")
        expected_points = 2 if self.kind is CorrectionKind.ROUTE else 3
        if len(self.points) < expected_points:
            raise ValueError(
                f"{self.kind.value} corrections require at least {expected_points} points"
            )
        expected_geometry = canonical_json(
            {"kind": self.kind.value, "points": [point.as_json() for point in self.points]}
        )
        if self.geometry_json != expected_geometry:
            raise ValueError("geometry_json must match the canonical correction geometry")

    @property
    def content_hash(self) -> str:
        return self.correction_id.removeprefix("correction-")


@dataclass(frozen=True, slots=True)
class AuthenticatedHuman:
    subject_id: str
    authentication_method: str
    authenticated: bool

    def require_authenticated(self) -> None:
        if (
            not self.authenticated
            or not self.subject_id.strip()
            or not self.authentication_method.strip()
        ):
            raise PermissionError("correction approval requires an authenticated human")


@dataclass(frozen=True, slots=True)
class CorrectionApproval:
    approval_id: str
    submission: CorrectionSubmission
    approved_by: str
    authentication_method: str
    approved_at: datetime
    graph_receipt: GraphWriteReceipt | None
    graph_error_type: str | None = None
    graph_detail: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.approval_id, "approval_id")
        if not self.approval_id.startswith("approval-"):
            raise ValueError("approval_id must contain its content digest")
        _require_hash(self.approval_id.removeprefix("approval-"), "approval content")
        _require_aware(self.approved_at, "approved_at")
        if not self.approved_by.strip() or not self.authentication_method.strip():
            raise ValueError("approval identity must not be empty")

    @property
    def graph_provider_complete(self) -> bool:
        receipt = self.graph_receipt
        return (
            receipt is not None
            and receipt.storage is GraphStorage.FALKORDB
            and receipt.provider_state
            in {ProviderState.HEALTHY, ProviderState.END_TO_END_VERIFIED}
        )


@dataclass(frozen=True, slots=True)
class TrainingCorrection:
    """Approved correction projection intentionally owned by the training boundary."""

    correction_id: str
    episode_id: str
    failure_id: str
    kind: CorrectionKind
    points: tuple[CorrectionPoint, ...]
    description: str
    approved_by: str
    approved_at: datetime
