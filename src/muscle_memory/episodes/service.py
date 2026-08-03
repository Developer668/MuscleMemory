"""Episode ingestion, immutable closure, replay, and correction approval."""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from muscle_memory.episodes.journal import EpisodeJournal, VolatileEpisodeJournal
from muscle_memory.episodes.models import (
    AuthenticatedHuman,
    CorrectionApproval,
    CorrectionKind,
    CorrectionPoint,
    CorrectionSubmission,
    EpisodeAppendReceipt,
    EpisodeClosure,
    EpisodeIdentity,
    EpisodeLifecycleState,
    GraphPersistenceReport,
    ReplayRecord,
    TelemetryDeliveryReport,
    TrainingCorrection,
    canonical_json,
    content_digest,
)
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    CorrectionMemoryRecord,
    EpisodeMemoryRecord,
    EpisodeOutcome,
    EvaluatedPolicyVersion,
    FailureMemoryRecord,
    GraphMemory,
    GraphWriteReceipt,
    ObstacleMemoryRecord,
    WorldMemoryRecord,
    WorldSplit,
)
from muscle_memory.telemetry import (
    DuplicateTelemetryRecordError,
    EpisodeTelemetryRecord,
    LaserDataTelemetryEnvelope,
    OutOfOrderTelemetryError,
    TelemetryAppendResult,
    TelemetryDelivery,
    TelemetryMutationError,
)

NUMERIC_TELEMETRY_HZ = 20
_CADENCE_TOLERANCE_SECONDS = 1e-6


class EpisodeServiceError(RuntimeError):
    """Base error for the episode lifecycle boundary."""


class EpisodeNotFoundError(EpisodeServiceError):
    pass


class DuplicateEpisodeError(EpisodeServiceError):
    pass


class EpisodeClosedError(EpisodeServiceError):
    pass


class EpisodeIdentityError(EpisodeServiceError):
    pass


class TelemetryCadenceError(EpisodeServiceError):
    pass


class CorrectionNotFoundError(EpisodeServiceError):
    pass


class CorrectionAlreadyApprovedError(EpisodeServiceError):
    pass


class TelemetryBackend(Protocol):
    async def append(self, record: EpisodeTelemetryRecord) -> TelemetryAppendResult: ...


class TelemetryRecordStore(Protocol):
    def records_for(self, episode_id: str) -> tuple[EpisodeTelemetryRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class EpisodeGraphPrerequisites:
    """Trusted parents that must exist before an episode enters graph memory."""

    world: WorldMemoryRecord
    obstacles: tuple[ObstacleMemoryRecord, ...]
    policy: EvaluatedPolicyVersion


class EpisodeGraphPrerequisiteResolver(Protocol):
    def resolve(
        self,
        identity: EpisodeIdentity,
        result: PolicyEpisodeResult,
        *,
        recorded_at: datetime,
    ) -> EpisodeGraphPrerequisites: ...


@dataclass(slots=True)
class _EpisodeSession:
    identity: EpisodeIdentity
    state: EpisodeLifecycleState
    receipts: list[EpisodeAppendReceipt]
    closure: EpisodeClosure | None = None


class EpisodeService:
    """Own the lifecycle around telemetry and post-episode graph memory.

    Graph memory is called only after a session has transitioned to ``closed``.
    It is never reachable from telemetry ingestion or any policy/control method.
    """

    def __init__(
        self,
        *,
        telemetry_backend: TelemetryBackend,
        telemetry_store: TelemetryRecordStore,
        graph_memory: GraphMemory,
        journal: EpisodeJournal | None = None,
        graph_prerequisites: EpisodeGraphPrerequisiteResolver | None = None,
    ) -> None:
        self._telemetry_backend = telemetry_backend
        self._telemetry_store = telemetry_store
        self._graph_memory = graph_memory
        self._journal = journal or VolatileEpisodeJournal()
        self._graph_prerequisites = graph_prerequisites
        self._sessions: dict[str, _EpisodeSession] = {}
        self._corrections: dict[str, CorrectionSubmission] = {}
        self._approvals: dict[str, CorrectionApproval] = {}
        self._lock = asyncio.Lock()
        self._restore_journal()

    def _restore_journal(self) -> None:
        """Validate and restore append-only lifecycle facts after a process restart."""

        for identity in self._journal.identities():
            if identity.episode_id in self._sessions:
                raise DuplicateEpisodeError(
                    f"episode {identity.episode_id!r} appears twice in the journal"
                )
            records = self._telemetry_store.records_for(identity.episode_id)
            if records:
                self._validate_record_set(identity, records)
            receipts = list(self._journal.receipts_for(identity.episode_id))
            if len(receipts) > len(records):
                raise EpisodeServiceError(
                    "episode journal has more delivery receipts than telemetry records"
                )
            for sequence, receipt in enumerate(receipts):
                if receipt.episode_id != identity.episode_id or receipt.sequence != sequence:
                    raise EpisodeServiceError(
                        "episode journal delivery receipts are not a contiguous prefix"
                    )
            closure = self._journal.closure_for(identity.episode_id)
            if closure is not None:
                if closure.identity != identity:
                    raise EpisodeIdentityError(
                        "journal closure does not match its immutable episode identity"
                    )
                if not records:
                    raise EpisodeServiceError(
                        "a journaled episode closure requires durable telemetry"
                    )
                state = EpisodeLifecycleState.CLOSED
            else:
                state = EpisodeLifecycleState.OPEN
            self._sessions[identity.episode_id] = _EpisodeSession(
                identity=identity,
                state=state,
                receipts=receipts,
                closure=closure,
            )
            self._reconcile_unjournaled_receipts(
                self._sessions[identity.episode_id],
                records,
            )

        for correction in self._journal.corrections():
            session = self._sessions.get(correction.episode_id)
            if (
                session is None
                or session.state is not EpisodeLifecycleState.CLOSED
                or session.identity.world_split is not WorldSplit.TRAINING
            ):
                raise EpisodeServiceError("journaled corrections require a closed training episode")
            if correction.correction_id in self._corrections:
                raise EpisodeServiceError("duplicate correction in episode journal")
            self._corrections[correction.correction_id] = correction

        for approval in self._journal.approvals():
            correction_id = approval.submission.correction_id
            if self._corrections.get(correction_id) != approval.submission:
                raise EpisodeServiceError(
                    "journaled approval is detached from its correction submission"
                )
            if correction_id in self._approvals:
                raise EpisodeServiceError("duplicate correction approval in episode journal")
            self._approvals[correction_id] = approval

        self._retry_incomplete_graph_deliveries()
        self._retry_incomplete_correction_deliveries()

    async def open_episode(self, identity: EpisodeIdentity) -> EpisodeIdentity:
        async with self._lock:
            if identity.episode_id in self._sessions:
                raise DuplicateEpisodeError(
                    f"episode {identity.episode_id!r} has already been opened"
                )
            if self._telemetry_store.records_for(identity.episode_id):
                raise DuplicateEpisodeError(
                    f"episode {identity.episode_id!r} already has immutable telemetry"
                )
            self._journal.record_identity(identity)
            self._sessions[identity.episode_id] = _EpisodeSession(
                identity=identity,
                state=EpisodeLifecycleState.OPEN,
                receipts=[],
            )
            return identity

    def episode_state(self, episode_id: str) -> EpisodeLifecycleState:
        return self._session(episode_id).state

    def closure_for(self, episode_id: str) -> EpisodeClosure | None:
        return self._session(episode_id).closure

    async def append_telemetry(self, record: EpisodeTelemetryRecord) -> EpisodeAppendReceipt:
        async with self._lock:
            session = self._session(record.episode_id)
            self._require_open(session)
            records = self._telemetry_store.records_for(record.episode_id)
            receipt_count = len(session.receipts)
            if (
                record.sequence < len(records)
                and records[record.sequence] == record
                and record.sequence >= receipt_count
            ):
                self._reconcile_unjournaled_receipts(session, records)
                if record.sequence < len(session.receipts):
                    return session.receipts[record.sequence]
            self._validate_append(session.identity, record, records)
            result = await self._telemetry_backend.append(record)
            persisted = self._telemetry_store.records_for(record.episode_id)
            if len(persisted) != len(records) + 1 or persisted[-1] != record:
                raise EpisodeServiceError(
                    "telemetry backend returned a receipt without the exact durable record"
                )
            receipt = EpisodeAppendReceipt(
                episode_id=record.episode_id,
                sequence=record.sequence,
                event_id=result.event_id,
                delivery=result.delivery,
                provider_state=result.provider_state,
                pending_local_records=result.pending_local_records,
            )
            self._journal.record_receipt(receipt)
            session.receipts.append(receipt)
            return receipt

    async def close_episode(
        self,
        result: PolicyEpisodeResult,
        *,
        closed_at: datetime | None = None,
    ) -> EpisodeClosure:
        async with self._lock:
            session = self._session(result.episode_id)
            self._require_open(session)
            self._validate_result_identity(session.identity, result)
            records = self._telemetry_store.records_for(result.episode_id)
            if not records:
                raise EpisodeServiceError("an episode cannot close without telemetry")
            self._validate_record_set(session.identity, records)
            self._reconcile_unjournaled_receipts(session, records)
            ended_at = closed_at or datetime.now(UTC)
            if ended_at.tzinfo is None or ended_at.utcoffset() is None:
                raise ValueError("closed_at must be timezone-aware")
            digest = telemetry_digest(records)
            failures = self._failure_records(result, records, ended_at)
            delivery = self._delivery_report(records, session.receipts)
            prerequisites = self._resolve_graph_prerequisites(
                session.identity,
                result,
                recorded_at=ended_at,
            )
            expected_graph_records = (
                1
                + len(failures)
                + (
                    0
                    if prerequisites is None
                    else 2 + len(prerequisites.obstacles)
                )
            )
            pending_graph = GraphPersistenceReport(
                expected_records=expected_graph_records,
                receipts=(),
                error_type="GraphDeliveryPending",
                detail="measured closure is durable; graph delivery has not yet been attempted",
            )
            measured_closure = EpisodeClosure(
                identity=session.identity,
                result=result,
                telemetry_digest=digest,
                telemetry=delivery,
                failures=failures,
                graph=pending_graph,
                closed_at=ended_at,
            )

            # Commit the measured terminal fact before any external graph side effect.
            session.state = EpisodeLifecycleState.CLOSED
            try:
                self._journal.record_closure(measured_closure)
            except BaseException:
                session.state = EpisodeLifecycleState.OPEN
                raise
            session.closure = measured_closure

            graph_report = self._persist_closed_episode(
                identity=session.identity,
                result=result,
                failures=failures,
                telemetry_digest_value=digest,
                ended_at=ended_at,
                prerequisites=prerequisites,
                expected_records=expected_graph_records,
            )
            self._journal.record_graph_delivery(result.episode_id, graph_report)
            closure = replace(measured_closure, graph=graph_report)
            session.closure = closure
            return closure

    async def replay_episode(self, episode_id: str) -> tuple[ReplayRecord, ...]:
        async with self._lock:
            session = self._session(episode_id)
            if session.state is not EpisodeLifecycleState.CLOSED:
                raise EpisodeServiceError("only closed episodes can be replayed")
            records = self._telemetry_store.records_for(episode_id)
            self._validate_record_set(session.identity, records)
            return tuple(
                ReplayRecord(record=record, frame_id=record.frame_id) for record in records
            )

    async def submit_route_correction(
        self,
        *,
        episode_id: str,
        failure_id: str,
        points: tuple[CorrectionPoint, ...],
        description: str,
        submitted_by: str,
        created_at: datetime | None = None,
    ) -> CorrectionSubmission:
        return await self._submit_correction(
            episode_id=episode_id,
            failure_id=failure_id,
            kind=CorrectionKind.ROUTE,
            points=points,
            description=description,
            submitted_by=submitted_by,
            created_at=created_at,
        )

    async def submit_keep_out_correction(
        self,
        *,
        episode_id: str,
        failure_id: str,
        polygon: tuple[CorrectionPoint, ...],
        description: str,
        submitted_by: str,
        created_at: datetime | None = None,
    ) -> CorrectionSubmission:
        return await self._submit_correction(
            episode_id=episode_id,
            failure_id=failure_id,
            kind=CorrectionKind.KEEP_OUT,
            points=polygon,
            description=description,
            submitted_by=submitted_by,
            created_at=created_at,
        )

    async def _submit_correction(
        self,
        *,
        episode_id: str,
        failure_id: str,
        kind: CorrectionKind,
        points: tuple[CorrectionPoint, ...],
        description: str,
        submitted_by: str,
        created_at: datetime | None,
    ) -> CorrectionSubmission:
        async with self._lock:
            session = self._session(episode_id)
            if session.state is not EpisodeLifecycleState.CLOSED or session.closure is None:
                raise EpisodeServiceError("corrections require a closed episode")
            if session.identity.world_split is not WorldSplit.TRAINING:
                raise EpisodeServiceError("held-out episodes cannot produce training corrections")
            if failure_id not in {failure.failure_id for failure in session.closure.failures}:
                raise EpisodeServiceError("correction failure_id does not belong to the episode")
            geometry_json = canonical_json(
                {"kind": kind.value, "points": [point.as_json() for point in points]}
            )
            digest = content_digest(
                {
                    "description": description.strip(),
                    "episode_id": episode_id,
                    "failure_id": failure_id,
                    "geometry_json": geometry_json,
                    "submitted_by": submitted_by.strip(),
                }
            )
            submission = CorrectionSubmission(
                correction_id=f"correction-{digest}",
                episode_id=episode_id,
                failure_id=failure_id,
                kind=kind,
                points=points,
                geometry_json=geometry_json,
                description=description.strip(),
                submitted_by=submitted_by.strip(),
                created_at=created_at or datetime.now(UTC),
            )
            existing = self._corrections.get(submission.correction_id)
            if existing is not None:
                return existing
            self._journal.record_correction(submission)
            self._corrections[submission.correction_id] = submission
            return submission

    async def approve_correction(
        self,
        correction_id: str,
        *,
        approver: AuthenticatedHuman,
        approved_at: datetime | None = None,
    ) -> CorrectionApproval:
        approver.require_authenticated()
        async with self._lock:
            submission = self._corrections.get(correction_id)
            if submission is None:
                raise CorrectionNotFoundError(f"correction {correction_id!r} was not found")
            if correction_id in self._approvals:
                raise CorrectionAlreadyApprovedError(
                    f"correction {correction_id!r} has already been approved"
                )
            when = approved_at or datetime.now(UTC)
            approval_digest = content_digest(
                {
                    "approved_at": when.isoformat(),
                    "approved_by": approver.subject_id,
                    "authentication_method": approver.authentication_method,
                    "correction_id": correction_id,
                }
            )
            approval = CorrectionApproval(
                approval_id=f"approval-{approval_digest}",
                submission=submission,
                approved_by=approver.subject_id,
                authentication_method=approver.authentication_method,
                approved_at=when,
                graph_receipt=None,
            )
            # Store the authenticated approval before any fallible provider write.
            self._journal.record_approval(approval)
            self._approvals[correction_id] = approval
            approval = self._deliver_correction(approval)
            self._journal.record_approval_delivery(approval)
            self._approvals[correction_id] = approval
            return approval

    def _approved_corrections_for_training(
        self, episode_id: str | None = None
    ) -> tuple[TrainingCorrection, ...]:
        """Internal projection consumed only by ``episodes.training``."""

        approved = (
            approval
            for approval in self._approvals.values()
            if episode_id is None or approval.submission.episode_id == episode_id
        )
        return tuple(
            TrainingCorrection(
                correction_id=approval.submission.correction_id,
                episode_id=approval.submission.episode_id,
                failure_id=approval.submission.failure_id,
                kind=approval.submission.kind,
                points=approval.submission.points,
                description=approval.submission.description,
                approved_by=approval.approved_by,
                approved_at=approval.approved_at,
            )
            for approval in sorted(approved, key=lambda item: item.approval_id)
        )

    def _session(self, episode_id: str) -> _EpisodeSession:
        try:
            return self._sessions[episode_id]
        except KeyError as exc:
            raise EpisodeNotFoundError(f"episode {episode_id!r} was not opened") from exc

    @staticmethod
    def _require_open(session: _EpisodeSession) -> None:
        if session.state is EpisodeLifecycleState.CLOSED:
            raise EpisodeClosedError(f"episode {session.identity.episode_id!r} is already closed")

    @staticmethod
    def _validate_append(
        identity: EpisodeIdentity,
        record: EpisodeTelemetryRecord,
        records: tuple[EpisodeTelemetryRecord, ...],
    ) -> None:
        record.verify_integrity()
        EpisodeService._validate_record_identity(identity, record)
        expected_sequence = len(records)
        if record.sequence < expected_sequence:
            existing = records[record.sequence]
            if existing == record:
                raise DuplicateTelemetryRecordError(
                    f"episode {record.episode_id!r} sequence {record.sequence} already exists"
                )
            raise TelemetryMutationError(
                f"episode {record.episode_id!r} sequence {record.sequence} is immutable"
            )
        if record.sequence > expected_sequence:
            raise OutOfOrderTelemetryError(
                f"expected sequence {expected_sequence}, received {record.sequence}"
            )
        expected_time = record.sequence / NUMERIC_TELEMETRY_HZ
        if not math.isclose(
            record.sim_time_seconds,
            expected_time,
            rel_tol=0.0,
            abs_tol=_CADENCE_TOLERANCE_SECONDS,
        ):
            raise TelemetryCadenceError(
                f"sequence {record.sequence} must be sampled at {expected_time:.6f}s"
            )

    @staticmethod
    def _validate_record_identity(
        identity: EpisodeIdentity, record: EpisodeTelemetryRecord
    ) -> None:
        actual = (
            record.episode_id,
            record.world_id,
            record.policy_id,
            record.robot_checksum,
            record.world_hash,
            record.policy_hash,
        )
        expected = (
            identity.episode_id,
            identity.world_id,
            identity.policy_id,
            identity.robot_checksum,
            identity.world_hash,
            identity.policy_hash,
        )
        if actual != expected:
            raise EpisodeIdentityError(
                "telemetry cannot mutate the episode robot, world, or policy identity"
            )

    @staticmethod
    def _validate_record_set(
        identity: EpisodeIdentity, records: tuple[EpisodeTelemetryRecord, ...]
    ) -> None:
        for sequence, record in enumerate(records):
            EpisodeService._validate_record_identity(identity, record)
            if record.sequence != sequence:
                raise EpisodeServiceError("replay telemetry sequence is not contiguous")
            expected_time = sequence / NUMERIC_TELEMETRY_HZ
            if not math.isclose(
                record.sim_time_seconds,
                expected_time,
                rel_tol=0.0,
                abs_tol=_CADENCE_TOLERANCE_SECONDS,
            ):
                raise EpisodeServiceError("replay telemetry violates the 20 Hz clock")
            record.verify_integrity()

    @staticmethod
    def _validate_result_identity(identity: EpisodeIdentity, result: PolicyEpisodeResult) -> None:
        actual = (
            result.episode_id,
            result.robot_checksum,
            result.world_id,
            result.world_hash,
            result.world_split,
            result.policy_id,
            result.policy_hash,
        )
        expected = (
            identity.episode_id,
            identity.robot_checksum,
            identity.world_id,
            identity.world_hash,
            identity.world_split.value,
            identity.policy_id,
            identity.policy_hash,
        )
        if actual != expected:
            raise EpisodeIdentityError("measured result does not match the opened episode identity")

    @staticmethod
    def _delivery_report(
        records: tuple[EpisodeTelemetryRecord, ...],
        receipts: list[EpisodeAppendReceipt],
    ) -> TelemetryDeliveryReport:
        provider_count = sum(
            receipt.delivery is TelemetryDelivery.LASERDATA_AND_DURABLE_CACHE
            for receipt in receipts
        )
        local_count = sum(
            receipt.delivery is TelemetryDelivery.DURABLE_LOCAL_CACHE_ONLY for receipt in receipts
        )
        pending = receipts[-1].pending_local_records if receipts else len(records)
        return TelemetryDeliveryReport(
            total_records=len(records),
            provider_confirmed_records=provider_count,
            local_only_records=local_count,
            records_without_receipts=max(0, len(records) - len(receipts)),
            pending_local_records=pending,
            provider_states=tuple(dict.fromkeys(receipt.provider_state for receipt in receipts)),
        )

    @staticmethod
    def _failure_records(
        result: PolicyEpisodeResult,
        records: tuple[EpisodeTelemetryRecord, ...],
        detected_at: datetime,
    ) -> tuple[FailureMemoryRecord, ...]:
        event_failures = tuple(
            record.failure_type for record in records if record.failure_type is not None
        )
        if result.success and event_failures:
            raise EpisodeServiceError(
                "a successful result cannot discard typed telemetry failure events"
            )
        failures: list[FailureMemoryRecord] = []
        for reason in dict.fromkeys((*result.failed_reasons, *event_failures)):
            category = reason.lower()
            failure_digest = content_digest(
                {
                    "category": category,
                    "episode_id": result.episode_id,
                    "minimum_clearance_m": result.minimum_obstacle_clearance_m,
                }
            )
            failures.append(
                FailureMemoryRecord(
                    failure_id=f"failure-{failure_digest}",
                    episode_id=result.episode_id,
                    category=category,
                    obstacle_id=None,
                    severity=_failure_severity(category, result),
                    summary=reason.replace("_", " ").lower(),
                    detected_at=detected_at,
                )
            )
        return tuple(failures)

    def _persist_closed_episode(
        self,
        *,
        identity: EpisodeIdentity,
        result: PolicyEpisodeResult,
        failures: tuple[FailureMemoryRecord, ...],
        telemetry_digest_value: str,
        ended_at: datetime,
        prerequisites: EpisodeGraphPrerequisites | None,
        expected_records: int,
    ) -> GraphPersistenceReport:
        receipts: list[GraphWriteReceipt] = []
        try:
            if prerequisites is not None:
                receipts.append(self._graph_memory.record_world(prerequisites.world))
                for obstacle in prerequisites.obstacles:
                    receipts.append(self._graph_memory.record_obstacle(obstacle))
                receipts.append(
                    self._graph_memory.record_evaluated_policy(prerequisites.policy)
                )
            episode_record = EpisodeMemoryRecord(
                episode_id=identity.episode_id,
                robot_checksum=identity.robot_checksum,
                world_id=identity.world_id,
                world_hash=identity.world_hash,
                world_split=identity.world_split,
                policy_id=identity.policy_id,
                policy_hash=identity.policy_hash,
                outcome=EpisodeOutcome.SUCCEEDED if result.success else EpisodeOutcome.FAILED,
                completion_time_seconds=result.simulated_duration_seconds,
                collision_count=result.body_collisions,
                fall_count=result.falls,
                minimum_clearance_m=result.minimum_obstacle_clearance_m,
                human_interventions=result.human_interventions,
                telemetry_digest=telemetry_digest_value,
                ended_at=ended_at,
            )
            receipts.append(self._graph_memory.record_episode(episode_record))
            for failure in failures:
                receipts.append(self._graph_memory.record_failure(failure))
        except Exception as exc:
            return GraphPersistenceReport(
                expected_records=expected_records,
                receipts=tuple(receipts),
                error_type=type(exc).__name__,
                detail="closed episode graph persistence was only partially completed",
            )
        return GraphPersistenceReport(
            expected_records=expected_records,
            receipts=tuple(receipts),
        )

    def _resolve_graph_prerequisites(
        self,
        identity: EpisodeIdentity,
        result: PolicyEpisodeResult,
        *,
        recorded_at: datetime,
    ) -> EpisodeGraphPrerequisites | None:
        resolver = self._graph_prerequisites
        if resolver is None:
            return None
        prerequisites = resolver.resolve(
            identity,
            result,
            recorded_at=recorded_at,
        )
        if (
            prerequisites.world.world_id != identity.world_id
            or prerequisites.world.world_hash != identity.world_hash
            or prerequisites.world.split is not identity.world_split
            or prerequisites.policy.policy_id != identity.policy_id
            or prerequisites.policy.checkpoint_hash != identity.policy_hash
            or any(obstacle.world_id != identity.world_id for obstacle in prerequisites.obstacles)
        ):
            raise EpisodeIdentityError(
                "trusted graph prerequisites do not match the episode identity"
            )
        return prerequisites

    def _retry_incomplete_graph_deliveries(self) -> None:
        for session in self._sessions.values():
            closure = session.closure
            if closure is None or closure.graph.complete:
                continue
            prerequisites = self._resolve_graph_prerequisites(
                closure.identity,
                closure.result,
                recorded_at=closure.closed_at,
            )
            expected = 1 + len(closure.failures) + (
                0 if prerequisites is None else 2 + len(prerequisites.obstacles)
            )
            if expected != closure.graph.expected_records:
                raise EpisodeServiceError(
                    "retry graph prerequisites changed after measured episode closure"
                )
            report = self._persist_closed_episode(
                identity=closure.identity,
                result=closure.result,
                failures=closure.failures,
                telemetry_digest_value=closure.telemetry_digest,
                ended_at=closure.closed_at,
                prerequisites=prerequisites,
                expected_records=expected,
            )
            self._journal.record_graph_delivery(closure.identity.episode_id, report)
            session.closure = replace(closure, graph=report)

    def _retry_incomplete_correction_deliveries(self) -> None:
        """Finish approvals interrupted between graph write and delivery journal."""

        for correction_id, approval in tuple(self._approvals.items()):
            if approval.graph_receipt is not None or approval.graph_error_type is not None:
                continue
            delivered = self._deliver_correction(approval)
            self._journal.record_approval_delivery(delivered)
            self._approvals[correction_id] = delivered

    def _deliver_correction(self, approval: CorrectionApproval) -> CorrectionApproval:
        submission = approval.submission
        graph_payload = canonical_json(
            {
                "description": submission.description,
                "geometry": submission.geometry_json,
            }
        )
        try:
            graph_receipt = self._graph_memory.record_correction(
                CorrectionMemoryRecord(
                    correction_id=submission.correction_id,
                    failure_id=submission.failure_id,
                    kind=submission.kind.value,
                    description=graph_payload,
                    approved=True,
                    approved_by=approval.approved_by,
                    approved_at=approval.approved_at,
                    created_at=submission.created_at,
                )
            )
        except Exception as exc:
            return replace(
                approval,
                graph_error_type=type(exc).__name__,
                graph_detail="approved correction was not persisted to graph memory",
            )
        return replace(approval, graph_receipt=graph_receipt)

    def _reconcile_unjournaled_receipts(
        self,
        session: _EpisodeSession,
        records: tuple[EpisodeTelemetryRecord, ...],
    ) -> None:
        """Fill a durable receipt prefix interrupted after telemetry persistence."""

        recover = getattr(self._telemetry_backend, "recover_append_result", None)
        if not callable(recover):
            return
        for record in records[len(session.receipts) :]:
            result = recover(record)
            expected_event_id = LaserDataTelemetryEnvelope.from_domain(record).event_id
            if result.event_id != expected_event_id:
                raise EpisodeServiceError(
                    "recovered telemetry receipt does not match its durable event"
                )
            receipt = EpisodeAppendReceipt(
                episode_id=record.episode_id,
                sequence=record.sequence,
                event_id=result.event_id,
                delivery=result.delivery,
                provider_state=result.provider_state,
                pending_local_records=result.pending_local_records,
            )
            self._journal.record_receipt(receipt)
            session.receipts.append(receipt)


def telemetry_digest(records: tuple[EpisodeTelemetryRecord, ...]) -> str:
    """Hash exact ordered provider envelopes without mutating any record."""

    digest = hashlib.sha256()
    for record in records:
        envelope = LaserDataTelemetryEnvelope.from_domain(record)
        encoded = envelope.canonical_json().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _failure_severity(category: str, result: PolicyEpisodeResult) -> float:
    if category in {"fell", "body_collision", "package_slipped"}:
        return 1.0
    if category == "insufficient_obstacle_clearance":
        return min(1.0, max(0.25, (0.25 - result.minimum_obstacle_clearance_m) / 0.25))
    if category == "excessive_tray_tilt":
        return 0.9
    return 0.7
