"""SQLite-backed immutable coordinator domain store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock

from muscle_memory.coordinator.models import (
    TERMINAL_EPISODE_STATES,
    ApprovalRequiredError,
    CoordinatorIntegrityError,
    CoordinatorStateError,
    EpisodeKind,
    EpisodeState,
    EpisodeTransition,
    HeldOutEvaluationEpisodeMetadata,
    NumericPolicyDecision,
    PolicyAction,
    PolicyAliasEvent,
    PolicyGateMetrics,
    ProviderEvidenceReference,
    TrainingEpisodeMetadata,
    WorkflowStepAudit,
    WorkflowStepState,
    canonical_json,
    isoformat_utc,
    sha256_text,
)
from muscle_memory.graph_memory.models import EvaluatedPolicyVersion
from muscle_memory.orchestration.approvals import HumanDecision, HumanVerdict
from muscle_memory.orchestration.contracts import (
    FIXED_PIPELINE,
    ApprovalKind,
    ApprovalRequirement,
    ExecutionPlan,
    PipelineCommand,
    PipelineStep,
)
from muscle_memory.orchestration.evidence import (
    EvaluationEvidence,
    GuildEvidenceBundle,
    validate_evidence_plan_binding,
)

_IMMUTABLE_TABLES = (
    "episodes",
    "held_out_episode_scopes",
    "training_episode_sessions",
    "training_episode_receipts",
    "training_episode_closures",
    "training_correction_submissions",
    "training_correction_approvals",
    "training_correction_graph_deliveries",
    "training_correction_rejections",
    "episode_transitions",
    "provider_evidence",
    "workflow_runs",
    "workflow_guild_evidence",
    "workflow_reviews",
    "workflow_run_snapshots",
    "approval_requirements",
    "human_decisions",
    "workflow_step_audits",
    "rocketride_callback_results",
    "evaluated_checkpoints",
    "numeric_policy_decisions",
    "policy_alias_events",
)

_ALLOWED_EPISODE_TRANSITIONS = {
    EpisodeState.CREATED: frozenset({EpisodeState.RUNNING, EpisodeState.ABORTED}),
    EpisodeState.RUNNING: TERMINAL_EPISODE_STATES,
}


class CoordinatorStore:
    """Durable facts and append-only state transitions owned by the coordinator."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._configure()

    def _configure(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('training', 'held_out_evaluation')),
                    robot_checksum TEXT NOT NULL,
                    world_hash TEXT NOT NULL,
                    policy_hash TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS held_out_episode_scopes (
                    episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
                    held_out_world_set_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_episode_sessions (
                    episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
                    identity_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS training_episode_receipts (
                    episode_id TEXT NOT NULL REFERENCES training_episode_sessions(episode_id),
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    receipt_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (episode_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS training_episode_closures (
                    episode_id TEXT PRIMARY KEY REFERENCES training_episode_sessions(episode_id),
                    closure_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS training_correction_submissions (
                    correction_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL REFERENCES training_episode_closures(episode_id),
                    submission_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS training_correction_approvals (
                    correction_id TEXT PRIMARY KEY
                        REFERENCES training_correction_submissions(correction_id),
                    approval_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS training_correction_graph_deliveries (
                    correction_id TEXT PRIMARY KEY
                        REFERENCES training_correction_approvals(correction_id),
                    delivery_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS training_correction_rejections (
                    correction_id TEXT PRIMARY KEY
                        REFERENCES training_correction_submissions(correction_id),
                    rejection_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS provider_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL,
                    provider_object_id TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    observed_at_utc TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS episode_transitions (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    state TEXT NOT NULL CHECK (
                        state IN ('created', 'running', 'succeeded', 'failed', 'aborted')
                    ),
                    occurred_at_utc TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    evidence_id TEXT REFERENCES provider_evidence(evidence_id),
                    content_hash TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (episode_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    plan_digest TEXT NOT NULL UNIQUE,
                    plan_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS workflow_guild_evidence (
                    run_id TEXT PRIMARY KEY REFERENCES workflow_runs(run_id),
                    evidence_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS workflow_reviews (
                    run_id TEXT PRIMARY KEY REFERENCES workflow_runs(run_id),
                    review_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS workflow_run_snapshots (
                    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    run_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (run_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS approval_requirements (
                    requirement_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
                    plan_digest TEXT NOT NULL,
                    step TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS human_decisions (
                    requirement_id TEXT PRIMARY KEY
                        REFERENCES approval_requirements(requirement_id),
                    plan_digest TEXT NOT NULL,
                    human_subject TEXT NOT NULL,
                    verdict TEXT NOT NULL CHECK (verdict IN ('approve', 'reject')),
                    decided_at_utc TEXT NOT NULL,
                    note TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS workflow_step_audits (
                    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    step TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN (
                            'awaiting_approval', 'started', 'completed', 'failed', 'blocked'
                        )
                    ),
                    occurred_at_utc TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    evidence_id TEXT REFERENCES provider_evidence(evidence_id),
                    content_hash TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (run_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS rocketride_callback_results (
                    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
                    step TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (run_id, step),
                    UNIQUE (run_id, request_sha256)
                );

                CREATE TABLE IF NOT EXISTS evaluated_checkpoints (
                    policy_id TEXT PRIMARY KEY,
                    checkpoint_hash TEXT NOT NULL UNIQUE,
                    record_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS numeric_policy_decisions (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
                    plan_digest TEXT NOT NULL,
                    action TEXT NOT NULL CHECK (action IN ('promote', 'roll_back')),
                    alias TEXT NOT NULL,
                    from_policy_id TEXT REFERENCES evaluated_checkpoints(policy_id),
                    target_policy_id TEXT NOT NULL REFERENCES evaluated_checkpoints(policy_id),
                    evaluation_evidence_hash TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    promotion_gate_passed INTEGER NOT NULL CHECK (
                        promotion_gate_passed IN (0, 1)
                    ),
                    decided_at_utc TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS policy_alias_events (
                    alias TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    target_policy_id TEXT NOT NULL REFERENCES evaluated_checkpoints(policy_id),
                    action TEXT NOT NULL CHECK (action IN ('initialize', 'promote', 'roll_back')),
                    occurred_at_utc TEXT NOT NULL,
                    numeric_decision_id TEXT UNIQUE
                        REFERENCES numeric_policy_decisions(decision_id),
                    approval_requirement_id TEXT
                        REFERENCES human_decisions(requirement_id),
                    content_hash TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (alias, sequence),
                    CHECK (
                        (action = 'initialize' AND numeric_decision_id IS NULL
                            AND approval_requirement_id IS NULL)
                        OR
                        (action IN ('promote', 'roll_back') AND numeric_decision_id IS NOT NULL
                            AND approval_requirement_id IS NOT NULL)
                    )
                );
                """
            )
            for table in _IMMUTABLE_TABLES:
                self._connection.executescript(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} history is immutable');
                    END;

                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} history is append-only');
                    END;
                    """
                )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def register_training_episode(
        self,
        metadata: TrainingEpisodeMetadata,
    ) -> TrainingEpisodeMetadata:
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?",
                (metadata.episode_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["kind"]) != EpisodeKind.TRAINING.value
                    or str(existing["content_hash"]) != metadata.content_hash
                ):
                    raise CoordinatorIntegrityError(f"episode {metadata.episode_id!r} is immutable")
                return self._training_episode_from_row(existing)
            self._insert_episode(
                connection,
                metadata.episode_id,
                EpisodeKind.TRAINING,
                metadata.robot_checksum,
                metadata.world_hash,
                metadata.policy_hash,
                metadata.created_at,
                metadata.content_hash,
            )
            self._insert_initial_episode_transition(
                connection,
                metadata.episode_id,
                metadata.created_at,
            )
        return metadata

    def register_held_out_evaluation_episode(
        self,
        metadata: HeldOutEvaluationEpisodeMetadata,
    ) -> HeldOutEvaluationEpisodeMetadata:
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT episodes.*, scopes.held_out_world_set_id
                FROM episodes
                LEFT JOIN held_out_episode_scopes AS scopes
                    ON scopes.episode_id = episodes.episode_id
                WHERE episodes.episode_id = ?
                """,
                (metadata.episode_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["kind"]) != EpisodeKind.HELD_OUT_EVALUATION.value
                    or str(existing["content_hash"]) != metadata.content_hash
                    or str(existing["held_out_world_set_id"]) != metadata.held_out_world_set_id
                ):
                    raise CoordinatorIntegrityError(f"episode {metadata.episode_id!r} is immutable")
                return self._held_out_episode_from_row(existing)
            self._insert_episode(
                connection,
                metadata.episode_id,
                EpisodeKind.HELD_OUT_EVALUATION,
                metadata.robot_checksum,
                metadata.world_hash,
                metadata.policy_hash,
                metadata.created_at,
                metadata.content_hash,
            )
            connection.execute(
                """
                INSERT INTO held_out_episode_scopes (
                    episode_id, held_out_world_set_id
                ) VALUES (?, ?)
                """,
                (metadata.episode_id, metadata.held_out_world_set_id),
            )
            self._insert_initial_episode_transition(
                connection,
                metadata.episode_id,
                metadata.created_at,
            )
        return metadata

    @staticmethod
    def _insert_episode(
        connection: sqlite3.Connection,
        episode_id: str,
        kind: EpisodeKind,
        robot_checksum: str,
        world_hash: str,
        policy_hash: str,
        created_at: datetime,
        content_hash: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, kind, robot_checksum, world_hash, policy_hash,
                created_at_utc, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                kind.value,
                robot_checksum,
                world_hash,
                policy_hash,
                isoformat_utc(created_at),
                content_hash,
            ),
        )

    def _insert_initial_episode_transition(
        self,
        connection: sqlite3.Connection,
        episode_id: str,
        occurred_at: datetime,
    ) -> None:
        transition = self._episode_transition(
            episode_id=episode_id,
            sequence=0,
            state=EpisodeState.CREATED,
            occurred_at=occurred_at,
            details_json="{}",
            evidence_id=None,
        )
        self._insert_episode_transition(connection, transition)

    def training_episode(self, episode_id: str) -> TrainingEpisodeMetadata | None:
        """Return only training metadata; evaluation identifiers cannot cross this API."""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ? AND kind = ?",
                (episode_id, EpisodeKind.TRAINING.value),
            ).fetchone()
        return None if row is None else self._training_episode_from_row(row)

    def training_episodes(self) -> tuple[TrainingEpisodeMetadata, ...]:
        """List only training episodes, never held-out episode or world-set identifiers."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM episodes WHERE kind = ? ORDER BY episode_id",
                (EpisodeKind.TRAINING.value,),
            ).fetchall()
        return tuple(self._training_episode_from_row(row) for row in rows)

    def record_training_episode_session(
        self,
        episode_id: str,
        identity_json: str,
    ) -> None:
        """Persist an exact training identity without exposing held-out storage."""

        content_hash = self._canonical_payload_hash(identity_json)
        with self._transaction() as connection:
            self._require_training_episode(connection, episode_id)
            self._insert_immutable_payload(
                connection,
                table="training_episode_sessions",
                identity_column="episode_id",
                identity=episode_id,
                payload_column="identity_json",
                payload_json=identity_json,
                content_hash=content_hash,
            )

    def training_episode_sessions(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT identity_json FROM training_episode_sessions ORDER BY episode_id"
            ).fetchall()
        return tuple(str(row["identity_json"]) for row in rows)

    def record_training_episode_receipt(
        self,
        episode_id: str,
        sequence: int,
        receipt_json: str,
    ) -> None:
        if sequence < 0:
            raise ValueError("training episode receipt sequence must be non-negative")
        content_hash = self._canonical_payload_hash(receipt_json)
        with self._transaction() as connection:
            self._require_training_session(connection, episode_id)
            existing = connection.execute(
                """
                SELECT * FROM training_episode_receipts
                WHERE episode_id = ? AND sequence = ?
                """,
                (episode_id, sequence),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise CoordinatorIntegrityError(
                        "training episode delivery receipt is immutable"
                    )
                return
            expected = connection.execute(
                """
                SELECT COUNT(*) AS count FROM training_episode_receipts
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
            if expected is None or int(expected["count"]) != sequence:
                raise CoordinatorStateError("training episode delivery receipts must be contiguous")
            connection.execute(
                """
                INSERT INTO training_episode_receipts (
                    episode_id, sequence, receipt_json, content_hash
                ) VALUES (?, ?, ?, ?)
                """,
                (episode_id, sequence, receipt_json, content_hash),
            )

    def training_episode_receipts(self, episode_id: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT receipt_json FROM training_episode_receipts
                WHERE episode_id = ? ORDER BY sequence
                """,
                (episode_id,),
            ).fetchall()
        return tuple(str(row["receipt_json"]) for row in rows)

    def record_training_episode_closure(
        self,
        episode_id: str,
        closure_json: str,
    ) -> None:
        content_hash = self._canonical_payload_hash(closure_json)
        with self._transaction() as connection:
            self._require_training_session(connection, episode_id)
            self._insert_immutable_payload(
                connection,
                table="training_episode_closures",
                identity_column="episode_id",
                identity=episode_id,
                payload_column="closure_json",
                payload_json=closure_json,
                content_hash=content_hash,
            )

    def training_episode_closure(self, episode_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT closure_json FROM training_episode_closures
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
        return None if row is None else str(row["closure_json"])

    def record_training_correction_submission(
        self,
        correction_id: str,
        episode_id: str,
        submission_json: str,
    ) -> None:
        content_hash = self._canonical_payload_hash(submission_json)
        with self._transaction() as connection:
            self._require_training_closure(connection, episode_id)
            existing = connection.execute(
                """
                SELECT * FROM training_correction_submissions
                WHERE correction_id = ?
                """,
                (correction_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["episode_id"]) != episode_id
                    or str(existing["content_hash"]) != content_hash
                ):
                    raise CoordinatorIntegrityError("training correction is immutable")
                return
            connection.execute(
                """
                INSERT INTO training_correction_submissions (
                    correction_id, episode_id, submission_json, content_hash
                ) VALUES (?, ?, ?, ?)
                """,
                (correction_id, episode_id, submission_json, content_hash),
            )

    def training_correction_submissions(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT submission_json FROM training_correction_submissions
                ORDER BY correction_id
                """
            ).fetchall()
        return tuple(str(row["submission_json"]) for row in rows)

    def record_training_correction_approval(
        self,
        correction_id: str,
        approval_json: str,
    ) -> None:
        content_hash = self._canonical_payload_hash(approval_json)
        with self._transaction() as connection:
            rejection = connection.execute(
                """
                SELECT 1 FROM training_correction_rejections
                WHERE correction_id = ?
                """,
                (correction_id,),
            ).fetchone()
            if rejection is not None:
                raise CoordinatorStateError("a rejected correction cannot be approved")
            row = connection.execute(
                """
                SELECT episode_id FROM training_correction_submissions
                WHERE correction_id = ?
                """,
                (correction_id,),
            ).fetchone()
            if row is None:
                raise CoordinatorStateError("cannot approve an unknown training correction")
            self._insert_immutable_payload(
                connection,
                table="training_correction_approvals",
                identity_column="correction_id",
                identity=correction_id,
                payload_column="approval_json",
                payload_json=approval_json,
                content_hash=content_hash,
            )

    def training_correction_approvals(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT approval_json FROM training_correction_approvals
                ORDER BY correction_id
                """
            ).fetchall()
        return tuple(str(row["approval_json"]) for row in rows)

    def record_training_correction_graph_delivery(
        self,
        correction_id: str,
        delivery_json: str,
    ) -> None:
        content_hash = self._canonical_payload_hash(delivery_json)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM training_correction_approvals
                WHERE correction_id = ?
                """,
                (correction_id,),
            ).fetchone()
            if row is None:
                raise CoordinatorStateError(
                    "correction graph delivery requires a durable approval"
                )
            self._insert_immutable_payload(
                connection,
                table="training_correction_graph_deliveries",
                identity_column="correction_id",
                identity=correction_id,
                payload_column="delivery_json",
                payload_json=delivery_json,
                content_hash=content_hash,
            )

    def training_correction_graph_deliveries(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT delivery_json FROM training_correction_graph_deliveries
                ORDER BY correction_id
                """
            ).fetchall()
        return tuple(str(row["delivery_json"]) for row in rows)

    def record_training_correction_rejection(
        self,
        correction_id: str,
        rejection_json: str,
    ) -> None:
        content_hash = self._canonical_payload_hash(rejection_json)
        with self._transaction() as connection:
            approval = connection.execute(
                """
                SELECT 1 FROM training_correction_approvals
                WHERE correction_id = ?
                """,
                (correction_id,),
            ).fetchone()
            if approval is not None:
                raise CoordinatorStateError("an approved correction cannot be rejected")
            row = connection.execute(
                """
                SELECT 1 FROM training_correction_submissions
                WHERE correction_id = ?
                """,
                (correction_id,),
            ).fetchone()
            if row is None:
                raise CoordinatorStateError("cannot reject an unknown training correction")
            self._insert_immutable_payload(
                connection,
                table="training_correction_rejections",
                identity_column="correction_id",
                identity=correction_id,
                payload_column="rejection_json",
                payload_json=rejection_json,
                content_hash=content_hash,
            )

    def training_correction_rejections(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT rejection_json FROM training_correction_rejections
                ORDER BY correction_id
                """
            ).fetchall()
        return tuple(str(row["rejection_json"]) for row in rows)

    def held_out_evaluation_episode(
        self,
        episode_id: str,
    ) -> HeldOutEvaluationEpisodeMetadata | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT episodes.*, scopes.held_out_world_set_id
                FROM episodes
                JOIN held_out_episode_scopes AS scopes
                    ON scopes.episode_id = episodes.episode_id
                WHERE episodes.episode_id = ? AND episodes.kind = ?
                """,
                (episode_id, EpisodeKind.HELD_OUT_EVALUATION.value),
            ).fetchone()
        return None if row is None else self._held_out_episode_from_row(row)

    def transition_episode(
        self,
        episode_id: str,
        state: EpisodeState,
        *,
        occurred_at: datetime,
        details: Mapping[str, object] | None = None,
        evidence_id: str | None = None,
    ) -> EpisodeTransition:
        details_json = canonical_json({} if details is None else details)
        with self._transaction() as connection:
            self._require_episode(connection, episode_id)
            latest_row = connection.execute(
                """
                SELECT * FROM episode_transitions
                WHERE episode_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            if latest_row is None:
                raise CoordinatorIntegrityError("episode is missing its initial transition")
            latest = self._episode_transition_from_row(latest_row)
            if latest.state is state:
                candidate = self._episode_transition(
                    episode_id=episode_id,
                    sequence=latest.sequence,
                    state=state,
                    occurred_at=occurred_at,
                    details_json=details_json,
                    evidence_id=evidence_id,
                )
                if candidate.content_hash == latest.content_hash:
                    return latest
                raise CoordinatorStateError(
                    f"episode {episode_id!r} already entered {state.value!r}"
                )
            allowed = _ALLOWED_EPISODE_TRANSITIONS.get(latest.state, frozenset())
            if state not in allowed:
                raise CoordinatorStateError(
                    f"cannot transition episode from {latest.state.value!r} to {state.value!r}"
                )
            if evidence_id is not None:
                self._require_evidence(connection, evidence_id)
            transition = self._episode_transition(
                episode_id=episode_id,
                sequence=latest.sequence + 1,
                state=state,
                occurred_at=occurred_at,
                details_json=details_json,
                evidence_id=evidence_id,
            )
            self._insert_episode_transition(connection, transition)
            return transition

    def episode_history(self, episode_id: str) -> tuple[EpisodeTransition, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM episode_transitions
                WHERE episode_id = ?
                ORDER BY sequence
                """,
                (episode_id,),
            ).fetchall()
        return tuple(self._episode_transition_from_row(row) for row in rows)

    def episode_state(self, episode_id: str) -> EpisodeState:
        history = self.episode_history(episode_id)
        if not history:
            raise KeyError(episode_id)
        return history[-1].state

    def record_provider_evidence(
        self,
        evidence: ProviderEvidenceReference,
    ) -> ProviderEvidenceReference:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM provider_evidence WHERE evidence_id = ?",
                (evidence.evidence_id,),
            ).fetchone()
            if row is not None:
                if str(row["content_hash"]) != evidence.content_hash:
                    raise CoordinatorIntegrityError(
                        f"provider evidence {evidence.evidence_id!r} is immutable"
                    )
                return self._provider_evidence_from_row(row)
            connection.execute(
                """
                INSERT INTO provider_evidence (
                    evidence_id, provider, evidence_kind, provider_object_id,
                    artifact_hash, observed_at_utc, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.provider,
                    evidence.evidence_kind,
                    evidence.provider_object_id,
                    evidence.artifact_hash,
                    isoformat_utc(evidence.observed_at),
                    evidence.content_hash,
                ),
            )
        return evidence

    def provider_evidence(self, evidence_id: str) -> ProviderEvidenceReference | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM provider_evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        return None if row is None else self._provider_evidence_from_row(row)

    def register_workflow(
        self,
        plan: ExecutionPlan,
        *,
        created_at: datetime,
    ) -> str:
        plan_json = self._plan_json(plan)
        content_hash = sha256_text(
            canonical_json(
                {
                    "plan": json.loads(plan_json),
                    "created_at": isoformat_utc(created_at),
                }
            )
        )
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?",
                (plan.run_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["plan_digest"]) != plan.digest
                    or str(existing["content_hash"]) != content_hash
                ):
                    raise CoordinatorIntegrityError(f"workflow {plan.run_id!r} is immutable")
                return plan.digest
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    run_id, plan_digest, plan_json, created_at_utc, content_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    plan.run_id,
                    plan.digest,
                    plan_json,
                    isoformat_utc(created_at),
                    content_hash,
                ),
            )
            for requirement in plan.approval_requirements:
                self._insert_approval_requirement(connection, requirement)
        return plan.digest

    def workflow_plan(self, run_id: str) -> ExecutionPlan | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT plan_json FROM workflow_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else self._execution_plan(str(row["plan_json"]))

    def workflow_plans(self) -> tuple[ExecutionPlan, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT plan_json FROM workflow_runs ORDER BY run_id"
            ).fetchall()
        return tuple(self._execution_plan(str(row["plan_json"])) for row in rows)

    def record_workflow_guild_evidence(
        self,
        run_id: str,
        bundle: GuildEvidenceBundle,
    ) -> GuildEvidenceBundle:
        evidence_json = canonical_json(bundle.model_dump(mode="json"))
        content_hash = sha256_text(evidence_json)
        with self._transaction() as connection:
            row = self._workflow_row(connection, run_id)
            if row is None:
                raise KeyError(run_id)
            plan = self._execution_plan(str(row["plan_json"]))
            validate_evidence_plan_binding(bundle, plan)
            for evidence_id, expected_kind, expected_hash in bundle.artifact_hashes():
                evidence = connection.execute(
                    "SELECT * FROM provider_evidence WHERE evidence_id = ?",
                    (evidence_id,),
                ).fetchone()
                if evidence is None:
                    raise CoordinatorIntegrityError(
                        f"Guild evidence reference {evidence_id!r} is not registered"
                    )
                if (
                    str(evidence["evidence_kind"]) != expected_kind
                    or str(evidence["artifact_hash"]) != expected_hash
                ):
                    raise CoordinatorIntegrityError(
                        f"Guild evidence reference {evidence_id!r} does not match its artifact"
                    )
            self._insert_immutable_payload(
                connection,
                table="workflow_guild_evidence",
                identity_column="run_id",
                identity=run_id,
                payload_column="evidence_json",
                payload_json=evidence_json,
                content_hash=content_hash,
            )
        return bundle

    def workflow_guild_evidence(self, run_id: str) -> GuildEvidenceBundle | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT evidence_json FROM workflow_guild_evidence WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        bundle = GuildEvidenceBundle.model_validate_json(str(row["evidence_json"]))
        plan = self.workflow_plan(run_id)
        if plan is None:
            raise CoordinatorIntegrityError("Guild evidence is detached from its plan")
        validate_evidence_plan_binding(bundle, plan)
        return bundle

    def pending_approval_requirements(
        self,
    ) -> tuple[tuple[ApprovalRequirement, datetime], ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT requirements.*, runs.created_at_utc
                FROM approval_requirements AS requirements
                JOIN workflow_runs AS runs ON runs.run_id = requirements.run_id
                LEFT JOIN human_decisions AS decisions
                    ON decisions.requirement_id = requirements.requirement_id
                WHERE decisions.requirement_id IS NULL
                ORDER BY runs.created_at_utc, requirements.requirement_id
                """
            ).fetchall()
        return tuple(
            (
                ApprovalRequirement(
                    requirement_id=str(row["requirement_id"]),
                    run_id=str(row["run_id"]),
                    plan_digest=str(row["plan_digest"]),
                    step=PipelineStep(str(row["step"])),
                    kind=ApprovalKind(str(row["kind"])),
                    summary=str(row["summary"]),
                ),
                datetime.fromisoformat(str(row["created_at_utc"])),
            )
            for row in rows
        )

    def record_workflow_review(self, run_id: str, review_json: str) -> None:
        content_hash = self._canonical_payload_hash(review_json)
        with self._transaction() as connection:
            if self._workflow_row(connection, run_id) is None:
                raise KeyError(run_id)
            self._insert_immutable_payload(
                connection,
                table="workflow_reviews",
                identity_column="run_id",
                identity=run_id,
                payload_column="review_json",
                payload_json=review_json,
                content_hash=content_hash,
            )

    def workflow_reviews(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT review_json FROM workflow_reviews ORDER BY run_id"
            ).fetchall()
        return tuple(str(row["review_json"]) for row in rows)

    def record_workflow_run_snapshot(self, run_id: str, run_json: str) -> None:
        content_hash = self._canonical_payload_hash(run_json)
        with self._transaction() as connection:
            if self._workflow_row(connection, run_id) is None:
                raise KeyError(run_id)
            existing = connection.execute(
                """
                SELECT * FROM workflow_run_snapshots
                WHERE run_id = ? AND content_hash = ?
                """,
                (run_id, content_hash),
            ).fetchone()
            if existing is not None:
                return
            latest = connection.execute(
                """
                SELECT sequence FROM workflow_run_snapshots
                WHERE run_id = ? ORDER BY sequence DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            sequence = 0 if latest is None else int(latest["sequence"]) + 1
            connection.execute(
                """
                INSERT INTO workflow_run_snapshots (
                    run_id, sequence, run_json, content_hash
                ) VALUES (?, ?, ?, ?)
                """,
                (run_id, sequence, run_json, content_hash),
            )

    def latest_workflow_run_snapshots(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT snapshots.run_json
                FROM workflow_run_snapshots AS snapshots
                JOIN (
                    SELECT run_id, MAX(sequence) AS sequence
                    FROM workflow_run_snapshots GROUP BY run_id
                ) AS latest
                    ON latest.run_id = snapshots.run_id
                    AND latest.sequence = snapshots.sequence
                ORDER BY snapshots.run_id
                """
            ).fetchall()
        return tuple(str(row["run_json"]) for row in rows)

    def record_human_decision(self, decision: HumanDecision) -> HumanDecision:
        if not decision.human_subject.strip():
            raise ValueError("human_subject must not be empty")
        isoformat_utc(decision.decided_at)
        decision_hash = self._human_decision_hash(decision)
        with self._transaction() as connection:
            requirement = connection.execute(
                "SELECT * FROM approval_requirements WHERE requirement_id = ?",
                (decision.requirement_id,),
            ).fetchone()
            if requirement is None:
                raise ApprovalRequiredError("approval requirement is not registered")
            if (
                str(requirement["plan_digest"]) != decision.plan_digest
                or decision.plan_digest == ""
            ):
                raise CoordinatorIntegrityError(
                    "human decision belongs to a different execution plan"
                )
            existing = connection.execute(
                "SELECT * FROM human_decisions WHERE requirement_id = ?",
                (decision.requirement_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != decision_hash:
                    raise CoordinatorIntegrityError("human decisions are immutable once recorded")
                return self._human_decision_from_row(existing)
            connection.execute(
                """
                INSERT INTO human_decisions (
                    requirement_id, plan_digest, human_subject, verdict,
                    decided_at_utc, note, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.requirement_id,
                    decision.plan_digest,
                    decision.human_subject,
                    decision.verdict.value,
                    isoformat_utc(decision.decided_at),
                    decision.note,
                    decision_hash,
                ),
            )
        return decision

    def human_decision_for(self, requirement_id: str) -> HumanDecision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM human_decisions WHERE requirement_id = ?",
                (requirement_id,),
            ).fetchone()
        return None if row is None else self._human_decision_from_row(row)

    def record_workflow_step(
        self,
        run_id: str,
        step: PipelineStep,
        state: WorkflowStepState,
        *,
        occurred_at: datetime,
        details: Mapping[str, object] | None = None,
        evidence_id: str | None = None,
    ) -> WorkflowStepAudit:
        details_json = canonical_json({} if details is None else details)
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            if evidence_id is not None:
                self._require_evidence(connection, evidence_id)
            rows = connection.execute(
                """
                SELECT * FROM workflow_step_audits
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
            history = tuple(self._workflow_audit_from_row(row) for row in rows)
            latest_same = next(
                (audit for audit in reversed(history) if audit.step == step.value),
                None,
            )
            if latest_same is not None and latest_same.state is state:
                candidate = self._workflow_audit(
                    run_id=run_id,
                    sequence=latest_same.sequence,
                    step=step,
                    state=state,
                    occurred_at=occurred_at,
                    details_json=details_json,
                    evidence_id=evidence_id,
                )
                if candidate.content_hash == latest_same.content_hash:
                    return latest_same
                raise CoordinatorStateError(
                    f"workflow step {step.value!r} already entered {state.value!r}"
                )

            requirement_row = connection.execute(
                """
                SELECT requirements.*, decisions.verdict
                FROM approval_requirements AS requirements
                LEFT JOIN human_decisions AS decisions
                    ON decisions.requirement_id = requirements.requirement_id
                WHERE requirements.run_id = ? AND requirements.step = ?
                """,
                (run_id, step.value),
            ).fetchone()
            self._validate_step_approval(state, requirement_row)
            self._validate_step_progression(step, state, history, latest_same)

            sequence = 0 if not history else history[-1].sequence + 1
            audit = self._workflow_audit(
                run_id=run_id,
                sequence=sequence,
                step=step,
                state=state,
                occurred_at=occurred_at,
                details_json=details_json,
                evidence_id=evidence_id,
            )
            connection.execute(
                """
                INSERT INTO workflow_step_audits (
                    run_id, sequence, step, state, occurred_at_utc,
                    details_json, evidence_id, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.run_id,
                    audit.sequence,
                    audit.step,
                    audit.state.value,
                    isoformat_utc(audit.occurred_at),
                    audit.details_json,
                    audit.evidence_id,
                    audit.content_hash,
                ),
            )
            return audit

    def workflow_history(self, run_id: str) -> tuple[WorkflowStepAudit, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM workflow_step_audits
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        return tuple(self._workflow_audit_from_row(row) for row in rows)

    def record_rocketride_callback_result(
        self,
        run_id: str,
        step: PipelineStep,
        request_sha256: str,
        result_json: str,
    ) -> None:
        if len(request_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in request_sha256
        ):
            raise ValueError("callback request checksum must be a lowercase SHA-256 digest")
        content_hash = self._canonical_payload_hash(result_json)
        with self._transaction() as connection:
            if self._workflow_row(connection, run_id) is None:
                raise KeyError(run_id)
            existing = connection.execute(
                """
                SELECT * FROM rocketride_callback_results
                WHERE run_id = ? AND step = ?
                """,
                (run_id, step.value),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["request_sha256"]) == request_sha256
                    and str(existing["content_hash"]) == content_hash
                ):
                    return
                raise CoordinatorIntegrityError(
                    "RocketRide callback step is immutable once completed"
                )
            connection.execute(
                """
                INSERT INTO rocketride_callback_results (
                    run_id, step, request_sha256, result_json, content_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, step.value, request_sha256, result_json, content_hash),
            )

    def rocketride_callback_results(
        self,
        run_id: str,
    ) -> tuple[tuple[PipelineStep, str, str], ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT step, request_sha256, result_json
                FROM rocketride_callback_results
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        by_step = {
            PipelineStep(str(row["step"])): (
                str(row["request_sha256"]),
                str(row["result_json"]),
            )
            for row in rows
        }
        completed = tuple(step for step in FIXED_PIPELINE if step in by_step)
        if completed != FIXED_PIPELINE[: len(completed)]:
            raise CoordinatorIntegrityError(
                "durable RocketRide callback results are not a fixed-pipeline prefix"
            )
        return tuple((step, *by_step[step]) for step in completed)

    def numeric_policy_decision_for_run(self, run_id: str) -> NumericPolicyDecision | None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM numeric_policy_decisions WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        if len(rows) > 1:
            raise CoordinatorIntegrityError("workflow has multiple numeric policy decisions")
        return None if not rows else self._numeric_decision_from_row(rows[0])

    def register_evaluated_checkpoint(
        self,
        checkpoint: EvaluatedPolicyVersion,
    ) -> EvaluatedPolicyVersion:
        record_json = canonical_json(checkpoint.model_dump(mode="json"))
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM evaluated_checkpoints WHERE policy_id = ?",
                (checkpoint.policy_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != checkpoint.content_hash:
                    raise CoordinatorIntegrityError(
                        f"evaluated policy {checkpoint.policy_id!r} is immutable"
                    )
                return EvaluatedPolicyVersion.model_validate_json(str(existing["record_json"]))
            connection.execute(
                """
                INSERT INTO evaluated_checkpoints (
                    policy_id, checkpoint_hash, record_json, content_hash
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    checkpoint.policy_id,
                    checkpoint.checkpoint_hash,
                    record_json,
                    checkpoint.content_hash,
                ),
            )
        return checkpoint

    def evaluated_checkpoints(self) -> tuple[EvaluatedPolicyVersion, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_json FROM evaluated_checkpoints ORDER BY policy_id"
            ).fetchall()
        return tuple(
            EvaluatedPolicyVersion.model_validate_json(str(row["record_json"])) for row in rows
        )

    def initialize_policy_alias(
        self,
        alias: str,
        policy_id: str,
        *,
        occurred_at: datetime,
    ) -> PolicyAliasEvent:
        with self._transaction() as connection:
            self._require_policy(connection, policy_id)
            existing = connection.execute(
                """
                SELECT * FROM policy_alias_events
                WHERE alias = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (alias,),
            ).fetchone()
            event = self._policy_alias_event(
                alias=alias,
                sequence=0,
                target_policy_id=policy_id,
                action="initialize",
                occurred_at=occurred_at,
                numeric_decision_id=None,
                approval_requirement_id=None,
            )
            if existing is not None:
                existing_event = self._policy_alias_event_from_row(existing)
                if existing_event.content_hash == event.content_hash:
                    return existing_event
                raise CoordinatorStateError(f"policy alias {alias!r} is already initialized")
            self._insert_alias_event(connection, event)
            return event

    def record_numeric_policy_decision(
        self,
        decision: NumericPolicyDecision,
    ) -> NumericPolicyDecision:
        metrics_json = canonical_json(decision.metrics.as_mapping())
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?",
                (decision.run_id,),
            ).fetchone()
            if run is None or str(run["plan_digest"]) != decision.plan_digest:
                raise CoordinatorIntegrityError(
                    "numeric policy decision belongs to an unknown execution plan"
                )
            self._require_policy(connection, decision.target_policy_id)
            if decision.from_policy_id is not None:
                self._require_policy(connection, decision.from_policy_id)
            self._validate_numeric_decision_evidence(connection, decision)
            existing = connection.execute(
                "SELECT * FROM numeric_policy_decisions WHERE decision_id = ?",
                (decision.decision_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != decision.content_hash:
                    raise CoordinatorIntegrityError(
                        f"numeric decision {decision.decision_id!r} is immutable"
                    )
                return self._numeric_decision_from_row(existing)
            current = self._current_policy(connection, decision.alias)
            if current != decision.from_policy_id:
                raise CoordinatorStateError(
                    "numeric policy decision does not match the current alias target"
                )
            connection.execute(
                """
                INSERT INTO numeric_policy_decisions (
                    decision_id, run_id, plan_digest, action, alias, from_policy_id,
                    target_policy_id, evaluation_evidence_hash, metrics_json,
                    promotion_gate_passed, decided_at_utc, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.run_id,
                    decision.plan_digest,
                    decision.action.value,
                    decision.alias,
                    decision.from_policy_id,
                    decision.target_policy_id,
                    decision.evaluation_evidence_hash,
                    metrics_json,
                    int(decision.metrics.passes_promotion_gate),
                    isoformat_utc(decision.decided_at),
                    decision.content_hash,
                ),
            )
        return decision

    def apply_policy_action(
        self,
        decision_id: str,
        approval_requirement_id: str,
        *,
        occurred_at: datetime,
    ) -> PolicyAliasEvent:
        with self._transaction() as connection:
            decision_row = connection.execute(
                "SELECT * FROM numeric_policy_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if decision_row is None:
                raise CoordinatorStateError("numeric policy decision is not recorded")
            decision = self._numeric_decision_from_row(decision_row)
            self._validate_numeric_decision_evidence(connection, decision)
            approval_row = connection.execute(
                """
                SELECT requirements.*, decisions.verdict, decisions.plan_digest AS decision_plan
                FROM approval_requirements AS requirements
                JOIN human_decisions AS decisions
                    ON decisions.requirement_id = requirements.requirement_id
                WHERE requirements.requirement_id = ?
                """,
                (approval_requirement_id,),
            ).fetchone()
            if approval_row is None:
                raise ApprovalRequiredError("policy action requires a human decision")
            expected_kind = (
                ApprovalKind.POLICY_PROMOTION
                if decision.action is PolicyAction.PROMOTE
                else ApprovalKind.POLICY_ROLLBACK
            )
            if (
                str(approval_row["kind"]) != expected_kind.value
                or str(approval_row["run_id"]) != decision.run_id
                or str(approval_row["plan_digest"]) != decision.plan_digest
                or str(approval_row["decision_plan"]) != decision.plan_digest
            ):
                raise ApprovalRequiredError(
                    "human decision does not approve this measured policy action"
                )
            if str(approval_row["verdict"]) != HumanVerdict.APPROVE.value:
                raise ApprovalRequiredError("human decision rejected the policy action")

            prior_event = connection.execute(
                """
                SELECT * FROM policy_alias_events
                WHERE numeric_decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
            current_row = connection.execute(
                """
                SELECT * FROM policy_alias_events
                WHERE alias = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (decision.alias,),
            ).fetchone()
            current = (
                None if current_row is None else self._policy_alias_event_from_row(current_row)
            )
            if prior_event is not None:
                existing = self._policy_alias_event_from_row(prior_event)
                candidate = self._policy_alias_event(
                    alias=decision.alias,
                    sequence=existing.sequence,
                    target_policy_id=decision.target_policy_id,
                    action=decision.action.value,
                    occurred_at=occurred_at,
                    numeric_decision_id=decision.decision_id,
                    approval_requirement_id=approval_requirement_id,
                )
                if candidate.content_hash == existing.content_hash:
                    return existing
                raise CoordinatorStateError("policy decision was already applied")
            current_policy = None if current is None else current.target_policy_id
            if current_policy != decision.from_policy_id:
                raise CoordinatorStateError(
                    "policy alias changed after the numeric decision was recorded"
                )
            sequence = 0 if current is None else current.sequence + 1
            event = self._policy_alias_event(
                alias=decision.alias,
                sequence=sequence,
                target_policy_id=decision.target_policy_id,
                action=decision.action.value,
                occurred_at=occurred_at,
                numeric_decision_id=decision.decision_id,
                approval_requirement_id=approval_requirement_id,
            )
            self._insert_alias_event(connection, event)
            return event

    def current_policy(self, alias: str) -> str | None:
        with self._lock:
            return self._current_policy(self._connection, alias)

    def policy_alias_history(self, alias: str) -> tuple[PolicyAliasEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM policy_alias_events
                WHERE alias = ?
                ORDER BY sequence
                """,
                (alias,),
            ).fetchall()
        return tuple(self._policy_alias_event_from_row(row) for row in rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> CoordinatorStore:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    @staticmethod
    def _training_episode_from_row(row: sqlite3.Row) -> TrainingEpisodeMetadata:
        metadata = TrainingEpisodeMetadata(
            episode_id=str(row["episode_id"]),
            robot_checksum=str(row["robot_checksum"]),
            world_hash=str(row["world_hash"]),
            policy_hash=str(row["policy_hash"]),
            created_at=datetime.fromisoformat(str(row["created_at_utc"])),
        )
        if metadata.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("training episode content hash mismatch")
        return metadata

    @staticmethod
    def _held_out_episode_from_row(
        row: sqlite3.Row,
    ) -> HeldOutEvaluationEpisodeMetadata:
        metadata = HeldOutEvaluationEpisodeMetadata(
            episode_id=str(row["episode_id"]),
            robot_checksum=str(row["robot_checksum"]),
            world_hash=str(row["world_hash"]),
            policy_hash=str(row["policy_hash"]),
            held_out_world_set_id=str(row["held_out_world_set_id"]),
            created_at=datetime.fromisoformat(str(row["created_at_utc"])),
        )
        if metadata.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("held-out episode content hash mismatch")
        return metadata

    @staticmethod
    def _episode_transition(
        *,
        episode_id: str,
        sequence: int,
        state: EpisodeState,
        occurred_at: datetime,
        details_json: str,
        evidence_id: str | None,
    ) -> EpisodeTransition:
        content_hash = sha256_text(
            canonical_json(
                {
                    "episode_id": episode_id,
                    "sequence": sequence,
                    "state": state.value,
                    "occurred_at": isoformat_utc(occurred_at),
                    "details": json.loads(details_json),
                    "evidence_id": evidence_id,
                }
            )
        )
        return EpisodeTransition(
            episode_id=episode_id,
            sequence=sequence,
            state=state,
            occurred_at=occurred_at,
            details_json=details_json,
            evidence_id=evidence_id,
            content_hash=content_hash,
        )

    @staticmethod
    def _insert_episode_transition(
        connection: sqlite3.Connection,
        transition: EpisodeTransition,
    ) -> None:
        connection.execute(
            """
            INSERT INTO episode_transitions (
                episode_id, sequence, state, occurred_at_utc, details_json,
                evidence_id, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.episode_id,
                transition.sequence,
                transition.state.value,
                isoformat_utc(transition.occurred_at),
                transition.details_json,
                transition.evidence_id,
                transition.content_hash,
            ),
        )

    def _episode_transition_from_row(self, row: sqlite3.Row) -> EpisodeTransition:
        transition = self._episode_transition(
            episode_id=str(row["episode_id"]),
            sequence=int(row["sequence"]),
            state=EpisodeState(str(row["state"])),
            occurred_at=datetime.fromisoformat(str(row["occurred_at_utc"])),
            details_json=str(row["details_json"]),
            evidence_id=None if row["evidence_id"] is None else str(row["evidence_id"]),
        )
        if transition.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("episode transition content hash mismatch")
        return transition

    @staticmethod
    def _provider_evidence_from_row(row: sqlite3.Row) -> ProviderEvidenceReference:
        evidence = ProviderEvidenceReference(
            evidence_id=str(row["evidence_id"]),
            provider=str(row["provider"]),
            evidence_kind=str(row["evidence_kind"]),
            provider_object_id=str(row["provider_object_id"]),
            artifact_hash=str(row["artifact_hash"]),
            observed_at=datetime.fromisoformat(str(row["observed_at_utc"])),
        )
        if evidence.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("provider evidence content hash mismatch")
        return evidence

    @staticmethod
    def _plan_json(plan: ExecutionPlan) -> str:
        return canonical_json(
            {
                "run_id": plan.run_id,
                "plan_digest": plan.digest,
                "commands": [
                    {
                        "step": command.step.value,
                        "payload_json": command.payload_json,
                        "payload_sha256": command.payload_sha256,
                    }
                    for command in plan.commands
                ],
            }
        )

    @staticmethod
    def _execution_plan(plan_json: str) -> ExecutionPlan:
        try:
            value = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            raise CoordinatorIntegrityError("stored workflow plan is not JSON") from exc
        if not isinstance(value, dict) or canonical_json(value) != plan_json:
            raise CoordinatorIntegrityError("stored workflow plan is not canonical")
        raw_commands = value.get("commands")
        if not isinstance(raw_commands, list):
            raise CoordinatorIntegrityError("stored workflow plan has no commands")
        commands: list[PipelineCommand] = []
        for item in raw_commands:
            if not isinstance(item, dict):
                raise CoordinatorIntegrityError("stored workflow command is invalid")
            try:
                command = PipelineCommand(
                    step=PipelineStep(str(item["step"])),
                    payload_json=str(item["payload_json"]),
                    payload_sha256=str(item["payload_sha256"]),
                )
            except (KeyError, ValueError) as exc:
                raise CoordinatorIntegrityError(
                    "stored workflow command failed validation"
                ) from exc
            commands.append(command)
        try:
            return ExecutionPlan(
                run_id=str(value["run_id"]),
                commands=tuple(commands),
                digest=str(value["plan_digest"]),
            )
        except (KeyError, ValueError) as exc:
            raise CoordinatorIntegrityError("stored workflow plan failed validation") from exc

    @staticmethod
    def _insert_approval_requirement(
        connection: sqlite3.Connection,
        requirement: ApprovalRequirement,
    ) -> None:
        content_hash = sha256_text(
            canonical_json(
                {
                    "requirement_id": requirement.requirement_id,
                    "run_id": requirement.run_id,
                    "plan_digest": requirement.plan_digest,
                    "step": requirement.step.value,
                    "kind": requirement.kind.value,
                    "summary": requirement.summary,
                }
            )
        )
        connection.execute(
            """
            INSERT INTO approval_requirements (
                requirement_id, run_id, plan_digest, step, kind, summary, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                requirement.requirement_id,
                requirement.run_id,
                requirement.plan_digest,
                requirement.step.value,
                requirement.kind.value,
                requirement.summary,
                content_hash,
            ),
        )

    @staticmethod
    def _human_decision_hash(decision: HumanDecision) -> str:
        return decision.decision_id

    @staticmethod
    def _human_decision_from_row(row: sqlite3.Row) -> HumanDecision:
        return HumanDecision(
            requirement_id=str(row["requirement_id"]),
            plan_digest=str(row["plan_digest"]),
            human_subject=str(row["human_subject"]),
            verdict=HumanVerdict(str(row["verdict"])),
            decided_at=datetime.fromisoformat(str(row["decided_at_utc"])),
            note=str(row["note"]),
        )

    @staticmethod
    def _validate_step_approval(
        state: WorkflowStepState,
        requirement: sqlite3.Row | None,
    ) -> None:
        verdict = None if requirement is None else requirement["verdict"]
        if state is WorkflowStepState.AWAITING_APPROVAL:
            if requirement is None:
                raise CoordinatorStateError("workflow step has no approval requirement")
            if verdict is not None:
                raise CoordinatorStateError("workflow approval already has a human decision")
            return
        if state is WorkflowStepState.BLOCKED:
            if verdict != HumanVerdict.REJECT.value:
                raise CoordinatorStateError("blocked step requires a rejected human decision")
            return
        if requirement is not None and verdict != HumanVerdict.APPROVE.value:
            raise ApprovalRequiredError("workflow step is blocked on human approval")

    @staticmethod
    def _validate_step_progression(
        step: PipelineStep,
        state: WorkflowStepState,
        history: tuple[WorkflowStepAudit, ...],
        latest_same: WorkflowStepAudit | None,
    ) -> None:
        step_index = FIXED_PIPELINE.index(step)
        completed = {
            PipelineStep(audit.step)
            for audit in history
            if audit.state is WorkflowStepState.COMPLETED
        }
        missing_prior = [prior for prior in FIXED_PIPELINE[:step_index] if prior not in completed]
        if missing_prior:
            raise CoordinatorStateError(
                f"workflow cannot enter {step.value!r} before prior steps complete"
            )
        if state is WorkflowStepState.AWAITING_APPROVAL:
            if latest_same is not None:
                raise CoordinatorStateError("approval wait must precede step execution")
            return
        if state is WorkflowStepState.BLOCKED:
            if latest_same is not None and latest_same.state is WorkflowStepState.COMPLETED:
                raise CoordinatorStateError("a completed workflow step cannot be blocked")
            return
        if state is WorkflowStepState.STARTED:
            if latest_same is not None and latest_same.state not in {
                WorkflowStepState.AWAITING_APPROVAL,
                WorkflowStepState.FAILED,
            }:
                raise CoordinatorStateError("workflow step cannot be started from its state")
            return
        if latest_same is None or latest_same.state is not WorkflowStepState.STARTED:
            raise CoordinatorStateError(f"workflow step must be started before it is {state.value}")

    @staticmethod
    def _workflow_audit(
        *,
        run_id: str,
        sequence: int,
        step: PipelineStep,
        state: WorkflowStepState,
        occurred_at: datetime,
        details_json: str,
        evidence_id: str | None,
    ) -> WorkflowStepAudit:
        content_hash = sha256_text(
            canonical_json(
                {
                    "run_id": run_id,
                    "sequence": sequence,
                    "step": step.value,
                    "state": state.value,
                    "occurred_at": isoformat_utc(occurred_at),
                    "details": json.loads(details_json),
                    "evidence_id": evidence_id,
                }
            )
        )
        return WorkflowStepAudit(
            run_id=run_id,
            sequence=sequence,
            step=step.value,
            state=state,
            occurred_at=occurred_at,
            details_json=details_json,
            evidence_id=evidence_id,
            content_hash=content_hash,
        )

    def _workflow_audit_from_row(self, row: sqlite3.Row) -> WorkflowStepAudit:
        audit = self._workflow_audit(
            run_id=str(row["run_id"]),
            sequence=int(row["sequence"]),
            step=PipelineStep(str(row["step"])),
            state=WorkflowStepState(str(row["state"])),
            occurred_at=datetime.fromisoformat(str(row["occurred_at_utc"])),
            details_json=str(row["details_json"]),
            evidence_id=None if row["evidence_id"] is None else str(row["evidence_id"]),
        )
        if audit.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("workflow audit content hash mismatch")
        return audit

    @staticmethod
    def _numeric_decision_from_row(row: sqlite3.Row) -> NumericPolicyDecision:
        metrics_payload = json.loads(str(row["metrics_json"]))
        if not isinstance(metrics_payload, dict):
            raise CoordinatorIntegrityError("numeric decision metrics are not an object")
        metrics = PolicyGateMetrics(
            held_out_success_rate=float(metrics_payload["held_out_success_rate"]),
            collision_rate=float(metrics_payload["collision_rate"]),
            fall_count=int(metrics_payload["fall_count"]),
            median_clearance_m=float(metrics_payload["median_clearance_m"]),
            success_rate_delta=float(metrics_payload["success_rate_delta"]),
            collision_reduction_fraction=float(metrics_payload["collision_reduction_fraction"]),
            path_efficiency_regression_fraction=float(
                metrics_payload["path_efficiency_regression_fraction"]
            ),
        )
        decision = NumericPolicyDecision(
            decision_id=str(row["decision_id"]),
            run_id=str(row["run_id"]),
            plan_digest=str(row["plan_digest"]),
            action=PolicyAction(str(row["action"])),
            alias=str(row["alias"]),
            from_policy_id=(None if row["from_policy_id"] is None else str(row["from_policy_id"])),
            target_policy_id=str(row["target_policy_id"]),
            evaluation_evidence_hash=str(row["evaluation_evidence_hash"]),
            metrics=metrics,
            decided_at=datetime.fromisoformat(str(row["decided_at_utc"])),
        )
        if decision.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("numeric policy decision content hash mismatch")
        gate_value = bool(int(row["promotion_gate_passed"]))
        if gate_value is not metrics.passes_promotion_gate:
            raise CoordinatorIntegrityError("stored promotion gate result does not match metrics")
        return decision

    @staticmethod
    def _policy_alias_event(
        *,
        alias: str,
        sequence: int,
        target_policy_id: str,
        action: str,
        occurred_at: datetime,
        numeric_decision_id: str | None,
        approval_requirement_id: str | None,
    ) -> PolicyAliasEvent:
        content_hash = sha256_text(
            canonical_json(
                {
                    "alias": alias,
                    "sequence": sequence,
                    "target_policy_id": target_policy_id,
                    "action": action,
                    "occurred_at": isoformat_utc(occurred_at),
                    "numeric_decision_id": numeric_decision_id,
                    "approval_requirement_id": approval_requirement_id,
                }
            )
        )
        return PolicyAliasEvent(
            alias=alias,
            sequence=sequence,
            target_policy_id=target_policy_id,
            action=action,
            occurred_at=occurred_at,
            numeric_decision_id=numeric_decision_id,
            approval_requirement_id=approval_requirement_id,
            content_hash=content_hash,
        )

    def _policy_alias_event_from_row(self, row: sqlite3.Row) -> PolicyAliasEvent:
        event = self._policy_alias_event(
            alias=str(row["alias"]),
            sequence=int(row["sequence"]),
            target_policy_id=str(row["target_policy_id"]),
            action=str(row["action"]),
            occurred_at=datetime.fromisoformat(str(row["occurred_at_utc"])),
            numeric_decision_id=(
                None if row["numeric_decision_id"] is None else str(row["numeric_decision_id"])
            ),
            approval_requirement_id=(
                None
                if row["approval_requirement_id"] is None
                else str(row["approval_requirement_id"])
            ),
        )
        if event.content_hash != str(row["content_hash"]):
            raise CoordinatorIntegrityError("policy alias event content hash mismatch")
        return event

    @staticmethod
    def _insert_alias_event(
        connection: sqlite3.Connection,
        event: PolicyAliasEvent,
    ) -> None:
        connection.execute(
            """
            INSERT INTO policy_alias_events (
                alias, sequence, target_policy_id, action, occurred_at_utc,
                numeric_decision_id, approval_requirement_id, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.alias,
                event.sequence,
                event.target_policy_id,
                event.action,
                isoformat_utc(event.occurred_at),
                event.numeric_decision_id,
                event.approval_requirement_id,
                event.content_hash,
            ),
        )

    @staticmethod
    def _current_policy(connection: sqlite3.Connection, alias: str) -> str | None:
        row = connection.execute(
            """
            SELECT target_policy_id
            FROM policy_alias_events
            WHERE alias = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (alias,),
        ).fetchone()
        return None if row is None else str(row["target_policy_id"])

    @classmethod
    def _validate_numeric_decision_evidence(
        cls,
        connection: sqlite3.Connection,
        decision: NumericPolicyDecision,
    ) -> None:
        run = connection.execute(
            "SELECT plan_json FROM workflow_runs WHERE run_id = ?",
            (decision.run_id,),
        ).fetchone()
        evidence_row = connection.execute(
            "SELECT evidence_json FROM workflow_guild_evidence WHERE run_id = ?",
            (decision.run_id,),
        ).fetchone()
        if run is None or evidence_row is None:
            raise CoordinatorIntegrityError(
                "numeric policy decision requires trusted workflow evaluation evidence"
            )
        plan = cls._execution_plan(str(run["plan_json"]))
        bundle = GuildEvidenceBundle.model_validate_json(str(evidence_row["evidence_json"]))
        validate_evidence_plan_binding(bundle, plan)
        evaluation = bundle.evaluation.evaluation_evidence
        final_command = plan.commands[FIXED_PIPELINE.index(PipelineStep.PROMOTE_OR_ROLL_BACK)]
        if (
            decision.plan_digest != plan.digest
            or decision.action.value != final_command.payload.get("action")
            or decision.from_policy_id != evaluation.baseline.policy_id
            or decision.target_policy_id != evaluation.candidate.policy_id
        ):
            raise CoordinatorIntegrityError(
                "numeric policy decision identities do not match trusted evaluation evidence"
            )

        expected_artifact_hash = next(
            artifact_hash
            for _evidence_id, kind, artifact_hash in bundle.artifact_hashes()
            if kind == "guild_evaluation_evidence"
        )
        if decision.evaluation_evidence_hash != expected_artifact_hash:
            raise CoordinatorIntegrityError(
                "numeric policy decision does not reference the trusted evaluation artifact"
            )

        baseline = cls._evaluated_checkpoint_row(
            connection,
            evaluation.baseline.policy_id,
        )
        candidate = cls._evaluated_checkpoint_row(
            connection,
            evaluation.candidate.policy_id,
        )
        if (
            baseline.evaluation_split != "held_out"
            or candidate.evaluation_split != "held_out"
            or baseline.checkpoint_hash != evaluation.baseline.policy_checksum
            or candidate.checkpoint_hash != evaluation.candidate.policy_checksum
            or baseline.evaluation_evidence_hash != evaluation.baseline.evaluation_id
            or candidate.evaluation_evidence_hash != evaluation.candidate.evaluation_id
        ):
            raise CoordinatorIntegrityError(
                "evaluated checkpoints do not match trusted evaluation evidence"
            )
        cls._require_checkpoint_metrics(baseline, evaluation, candidate=False)
        cls._require_checkpoint_metrics(candidate, evaluation, candidate=True)
        if decision.metrics != cls._policy_gate_metrics(evaluation):
            raise CoordinatorIntegrityError(
                "numeric policy metrics were not exactly recomputed from trusted evidence"
            )

    @staticmethod
    def _evaluated_checkpoint_row(
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> EvaluatedPolicyVersion:
        row = connection.execute(
            "SELECT record_json FROM evaluated_checkpoints WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        if row is None:
            raise CoordinatorIntegrityError("trusted evaluation references an unknown checkpoint")
        return EvaluatedPolicyVersion.model_validate_json(str(row["record_json"]))

    @staticmethod
    def _require_checkpoint_metrics(
        checkpoint: EvaluatedPolicyVersion,
        evaluation: EvaluationEvidence,
        *,
        candidate: bool,
    ) -> None:
        decoded = json.loads(checkpoint.metrics_json)
        if not isinstance(decoded, dict):
            raise CoordinatorIntegrityError("checkpoint metrics are not a JSON object")
        source = evaluation.candidate if candidate else evaluation.baseline
        expected: dict[str, int | float] = {
            "success_rate": source.success_rate,
            "collision_rate": source.collision_rate,
        }
        if candidate:
            expected.update(
                {
                    "falls": evaluation.candidate.falls,
                    "median_clearance_m": evaluation.candidate.median_clearance_m,
                    "path_efficiency_regression_fraction": (
                        evaluation.candidate.path_efficiency_regression_fraction
                    ),
                }
            )
        if any(decoded.get(key) != value for key, value in expected.items()):
            raise CoordinatorIntegrityError(
                "checkpoint metrics do not match trusted evaluation evidence"
            )

    @staticmethod
    def _policy_gate_metrics(evaluation: EvaluationEvidence) -> PolicyGateMetrics:
        collision_reduction = (
            0.0
            if evaluation.baseline.collision_rate == 0.0
            else (
                evaluation.baseline.collision_rate - evaluation.candidate.collision_rate
            )
            / evaluation.baseline.collision_rate
        )
        return PolicyGateMetrics(
            held_out_success_rate=evaluation.candidate.success_rate,
            collision_rate=evaluation.candidate.collision_rate,
            fall_count=evaluation.candidate.falls,
            median_clearance_m=evaluation.candidate.median_clearance_m,
            success_rate_delta=(
                evaluation.candidate.success_rate - evaluation.baseline.success_rate
            ),
            collision_reduction_fraction=collision_reduction,
            path_efficiency_regression_fraction=(
                evaluation.candidate.path_efficiency_regression_fraction
            ),
        )

    @staticmethod
    def _canonical_payload_hash(payload_json: str) -> str:
        try:
            decoded = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("coordinator episode payload is not valid JSON") from exc
        if not isinstance(decoded, dict) or canonical_json(decoded) != payload_json:
            raise ValueError("coordinator episode payload must be a canonical JSON object")
        return sha256_text(payload_json)

    @staticmethod
    def _insert_immutable_payload(
        connection: sqlite3.Connection,
        *,
        table: str,
        identity_column: str,
        identity: str,
        payload_column: str,
        payload_json: str,
        content_hash: str,
    ) -> None:
        allowed = {
            (
                "training_episode_sessions",
                "episode_id",
                "identity_json",
            ),
            (
                "training_episode_closures",
                "episode_id",
                "closure_json",
            ),
            (
                "training_correction_approvals",
                "correction_id",
                "approval_json",
            ),
            (
                "training_correction_graph_deliveries",
                "correction_id",
                "delivery_json",
            ),
            (
                "training_correction_rejections",
                "correction_id",
                "rejection_json",
            ),
            ("workflow_guild_evidence", "run_id", "evidence_json"),
            ("workflow_reviews", "run_id", "review_json"),
        }
        if (table, identity_column, payload_column) not in allowed:
            raise ValueError("unsupported coordinator payload table")
        row = connection.execute(
            f"SELECT content_hash FROM {table} WHERE {identity_column} = ?",
            (identity,),
        ).fetchone()
        if row is not None:
            if str(row["content_hash"]) != content_hash:
                raise CoordinatorIntegrityError(f"{table.replace('_', ' ')} is immutable")
            return
        connection.execute(
            f"""
            INSERT INTO {table} (
                {identity_column}, {payload_column}, content_hash
            ) VALUES (?, ?, ?)
            """,
            (identity, payload_json, content_hash),
        )

    @staticmethod
    def _require_training_episode(
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT kind FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        if str(row["kind"]) != EpisodeKind.TRAINING.value:
            raise CoordinatorStateError(
                "operational episode journaling cannot access held-out evaluations"
            )

    @classmethod
    def _require_training_session(
        cls,
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> None:
        cls._require_training_episode(connection, episode_id)
        row = connection.execute(
            "SELECT 1 FROM training_episode_sessions WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise CoordinatorStateError("training episode session is not registered")

    @classmethod
    def _require_training_closure(
        cls,
        connection: sqlite3.Connection,
        episode_id: str,
    ) -> None:
        cls._require_training_session(connection, episode_id)
        row = connection.execute(
            "SELECT 1 FROM training_episode_closures WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise CoordinatorStateError("training episode is not durably closed")

    @staticmethod
    def _require_episode(connection: sqlite3.Connection, episode_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)

    @staticmethod
    def _workflow_row(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> sqlite3.Row | None:
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM workflow_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row

    @staticmethod
    def _require_evidence(connection: sqlite3.Connection, evidence_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM provider_evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        if row is None:
            raise CoordinatorIntegrityError(f"provider evidence {evidence_id!r} is not registered")

    @staticmethod
    def _require_policy(connection: sqlite3.Connection, policy_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM evaluated_checkpoints WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        if row is None:
            raise CoordinatorIntegrityError(f"evaluated policy {policy_id!r} is not registered")
