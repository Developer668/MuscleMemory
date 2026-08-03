"""Coordinator-backed journal for operational training episodes."""

from __future__ import annotations

import json
from dataclasses import replace

from pydantic import TypeAdapter

from muscle_memory.coordinator import (
    CoordinatorStateError,
    CoordinatorStore,
    EpisodeState,
    TrainingEpisodeMetadata,
)
from muscle_memory.coordinator.models import canonical_json
from muscle_memory.episodes import (
    CorrectionApproval,
    CorrectionSubmission,
    EpisodeAbort,
    EpisodeAppendReceipt,
    EpisodeClosure,
    EpisodeIdentity,
    GraphPersistenceReport,
)
from muscle_memory.graph_memory import WorldSplit

_IDENTITY_ADAPTER = TypeAdapter(EpisodeIdentity)
_RECEIPT_ADAPTER = TypeAdapter(EpisodeAppendReceipt)
_CLOSURE_ADAPTER = TypeAdapter(EpisodeClosure)
_GRAPH_REPORT_ADAPTER = TypeAdapter(GraphPersistenceReport)
_CORRECTION_ADAPTER = TypeAdapter(CorrectionSubmission)
_APPROVAL_ADAPTER = TypeAdapter(CorrectionApproval)


def _encode[T](adapter: TypeAdapter[T], value: T) -> str:
    return canonical_json(adapter.dump_python(value, mode="json"))


class CoordinatorEpisodeJournal:
    """Persist complete lifecycle facts while excluding held-out evaluations.

    Held-out execution must use the evaluation-only coordinator API. This journal
    deliberately has no resolver for held-out world sets and refuses such identities.
    """

    def __init__(self, store: CoordinatorStore, *, expected_robot_checksum: str) -> None:
        self._store = store
        self._expected_robot_checksum = expected_robot_checksum
        self._recover_transitions()

    def identities(self) -> tuple[EpisodeIdentity, ...]:
        identities = tuple(
            _IDENTITY_ADAPTER.validate_json(payload)
            for payload in self._store.training_episode_sessions()
        )
        for identity in identities:
            self._require_fixed_robot(identity)
        return identities

    def receipts_for(self, episode_id: str) -> tuple[EpisodeAppendReceipt, ...]:
        return tuple(
            _RECEIPT_ADAPTER.validate_json(payload)
            for payload in self._store.training_episode_receipts(episode_id)
        )

    def closure_for(self, episode_id: str) -> EpisodeClosure | None:
        payload = self._store.training_episode_closure(episode_id)
        if payload is None:
            return None
        closure = _CLOSURE_ADAPTER.validate_json(payload)
        deliveries = self._store.training_episode_graph_deliveries(episode_id)
        if not deliveries:
            return closure
        return replace(
            closure,
            graph=_GRAPH_REPORT_ADAPTER.validate_json(deliveries[-1]),
        )

    def abort_for(self, episode_id: str) -> EpisodeAbort | None:
        history = self._store.episode_history(episode_id)
        if not history or history[-1].state is not EpisodeState.ABORTED:
            return None
        terminal = history[-1]
        details = json.loads(terminal.details_json)
        error_type = details.get("error_type")
        if not isinstance(error_type, str):
            raise CoordinatorStateError("aborted episode is missing its error type")
        return EpisodeAbort(
            episode_id=episode_id,
            error_type=error_type,
            aborted_at=terminal.occurred_at,
        )

    def corrections(self) -> tuple[CorrectionSubmission, ...]:
        return tuple(
            _CORRECTION_ADAPTER.validate_json(payload)
            for payload in self._store.training_correction_submissions()
        )

    def approvals(self) -> tuple[CorrectionApproval, ...]:
        approvals = {
            approval.submission.correction_id: approval
            for approval in (
                _APPROVAL_ADAPTER.validate_json(payload)
                for payload in self._store.training_correction_approvals()
            )
        }
        for payload in self._store.training_correction_graph_deliveries():
            delivered = _APPROVAL_ADAPTER.validate_json(payload)
            correction_id = delivered.submission.correction_id
            base = approvals.get(correction_id)
            if base is None or base.approval_id != delivered.approval_id:
                raise CoordinatorStateError(
                    "correction graph delivery is detached from its approval"
                )
            approvals[correction_id] = delivered
        return tuple(approvals[key] for key in sorted(approvals))

    def record_identity(self, identity: EpisodeIdentity) -> None:
        self._require_fixed_robot(identity)
        if identity.world_split is not WorldSplit.TRAINING:
            raise CoordinatorStateError(
                "operational episode journal accepts training episodes only"
            )
        self._store.register_training_episode(
            TrainingEpisodeMetadata(
                episode_id=identity.episode_id,
                robot_checksum=identity.robot_checksum,
                world_hash=identity.world_hash,
                policy_hash=identity.policy_hash,
                created_at=identity.opened_at,
            )
        )
        self._store.record_training_episode_session(
            identity.episode_id,
            _encode(_IDENTITY_ADAPTER, identity),
        )
        state = self._store.episode_state(identity.episode_id)
        if state is EpisodeState.CREATED:
            self._store.transition_episode(
                identity.episode_id,
                EpisodeState.RUNNING,
                occurred_at=identity.opened_at,
                details={"world_id": identity.world_id, "policy_id": identity.policy_id},
            )
        elif state is not EpisodeState.RUNNING:
            raise CoordinatorStateError("episode identity already has a terminal state")

    def record_receipt(self, receipt: EpisodeAppendReceipt) -> None:
        self._store.record_training_episode_receipt(
            receipt.episode_id,
            receipt.sequence,
            _encode(_RECEIPT_ADAPTER, receipt),
        )

    def record_receipt_delivery(self, receipt: EpisodeAppendReceipt) -> None:
        self._store.record_training_episode_receipt_delivery(
            receipt.episode_id,
            receipt.sequence,
            _encode(_RECEIPT_ADAPTER, receipt),
        )

    def record_closure(self, closure: EpisodeClosure) -> None:
        if closure.identity.world_split is not WorldSplit.TRAINING:
            raise CoordinatorStateError(
                "operational episode journal cannot close a held-out evaluation"
            )
        self._store.record_training_episode_closure(
            closure.identity.episode_id,
            _encode(_CLOSURE_ADAPTER, closure),
        )
        state = self._store.episode_state(closure.identity.episode_id)
        terminal = EpisodeState.SUCCEEDED if closure.result.success else EpisodeState.FAILED
        if state is EpisodeState.RUNNING:
            self._store.transition_episode(
                closure.identity.episode_id,
                terminal,
                occurred_at=closure.closed_at,
                details={
                    "failure_ids": [failure.failure_id for failure in closure.failures],
                    "graph_provider_complete": closure.graph.provider_complete,
                    "telemetry_digest": closure.telemetry_digest,
                },
            )
        elif state is not terminal:
            raise CoordinatorStateError("durable closure conflicts with episode state")

    def record_abort(self, abort: EpisodeAbort) -> None:
        if self._store.training_episode_closure(abort.episode_id) is not None:
            raise CoordinatorStateError("a closed episode cannot be aborted")
        state = self._store.episode_state(abort.episode_id)
        if state is not EpisodeState.RUNNING:
            raise CoordinatorStateError("only a running episode can be aborted")
        self._store.transition_episode(
            abort.episode_id,
            EpisodeState.ABORTED,
            occurred_at=abort.aborted_at,
            details={"error_type": abort.error_type},
        )

    def record_graph_delivery(
        self,
        episode_id: str,
        report: GraphPersistenceReport,
    ) -> None:
        closure = self.closure_for(episode_id)
        if closure is None:
            raise CoordinatorStateError("graph delivery requires a durable episode closure")
        if report.expected_records != closure.graph.expected_records:
            raise CoordinatorStateError("graph delivery record count conflicts with closure")
        self._store.record_training_episode_graph_delivery(
            episode_id,
            _encode(_GRAPH_REPORT_ADAPTER, report),
        )

    def record_correction(self, correction: CorrectionSubmission) -> None:
        self._store.record_training_correction_submission(
            correction.correction_id,
            correction.episode_id,
            _encode(_CORRECTION_ADAPTER, correction),
        )

    def record_approval(self, approval: CorrectionApproval) -> None:
        self._store.record_training_correction_approval(
            approval.submission.correction_id,
            _encode(_APPROVAL_ADAPTER, approval),
        )

    def record_approval_delivery(self, approval: CorrectionApproval) -> None:
        self._store.record_training_correction_graph_delivery(
            approval.submission.correction_id,
            _encode(_APPROVAL_ADAPTER, approval),
        )

    def _require_fixed_robot(self, identity: EpisodeIdentity) -> None:
        if identity.robot_checksum != self._expected_robot_checksum:
            raise CoordinatorStateError(
                "episode robot checksum does not match the qualified MM-01 bundle"
            )

    def _recover_transitions(self) -> None:
        """Finish idempotent coordinator transitions after a prior process crash."""

        for payload in self._store.training_episode_sessions():
            identity = _IDENTITY_ADAPTER.validate_json(payload)
            self._require_fixed_robot(identity)
            closure_payload = self._store.training_episode_closure(identity.episode_id)
            state = self._store.episode_state(identity.episode_id)
            if closure_payload is None:
                if state is EpisodeState.CREATED:
                    self._store.transition_episode(
                        identity.episode_id,
                        EpisodeState.RUNNING,
                        occurred_at=identity.opened_at,
                        details={
                            "world_id": identity.world_id,
                            "policy_id": identity.policy_id,
                        },
                    )
                elif state not in {EpisodeState.RUNNING, EpisodeState.ABORTED}:
                    raise CoordinatorStateError(
                        "terminal coordinator episode is missing its durable closure"
                    )
                continue
            closure = _CLOSURE_ADAPTER.validate_json(closure_payload)
            expected = EpisodeState.SUCCEEDED if closure.result.success else EpisodeState.FAILED
            if state is EpisodeState.RUNNING:
                self._store.transition_episode(
                    identity.episode_id,
                    expected,
                    occurred_at=closure.closed_at,
                    details={
                        "failure_ids": [failure.failure_id for failure in closure.failures],
                        "graph_provider_complete": closure.graph.provider_complete,
                        "telemetry_digest": closure.telemetry_digest,
                    },
                )
            elif state is not expected:
                raise CoordinatorStateError(
                    "durable closure conflicts with coordinator terminal state"
                )


__all__ = ["CoordinatorEpisodeJournal"]
