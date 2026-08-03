"""Coordinator-backed journal for operational training episodes."""

from __future__ import annotations

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
    EpisodeAppendReceipt,
    EpisodeClosure,
    EpisodeIdentity,
)
from muscle_memory.graph_memory import WorldSplit

_IDENTITY_ADAPTER = TypeAdapter(EpisodeIdentity)
_RECEIPT_ADAPTER = TypeAdapter(EpisodeAppendReceipt)
_CLOSURE_ADAPTER = TypeAdapter(EpisodeClosure)
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
        return None if payload is None else _CLOSURE_ADAPTER.validate_json(payload)

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
                elif state is not EpisodeState.RUNNING:
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
