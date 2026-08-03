"""Coordinator-backed human approval ledger for RocketRide gates."""

from __future__ import annotations

from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.orchestration.approvals import HumanDecision


class CoordinatorApprovalLedger:
    def __init__(self, store: CoordinatorStore) -> None:
        self._store = store

    def record(self, decision: HumanDecision) -> None:
        self._store.record_human_decision(decision)

    def decision_for(self, requirement_id: str) -> HumanDecision | None:
        return self._store.human_decision_for(requirement_id)


__all__ = ["CoordinatorApprovalLedger"]
