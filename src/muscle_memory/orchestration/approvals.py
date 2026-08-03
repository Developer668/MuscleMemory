"""Append-only human approval records for gated orchestration changes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol

from muscle_memory.orchestration.contracts import (
    ApprovalRequirement,
    ContractViolationError,
    canonical_json,
    sha256_text,
)


class HumanVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class HumanDecision:
    requirement_id: str
    plan_digest: str
    human_subject: str
    verdict: HumanVerdict
    decided_at: datetime
    note: str = ""

    @property
    def decision_id(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "requirement_id": self.requirement_id,
                    "plan_digest": self.plan_digest,
                    "human_subject": self.human_subject,
                    "verdict": self.verdict.value,
                    "decided_at": self.decided_at.isoformat(),
                    "note": self.note,
                }
            )
        )

    @classmethod
    def create(
        cls,
        requirement: ApprovalRequirement,
        *,
        human_subject: str,
        verdict: HumanVerdict,
        note: str = "",
    ) -> HumanDecision:
        if not human_subject.strip():
            raise ContractViolationError("human approval requires an authenticated subject")
        return cls(
            requirement_id=requirement.requirement_id,
            plan_digest=requirement.plan_digest,
            human_subject=human_subject,
            verdict=verdict,
            decided_at=datetime.now(UTC),
            note=note,
        )


class ApprovalLedger(Protocol):
    def record(self, decision: HumanDecision) -> None: ...

    def decision_for(self, requirement_id: str) -> HumanDecision | None: ...


class InMemoryApprovalLedger:
    """Append-only local ledger; provider agents have no write method on it."""

    def __init__(self) -> None:
        self._decisions: dict[str, HumanDecision] = {}
        self._lock = RLock()

    def record(self, decision: HumanDecision) -> None:
        with self._lock:
            if decision.requirement_id in self._decisions:
                raise ContractViolationError("approval decisions are immutable once recorded")
            self._decisions[decision.requirement_id] = decision

    def decision_for(self, requirement_id: str) -> HumanDecision | None:
        with self._lock:
            return self._decisions.get(requirement_id)

    @property
    def decisions(self) -> tuple[HumanDecision, ...]:
        with self._lock:
            return tuple(self._decisions.values())
