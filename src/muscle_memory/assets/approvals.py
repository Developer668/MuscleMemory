"""Persistent, append-only human decisions for asset physics proposals."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from muscle_memory.assets.models import (
    FrozenAssetModel,
    PhysicalField,
    PhysicalProposal,
    canonical_json_bytes,
    sha256_bytes,
)


class AssetApprovalError(RuntimeError):
    """An approval record was invalid, missing, or mutated."""


class HumanVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class AssetApprovalRequirement(FrozenAssetModel):
    """A blocking review request that agent code cannot self-resolve."""

    requirement_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_fields: tuple[PhysicalField, ...] = Field(min_length=1)
    blocking: Literal[True] = True
    reason: str = Field(min_length=1)

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        proposal: PhysicalProposal,
    ) -> AssetApprovalRequirement:
        review_fields = proposal.review_fields
        if not review_fields:
            raise AssetApprovalError("a non-blocking physical proposal needs no approval request")
        payload = {
            "blocking": True,
            "bundle_id": bundle_id,
            "proposal_sha256": proposal.proposal_sha256,
            "reason": "Safety-critical obstacle physics require a recorded human decision.",
            "review_fields": [field.value for field in review_fields],
            "schema_version": 1,
        }
        return cls(
            requirement_id=sha256_bytes(canonical_json_bytes(payload)),
            bundle_id=bundle_id,
            proposal_sha256=proposal.proposal_sha256,
            review_fields=review_fields,
            reason=str(payload["reason"]),
        )

    @model_validator(mode="after")
    def validate_content_address(self) -> AssetApprovalRequirement:
        payload = {
            "blocking": self.blocking,
            "bundle_id": self.bundle_id,
            "proposal_sha256": self.proposal_sha256,
            "reason": self.reason,
            "review_fields": [field.value for field in self.review_fields],
            "schema_version": 1,
        }
        if self.requirement_id != sha256_bytes(canonical_json_bytes(payload)):
            raise ValueError("approval requirement content address does not match its fields")
        return self


class HumanAssetDecision(FrozenAssetModel):
    """An authenticated human verdict bound to one exact physical proposal."""

    requirement_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_subject: str = Field(min_length=1)
    verdict: HumanVerdict
    decided_at: datetime
    note: str = ""

    @classmethod
    def create(
        cls,
        requirement: AssetApprovalRequirement,
        *,
        human_subject: str,
        verdict: HumanVerdict,
        note: str = "",
    ) -> HumanAssetDecision:
        if not human_subject.strip():
            raise AssetApprovalError("human approval requires an authenticated subject")
        return cls(
            requirement_id=requirement.requirement_id,
            proposal_sha256=requirement.proposal_sha256,
            human_subject=human_subject.strip(),
            verdict=verdict,
            decided_at=datetime.now(UTC),
            note=note,
        )

    @model_validator(mode="after")
    def require_timezone(self) -> HumanAssetDecision:
        if self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            raise ValueError("approval decisions require a timezone-aware timestamp")
        return self


class AssetApprovalLedger:
    """Filesystem ledger with immutable request and decision objects."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._requirements = root / "requirements"
        self._decisions = root / "decisions"
        self._requirements.mkdir(parents=True, exist_ok=True)
        self._decisions.mkdir(parents=True, exist_ok=True)

    def submit(self, requirement: AssetApprovalRequirement) -> None:
        data = canonical_json_bytes(requirement.model_dump(mode="json"))
        self._write_once(
            self._requirements / f"{requirement.requirement_id}.json",
            data,
            label="approval requirement",
        )

    def record_human_decision(self, decision: HumanAssetDecision) -> None:
        requirement = self.requirement_for(decision.requirement_id)
        if requirement is None:
            raise AssetApprovalError("cannot decide an unknown approval requirement")
        if requirement.proposal_sha256 != decision.proposal_sha256:
            raise AssetApprovalError("human decision does not match the reviewed proposal")
        data = canonical_json_bytes(decision.model_dump(mode="json"))
        path = self._decisions / f"{decision.requirement_id}.json"
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError as exc:
            raise AssetApprovalError("human approval decisions are immutable") from exc

    def requirement_for(self, requirement_id: str) -> AssetApprovalRequirement | None:
        path = self._requirements / f"{requirement_id}.json"
        if not path.exists():
            return None
        try:
            requirement = AssetApprovalRequirement.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise AssetApprovalError("approval requirement failed validation") from exc
        if requirement.requirement_id != requirement_id:
            raise AssetApprovalError("approval requirement is stored under the wrong ID")
        return requirement

    def decision_for(self, requirement_id: str) -> HumanAssetDecision | None:
        path = self._decisions / f"{requirement_id}.json"
        if not path.exists():
            return None
        try:
            decision = HumanAssetDecision.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise AssetApprovalError("human approval decision failed validation") from exc
        if decision.requirement_id != requirement_id:
            raise AssetApprovalError("human decision is stored under the wrong requirement")
        requirement = self.requirement_for(requirement_id)
        if requirement is None or requirement.proposal_sha256 != decision.proposal_sha256:
            raise AssetApprovalError("human decision is detached from its requirement")
        return decision

    @staticmethod
    def _write_once(path: Path, data: bytes, *, label: str) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise AssetApprovalError(f"could not verify existing {label}") from exc
            if existing != data:
                raise AssetApprovalError(f"immutable {label} has different content") from None
