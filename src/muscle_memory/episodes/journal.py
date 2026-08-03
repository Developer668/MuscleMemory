"""Persistence boundary for recoverable operational episode state."""

from __future__ import annotations

from typing import Protocol

from muscle_memory.episodes.models import (
    CorrectionApproval,
    CorrectionSubmission,
    EpisodeAppendReceipt,
    EpisodeClosure,
    EpisodeIdentity,
)


class EpisodeJournal(Protocol):
    """Append-only facts needed to reconstruct an ``EpisodeService``."""

    def identities(self) -> tuple[EpisodeIdentity, ...]: ...

    def receipts_for(self, episode_id: str) -> tuple[EpisodeAppendReceipt, ...]: ...

    def closure_for(self, episode_id: str) -> EpisodeClosure | None: ...

    def corrections(self) -> tuple[CorrectionSubmission, ...]: ...

    def approvals(self) -> tuple[CorrectionApproval, ...]: ...

    def record_identity(self, identity: EpisodeIdentity) -> None: ...

    def record_receipt(self, receipt: EpisodeAppendReceipt) -> None: ...

    def record_closure(self, closure: EpisodeClosure) -> None: ...

    def record_correction(self, correction: CorrectionSubmission) -> None: ...

    def record_approval(self, approval: CorrectionApproval) -> None: ...

    def record_approval_delivery(self, approval: CorrectionApproval) -> None: ...


class VolatileEpisodeJournal:
    """No-op journal retained for focused unit tests and explicitly ephemeral use."""

    def identities(self) -> tuple[EpisodeIdentity, ...]:
        return ()

    def receipts_for(self, episode_id: str) -> tuple[EpisodeAppendReceipt, ...]:
        del episode_id
        return ()

    def closure_for(self, episode_id: str) -> EpisodeClosure | None:
        del episode_id
        return None

    def corrections(self) -> tuple[CorrectionSubmission, ...]:
        return ()

    def approvals(self) -> tuple[CorrectionApproval, ...]:
        return ()

    def record_identity(self, identity: EpisodeIdentity) -> None:
        del identity

    def record_receipt(self, receipt: EpisodeAppendReceipt) -> None:
        del receipt

    def record_closure(self, closure: EpisodeClosure) -> None:
        del closure

    def record_correction(self, correction: CorrectionSubmission) -> None:
        del correction

    def record_approval(self, approval: CorrectionApproval) -> None:
        del approval

    def record_approval_delivery(self, approval: CorrectionApproval) -> None:
        del approval


__all__ = ["EpisodeJournal", "VolatileEpisodeJournal"]
