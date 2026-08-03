"""Development evidence gate that protects the frozen held-out split."""

from __future__ import annotations

import json
from pathlib import Path


def assert_development_gate(evidence_path: Path, candidate_hash: str) -> None:
    """Reject held-out access unless the exact candidate passed development."""
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_hash = str(payload["candidate_policy_sha256"])
        selection_status = str(payload["selection_status"])
        promotable = bool(payload["promotion_preview"]["promotable"])
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("development evidence is missing or invalid") from error
    if evidence_hash != candidate_hash:
        raise RuntimeError("development evidence belongs to a different checkpoint")
    if not promotable or selection_status != "eligible_for_blinded_heldout_evaluation":
        raise RuntimeError("candidate failed the development gate; held-out access denied")
