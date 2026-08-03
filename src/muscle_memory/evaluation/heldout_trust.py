"""Reviewed trust root and durable one-shot claim for held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from muscle_memory.coordinator.models import canonical_json
from muscle_memory.evaluation.development import (
    DevelopmentEvidenceError,
    DevelopmentEvidenceReceipt,
    verify_development_evidence,
)
from muscle_memory.paths import REPOSITORY_ROOT


class HeldOutTrustRootError(DevelopmentEvidenceError):
    """The checked-in evidence chain differs from its reviewed trust root."""


class HeldOutAccessAlreadyConsumedError(RuntimeError):
    """The immutable candidate/world-set pair has already consumed its one shot."""


@dataclass(frozen=True, slots=True)
class HeldOutTrustRoot:
    """Human-reviewed identities which cannot be supplied through the CLI."""

    development_lock_path: Path
    development_lock_sha256: str
    development_lock_canonical_sha256: str
    development_evidence_path: Path
    development_evidence_sha256: str
    development_evidence_canonical_sha256: str
    candidate_checkpoint_path: Path
    candidate_checkpoint_sha256: str
    training_dataset_path: Path
    training_dataset_sha256: str
    training_evidence_path: Path
    training_evidence_sha256: str
    training_evidence_canonical_sha256: str
    development_round1_path: Path
    development_round1_sha256: str
    development_round1_canonical_sha256: str
    heldout_bundle_path: Path
    heldout_bundle_sha256: str
    heldout_bundle_canonical_sha256: str
    heldout_output_path: Path
    consumption_receipt_path: Path


# Review boundary: changing any path or digest below authorizes a new evidence chain.
# No training or evaluation command updates this object automatically.
DELIVERY_V2_HELDOUT_TRUST_ROOT = HeldOutTrustRoot(
    development_lock_path=REPOSITORY_ROOT / "evidence/policy/delivery-v2/lock.json",
    development_lock_sha256=("c91e1c485b55980402cece14a30671fe48fe0e0c81fae2b911eeea5974c1922c"),
    development_lock_canonical_sha256=(
        "53731391d2139454fec498738f6258fe98ad91775ba45c1af31effa46bd0acc1"
    ),
    development_evidence_path=REPOSITORY_ROOT
    / "evidence/policy/delivery-v2/development-evaluation.json",
    development_evidence_sha256=(
        "cf38e707b0c0ef098acac1f3f28066e4e650a5973e5e78a7b4cbe19c373460e9"
    ),
    development_evidence_canonical_sha256=(
        "b080b5e91e7ab7233ba9c2b95d509c89ab9cdf97b3aab80f73c4a17741d55c0a"
    ),
    candidate_checkpoint_path=REPOSITORY_ROOT / "models/policy/delivery-v2.npz",
    candidate_checkpoint_sha256=(
        "cb449b5fbaad448d2575fc8da7fd6151cec3412315d5ce0289aa8739db81937c"
    ),
    training_dataset_path=REPOSITORY_ROOT / "artifacts/policy/expert-v1.npz",
    training_dataset_sha256=("d3c7aa08ae467f0bf17eca13c116037ab2049e0da7d1ff95b2deb489252e20ef"),
    training_evidence_path=REPOSITORY_ROOT / "evidence/policy/delivery-v2/training.json",
    training_evidence_sha256=("ad11af923e20f3e6a1f412e313f370e60d449ab70af4d0388ba419922a9bd7ae"),
    training_evidence_canonical_sha256=(
        "dfaf743fe42501833a2ff4647c4c49f36734e0dea7eb3d19e4833bf94417cfaa"
    ),
    development_round1_path=REPOSITORY_ROOT
    / "evidence/policy/delivery-v2/development-evaluation-round1.json",
    development_round1_sha256=("72bfd477181e6ef54648890fcbd099eb48e6f280f2d60031b473e53b335504bf"),
    development_round1_canonical_sha256=(
        "973f7136eaa9ae18296f8ddee85a983eb2b799d5b95f4c8762b1c064002e04f1"
    ),
    heldout_bundle_path=REPOSITORY_ROOT / "config/worlds/heldout-v1.json",
    heldout_bundle_sha256=("e100a7ae654eebb7bbdf84ab8869a81da4e52c8b0adc6d350309ab93aae82165"),
    heldout_bundle_canonical_sha256=(
        "e3a0328eb1f4805e7023ef30b1b04af28d471d13c2e0f0f0d832ed2448bc6c7c"
    ),
    heldout_output_path=REPOSITORY_ROOT / "evidence/policy/delivery-v2/heldout-evaluation.json",
    consumption_receipt_path=REPOSITORY_ROOT
    / "evidence/policy/delivery-v2/heldout-consumption.json",
)


@dataclass(frozen=True, slots=True)
class TrustedHeldOutInputs:
    """Verified identities atomically recorded before held-out code is imported."""

    candidate_checkpoint_sha256: str
    heldout_bundle_sha256: str
    development_lock_sha256: str
    output_path: Path


def verify_checked_in_trust_root() -> DevelopmentEvidenceReceipt:
    """Verify every reviewed file identity without importing held-out code."""

    root = DELIVERY_V2_HELDOUT_TRUST_ROOT
    _verify_file(root.development_lock_path, root.development_lock_sha256)
    _verify_json_file(
        root.development_lock_path,
        root.development_lock_canonical_sha256,
    )
    _verify_file(root.development_evidence_path, root.development_evidence_sha256)
    _verify_json_file(
        root.development_evidence_path,
        root.development_evidence_canonical_sha256,
    )
    _verify_file(root.candidate_checkpoint_path, root.candidate_checkpoint_sha256)
    _verify_file(root.training_dataset_path, root.training_dataset_sha256)
    _verify_file(root.training_evidence_path, root.training_evidence_sha256)
    _verify_json_file(
        root.training_evidence_path,
        root.training_evidence_canonical_sha256,
    )
    _verify_file(root.development_round1_path, root.development_round1_sha256)
    _verify_json_file(
        root.development_round1_path,
        root.development_round1_canonical_sha256,
    )
    _verify_file(root.heldout_bundle_path, root.heldout_bundle_sha256)
    _verify_json_file(root.heldout_bundle_path, root.heldout_bundle_canonical_sha256)
    receipt = verify_development_evidence(
        root.development_evidence_path,
        root.candidate_checkpoint_path,
        lock_path=root.development_lock_path,
    )
    if (
        receipt.evidence_sha256 != root.development_evidence_sha256
        or receipt.evidence_canonical_sha256 != root.development_evidence_canonical_sha256
        or receipt.checkpoint_sha256 != root.candidate_checkpoint_sha256
    ):
        raise HeldOutTrustRootError(
            "verified development evidence differs from the reviewed trust root"
        )
    return receipt


def consume_checked_in_heldout_access() -> TrustedHeldOutInputs:
    """Claim the reviewed candidate/world-set pair exactly once before import."""

    receipt = verify_checked_in_trust_root()
    if not receipt.promotable:
        raise DevelopmentEvidenceError(
            "candidate failed the development gate; held-out access denied"
        )
    root = DELIVERY_V2_HELDOUT_TRUST_ROOT
    if root.heldout_output_path.exists():
        raise HeldOutAccessAlreadyConsumedError(
            "canonical held-out evidence already exists for this candidate"
        )
    inputs = TrustedHeldOutInputs(
        candidate_checkpoint_sha256=root.candidate_checkpoint_sha256,
        heldout_bundle_sha256=root.heldout_bundle_sha256,
        development_lock_sha256=root.development_lock_sha256,
        output_path=root.heldout_output_path,
    )
    _atomically_consume(
        root.consumption_receipt_path,
        inputs,
    )
    return inputs


def _atomically_consume(path: Path, inputs: TrustedHeldOutInputs) -> None:
    try:
        output_identity = str(inputs.output_path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        output_identity = str(inputs.output_path.resolve())
    payload = {
        "schema_version": 1,
        "state": "consumed_before_heldout_import",
        "candidate_checkpoint_sha256": inputs.candidate_checkpoint_sha256,
        "heldout_bundle_sha256": inputs.heldout_bundle_sha256,
        "development_lock_sha256": inputs.development_lock_sha256,
        "heldout_output_path": output_identity,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise HeldOutAccessAlreadyConsumedError(
                "candidate checkpoint and held-out bundle were already consumed"
            ) from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_file(path: Path, expected_sha256: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HeldOutTrustRootError(f"reviewed file is unavailable: {path.name}") from exc
    if actual != expected_sha256:
        raise HeldOutTrustRootError(
            f"reviewed file differs from its trust-root identity: {path.name}"
        )


def _verify_json_file(path: Path, expected_canonical_sha256: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        encoded = canonical_json(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HeldOutTrustRootError(f"reviewed JSON is invalid: {path.name}") from exc
    actual = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if actual != expected_canonical_sha256:
        raise HeldOutTrustRootError(
            f"reviewed JSON differs from its canonical trust-root identity: {path.name}"
        )


__all__ = [
    "DELIVERY_V2_HELDOUT_TRUST_ROOT",
    "HeldOutAccessAlreadyConsumedError",
    "HeldOutTrustRootError",
    "TrustedHeldOutInputs",
    "consume_checked_in_heldout_access",
    "verify_checked_in_trust_root",
]
