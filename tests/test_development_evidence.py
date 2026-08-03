from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from muscle_memory.coordinator.models import canonical_json
from muscle_memory.evaluation.development import (
    DevelopmentEvidenceError,
    assert_development_gate,
    verify_development_evidence,
)
from muscle_memory.evaluation.heldout_trust import (
    DELIVERY_V2_HELDOUT_TRUST_ROOT,
    HeldOutAccessAlreadyConsumedError,
    TrustedHeldOutInputs,
    _atomically_consume,
    verify_checked_in_trust_root,
)
from muscle_memory.paths import (
    POLICY_V2_CHECKPOINT,
    POLICY_V2_DEVELOPMENT_EVIDENCE,
)

LOCK = POLICY_V2_DEVELOPMENT_EVIDENCE.with_name("lock.json")
EXPECTED_EVIDENCE_SHA256 = "cf38e707b0c0ef098acac1f3f28066e4e650a5973e5e78a7b4cbe19c373460e9"
EXPECTED_CANONICAL_SHA256 = "b080b5e91e7ab7233ba9c2b95d509c89ab9cdf97b3aab80f73c4a17741d55c0a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bound_fixture(
    tmp_path: Path,
    *,
    evidence_payload: dict[str, object] | None = None,
    lock_payload: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    evidence = tmp_path / "development-evaluation.json"
    if evidence_payload is None:
        evidence.write_bytes(POLICY_V2_DEVELOPMENT_EVIDENCE.read_bytes())
    else:
        evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")
    lock = dict(lock_payload or json.loads(LOCK.read_text(encoding="utf-8")))
    lock["checkpoint_path"] = str(POLICY_V2_CHECKPOINT.resolve())
    lock["final_development_evidence_path"] = str(evidence.resolve())
    lock["final_development_evidence_sha256"] = _sha256(evidence)
    decoded_evidence = json.loads(evidence.read_text(encoding="utf-8"))
    lock["final_development_evidence_canonical_sha256"] = hashlib.sha256(
        canonical_json(decoded_evidence).encode("utf-8")
    ).hexdigest()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return evidence, lock_path


def test_checked_in_rejected_evidence_is_fully_verified_then_denied() -> None:
    assert _sha256(POLICY_V2_DEVELOPMENT_EVIDENCE) == EXPECTED_EVIDENCE_SHA256
    receipt = verify_development_evidence(
        POLICY_V2_DEVELOPMENT_EVIDENCE,
        POLICY_V2_CHECKPOINT,
        lock_path=LOCK,
    )

    assert receipt.evidence_sha256 == EXPECTED_EVIDENCE_SHA256
    assert receipt.evidence_canonical_sha256 == EXPECTED_CANONICAL_SHA256
    assert receipt.paired_world_count == 12
    assert not receipt.promotable
    assert receipt.selection_status == "rejected_before_heldout"
    trusted_receipt = verify_checked_in_trust_root()
    assert trusted_receipt == receipt
    with pytest.raises(DevelopmentEvidenceError, match="held-out access denied"):
        assert_development_gate(
            POLICY_V2_DEVELOPMENT_EVIDENCE,
            POLICY_V2_CHECKPOINT,
            lock_path=LOCK,
        )


def test_evidence_scope_explains_training_world_mechanics_without_heldout_use() -> None:
    payload = json.loads(POLICY_V2_DEVELOPMENT_EVIDENCE.read_text(encoding="utf-8"))

    assert payload["evaluation_scope"] == "generated_disjoint_development"
    assert payload["provenance"] == {
        "world_generation_mechanics": "training_world_generation",
        "evaluation_dataset_membership": "disjoint_development_only",
        "heldout_world_access": "never",
    }
    assert {item["world_split"] for item in payload["baseline_results"]} == {"training"}
    assert {item["world_split"] for item in payload["candidate_results"]} == {"training"}


def test_heldout_cli_imports_nothing_heldout_before_rejected_gate() -> None:
    script = """
import sys
import ops.policy.evaluate_heldout as cli
from muscle_memory.evaluation.development import DevelopmentEvidenceError
from muscle_memory.evaluation.heldout_trust import DELIVERY_V2_HELDOUT_TRUST_ROOT
assert 'muscle_memory.evaluation.heldout' not in sys.modules
receipt = DELIVERY_V2_HELDOUT_TRUST_ROOT.consumption_receipt_path
output = DELIVERY_V2_HELDOUT_TRUST_ROOT.heldout_output_path
assert not receipt.exists()
assert not output.exists()
sys.argv = ['evaluate_heldout']
try:
    cli.main()
except DevelopmentEvidenceError as exc:
    assert 'held-out access denied' in str(exc)
else:
    raise AssertionError('rejected development evidence opened held-out evaluation')
assert 'muscle_memory.evaluation.heldout' not in sys.modules
assert not receipt.exists()
assert not output.exists()
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_rejects_fabricated_lock_and_alternate_output_before_import(
    tmp_path: Path,
) -> None:
    fabricated_lock = tmp_path / "fabricated-lock.json"
    fabricated_lock.write_text("{}", encoding="utf-8")
    alternate_output = tmp_path / "alternate-output.json"
    script = f"""
import sys
import ops.policy.evaluate_heldout as cli
assert 'muscle_memory.evaluation.heldout' not in sys.modules
for arguments in (
    ['--development-lock', {str(fabricated_lock)!r}],
    ['--output', {str(alternate_output)!r}],
    ['--checkpoint', 'alternate.npz'],
    ['--development-evidence', 'alternate.json'],
):
    try:
        cli._parser().parse_args(arguments)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(f'CLI accepted untrusted path override: {{arguments}}')
    assert 'muscle_memory.evaluation.heldout' not in sys.modules
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not alternate_output.exists()


def test_atomic_consumption_rejects_repeat_independent_of_output_path(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "consumption.json"
    first = TrustedHeldOutInputs(
        candidate_checkpoint_sha256="a" * 64,
        heldout_bundle_sha256="b" * 64,
        development_lock_sha256="c" * 64,
        output_path=tmp_path / "first-output.json",
    )
    second = TrustedHeldOutInputs(
        candidate_checkpoint_sha256="a" * 64,
        heldout_bundle_sha256="b" * 64,
        development_lock_sha256="c" * 64,
        output_path=tmp_path / "different-output.json",
    )

    _atomically_consume(receipt, first)
    original = receipt.read_bytes()
    with pytest.raises(HeldOutAccessAlreadyConsumedError, match="already consumed"):
        _atomically_consume(receipt, second)

    assert receipt.read_bytes() == original
    payload = json.loads(original)
    assert payload["candidate_checkpoint_sha256"] == "a" * 64
    assert payload["heldout_bundle_sha256"] == "b" * 64


def test_rejected_checked_in_candidate_has_no_consumption_receipt() -> None:
    assert not DELIVERY_V2_HELDOUT_TRUST_ROOT.consumption_receipt_path.exists()
    assert not DELIVERY_V2_HELDOUT_TRUST_ROOT.heldout_output_path.exists()


def test_arbitrary_development_evidence_without_a_lock_is_rejected(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "arbitrary.json"
    evidence.write_bytes(POLICY_V2_DEVELOPMENT_EVIDENCE.read_bytes())

    with pytest.raises(DevelopmentEvidenceError, match="lock is unavailable"):
        verify_development_evidence(evidence, POLICY_V2_CHECKPOINT)


def test_lock_must_bind_the_exact_evidence_path(tmp_path: Path) -> None:
    first, lock = _write_bound_fixture(tmp_path)
    second = tmp_path / "other.json"
    second.write_bytes(first.read_bytes())

    with pytest.raises(DevelopmentEvidenceError, match="path does not equal"):
        verify_development_evidence(
            second,
            POLICY_V2_CHECKPOINT,
            lock_path=lock,
        )


def test_strict_result_schema_rejects_integer_for_boolean(tmp_path: Path) -> None:
    payload = json.loads(POLICY_V2_DEVELOPMENT_EVIDENCE.read_text(encoding="utf-8"))
    payload["candidate_results"][0]["success"] = 1
    evidence, lock = _write_bound_fixture(tmp_path, evidence_payload=payload)

    with pytest.raises(DevelopmentEvidenceError, match="unsupported strict schema"):
        verify_development_evidence(
            evidence,
            POLICY_V2_CHECKPOINT,
            lock_path=lock,
        )


def test_rehashed_aggregate_tampering_is_recomputed_and_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(POLICY_V2_DEVELOPMENT_EVIDENCE.read_text(encoding="utf-8"))
    payload["promotion_preview"]["candidate"]["success_count"] += 1
    evidence, lock = _write_bound_fixture(tmp_path, evidence_payload=payload)

    with pytest.raises(DevelopmentEvidenceError, match="recomputed measurements"):
        verify_development_evidence(
            evidence,
            POLICY_V2_CHECKPOINT,
            lock_path=lock,
        )


def test_rehashed_episode_measurement_tampering_is_recomputed_and_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(POLICY_V2_DEVELOPMENT_EVIDENCE.read_text(encoding="utf-8"))
    payload["candidate_results"][0]["energy_joules"] += 1.0
    evidence, lock = _write_bound_fixture(tmp_path, evidence_payload=payload)

    with pytest.raises(DevelopmentEvidenceError, match="recomputed measurements"):
        verify_development_evidence(
            evidence,
            POLICY_V2_CHECKPOINT,
            lock_path=lock,
        )


def test_checkpoint_bytes_must_match_the_lock(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tampered.npz"
    checkpoint.write_bytes(POLICY_V2_CHECKPOINT.read_bytes() + b"tampered")
    evidence, lock_path = _write_bound_fixture(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["checkpoint_path"] = str(checkpoint.resolve())
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(DevelopmentEvidenceError, match="locked hash"):
        verify_development_evidence(evidence, checkpoint, lock_path=lock_path)


def test_lock_schema_rejects_unexpected_fields(tmp_path: Path) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock["untrusted_override"] = True
    evidence, lock_path = _write_bound_fixture(tmp_path, lock_payload=lock)

    with pytest.raises(DevelopmentEvidenceError, match="unsupported strict schema"):
        verify_development_evidence(
            evidence,
            POLICY_V2_CHECKPOINT,
            lock_path=lock_path,
        )


def test_training_dataset_identity_is_cross_checked_through_metadata(
    tmp_path: Path,
) -> None:
    evidence, lock_path = _write_bound_fixture(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = Path(lock["training_evidence_path"])
    if not source.is_absolute():
        source = Path.cwd() / source
    training = json.loads(source.read_text(encoding="utf-8"))
    training["dataset_sha256"] = "f" * 64
    fabricated_training = tmp_path / "fabricated-training.json"
    fabricated_training.write_text(json.dumps(training), encoding="utf-8")
    lock["training_evidence_path"] = str(fabricated_training.resolve())
    lock["training_evidence_sha256"] = _sha256(fabricated_training)
    lock["training_evidence_canonical_sha256"] = hashlib.sha256(
        canonical_json(training).encode("utf-8")
    ).hexdigest()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(DevelopmentEvidenceError, match="lineage identities differ"):
        verify_development_evidence(
            evidence,
            POLICY_V2_CHECKPOINT,
            lock_path=lock_path,
        )
