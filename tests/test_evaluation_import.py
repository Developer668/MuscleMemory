from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from muscle_memory.backend.evaluation_import import (
    HeldOutEvaluationAdmissionError,
    admit_held_out_evaluation,
    canonical_artifact_sha256,
)
from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.paths import (
    POLICY_V1_CHECKPOINT,
    POLICY_V2_CHECKPOINT,
    REPOSITORY_ROOT,
)
from muscle_memory.policy.network import BehaviorClonedPolicy
from muscle_memory.runtime import build_api_backend

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
V1_EVIDENCE = REPOSITORY_ROOT / "evidence/policy/delivery-v1/heldout-evaluation.json"
V1_EVIDENCE_RAW_SHA256 = "faf66fc609c19ccaf0fe8c9466968fbb3f1165328b2fc9b15800177859e191be"
V1_EVIDENCE_CANONICAL_SHA256 = "ea0a93bece3cc6fbad3d1ae66cdbc67b906f368ceb85cc59e3d78ea3aeead684"
_RESULTS = TypeAdapter(tuple[PolicyEpisodeResult, ...])


def _artifact(tmp_path: Path) -> Path:
    payload = json.loads(V1_EVIDENCE.read_text(encoding="utf-8"))
    payload["candidate_checkpoint_sha256"] = hashlib.sha256(
        POLICY_V1_CHECKPOINT.read_bytes()
    ).hexdigest()
    path = tmp_path / "heldout-evaluation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _environment(tmp_path: Path, artifact: Path) -> dict[str, str]:
    return {
        "MUSCLE_MEMORY_COORDINATOR_DB_PATH": str(tmp_path / "coordinator.sqlite3"),
        "MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "graph.jsonl"),
        "MUSCLE_MEMORY_TELEMETRY_SPOOL": str(tmp_path / "laser.sqlite3"),
        "MM_ASSET_CACHE_DIR": str(tmp_path / "assets"),
        "MM_ASSET_APPROVAL_LEDGER_DIR": str(tmp_path / "approvals"),
        "MM_HELDOUT_EVALUATION_ARTIFACT": str(artifact),
        "MM_HELDOUT_EVALUATION_ARTIFACT_SHA256": canonical_artifact_sha256(artifact),
        "MM_HELDOUT_CANDIDATE_CHECKPOINT": str(POLICY_V1_CHECKPOINT),
        "MM_HELDOUT_EVALUATED_AT": NOW.isoformat(),
    }


def _sequential_artifact(tmp_path: Path) -> tuple[Path, BehaviorClonedPolicy]:
    payload = json.loads(V1_EVIDENCE.read_text(encoding="utf-8"))
    candidate_policy = BehaviorClonedPolicy.load(POLICY_V2_CHECKPOINT)
    scope = candidate_policy.policy_hash[:16]
    for index, result in enumerate(payload["baseline_results"]):
        result["episode_id"] = f"heldout-{scope}-baseline-{index:02d}"
    for index, result in enumerate(payload["candidate_results"]):
        result["episode_id"] = f"heldout-{scope}-candidate-{index:02d}"
        result["policy_id"] = candidate_policy.policy_id
        result["policy_hash"] = candidate_policy.policy_hash
    payload["candidate_checkpoint_sha256"] = candidate_policy.policy_hash
    payload["promotion_decision"] = asdict(
        evaluate_promotion(
            _RESULTS.validate_python(payload["baseline_results"]),
            _RESULTS.validate_python(payload["candidate_results"]),
        )
    )
    path = tmp_path / "heldout-evaluation-v2.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, candidate_policy


def test_checked_in_legacy_v1_artifact_is_admitted_unchanged(tmp_path: Path) -> None:
    original = V1_EVIDENCE.read_bytes()
    assert hashlib.sha256(original).hexdigest() == V1_EVIDENCE_RAW_SHA256
    assert canonical_artifact_sha256(V1_EVIDENCE) == V1_EVIDENCE_CANONICAL_SHA256

    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    receipt = admit_held_out_evaluation(
        store,
        artifact_path=V1_EVIDENCE,
        expected_artifact_hash=V1_EVIDENCE_CANONICAL_SHA256,
        candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
        evaluated_at=NOW,
    )

    assert receipt.candidate_policy_id == "delivery-v1-bc"
    assert receipt.paired_world_count == 20
    assert len(store.held_out_evaluation_results_for_set("heldout-v1")) == 40
    assert V1_EVIDENCE.read_bytes() == original
    store.close()


def test_heldout_artifact_top_level_schemas_are_exact(tmp_path: Path) -> None:
    original = json.loads(V1_EVIDENCE.read_text(encoding="utf-8"))
    invalid_payloads: list[dict[str, object]] = []

    missing = dict(original)
    missing.pop("promotion_decision")
    invalid_payloads.append(missing)

    unexpected = dict(original)
    unexpected["unexpected"] = "not admitted"
    invalid_payloads.append(unexpected)

    mixed = dict(original)
    mixed["candidate_checkpoint_sha256"] = hashlib.sha256(
        POLICY_V1_CHECKPOINT.read_bytes()
    ).hexdigest()
    mixed.pop("baseline_results")
    invalid_payloads.append(mixed)

    boolean_version = dict(original)
    boolean_version["schema_version"] = True
    invalid_payloads.append(boolean_version)

    for index, payload in enumerate(invalid_payloads):
        artifact = tmp_path / f"invalid-{index}.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        store = CoordinatorStore(tmp_path / f"coordinator-{index}.sqlite3")
        with pytest.raises(
            HeldOutEvaluationAdmissionError,
            match="unsupported, missing, or unexpected fields",
        ):
            admit_held_out_evaluation(
                store,
                artifact_path=artifact,
                expected_artifact_hash=canonical_artifact_sha256(artifact),
                candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
                evaluated_at=NOW,
            )
        assert store.evaluated_checkpoints() == ()
        store.close()


def test_legacy_candidate_promotion_identity_must_match_checkpoint(
    tmp_path: Path,
) -> None:
    payload = json.loads(V1_EVIDENCE.read_text(encoding="utf-8"))
    payload["promotion_decision"]["candidate"]["policy_hash"] = "0" * 64
    artifact = tmp_path / "candidate-identity-tampered.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")

    with pytest.raises(
        HeldOutEvaluationAdmissionError,
        match="checkpoint identity does not match",
    ):
        admit_held_out_evaluation(
            store,
            artifact_path=artifact,
            expected_artifact_hash=canonical_artifact_sha256(artifact),
            candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluated_at=NOW,
        )
    assert store.evaluated_checkpoints() == ()
    store.close()


def test_heldout_result_boole_are_not_coerced_from_integers(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["candidate_results"][0]["success"] = int(payload["candidate_results"][0]["success"])
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")

    with pytest.raises(
        HeldOutEvaluationAdmissionError,
        match="invalid measured episode results",
    ):
        admit_held_out_evaluation(
            store,
            artifact_path=artifact,
            expected_artifact_hash=canonical_artifact_sha256(artifact),
            candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluated_at=NOW,
        )
    assert store.evaluated_checkpoints() == ()
    store.close()


def test_fresh_runtime_admits_checkpoint_and_forty_measured_results(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    environment = _environment(tmp_path, artifact)

    backend = build_api_backend(environment)
    checkpoints = {item.policy_id: item for item in backend.coordinator.evaluated_checkpoints()}
    assert set(checkpoints) == {"delivery-v0-direct-goal", "delivery-v1-bc"}
    assert {item.evaluation_evidence_hash for item in checkpoints.values()} == {
        environment["MM_HELDOUT_EVALUATION_ARTIFACT_SHA256"]
    }
    assert len(backend.coordinator.held_out_evaluation_results_for_set("heldout-v1")) == 40
    assert backend.coordinator.current_policy("stable") == "delivery-v0-direct-goal"
    asyncio.run(backend.shutdown())

    reopened = build_api_backend(environment)
    assert len(reopened.coordinator.held_out_evaluation_results_for_set("heldout-v1")) == 40
    assert reopened.coordinator.current_policy("stable") == "delivery-v0-direct-goal"
    asyncio.run(reopened.shutdown())


def test_rehashed_episode_tampering_cannot_enter_trusted_admission(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["candidate_results"][0]["body_collisions"] += 1
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    with pytest.raises(
        HeldOutEvaluationAdmissionError,
        match="promotion decision does not equal recomputed",
    ):
        admit_held_out_evaluation(
            store,
            artifact_path=artifact,
            expected_artifact_hash=canonical_artifact_sha256(artifact),
            candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluated_at=NOW,
        )
    assert store.evaluated_checkpoints() == ()
    assert store.held_out_evaluation_results_for_set("heldout-v1") == ()
    store.close()


def test_rehashed_terminal_lie_is_rejected_even_with_recomputed_decision(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    passing = next(item for item in payload["candidate_results"] if item["success"])
    passing["success"] = False
    payload["promotion_decision"] = asdict(
        evaluate_promotion(
            _RESULTS.validate_python(payload["baseline_results"]),
            _RESULTS.validate_python(payload["candidate_results"]),
        )
    )
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    with pytest.raises(
        HeldOutEvaluationAdmissionError,
        match="success and failed_reasons do not match canonical measurements",
    ):
        admit_held_out_evaluation(
            store,
            artifact_path=artifact,
            expected_artifact_hash=canonical_artifact_sha256(artifact),
            candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluated_at=NOW,
        )
    assert store.held_out_evaluation_results_for_set("heldout-v1") == ()
    store.close()


def test_sequential_candidate_artifacts_are_independently_scoped(
    tmp_path: Path,
) -> None:
    first = _artifact(tmp_path)
    second, second_policy = _sequential_artifact(tmp_path)
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")

    first_receipt = admit_held_out_evaluation(
        store,
        artifact_path=first,
        expected_artifact_hash=canonical_artifact_sha256(first),
        candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
        evaluated_at=NOW,
    )
    store.initialize_policy_alias(
        "already-promoted",
        first_receipt.candidate_policy_id,
        occurred_at=NOW,
    )
    assert (
        admit_held_out_evaluation(
            store,
            artifact_path=first,
            expected_artifact_hash=canonical_artifact_sha256(first),
            candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluated_at=NOW,
            stable_alias="already-promoted",
        ).candidate_policy_id
        == first_receipt.candidate_policy_id
    )

    second_receipt = admit_held_out_evaluation(
        store,
        artifact_path=second,
        expected_artifact_hash=canonical_artifact_sha256(second),
        candidate_checkpoint_path=POLICY_V2_CHECKPOINT,
        evaluated_at=NOW,
    )
    assert second_receipt.candidate_policy_id == second_policy.policy_id
    assert len(store.held_out_evaluation_results_for_set("heldout-v1")) == 80
    assert len(store.held_out_evaluation_results_for_artifact(first_receipt.artifact_hash)) == 40
    assert len(store.held_out_evaluation_results_for_artifact(second_receipt.artifact_hash)) == 40
    assert store.current_policy("stable") == first_receipt.baseline_policy_id
    assert store.current_policy("already-promoted") == first_receipt.candidate_policy_id
    store.close()


def test_external_canonical_hash_mismatch_fails_before_registration(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    store = CoordinatorStore(tmp_path / "coordinator.sqlite3")
    with pytest.raises(HeldOutEvaluationAdmissionError, match="independently configured"):
        admit_held_out_evaluation(
            store,
            artifact_path=artifact,
            expected_artifact_hash="0" * 64,
            candidate_checkpoint_path=POLICY_V1_CHECKPOINT,
            evaluated_at=NOW,
        )
    assert store.evaluated_checkpoints() == ()
    store.close()
