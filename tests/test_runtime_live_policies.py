from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from muscle_memory.backend.evaluation_import import canonical_artifact_sha256
from muscle_memory.paths import POLICY_V1_CHECKPOINT, REPOSITORY_ROOT
from muscle_memory.runtime import build_api_backend


def test_runtime_exposes_evidence_bound_baseline_and_candidate(tmp_path: Path) -> None:
    evidence = REPOSITORY_ROOT / "evidence/policy/delivery-v1/heldout-evaluation.json"
    environment = {
        "MUSCLE_MEMORY_COORDINATOR_DB_PATH": str(tmp_path / "coordinator.sqlite3"),
        "MUSCLE_MEMORY_FALKORDB_CACHE_PATH": str(tmp_path / "graph.jsonl"),
        "MUSCLE_MEMORY_TELEMETRY_SPOOL": str(tmp_path / "laser.sqlite3"),
        "MM_ASSET_CACHE_DIR": str(tmp_path / "assets"),
        "MM_ASSET_APPROVAL_LEDGER_DIR": str(tmp_path / "approvals"),
        "MM_HELDOUT_EVALUATION_ARTIFACT": str(evidence),
        "MM_HELDOUT_EVALUATION_ARTIFACT_SHA256": canonical_artifact_sha256(evidence),
        "MM_HELDOUT_CANDIDATE_CHECKPOINT": str(POLICY_V1_CHECKPOINT),
        "MM_HELDOUT_EVALUATED_AT": datetime(2026, 8, 3, tzinfo=UTC).isoformat(),
        "MM_LIVE_MAX_DURATION_SECONDS": "10",
        "MM_LIVE_RENDER_WIDTH": "160",
        "MM_LIVE_RENDER_HEIGHT": "120",
        "MM_LIVE_JPEG_QUALITY": "75",
    }

    backend = build_api_backend(environment)
    try:
        controller = backend.live_episode_controller
        options = controller.options()
        policies = {item.policy_id: item for item in options.policies}

        assert set(policies) == {"delivery-v0-direct-goal", "delivery-v1-bc"}
        assert options.default_policy_id == "delivery-v0-direct-goal"
        assert options.maximum_duration_seconds == 10.0
        assert policies["delivery-v0-direct-goal"].deployment_status == "stable_deployed"
        assert policies["delivery-v0-direct-goal"].is_default is True
        assert policies["delivery-v0-direct-goal"].evaluated_episode_count == 20
        assert policies["delivery-v1-bc"].deployment_status == "candidate_live_test"
        assert policies["delivery-v1-bc"].is_default is False
    finally:
        asyncio.run(backend.shutdown())
