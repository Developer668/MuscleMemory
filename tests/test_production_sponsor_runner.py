from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from muscle_memory.orchestration.contracts import FIXED_PIPELINE  # noqa: E402
from ops.sponsors.run_production_evidence import (  # noqa: E402
    ProductionEvidenceError,
    _commands,
    _require_live_guild_review,
    _require_live_rocketride_run,
    _reserve_output,
    _write_once,
)


def test_production_commands_are_fixed_and_rollback_only() -> None:
    commands = _commands(
        episode_id="episode-production-1",
        world_id="world-production-1",
        baseline_policy_id="delivery-v0-direct-goal",
        candidate_policy_id="delivery-v1-bc",
    )

    assert tuple(command.step for command in commands) == FIXED_PIPELINE
    assert commands[6].payload == {
        "baseline_policy_id": "delivery-v0-direct-goal",
        "candidate_policy_id": "delivery-v1-bc",
        "heldout_world_set_id": "heldout-v1",
    }
    assert commands[7].payload == {
        "action": "roll_back",
        "candidate_policy_id": "delivery-v1-bc",
    }


def test_production_evidence_is_create_once(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "run.json"
    payload = {"schema_version": 1, "run_id": "production-run-1"}

    _write_once(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(ProductionEvidenceError, match="already exists"):
        _write_once(path, payload)


def test_output_is_reserved_before_external_work(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "run.json"
    revision = "a" * 40

    reservation = _reserve_output(path, revision)

    assert not path.exists()
    assert json.loads(reservation.read_text(encoding="utf-8"))["expected_revision"] == revision
    with pytest.raises(ProductionEvidenceError, match="already exists"):
        _reserve_output(path, revision)


def test_live_provider_evidence_fails_closed() -> None:
    guild = {
        "provider": {"state": "end_to_end_verified"},
        "reviews": [
            {"provider_session_id": "world-session"},
            {"provider_session_id": "failure-session"},
            {"provider_session_id": "safety-session"},
        ],
    }
    _require_live_guild_review(guild)
    guild["provider"] = {"state": "cached"}
    with pytest.raises(ProductionEvidenceError, match="end-to-end"):
        _require_live_guild_review(guild)

    rocketride = {
        "provider": {"state": "end_to_end_verified"},
        "completed_steps": [
            {
                "provider_task_receipt_sha256": "b" * 64,
                "provider_run_id": f"provider-run-{index}",
            }
            for index in range(8)
        ],
    }
    _require_live_rocketride_run(rocketride, expected_steps=8)
    rocketride["completed_steps"][0]["provider_task_receipt_sha256"] = None
    with pytest.raises(ProductionEvidenceError, match="task receipt"):
        _require_live_rocketride_run(rocketride, expected_steps=8)

    rocketride["completed_steps"] = rocketride["completed_steps"][:7]
    rocketride["completed_steps"][0]["provider_task_receipt_sha256"] = "b" * 64
    rocketride["provider"] = {"state": "healthy"}
    _require_live_rocketride_run(rocketride, expected_steps=7)
