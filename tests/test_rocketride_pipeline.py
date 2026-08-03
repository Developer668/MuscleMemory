# ruff: noqa: E402
"""Static, callback, and SDK-protocol tests for the RocketRide artifact."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.rocketride.callback import CALLBACK_PATH, make_callback_server
from integrations.rocketride.live_verify import (
    LiveVerificationConfig,
    verify_live_provider,
)
from integrations.rocketride.protocol import (
    FIXED_STEPS,
    ApprovalRejectedError,
    ContractError,
    FixedStepDispatcher,
    HandlerExecutionError,
    SequenceError,
    SequenceLedger,
    StepEnvelope,
    canonical_json,
    sha256_text,
    validate_result,
)
from integrations.rocketride.runtime import ReviewedPipelineArtifact
from integrations.rocketride.validator import BUNDLE_ROOT, validate_bundle

PLAN_DIGEST = sha256_text("plan")
CALLBACK_TOKEN = "callback-test-token-0123456789abcdef"


def _approval(step: str, kind: str, decision: str = "decision") -> dict[str, object]:
    return {
        "decided_at": datetime.now(UTC).isoformat(),
        "decision_id": sha256_text(decision),
        "human_subject": "operator@example.test",
        "kind": kind,
        "plan_digest": PLAN_DIGEST,
        "requirement_id": sha256_text(f"requirement:{step}:{kind}"),
        "step": step,
        "verdict": "approve",
    }


def _encoded(
    step: str,
    payload: Mapping[str, object],
    *,
    evidence: list[dict[str, object]] | None = None,
) -> str:
    envelope: dict[str, object] = {
        "contract_version": 1,
        "payload": dict(payload),
        "plan_digest": PLAN_DIGEST,
        "run_id": "run-rocketride-test",
        "step": step,
    }
    if evidence is not None:
        envelope["approval_evidence"] = evidence
    return canonical_json(envelope)


def _payloads() -> dict[str, dict[str, object]]:
    return {
        "validate_world": {
            "uncertain_physical_properties": False,
            "world_id": "world-001",
        },
        "run_episode": {"episode_id": "episode-001", "world_id": "world-001"},
        "summarize_telemetry": {"episode_id": "episode-001"},
        "query_graph_memory": {"episode_id": "episode-001"},
        "select_curriculum": {
            "curriculum_change_requested": False,
            "episode_id": "episode-001",
        },
        "train_candidate_policy": {
            "candidate_policy_id": "candidate-001",
            "reward_change_requested": False,
        },
        "evaluate_candidate_policy": {
            "baseline_policy_id": "baseline-001",
            "candidate_policy_id": "candidate-001",
            "heldout_world_set_id": "heldout-v1",
        },
        "promote_or_roll_back": {
            "action": "promote",
            "candidate_policy_id": "candidate-001",
        },
    }


def _dispatcher(
    calls: list[str] | None = None,
    *,
    decisions: set[str] | None = None,
) -> FixedStepDispatcher:
    def handler(step: str) -> Any:
        def execute(payload: Mapping[str, object]) -> Mapping[str, object]:
            if calls is not None:
                calls.append(step)
            if step == "validate_world":
                return {"world_valid": True, "world_id": payload.get("world_id")}
            return {"accepted": True, "step": step}

        return execute

    handlers = {step: handler(step) for step in FIXED_STEPS}
    allowed = decisions or set()
    return FixedStepDispatcher(
        handlers,
        approval_verifier=lambda evidence: evidence.decision_id in allowed,
        sequence=SequenceLedger(),
    )


def test_reviewed_bundle_hashes_and_pipeline_shape_are_valid() -> None:
    evidence = validate_bundle()

    assert evidence["valid"] is True
    document = json.loads((BUNDLE_ROOT / "fixed-step.pipe").read_text(encoding="utf-8"))
    assert set(document) == {"pipeline"}
    pipeline = document["pipeline"]
    assert set(pipeline) == {"components", "project_id", "source", "viewport", "version"}
    assert pipeline["project_id"] == "00000000-0000-0000-0000-000000000000"
    assert pipeline["source"] == "fixed_step_source"
    assert pipeline["viewport"] == {"x": 0, "y": 0, "zoom": 1}
    assert tuple(component["provider"] for component in pipeline["components"]) == (
        "webhook",
        "tool_n8n",
        "response_text",
    )
    assert pipeline["components"][-1]["config"] == {"laneName": "text"}
    assert all("control" not in component for component in pipeline["components"])


def test_reviewed_artifact_builds_only_callback_sdk_environment() -> None:
    artifact = ReviewedPipelineArtifact.from_env(
        {
            "ROCKETRIDE_MM_COORDINATOR_URL": "https://coordinator.example.test",
            "ROCKETRIDE_MM_COORDINATOR_TOKEN": CALLBACK_TOKEN,
        }
    )

    assert artifact.pipeline_path == BUNDLE_ROOT / "fixed-step.pipe"
    assert len(artifact.pipeline_sha256) == 64
    assert artifact.sdk_environment == {
        "ROCKETRIDE_MM_COORDINATOR_URL": "https://coordinator.example.test",
        "ROCKETRIDE_MM_COORDINATOR_TOKEN": CALLBACK_TOKEN,
    }
    assert CALLBACK_TOKEN not in canonical_json(artifact.public_evidence)
    assert CALLBACK_TOKEN not in repr(artifact)


def test_dispatcher_executes_only_the_fixed_order_and_exact_retries_are_idempotent() -> None:
    calls: list[str] = []
    final_decision = sha256_text("final-approval")
    dispatcher = _dispatcher(calls, decisions={final_decision})
    payloads = _payloads()
    results: list[str] = []
    final_encoded = ""

    for step in FIXED_STEPS:
        evidence = None
        if step == "promote_or_roll_back":
            evidence = [_approval(step, "policy_promotion", "final-approval")]
        encoded = _encoded(step, payloads[step], evidence=evidence)
        if step == "promote_or_roll_back":
            final_encoded = encoded
        result_json = dispatcher.dispatch(encoded)
        result = validate_result(result_json, StepEnvelope.parse(encoded))
        assert result["step"] == step
        results.append(result_json)

    assert dispatcher.dispatch(final_encoded) == results[-1]
    assert calls == list(FIXED_STEPS)


def test_dispatcher_does_not_expose_handler_exception_text() -> None:
    secret = "domain-secret-should-not-cross-callback"

    def fail(_payload: Mapping[str, object]) -> Mapping[str, object]:
        raise RuntimeError(secret)

    dispatcher = FixedStepDispatcher(
        {step: fail for step in FIXED_STEPS},
        approval_verifier=lambda _evidence: False,
    )
    encoded = _encoded("validate_world", _payloads()["validate_world"])

    with pytest.raises(HandlerExecutionError) as error:
        dispatcher.dispatch(encoded)

    assert str(error.value) == "validate_world handler failed (RuntimeError)"
    assert secret not in str(error.value)


def test_dispatcher_rejects_reordering_unknown_tools_and_runtime_teacher_input() -> None:
    dispatcher = _dispatcher()

    with pytest.raises(SequenceError, match="expected validate_world"):
        dispatcher.dispatch(_encoded("run_episode", _payloads()["run_episode"]))

    unknown = json.loads(_encoded("run_episode", _payloads()["run_episode"]))
    unknown["step"] = "shell_command"
    with pytest.raises(ContractError, match="eight allowed"):
        StepEnvelope.parse(canonical_json(unknown))

    evaluation = _payloads()["evaluate_candidate_policy"] | {"expert_path": [[0.0, 0.0]]}
    with pytest.raises(ContractError, match="only policy ids"):
        StepEnvelope.parse(_encoded("evaluate_candidate_policy", evaluation))


def test_promotion_requires_ledger_verified_matching_human_evidence() -> None:
    payloads = _payloads()
    no_decisions = _dispatcher()
    for step in FIXED_STEPS[:-1]:
        no_decisions.dispatch(_encoded(step, payloads[step]))

    with pytest.raises(ApprovalRejectedError, match="requires exactly"):
        no_decisions.dispatch(_encoded("promote_or_roll_back", payloads["promote_or_roll_back"]))

    forged = _approval("promote_or_roll_back", "policy_promotion", "forged")
    with pytest.raises(ApprovalRejectedError, match="not found"):
        no_decisions.dispatch(
            _encoded(
                "promote_or_roll_back",
                payloads["promote_or_roll_back"],
                evidence=[forged],
            )
        )


def test_fake_callback_server_enforces_bearer_and_returns_typed_result() -> None:
    dispatcher = _dispatcher()
    server = make_callback_server(dispatcher, bearer_token=CALLBACK_TOKEN, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    envelope = _encoded("validate_world", _payloads()["validate_world"])
    body = canonical_json({"data": envelope}).encode()

    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=2) as response:
            health = json.load(response)
        assert health["state"] == "healthy"

        unauthorized = urllib.request.Request(
            f"{base}{CALLBACK_PATH}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(unauthorized, timeout=2)
        assert error.value.code == 401

        authorized = urllib.request.Request(
            f"{base}{CALLBACK_PATH}",
            data=body,
            headers={
                "Authorization": f"Bearer {CALLBACK_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(authorized, timeout=2) as response:
            result_json = response.read().decode()
        result = validate_result(result_json, StepEnvelope.parse(envelope))
        assert result["output"]["world_valid"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _FakeRocketRideClient:
    def __init__(self, events: list[object], **kwargs: object) -> None:
        self._events = events
        self._events.append(("init", kwargs))
        self._dispatcher = _dispatcher()

    async def __aenter__(self) -> _FakeRocketRideClient:
        self._events.append("enter")
        return self

    async def __aexit__(self, *args: object) -> None:
        self._events.append("exit")

    async def validate(self, pipeline: Mapping[str, object]) -> Mapping[str, object]:
        self._events.append(("validate", dict(pipeline)))
        assert set(pipeline) == {"pipeline"}
        inner = pipeline["pipeline"]
        assert isinstance(inner, Mapping)
        assert set(inner) == {"components", "project_id", "source", "viewport", "version"}
        assert inner["project_id"] == "00000000-0000-0000-0000-000000000000"
        return {"pipeline": {"components": inner["components"]}}

    async def use(self, *, filepath: str) -> Mapping[str, object]:
        self._events.append(("use", filepath))
        return {"token": "real-looking-fake-task-token"}

    async def send(
        self,
        token: str,
        payload: str,
        *,
        objinfo: Mapping[str, object],
        mimetype: str,
    ) -> Mapping[str, object]:
        self._events.append(("send", token, objinfo, mimetype))
        return {
            "result": [self._dispatcher.dispatch(payload)],
            "result_types": {"result": "text"},
        }

    async def terminate(self, token: str) -> None:
        self._events.append(("terminate", token))


def test_live_verifier_redacts_token_and_returns_hashes_with_fake_sdk(tmp_path: Path) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(
        _encoded("validate_world", _payloads()["validate_world"]),
        encoding="utf-8",
    )
    config = LiveVerificationConfig(
        uri="https://cloud.rocketride.ai",
        api_key="provider-secret",
        coordinator_url="https://coordinator.example.test",
        coordinator_token=CALLBACK_TOKEN,
        envelope_path=envelope_path,
    )
    events: list[object] = []
    assert "provider-secret" not in repr(config)
    assert CALLBACK_TOKEN not in repr(config)

    def factory(**kwargs: object) -> _FakeRocketRideClient:
        return _FakeRocketRideClient(events, **kwargs)

    exit_code, evidence = asyncio.run(verify_live_provider(config, client_factory=factory))

    assert exit_code == 0
    assert evidence["verified"] is True
    assert evidence["state"] == "end_to_end_verified"
    assert evidence["task_token_sha256"] == sha256_text("real-looking-fake-task-token")
    assert "real-looking-fake-task-token" not in canonical_json(evidence)
    assert len(str(evidence["pipeline_sha256"])) == 64
    assert events[-2:] == [("terminate", "real-looking-fake-task-token"), "exit"]


def test_live_verifier_reports_unconfigured_without_calling_sdk() -> None:
    exit_code, evidence = asyncio.run(verify_live_provider(environ={}))

    assert exit_code == 2
    assert evidence["verified"] is False
    assert evidence["state"] == "unconfigured"
    assert "ROCKETRIDE_URI" in evidence["missing"]
