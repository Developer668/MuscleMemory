"""Fail-closed protocol for one deterministic RocketRide pipeline step."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

FIXED_STEPS = (
    "validate_world",
    "run_episode",
    "summarize_telemetry",
    "query_graph_memory",
    "select_curriculum",
    "train_candidate_policy",
    "evaluate_candidate_policy",
    "promote_or_roll_back",
)

APPROVAL_KINDS = (
    "uncertain_physical_properties",
    "reward_change",
    "curriculum_change",
    "policy_promotion",
    "policy_rollback",
)

_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ContractError(ValueError):
    """A callback request or result violates the fixed protocol."""


class SequenceError(ContractError):
    """A request would skip, repeat differently, or reorder a fixed step."""


class ApprovalRejectedError(ContractError):
    """A gated step lacks evidence verified against the coordinator ledger."""


class HandlerExecutionError(RuntimeError):
    """An allowed domain handler failed after the request passed validation."""


def canonical_json(value: object) -> str:
    """Encode a value using the content-addressed orchestration JSON form."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("value must be finite, JSON-serializable data") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{name} is not a valid non-empty identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX_256.fullmatch(value) is None:
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ApprovalEvidence:
    requirement_id: str
    decision_id: str
    plan_digest: str
    step: str
    kind: str
    verdict: str
    human_subject: str
    decided_at: str

    @classmethod
    def parse(cls, raw: object) -> ApprovalEvidence:
        if not isinstance(raw, dict):
            raise ContractError("approval evidence must be a JSON object")
        required = {
            "requirement_id",
            "decision_id",
            "plan_digest",
            "step",
            "kind",
            "verdict",
            "human_subject",
            "decided_at",
        }
        if set(raw) != required:
            raise ContractError("approval evidence has missing or unexpected fields")
        evidence = cls(
            requirement_id=_require_sha256(raw["requirement_id"], "requirement_id"),
            decision_id=_require_sha256(raw["decision_id"], "decision_id"),
            plan_digest=_require_sha256(raw["plan_digest"], "approval plan_digest"),
            step=str(raw["step"]),
            kind=str(raw["kind"]),
            verdict=str(raw["verdict"]),
            human_subject=str(raw["human_subject"]),
            decided_at=str(raw["decided_at"]),
        )
        if evidence.step not in FIXED_STEPS:
            raise ContractError("approval evidence names an unknown step")
        if evidence.kind not in APPROVAL_KINDS:
            raise ContractError("approval evidence names an unknown approval kind")
        if evidence.verdict != "approve":
            raise ApprovalRejectedError("only an approved human decision can unlock a step")
        if not evidence.human_subject.strip():
            raise ContractError("approval evidence requires an authenticated human subject")
        try:
            decided_at = datetime.fromisoformat(evidence.decided_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError("approval decided_at must be an ISO-8601 timestamp") from exc
        if decided_at.tzinfo is None:
            raise ContractError("approval decided_at must include a timezone")
        return evidence


@dataclass(frozen=True, slots=True)
class StepEnvelope:
    contract_version: int
    run_id: str
    plan_digest: str
    step: str
    payload: dict[str, Any]
    approval_evidence: tuple[ApprovalEvidence, ...] = ()

    @classmethod
    def parse(cls, encoded: str) -> StepEnvelope:
        if not encoded or len(encoded.encode("utf-8")) > 1_048_576:
            raise ContractError("step envelope is empty or exceeds 1 MiB")
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ContractError("step envelope is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ContractError("step envelope must be a JSON object")
        if canonical_json(raw) != encoded:
            raise ContractError("step envelope must use canonical JSON encoding")
        allowed = {
            "contract_version",
            "run_id",
            "plan_digest",
            "step",
            "payload",
            "approval_evidence",
        }
        required = allowed - {"approval_evidence"}
        if not required.issubset(raw) or not set(raw).issubset(allowed):
            raise ContractError("step envelope has missing or unexpected fields")
        if raw["contract_version"] != 1:
            raise ContractError("unsupported step-envelope contract version")
        run_id = _require_identifier(raw["run_id"], "run_id")
        plan_digest = _require_sha256(raw["plan_digest"], "plan_digest")
        step = str(raw["step"])
        if step not in FIXED_STEPS:
            raise ContractError("step is not one of the eight allowed domain tools")
        payload = raw["payload"]
        if not isinstance(payload, dict):
            raise ContractError("payload must be a JSON object")
        _validate_step_payload(step, payload)
        evidence_raw = raw.get("approval_evidence", [])
        if not isinstance(evidence_raw, list):
            raise ContractError("approval_evidence must be a JSON array")
        evidence = tuple(ApprovalEvidence.parse(item) for item in evidence_raw)
        envelope = cls(
            contract_version=1,
            run_id=run_id,
            plan_digest=plan_digest,
            step=step,
            payload=cast(dict[str, Any], payload),
            approval_evidence=evidence,
        )
        envelope._validate_approval_shapes()
        return envelope

    @property
    def required_approval_kinds(self) -> tuple[str, ...]:
        if self.step == "validate_world" and self.payload["uncertain_physical_properties"]:
            return ("uncertain_physical_properties",)
        if self.step == "select_curriculum" and self.payload["curriculum_change_requested"]:
            return ("curriculum_change",)
        if self.step == "train_candidate_policy" and self.payload["reward_change_requested"]:
            return ("reward_change",)
        if self.step == "promote_or_roll_back":
            if self.payload["action"] == "promote":
                return ("policy_promotion",)
            return ("policy_rollback",)
        return ()

    def _validate_approval_shapes(self) -> None:
        expected = self.required_approval_kinds
        actual = tuple(item.kind for item in self.approval_evidence)
        if actual != expected:
            if expected:
                raise ApprovalRejectedError(
                    "gated step requires exactly the matching human approval evidence"
                )
            raise ContractError("ungated step must not carry unrelated approval evidence")
        for evidence in self.approval_evidence:
            if evidence.plan_digest != self.plan_digest or evidence.step != self.step:
                raise ApprovalRejectedError(
                    "approval evidence does not belong to this plan and step"
                )


def _validate_step_payload(step: str, payload: Mapping[str, object]) -> None:
    if step == "evaluate_candidate_policy":
        allowed = {
            "baseline_policy_id",
            "candidate_policy_id",
            "heldout_world_set_id",
        }
        if set(payload) != allowed:
            raise ContractError("evaluation accepts only policy ids and the held-out world-set id")
        for key in allowed:
            _require_identifier(payload[key], key)
    elif step == "validate_world":
        _require_boolean(payload, "uncertain_physical_properties")
    elif step == "select_curriculum":
        _require_boolean(payload, "curriculum_change_requested")
    elif step == "train_candidate_policy":
        _require_boolean(payload, "reward_change_requested")
    elif step == "promote_or_roll_back":
        if payload.get("action") not in {"promote", "roll_back"}:
            raise ContractError("final action must be promote or roll_back")


def _require_boolean(payload: Mapping[str, object], key: str) -> None:
    if not isinstance(payload.get(key), bool):
        raise ContractError(f"{key} must be an explicit boolean")


@dataclass(slots=True)
class _RunSequence:
    completed: list[str] = field(default_factory=list)
    request_hashes: dict[str, str] = field(default_factory=dict)
    result_json: dict[str, str] = field(default_factory=dict)
    in_flight: tuple[str, str] | None = None


class SequenceLedger:
    """Process-local order and idempotency guard; use a durable equivalent in production."""

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], _RunSequence] = {}
        self._lock = threading.Lock()

    def begin(self, envelope: StepEnvelope, request_sha256: str) -> str | None:
        key = (envelope.run_id, envelope.plan_digest)
        with self._lock:
            run = self._runs.setdefault(key, _RunSequence())
            index = FIXED_STEPS.index(envelope.step)
            if index < len(run.completed):
                if (
                    run.completed[index] == envelope.step
                    and run.request_hashes[envelope.step] == request_sha256
                ):
                    return run.result_json[envelope.step]
                raise SequenceError("completed step cannot be replayed with different content")
            if index != len(run.completed):
                expected = FIXED_STEPS[len(run.completed)]
                raise SequenceError(f"fixed pipeline expected {expected}, not {envelope.step}")
            if run.in_flight is not None:
                raise SequenceError("another step for this plan is already in flight")
            run.in_flight = (envelope.step, request_sha256)
            return None

    def complete(
        self,
        envelope: StepEnvelope,
        request_sha256: str,
        result_json: str,
    ) -> None:
        key = (envelope.run_id, envelope.plan_digest)
        with self._lock:
            run = self._runs[key]
            if run.in_flight != (envelope.step, request_sha256):
                raise SequenceError("step completion does not match the in-flight request")
            run.completed.append(envelope.step)
            run.request_hashes[envelope.step] = request_sha256
            run.result_json[envelope.step] = result_json
            run.in_flight = None

    def abort(self, envelope: StepEnvelope, request_sha256: str) -> None:
        key = (envelope.run_id, envelope.plan_digest)
        with self._lock:
            run = self._runs.get(key)
            if run is not None and run.in_flight == (envelope.step, request_sha256):
                run.in_flight = None


StepHandler = Callable[[Mapping[str, object]], Mapping[str, object]]
ApprovalVerifier = Callable[[ApprovalEvidence], bool]


class FixedStepDispatcher:
    """Dispatch exactly one validated step; this class contains no reasoning."""

    def __init__(
        self,
        handlers: Mapping[str, StepHandler],
        *,
        approval_verifier: ApprovalVerifier,
        sequence: SequenceLedger | None = None,
    ) -> None:
        if tuple(handlers) != FIXED_STEPS:
            raise ContractError("handlers must be supplied in exact fixed-pipeline order")
        self._handlers = dict(handlers)
        self._approval_verifier = approval_verifier
        self._sequence = sequence or SequenceLedger()

    def dispatch(self, encoded_envelope: str) -> str:
        envelope = StepEnvelope.parse(encoded_envelope)
        request_sha256 = sha256_text(encoded_envelope)
        cached = self._sequence.begin(envelope, request_sha256)
        if cached is not None:
            return cached
        try:
            for evidence in envelope.approval_evidence:
                if not self._approval_verifier(evidence):
                    raise ApprovalRejectedError(
                        "approval evidence was not found in the coordinator ledger"
                    )
            output = self._handlers[envelope.step](envelope.payload)
            if not isinstance(output, Mapping):
                raise ContractError("domain handler output must be a JSON object")
            output_dict = dict(output)
            output_json = canonical_json(output_dict)
            result = {
                "contract_version": 1,
                "output": output_dict,
                "output_sha256": sha256_text(output_json),
                "plan_digest": envelope.plan_digest,
                "request_sha256": request_sha256,
                "run_id": envelope.run_id,
                "status": "completed",
                "step": envelope.step,
            }
            result_json = canonical_json(result)
            self._sequence.complete(envelope, request_sha256, result_json)
            return result_json
        except (ContractError, ApprovalRejectedError):
            self._sequence.abort(envelope, request_sha256)
            raise
        except Exception as exc:
            self._sequence.abort(envelope, request_sha256)
            raise HandlerExecutionError(
                f"{envelope.step} handler failed ({type(exc).__name__})"
            ) from exc


def validate_result(encoded_result: str, envelope: StepEnvelope) -> dict[str, Any]:
    try:
        raw = json.loads(encoded_result)
    except json.JSONDecodeError as exc:
        raise ContractError("callback result is not valid JSON") from exc
    if not isinstance(raw, dict) or canonical_json(raw) != encoded_result:
        raise ContractError("callback result must be a canonical JSON object")
    required = {
        "contract_version",
        "output",
        "output_sha256",
        "plan_digest",
        "request_sha256",
        "run_id",
        "status",
        "step",
    }
    if set(raw) != required:
        raise ContractError("callback result has missing or unexpected fields")
    if raw["contract_version"] != 1 or raw["status"] != "completed":
        raise ContractError("callback result has an unsupported version or state")
    if (
        raw["run_id"] != envelope.run_id
        or raw["plan_digest"] != envelope.plan_digest
        or raw["step"] != envelope.step
    ):
        raise ContractError("callback result does not match its request")
    output = raw["output"]
    if not isinstance(output, dict):
        raise ContractError("callback result output must be a JSON object")
    if sha256_text(canonical_json(output)) != raw["output_sha256"]:
        raise ContractError("callback output checksum mismatch")
    return cast(dict[str, Any], raw)
