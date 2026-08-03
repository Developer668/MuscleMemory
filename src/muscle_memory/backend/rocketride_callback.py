"""Authenticated, restart-safe RocketRide callback bound to real domain state."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.coordinator.models import PolicyAction
from muscle_memory.episodes import EpisodeClosure, EpisodeService
from muscle_memory.episodes.training import TrainingCorrectionFeed
from muscle_memory.graph_memory import (
    CurriculumQuery,
    EvaluatedPolicyVersion,
    PolicyComparisonRecord,
    PolicyEvaluationRecord,
    ResilientGraphMemory,
)
from muscle_memory.orchestration.approvals import HumanDecision, HumanVerdict
from muscle_memory.orchestration.contracts import (
    FIXED_PIPELINE,
    ApprovalRequirement,
    ExecutionPlan,
    PipelineCommand,
    PipelineStep,
    canonical_json,
    sha256_text,
)
from muscle_memory.orchestration.evidence import (
    CandidateEvaluationEvidence,
    EvaluationPolicyEvidence,
    GuildEvidenceBundle,
)
from muscle_memory.orchestration.service import ReviewedExecution

CALLBACK_PATH = "/webhook/muscle-memory-fixed-step"
MAX_CALLBACK_BODY_BYTES = 1_100_000
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_REVIEWED_EXECUTION = TypeAdapter(ReviewedExecution)


class CallbackContractError(ValueError):
    pass


class CallbackUnauthorizedError(CallbackContractError):
    pass


class CallbackApprovalError(CallbackContractError):
    pass


class CallbackSequenceError(CallbackContractError):
    pass


class CallbackHandlerError(RuntimeError):
    pass


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
    def parse(cls, value: object) -> ApprovalEvidence:
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
        if not isinstance(value, dict) or set(value) != required:
            raise CallbackContractError("approval evidence has missing or unexpected fields")
        if any(
            not isinstance(value[name], str)
            for name in required
        ):
            raise CallbackContractError("approval evidence fields must be strings")
        evidence = cls(**cast(dict[str, str], value))
        if _HEX_256.fullmatch(evidence.requirement_id) is None:
            raise CallbackContractError("approval requirement id is invalid")
        if _HEX_256.fullmatch(evidence.decision_id) is None:
            raise CallbackContractError("approval decision id is invalid")
        if _HEX_256.fullmatch(evidence.plan_digest) is None:
            raise CallbackContractError("approval plan digest is invalid")
        return evidence


@dataclass(frozen=True, slots=True)
class StepEnvelope:
    run_id: str
    plan_digest: str
    step: PipelineStep
    payload: dict[str, Any]
    approval_evidence: tuple[ApprovalEvidence, ...]

    @classmethod
    def parse(cls, encoded: str) -> StepEnvelope:
        if not encoded or len(encoded.encode("utf-8")) > 1_048_576:
            raise CallbackContractError("step envelope is empty or exceeds 1 MiB")
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise CallbackContractError("step envelope is not valid JSON") from exc
        if not isinstance(raw, dict) or canonical_json(raw) != encoded:
            raise CallbackContractError("step envelope must be a canonical JSON object")
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
            raise CallbackContractError("step envelope has missing or unexpected fields")
        if raw["contract_version"] != 1:
            raise CallbackContractError("unsupported step envelope version")
        run_id = raw["run_id"]
        digest = raw["plan_digest"]
        if not isinstance(run_id, str) or _IDENTIFIER.fullmatch(run_id) is None:
            raise CallbackContractError("run id is invalid")
        if not isinstance(digest, str) or _HEX_256.fullmatch(digest) is None:
            raise CallbackContractError("plan digest is invalid")
        try:
            step = PipelineStep(str(raw["step"]))
        except ValueError as exc:
            raise CallbackContractError("step is not in the fixed pipeline") from exc
        payload = raw["payload"]
        if not isinstance(payload, dict):
            raise CallbackContractError("step payload must be a JSON object")
        try:
            PipelineCommand.create(step, payload)
        except ValueError as exc:
            raise CallbackContractError(str(exc)) from exc
        evidence_raw = raw.get("approval_evidence", [])
        if not isinstance(evidence_raw, list):
            raise CallbackContractError("approval evidence must be a JSON array")
        return cls(
            run_id=run_id,
            plan_digest=digest,
            step=step,
            payload=cast(dict[str, Any], payload),
            approval_evidence=tuple(ApprovalEvidence.parse(item) for item in evidence_raw),
        )


StepHandler = Callable[[ExecutionPlan, Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class CallbackCredential:
    bearer_token: str = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.bearer_token) < 32:
            raise ValueError("RocketRide callback token must contain at least 32 characters")


class FixedStepDispatcher:
    """Dispatch the reviewed fixed pipeline without adding agent reasoning."""

    def __init__(
        self,
        *,
        coordinator: CoordinatorStore,
        episodes: EpisodeService,
        graph_memory: ResilientGraphMemory,
        credential: CallbackCredential,
    ) -> None:
        self._coordinator = coordinator
        self._episodes = episodes
        self._graph_memory = graph_memory
        self._credential = credential
        self._lock = RLock()
        self._handlers: dict[PipelineStep, StepHandler] = {
            PipelineStep.VALIDATE_WORLD: self._validate_world,
            PipelineStep.RUN_EPISODE: self._run_episode,
            PipelineStep.SUMMARIZE_TELEMETRY: self._summarize_telemetry,
            PipelineStep.QUERY_GRAPH_MEMORY: self._query_graph_memory,
            PipelineStep.SELECT_CURRICULUM: self._select_curriculum,
            PipelineStep.TRAIN_CANDIDATE_POLICY: self._train_candidate_policy,
            PipelineStep.EVALUATE_CANDIDATE_POLICY: self._evaluate_candidate_policy,
            PipelineStep.PROMOTE_OR_ROLL_BACK: self._promote_or_roll_back,
        }
        if tuple(self._handlers) != FIXED_PIPELINE:
            raise RuntimeError("RocketRide handlers do not match the fixed pipeline")

    def dispatch(self, encoded: str, authorization: str) -> dict[str, object]:
        self.authenticate(authorization)
        envelope = StepEnvelope.parse(encoded)
        request_sha256 = sha256_text(encoded)
        with self._lock:
            plan = self._coordinator.workflow_plan(envelope.run_id)
            if plan is None or plan.digest != envelope.plan_digest:
                raise CallbackContractError("callback references an unknown execution plan")
            self._require_executable_review(plan)
            command = plan.commands[FIXED_PIPELINE.index(envelope.step)]
            if command.payload != envelope.payload:
                raise CallbackContractError("callback payload does not match the immutable plan")
            self._verify_approval(plan, envelope)

            prior = self._coordinator.rocketride_callback_results(plan.run_id)
            step_index = FIXED_PIPELINE.index(envelope.step)
            if step_index < len(prior):
                prior_step, prior_request, prior_result = prior[step_index]
                if prior_step is envelope.step and prior_request == request_sha256:
                    return cast(dict[str, object], json.loads(prior_result))
                raise CallbackSequenceError(
                    "completed callback step cannot be replayed with different content"
                )
            if step_index != len(prior):
                raise CallbackSequenceError(
                    f"fixed pipeline expected {FIXED_PIPELINE[len(prior)].value}"
                )
            try:
                output = dict(self._handlers[envelope.step](plan, envelope.payload))
                output.setdefault(
                    "operation_execution",
                    self._synchronous_operation_execution(
                        plan,
                        envelope.step,
                        request_sha256=request_sha256,
                    ),
                )
                self._validate_operation_execution(output["operation_execution"])
                output_json = canonical_json(output)
            except (CallbackContractError, CallbackApprovalError, CallbackSequenceError):
                raise
            except Exception as exc:
                raise CallbackHandlerError(
                    f"{envelope.step.value} domain handler failed ({type(exc).__name__})"
                ) from exc
            result: dict[str, object] = {
                "contract_version": 1,
                "output": output,
                "output_sha256": sha256_text(output_json),
                "plan_digest": plan.digest,
                "request_sha256": request_sha256,
                "run_id": plan.run_id,
                "status": "completed",
                "step": envelope.step.value,
            }
            result_json = canonical_json(result)
            self._coordinator.record_rocketride_callback_result(
                plan.run_id,
                envelope.step,
                request_sha256,
                result_json,
            )
            return result

    def authenticate(self, authorization: str) -> None:
        """Authenticate a callback before its request body is consumed."""

        expected = f"Bearer {self._credential.bearer_token}"
        if not hmac.compare_digest(authorization, expected):
            raise CallbackUnauthorizedError("RocketRide callback authentication failed")

    def _verify_approval(self, plan: ExecutionPlan, envelope: StepEnvelope) -> None:
        requirement = next(
            (
                item
                for item in plan.approval_requirements
                if item.step is envelope.step
            ),
            None,
        )
        if requirement is None:
            if envelope.approval_evidence:
                raise CallbackContractError("ungated step carried unrelated approval evidence")
            return
        if len(envelope.approval_evidence) != 1:
            raise CallbackApprovalError("gated step requires one human approval")
        evidence = envelope.approval_evidence[0]
        decision = self._coordinator.human_decision_for(requirement.requirement_id)
        if decision is None or decision.verdict is not HumanVerdict.APPROVE:
            raise CallbackApprovalError("approved human decision was not found")
        expected = self._approval_payload(requirement, decision.decision_id, decision)
        if evidence != ApprovalEvidence(**expected):
            raise CallbackApprovalError("approval evidence does not match the coordinator ledger")

    def _require_executable_review(self, plan: ExecutionPlan) -> None:
        encoded = self._coordinator.workflow_review(plan.run_id)
        if encoded is None:
            raise CallbackContractError("callback requires a durable Guild review")
        try:
            reviewed = _REVIEWED_EXECUTION.validate_json(encoded)
        except ValidationError as exc:
            raise CallbackContractError("durable Guild review is invalid") from exc
        if reviewed.plan != plan or not reviewed.guild_reviews.executable:
            raise CallbackContractError(
                "all three exact-plan Guild reviews must recommend proceed"
            )

    @staticmethod
    def _approval_payload(
        requirement: ApprovalRequirement,
        decision_id: str,
        decision: HumanDecision,
    ) -> dict[str, str]:
        return {
            "requirement_id": requirement.requirement_id,
            "decision_id": decision_id,
            "plan_digest": requirement.plan_digest,
            "step": requirement.step.value,
            "kind": requirement.kind.value,
            "verdict": decision.verdict.value,
            "human_subject": decision.human_subject,
            "decided_at": decision.decided_at.isoformat(),
        }

    def _validate_world(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        del payload
        bundle = self._required_evidence(plan)
        evidence = bundle.world.world_evidence
        checks = evidence.validation.model_dump(mode="json")
        world_valid = evidence.robot_checksum_unchanged and all(checks.values())
        if not world_valid:
            raise CallbackContractError("trusted world evidence did not pass every validation")
        return {
            "world_valid": True,
            "world_id": evidence.world_id,
            "world_digest": evidence.world_digest,
            "baseline_path_digest": evidence.baseline_path_digest,
            "validation": checks,
        }

    def _run_episode(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        del plan
        closure = self._required_closure(str(payload.get("episode_id", "")))
        if payload.get("world_id") != closure.identity.world_id:
            raise CallbackContractError("episode closure belongs to a different world")
        return {
            "episode_id": closure.identity.episode_id,
            "success": closure.result.success,
            "failed_reasons": list(closure.result.failed_reasons),
            "telemetry_digest": closure.telemetry_digest,
            "operation_execution": self._async_job_completion(
                job_kind="simulation_episode",
                source_id=closure.identity.episode_id,
                completion_artifact_sha256=closure.telemetry_digest,
                completed_at=closure.closed_at.isoformat(),
            ),
        }

    def _summarize_telemetry(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        del plan
        closure = self._required_closure(str(payload.get("episode_id", "")))
        return {
            "episode_id": closure.identity.episode_id,
            "telemetry_digest": closure.telemetry_digest,
            "total_records": closure.telemetry.total_records,
            "provider_confirmed_records": closure.telemetry.provider_confirmed_records,
            "failure_ids": [failure.failure_id for failure in closure.failures],
        }

    def _query_graph_memory(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        del plan
        closure = self._required_closure(str(payload.get("episode_id", "")))
        result = self._graph_memory.query_curriculum(
            CurriculumQuery(
                failure_categories=tuple(sorted({item.category for item in closure.failures})),
                exclude_trained_policy_ids=(closure.identity.policy_id,),
            )
        )
        return {
            "episode_id": closure.identity.episode_id,
            "lesson_ids": [lesson.lesson_id for lesson in result.lessons],
            "provider_state": result.provider_state.value,
            "storage": result.storage.value,
        }

    def _select_curriculum(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        evidence = self._required_evidence(plan).failure_curriculum.failure_curriculum_evidence
        if payload.get("curriculum_change_requested") is not evidence.curriculum_change_requested:
            raise CallbackContractError("curriculum selection does not match trusted evidence")
        return {
            "curriculum_change_requested": evidence.curriculum_change_requested,
            "failure_signatures": [item.signature for item in evidence.failure_patterns],
            "proposal_id": (
                None
                if evidence.proposed_curriculum is None
                else evidence.proposed_curriculum.proposal_id
            ),
            "proposal_digest": (
                None
                if evidence.proposed_curriculum is None
                else evidence.proposed_curriculum.proposal_digest
            ),
        }

    def _train_candidate_policy(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        evidence = self._required_evidence(plan).evaluation.evaluation_evidence
        candidate_id = str(payload.get("candidate_policy_id", ""))
        checkpoint = self._checkpoint(candidate_id)
        if candidate_id != evidence.candidate.policy_id:
            raise CallbackContractError("candidate checkpoint does not match trusted evidence")
        self._verify_checkpoint(checkpoint, evidence.candidate)
        curriculum = self._required_evidence(plan).failure_curriculum.failure_curriculum_evidence
        lesson_ids = tuple(
            sorted(
                {
                    lesson_id
                    for pattern in curriculum.failure_patterns
                    for lesson_id in pattern.lesson_ids
                }
            )
        )
        lineage_receipts = TrainingCorrectionFeed(self._episodes).record_policy_lineage(
            policy=checkpoint,
            lesson_ids=lesson_ids,
            evidence_hash=curriculum.graph_query_digest,
        )
        return {
            "candidate_policy_id": checkpoint.policy_id,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "immutable_checkpoint_confirmed": True,
            "training_lesson_ids": list(lesson_ids),
            "training_lineage_record_ids": [
                receipt.record_id
                for receipt in lineage_receipts
                if receipt.record_kind == "policy_training"
            ],
            "reward_change_requested": payload.get("reward_change_requested"),
            "operation_execution": self._async_job_completion(
                job_kind="candidate_training",
                source_id=checkpoint.policy_id,
                completion_artifact_sha256=checkpoint.checkpoint_hash,
                completed_at=checkpoint.evaluated_at.isoformat(),
            ),
        }

    def _evaluate_candidate_policy(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        evidence = self._required_evidence(plan).evaluation.evaluation_evidence
        baseline = self._checkpoint(str(payload.get("baseline_policy_id", "")))
        candidate = self._checkpoint(str(payload.get("candidate_policy_id", "")))
        self._verify_checkpoint(
            baseline,
            evidence.baseline,
            require_current_artifact_binding=False,
        )
        self._verify_checkpoint(candidate, evidence.candidate)
        comparison = PolicyEvaluationRecord(
            candidate_policy_id=candidate.policy_id,
            baseline_policy_id=baseline.policy_id,
            evidence_hash=candidate.evaluation_evidence_hash,
            action=evidence.proposed_action,
            success_rate_delta=(
                evidence.candidate.success_rate - evidence.baseline.success_rate
            ),
            collision_rate_delta=(
                evidence.candidate.collision_rate - evidence.baseline.collision_rate
            ),
            measured_at=candidate.evaluated_at,
        )
        evaluation_receipt = self._graph_memory.record_policy_evaluation(comparison)
        outperformance_receipt = None
        if evidence.proposed_action == PolicyAction.PROMOTE.value:
            outperformance_receipt = self._graph_memory.record_outperformance(
                PolicyComparisonRecord(
                    candidate_policy_id=candidate.policy_id,
                    baseline_policy_id=baseline.policy_id,
                    evidence_hash=candidate.evaluation_evidence_hash,
                    success_rate_delta=(
                        evidence.candidate.success_rate - evidence.baseline.success_rate
                    ),
                    collision_rate_delta=(
                        evidence.candidate.collision_rate - evidence.baseline.collision_rate
                    ),
                    measured_at=candidate.evaluated_at,
                )
            )
        return {
            "heldout_world_set_id": evidence.heldout_world_set_id,
            "paired_world_count": evidence.paired_world_count,
            "baseline_policy_id": baseline.policy_id,
            "baseline_evidence_hash": baseline.evaluation_evidence_hash,
            "candidate_policy_id": candidate.policy_id,
            "candidate_evidence_hash": candidate.evaluation_evidence_hash,
            "candidate_metrics": json.loads(candidate.metrics_json),
            "policy_evaluation_record_id": evaluation_receipt.record_id,
            "outperformance_record_id": (
                None if outperformance_receipt is None else outperformance_receipt.record_id
            ),
            "operation_execution": self._async_job_completion(
                job_kind="paired_policy_evaluation",
                source_id=candidate.policy_id,
                completion_artifact_sha256=candidate.evaluation_evidence_hash,
                completed_at=candidate.evaluated_at.isoformat(),
            ),
        }

    def _promote_or_roll_back(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        decision = self._coordinator.numeric_policy_decision_for_run(plan.run_id)
        if decision is None:
            raise CallbackContractError("numeric policy decision is not recorded")
        action = PolicyAction(str(payload.get("action", "")))
        evaluation = self._required_evidence(plan).evaluation.evaluation_evidence
        expected_target_policy_id = (
            evaluation.candidate.policy_id
            if action is PolicyAction.PROMOTE
            else evaluation.baseline.policy_id
        )
        if (
            decision.action is not action
            or payload.get("candidate_policy_id") != evaluation.candidate.policy_id
            or decision.target_policy_id != expected_target_policy_id
        ):
            raise CallbackContractError("numeric policy decision does not match the final command")
        requirement = next(
            item
            for item in plan.approval_requirements
            if item.step is PipelineStep.PROMOTE_OR_ROLL_BACK
        )
        event = self._coordinator.apply_policy_action(
            decision.decision_id,
            requirement.requirement_id,
            occurred_at=decision.decided_at,
        )
        return {
            "action": action.value,
            "alias": event.alias,
            "sequence": event.sequence,
            "target_policy_id": event.target_policy_id,
            "numeric_decision_id": decision.decision_id,
            "human_approval_requirement_id": requirement.requirement_id,
        }

    def _required_evidence(self, plan: ExecutionPlan) -> GuildEvidenceBundle:
        bundle = self._coordinator.workflow_guild_evidence(plan.run_id)
        if bundle is None:
            raise CallbackContractError("trusted Guild evidence is unavailable")
        return bundle

    def _required_closure(self, episode_id: str) -> EpisodeClosure:
        closure = self._episodes.closure_for(episode_id)
        if closure is None:
            raise CallbackContractError("a durable closed episode is required")
        return closure

    def _checkpoint(self, policy_id: str) -> EvaluatedPolicyVersion:
        checkpoint = next(
            (
                item
                for item in self._coordinator.evaluated_checkpoints()
                if item.policy_id == policy_id
            ),
            None,
        )
        if checkpoint is None:
            raise CallbackContractError("immutable evaluated checkpoint is required")
        return checkpoint

    @staticmethod
    def _verify_checkpoint(
        checkpoint: EvaluatedPolicyVersion,
        evidence: EvaluationPolicyEvidence | CandidateEvaluationEvidence,
        *,
        require_current_artifact_binding: bool = True,
    ) -> None:
        if (
            checkpoint.evaluation_split != "held_out"
            or checkpoint.policy_id != evidence.policy_id
            or checkpoint.checkpoint_hash != evidence.policy_checksum
            or (
                require_current_artifact_binding
                and checkpoint.evaluation_evidence_hash != evidence.evaluation_id
            )
        ):
            raise CallbackContractError("evaluated checkpoint does not match trusted evidence")
        metrics = json.loads(checkpoint.metrics_json)
        expected: dict[str, int | float] = {
            "success_rate": evidence.success_rate,
            "collision_rate": evidence.collision_rate,
        }
        if isinstance(evidence, CandidateEvaluationEvidence):
            expected.update(
                {
                    "falls": evidence.falls,
                    "median_clearance_m": evidence.median_clearance_m,
                    "path_efficiency_regression_fraction": (
                        evidence.path_efficiency_regression_fraction
                    ),
                }
            )
        if not isinstance(metrics, dict) or any(
            metrics.get(key) != value for key, value in expected.items()
        ):
            raise CallbackContractError("checkpoint metrics do not match trusted evidence")

    @staticmethod
    def _synchronous_operation_execution(
        plan: ExecutionPlan,
        step: PipelineStep,
        *,
        request_sha256: str,
    ) -> dict[str, object]:
        return {
            "contract_version": 1,
            "execution_mode": "synchronous_domain_operation",
            "invocation_id": sha256_text(
                f"{plan.digest}:{step.value}:{request_sha256}:operation"
            ),
            "provider": "rocketride.ai",
            "step": step.value,
        }

    @staticmethod
    def _async_job_completion(
        *,
        job_kind: str,
        source_id: str,
        completion_artifact_sha256: str,
        completed_at: str,
    ) -> dict[str, object]:
        if _HEX_256.fullmatch(completion_artifact_sha256) is None:
            raise CallbackContractError("async job completion hash is invalid")
        return {
            "completion_artifact_sha256": completion_artifact_sha256,
            "completed_at": completed_at,
            "contract_version": 1,
            "execution_mode": "admitted_async_job_completion",
            "job_id": sha256_text(
                f"{job_kind}:{source_id}:{completion_artifact_sha256}"
            ),
            "job_kind": job_kind,
            "provider": "muscle-memory-worker",
            "source_id": source_id,
        }

    @staticmethod
    def _validate_operation_execution(value: object) -> None:
        if not isinstance(value, dict) or value.get("contract_version") != 1:
            raise CallbackContractError("operation execution contract is missing")
        mode = value.get("execution_mode")
        if mode == "synchronous_domain_operation":
            required = {
                "contract_version",
                "execution_mode",
                "invocation_id",
                "provider",
                "step",
            }
            digest = value.get("invocation_id")
        elif mode == "admitted_async_job_completion":
            required = {
                "completion_artifact_sha256",
                "completed_at",
                "contract_version",
                "execution_mode",
                "job_id",
                "job_kind",
                "provider",
                "source_id",
            }
            digest = value.get("job_id")
            completion = value.get("completion_artifact_sha256")
            if not isinstance(completion, str) or _HEX_256.fullmatch(completion) is None:
                raise CallbackContractError("async job completion artifact is invalid")
        else:
            raise CallbackContractError("operation execution mode is invalid")
        if set(value) != required:
            raise CallbackContractError("operation execution contract has unexpected fields")
        if not isinstance(digest, str) or _HEX_256.fullmatch(digest) is None:
            raise CallbackContractError("operation execution identity is invalid")


__all__ = [
    "CALLBACK_PATH",
    "MAX_CALLBACK_BODY_BYTES",
    "CallbackApprovalError",
    "CallbackContractError",
    "CallbackCredential",
    "CallbackHandlerError",
    "CallbackSequenceError",
    "CallbackUnauthorizedError",
    "FixedStepDispatcher",
]
