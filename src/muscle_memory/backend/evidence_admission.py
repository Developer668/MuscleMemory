"""Server-owned admission of content-addressed Guild workflow evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Literal, cast

from pydantic import TypeAdapter, ValidationError

from muscle_memory.backend.graph_prerequisites import derive_training_world_artifacts
from muscle_memory.coordinator import CoordinatorStore, ProviderEvidenceReference
from muscle_memory.coordinator.models import EpisodeState, PolicyGateMetrics
from muscle_memory.episodes.journal import EpisodeJournal
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import CurriculumQuery, ResilientGraphMemory
from muscle_memory.orchestration.contracts import (
    FIXED_PIPELINE,
    ExecutionPlan,
    PipelineStep,
    canonical_json,
    sha256_text,
)
from muscle_memory.orchestration.evidence import (
    CandidateEvaluationEvidence,
    EvaluationEvidence,
    EvaluationPolicyEvidence,
    FailureCurriculumEvidence,
    FailurePatternEvidence,
    GuildEvidenceBundle,
    WorldEvidence,
    WorldObstacleEvidence,
    WorldValidationEvidence,
    validate_evidence_plan_binding,
)

_POLICY_RESULTS = TypeAdapter(tuple[PolicyEpisodeResult, ...])
_POLICY_RESULT = TypeAdapter(PolicyEpisodeResult)


class WorkflowEvidenceAdmissionError(ValueError):
    """Submitted evidence does not equal independently reproduced domain facts."""


class TrustedWorkflowEvidenceAdmitter:
    """Verify a workflow bundle before creating provider evidence references."""

    def __init__(
        self,
        *,
        coordinator: CoordinatorStore,
        journal: EpisodeJournal,
        graph_memory: ResilientGraphMemory,
    ) -> None:
        self._coordinator = coordinator
        self._journal = journal
        self._graph_memory = graph_memory

    def admit(
        self,
        plan: ExecutionPlan,
        bundle: GuildEvidenceBundle,
    ) -> GuildEvidenceBundle:
        validate_evidence_plan_binding(bundle, plan)
        expected = self.reproduce(
            plan,
            world_evidence_id=bundle.world.evidence_id,
            failure_curriculum_evidence_id=bundle.failure_curriculum.evidence_id,
            evaluation_evidence_id=bundle.evaluation.evidence_id,
        )
        if bundle != expected:
            raise WorkflowEvidenceAdmissionError(
                "submitted workflow evidence does not equal reproduced domain facts"
            )
        world = expected.world.world_evidence
        curriculum = expected.failure_curriculum.failure_curriculum_evidence
        evaluation = expected.evaluation.evaluation_evidence
        world_observed_at = self._world_observed_at(plan)
        curriculum_observed_at = self._curriculum_observed_at(plan)
        evaluation_observed_at = self._evaluation_observed_at(plan)

        observations = (
            (
                bundle.world.evidence_id,
                "guild_world_evidence",
                world.world_id,
                world_observed_at,
                "muscle-memory-world-validation",
            ),
            (
                bundle.failure_curriculum.evidence_id,
                "guild_failure_curriculum_evidence",
                curriculum.graph_query_digest,
                curriculum_observed_at,
                "muscle-memory-explicit-experience",
            ),
            (
                bundle.evaluation.evidence_id,
                "guild_evaluation_evidence",
                evaluation.heldout_world_set_id,
                evaluation_observed_at,
                "muscle-memory-heldout-evaluation",
            ),
        )
        artifact_hashes = {
            (evidence_id, kind): artifact_hash
            for evidence_id, kind, artifact_hash in bundle.artifact_hashes()
        }
        for evidence_id, kind, provider_object_id, observed_at, provider in observations:
            self._coordinator.record_provider_evidence(
                ProviderEvidenceReference(
                    evidence_id=evidence_id,
                    provider=provider,
                    evidence_kind=kind,
                    provider_object_id=provider_object_id,
                    artifact_hash=artifact_hashes[(evidence_id, kind)],
                    observed_at=observed_at,
                )
            )
        return bundle

    def reproduce(
        self,
        plan: ExecutionPlan,
        *,
        world_evidence_id: str,
        failure_curriculum_evidence_id: str,
        evaluation_evidence_id: str,
    ) -> GuildEvidenceBundle:
        """Reproduce a bundle solely from durable server-owned domain facts."""

        world, world_observed_at = self._world_evidence(plan)
        curriculum, curriculum_observed_at = self._failure_curriculum_evidence(plan)
        evaluation, evaluation_observed_at = self._evaluation_evidence(plan)
        del world_observed_at, curriculum_observed_at, evaluation_observed_at
        return GuildEvidenceBundle.model_validate(
            {
                "world": {
                    "evidence_id": world_evidence_id,
                    "world_evidence": world.model_dump(mode="json"),
                },
                "failure_curriculum": {
                    "evidence_id": failure_curriculum_evidence_id,
                    "failure_curriculum_evidence": curriculum.model_dump(mode="json"),
                },
                "evaluation": {
                    "evidence_id": evaluation_evidence_id,
                    "evaluation_evidence": evaluation.model_dump(mode="json"),
                },
            }
        )

    def _world_observed_at(self, plan: ExecutionPlan) -> datetime:
        episode_id = str(self._payload(plan, PipelineStep.RUN_EPISODE)["episode_id"])
        closure = self._journal.closure_for(episode_id)
        if closure is None:
            raise WorkflowEvidenceAdmissionError("workflow episode is not durably closed")
        return closure.closed_at

    def _curriculum_observed_at(self, plan: ExecutionPlan) -> datetime:
        source_policy_id = str(
            self._payload(plan, PipelineStep.EVALUATE_CANDIDATE_POLICY)[
                "baseline_policy_id"
            ]
        )
        values = tuple(
            closure.closed_at
            for identity in self._journal.identities()
            if identity.policy_id == source_policy_id
            and (closure := self._journal.closure_for(identity.episode_id)) is not None
        )
        if not values:
            raise WorkflowEvidenceAdmissionError("workflow has no source episode evidence")
        return max(values)

    def _evaluation_observed_at(self, plan: ExecutionPlan) -> datetime:
        payload = self._payload(plan, PipelineStep.EVALUATE_CANDIDATE_POLICY)
        ids = {str(payload["baseline_policy_id"]), str(payload["candidate_policy_id"])}
        values = tuple(
            checkpoint.evaluated_at
            for checkpoint in self._coordinator.evaluated_checkpoints()
            if checkpoint.policy_id in ids
        )
        if len(values) != 2:
            raise WorkflowEvidenceAdmissionError("workflow evaluation checkpoints are missing")
        return max(values)

    def _world_evidence(self, plan: ExecutionPlan) -> tuple[WorldEvidence, datetime]:
        episode_id = str(self._payload(plan, PipelineStep.RUN_EPISODE)["episode_id"])
        closure = self._journal.closure_for(episode_id)
        if closure is None or not closure.graph.complete:
            raise WorkflowEvidenceAdmissionError(
                "world evidence requires a durable closure with complete graph handoff"
            )
        derived = derive_training_world_artifacts(
            closure.result.world_seed,
            recorded_at=closure.closed_at,
        )
        if (
            derived.world.world_id != closure.identity.world_id
            or derived.world.world_hash != closure.identity.world_hash
        ):
            raise WorkflowEvidenceAdmissionError(
                "closed episode world is not the deterministic validated seed result"
            )
        records = {record.obstacle_id: record for record in derived.obstacles}
        obstacles = tuple(
            WorldObstacleEvidence(
                obstacle_id=f"{derived.definition.world_id}:{item.object_id}",
                proposal_digest=records[
                    f"{derived.definition.world_id}:{item.object_id}"
                ].obstacle_hash,
                dimensions_m=(
                    item.collider.dimensions.length_m,
                    item.collider.dimensions.width_m,
                    item.collider.dimensions.height_m,
                ),
                mass_kg=item.physical.mass_kg,
                friction=item.physical.sliding_friction,
                property_origin="catalog_confirmed",
                collision_geometry="primitive",
                render_mesh_used_for_collision=False,
            )
            for item in derived.definition.objects
        )
        return (
            WorldEvidence(
                world_id=derived.world.world_id,
                world_digest=derived.world.world_hash,
                baseline_path_digest=derived.baseline_path_digest,
                robot_checksum_unchanged=True,
                validation=WorldValidationEvidence(
                    no_overlapping_objects=True,
                    start_destination_connected=True,
                    passages_meet_minimum_clearance=True,
                    approved_colliders_only=True,
                    baseline_path_exists=True,
                    physical_parameters_within_safe_limits=True,
                ),
                obstacles=obstacles,
            ),
            closure.closed_at,
        )

    def _failure_curriculum_evidence(
        self,
        plan: ExecutionPlan,
    ) -> tuple[FailureCurriculumEvidence, datetime]:
        source_policy_id = str(
            self._payload(plan, PipelineStep.EVALUATE_CANDIDATE_POLICY)[
                "baseline_policy_id"
            ]
        )
        groups: dict[str, set[str]] = defaultdict(set)
        failure_ids: dict[str, set[str]] = defaultdict(set)
        observed: list[datetime] = []
        for identity in self._journal.identities():
            closure = self._journal.closure_for(identity.episode_id)
            if closure is None or identity.policy_id != source_policy_id:
                continue
            observed.append(closure.closed_at)
            for failure in closure.failures:
                groups[failure.category].add(identity.episode_id)
                failure_ids[failure.category].add(failure.failure_id)
        recurring = {
            category: episode_ids
            for category, episode_ids in groups.items()
            if len(episode_ids) >= 2
        }
        if not recurring:
            raise WorkflowEvidenceAdmissionError(
                "failure evidence requires at least two durable source episodes"
            )
        change_requested = bool(
            self._payload(plan, PipelineStep.SELECT_CURRICULUM)[
                "curriculum_change_requested"
            ]
        )
        if change_requested:
            raise WorkflowEvidenceAdmissionError(
                "curriculum changes require a separately admitted generated-world proposal"
            )
        query = CurriculumQuery(
            failure_categories=tuple(sorted(recurring)),
            exclude_trained_policy_ids=(source_policy_id,),
        )
        result = self._graph_memory.query_curriculum(query)
        semantic_query = {
            "lessons": [lesson.model_dump(mode="json") for lesson in result.lessons],
            "query": query.model_dump(mode="json"),
        }
        query_digest = sha256_text(canonical_json(semantic_query))
        approved_by_failure = {
            approval.submission.failure_id: approval.submission.correction_id
            for approval in self._journal.approvals()
        }
        patterns = tuple(
            FailurePatternEvidence(
                signature=category,
                source_episode_ids=tuple(sorted(episode_ids)),
                distinct_source_episode_count=len(episode_ids),
                obstacle_categories=tuple(
                    sorted(
                        {
                            lesson.obstacle_category
                            for lesson in result.lessons
                            if lesson.failure_category == category
                            and lesson.obstacle_category is not None
                        }
                    )
                ),
                approved_correction_ids=tuple(
                    sorted(
                        {
                            approved_by_failure[failure_id]
                            for failure_id in failure_ids[category]
                            if failure_id in approved_by_failure
                        }
                    )
                ),
                lesson_ids=tuple(
                    sorted(
                        lesson.lesson_id
                        for lesson in result.lessons
                        if lesson.failure_category == category
                    )
                ),
            )
            for category, episode_ids in sorted(recurring.items())
        )
        return (
            FailureCurriculumEvidence(
                source_split="training",
                source_policy_id=source_policy_id,
                graph_query_digest=query_digest,
                failure_patterns=patterns,
                curriculum_change_requested=False,
                proposed_curriculum=None,
            ),
            max(observed),
        )

    def _evaluation_evidence(self, plan: ExecutionPlan) -> tuple[EvaluationEvidence, datetime]:
        payload = self._payload(plan, PipelineStep.EVALUATE_CANDIDATE_POLICY)
        baseline_id = str(payload["baseline_policy_id"])
        candidate_id = str(payload["candidate_policy_id"])
        heldout_world_set_id = str(payload["heldout_world_set_id"])
        checkpoints = {
            checkpoint.policy_id: checkpoint
            for checkpoint in self._coordinator.evaluated_checkpoints()
        }
        try:
            baseline = checkpoints[baseline_id]
            candidate = checkpoints[candidate_id]
        except KeyError as exc:
            raise WorkflowEvidenceAdmissionError(
                "evaluation evidence references an unknown checkpoint"
            ) from exc
        if baseline.evaluation_split != "held_out" or candidate.evaluation_split != "held_out":
            raise WorkflowEvidenceAdmissionError(
                "workflow evaluation requires held-out evaluated checkpoints"
            )
        measured = self._coordinator.held_out_evaluation_results_for_set(
            heldout_world_set_id
        )
        artifact_hashes = {item.evaluation_artifact_hash for item in measured}
        if len(measured) != 40 or len(artifact_hashes) != 1:
            raise WorkflowEvidenceAdmissionError(
                "evaluation evidence requires forty results from one admitted artifact"
            )
        artifact_hash = next(iter(artifact_hashes))
        artifact = self._coordinator.held_out_evaluation_artifact(artifact_hash)
        if artifact is None or artifact.held_out_world_set_id != heldout_world_set_id:
            raise WorkflowEvidenceAdmissionError(
                "measured results do not reference an admitted held-out artifact"
            )
        artifact_payload = cast(dict[str, object], json.loads(artifact.artifact_json))
        try:
            baseline_results = _POLICY_RESULTS.validate_python(
                artifact_payload["baseline_results"]
            )
            candidate_results = _POLICY_RESULTS.validate_python(
                artifact_payload["candidate_results"]
            )
        except (KeyError, ValidationError) as exc:
            raise WorkflowEvidenceAdmissionError(
                "admitted held-out artifact no longer validates"
            ) from exc
        stored_by_episode = {item.episode_id: item.result_json for item in measured}
        artifact_results = (*baseline_results, *candidate_results)
        if len(stored_by_episode) != 40 or any(
            stored_by_episode.get(item.episode_id)
            != canonical_json(_POLICY_RESULT.dump_python(item, mode="json"))
            for item in artifact_results
        ):
            raise WorkflowEvidenceAdmissionError(
                "stored measured results do not equal the admitted artifact"
            )
        try:
            decision = evaluate_promotion(baseline_results, candidate_results)
        except ValueError as exc:
            raise WorkflowEvidenceAdmissionError(
                "held-out artifact does not contain an exact paired comparison"
            ) from exc
        if len(baseline_results) != 20 or len(candidate_results) != 20:
            raise WorkflowEvidenceAdmissionError(
                "evaluation evidence requires exactly twenty paired held-out worlds"
            )
        if (
            baseline_results[0].policy_id != baseline.policy_id
            or baseline_results[0].policy_hash != baseline.checkpoint_hash
            or candidate_results[0].policy_id != candidate.policy_id
            or candidate_results[0].policy_hash != candidate.checkpoint_hash
            or baseline.evaluation_evidence_hash != artifact_hash
            or candidate.evaluation_evidence_hash != artifact_hash
        ):
            raise WorkflowEvidenceAdmissionError(
                "evaluated checkpoints do not match the admitted measured results"
            )
        episode_id = str(self._payload(plan, PipelineStep.RUN_EPISODE)["episode_id"])
        source_closure = self._journal.closure_for(episode_id)
        if (
            source_closure is None
            or source_closure.identity.policy_id != baseline.policy_id
            or source_closure.identity.policy_hash != baseline.checkpoint_hash
            or any(
                item.robot_checksum != source_closure.identity.robot_checksum
                for item in artifact_results
            )
        ):
            raise WorkflowEvidenceAdmissionError(
                "workflow source closure does not exactly match the evaluation baseline"
            )
        if any(
            self._coordinator.episode_state(item.episode_id)
            not in {EpisodeState.SUCCEEDED, EpisodeState.FAILED}
            for item in measured
        ):
            raise WorkflowEvidenceAdmissionError(
                "held-out provenance contains an unfinished evaluation episode"
            )
        proposed_action = str(
            self._payload(plan, PipelineStep.PROMOTE_OR_ROLL_BACK)["action"]
        )
        if proposed_action not in {"promote", "roll_back"}:
            raise WorkflowEvidenceAdmissionError("policy action is invalid")
        evidence = EvaluationEvidence(
            heldout_world_set_id=heldout_world_set_id,
            heldout_world_set_digest=str(artifact_payload["heldout_bundle_sha256"]),
            paired_world_count=20,
            baseline=EvaluationPolicyEvidence(
                policy_id=baseline.policy_id,
                policy_checksum=baseline.checkpoint_hash,
                evaluation_id=baseline.evaluation_evidence_hash,
                success_rate=decision.baseline.success_rate,
                collision_rate=decision.baseline.collision_rate,
            ),
            candidate=CandidateEvaluationEvidence(
                policy_id=candidate.policy_id,
                policy_checksum=candidate.checkpoint_hash,
                evaluation_id=candidate.evaluation_evidence_hash,
                success_rate=decision.candidate.success_rate,
                collision_rate=decision.candidate.collision_rate,
                falls=decision.candidate.total_falls,
                median_clearance_m=decision.candidate.median_minimum_clearance_m,
                path_efficiency_regression_fraction=decision.path_efficiency_regression,
            ),
            proposed_action=cast(Literal["promote", "roll_back"], proposed_action),
        )
        if ("promote" if self._gate_metrics(evidence).passes_promotion_gate else "roll_back") != (
            evidence.proposed_action
        ):
            raise WorkflowEvidenceAdmissionError(
                "proposed policy action does not match the recomputed numeric gate"
            )
        return evidence, max(baseline.evaluated_at, candidate.evaluated_at)

    @staticmethod
    def _payload(plan: ExecutionPlan, step: PipelineStep) -> dict[str, object]:
        return plan.commands[FIXED_PIPELINE.index(step)].payload

    @staticmethod
    def _gate_metrics(evaluation: EvaluationEvidence) -> PolicyGateMetrics:
        baseline_collision = evaluation.baseline.collision_rate
        collision_reduction = (
            0.0
            if baseline_collision == 0.0
            else (baseline_collision - evaluation.candidate.collision_rate)
            / baseline_collision
        )
        return PolicyGateMetrics(
            held_out_success_rate=evaluation.candidate.success_rate,
            collision_rate=evaluation.candidate.collision_rate,
            fall_count=evaluation.candidate.falls,
            median_clearance_m=evaluation.candidate.median_clearance_m,
            success_rate_delta=(
                evaluation.candidate.success_rate - evaluation.baseline.success_rate
            ),
            collision_reduction_fraction=collision_reduction,
            path_efficiency_regression_fraction=(
                evaluation.candidate.path_efficiency_regression_fraction
            ),
        )


__all__ = ["TrustedWorkflowEvidenceAdmitter", "WorkflowEvidenceAdmissionError"]
