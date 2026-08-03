"""Strict, content-addressed evidence supplied to the three Guild specialists."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from muscle_memory.orchestration.contracts import (
    ContractViolationError,
    ExecutionPlan,
    PipelineStep,
    canonical_json,
    sha256_text,
)

_HASH = r"^[0-9a-f]{64}$"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
Hash256 = Annotated[str, Field(pattern=_HASH)]
Text100 = Annotated[str, Field(min_length=1, max_length=100)]
Text200 = Annotated[str, Field(min_length=1, max_length=200)]
Text300 = Annotated[str, Field(min_length=1, max_length=300)]


class EvidenceModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class WorldValidationEvidence(EvidenceModel):
    no_overlapping_objects: bool
    start_destination_connected: bool
    passages_meet_minimum_clearance: bool
    approved_colliders_only: bool
    baseline_path_exists: bool
    physical_parameters_within_safe_limits: bool


class WorldObstacleEvidence(EvidenceModel):
    obstacle_id: str = Field(min_length=1, max_length=200)
    proposal_digest: str = Field(pattern=_HASH)
    dimensions_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    mass_kg: FiniteFloat = Field(gt=0)
    friction: FiniteFloat = Field(ge=0, le=2)
    property_origin: Literal[
        "human_confirmed",
        "catalog_confirmed",
        "agent_proposed",
        "uncertain",
    ]
    collision_geometry: Literal["primitive", "convex"]
    render_mesh_used_for_collision: Literal[False]
    prior_human_approval_reference: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    @model_validator(mode="after")
    def positive_dimensions(self) -> Self:
        if any(value <= 0 for value in self.dimensions_m):
            raise ValueError("world evidence dimensions must be positive")
        return self


class WorldEvidence(EvidenceModel):
    world_id: str = Field(min_length=1, max_length=200)
    world_digest: str = Field(pattern=_HASH)
    baseline_path_digest: str = Field(pattern=_HASH)
    robot_checksum_unchanged: Literal[True]
    validation: WorldValidationEvidence
    obstacles: tuple[WorldObstacleEvidence, ...] = Field(min_length=1, max_length=100)


class FailurePatternEvidence(EvidenceModel):
    signature: Text300
    source_episode_ids: tuple[Text200, ...] = Field(min_length=2, max_length=1000)
    distinct_source_episode_count: int = Field(ge=2, le=1000)
    obstacle_categories: tuple[Text100, ...] = Field(max_length=25)
    approved_correction_ids: tuple[Text200, ...] = Field(max_length=100)
    lesson_ids: tuple[Text200, ...] = Field(max_length=100)

    @model_validator(mode="after")
    def exact_distinct_count(self) -> Self:
        if len(set(self.source_episode_ids)) != self.distinct_source_episode_count:
            raise ValueError("failure evidence distinct episode count does not match")
        return self


class CurriculumProposalEvidence(EvidenceModel):
    proposal_id: str = Field(min_length=1, max_length=200)
    proposal_digest: str = Field(pattern=_HASH)
    target_failure_signatures: tuple[Text300, ...] = Field(min_length=1, max_length=100)
    generated_world_digests: tuple[Hash256, ...] = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=2000)


class FailureCurriculumEvidence(EvidenceModel):
    source_split: Literal["training"]
    source_policy_id: str = Field(min_length=1, max_length=200)
    graph_query_digest: str = Field(pattern=_HASH)
    failure_patterns: tuple[FailurePatternEvidence, ...] = Field(min_length=1, max_length=100)
    curriculum_change_requested: bool
    proposed_curriculum: CurriculumProposalEvidence | None = None

    @model_validator(mode="after")
    def proposal_matches_request(self) -> Self:
        if self.curriculum_change_requested is (self.proposed_curriculum is None):
            raise ValueError("curriculum proposal must exactly match the change request")
        return self


class EvaluationPolicyEvidence(EvidenceModel):
    policy_id: str = Field(min_length=1, max_length=200)
    policy_checksum: str = Field(pattern=_HASH)
    evaluation_id: str = Field(min_length=1, max_length=200)
    success_rate: FiniteFloat = Field(ge=0, le=1)
    collision_rate: FiniteFloat = Field(ge=0, le=1)


class CandidateEvaluationEvidence(EvaluationPolicyEvidence):
    falls: int = Field(ge=0)
    median_clearance_m: FiniteFloat = Field(ge=0)
    path_efficiency_regression_fraction: FiniteFloat


class EvaluationEvidence(EvidenceModel):
    heldout_world_set_id: str = Field(min_length=1, max_length=200)
    heldout_world_set_digest: str = Field(pattern=_HASH)
    paired_world_count: Literal[20]
    baseline: EvaluationPolicyEvidence
    candidate: CandidateEvaluationEvidence
    proposed_action: Literal["promote", "roll_back"]

    @model_validator(mode="after")
    def distinct_policies(self) -> Self:
        if self.baseline.policy_id == self.candidate.policy_id:
            raise ValueError("evaluation evidence requires distinct policy ids")
        return self


class WorldEvidenceReference(EvidenceModel):
    evidence_id: str = Field(pattern=_IDENTIFIER)
    world_evidence: WorldEvidence


class FailureCurriculumEvidenceReference(EvidenceModel):
    evidence_id: str = Field(pattern=_IDENTIFIER)
    failure_curriculum_evidence: FailureCurriculumEvidence


class EvaluationEvidenceReference(EvidenceModel):
    evidence_id: str = Field(pattern=_IDENTIFIER)
    evaluation_evidence: EvaluationEvidence


class GuildEvidenceBundle(EvidenceModel):
    world: WorldEvidenceReference
    failure_curriculum: FailureCurriculumEvidenceReference
    evaluation: EvaluationEvidenceReference

    def artifact_hashes(self) -> tuple[tuple[str, str, str], ...]:
        entries = (
            (
                self.world.evidence_id,
                "guild_world_evidence",
                self.world.world_evidence.model_dump(mode="json"),
            ),
            (
                self.failure_curriculum.evidence_id,
                "guild_failure_curriculum_evidence",
                self.failure_curriculum.failure_curriculum_evidence.model_dump(mode="json"),
            ),
            (
                self.evaluation.evidence_id,
                "guild_evaluation_evidence",
                self.evaluation.evaluation_evidence.model_dump(mode="json"),
            ),
        )
        return tuple(
            (evidence_id, kind, sha256_text(canonical_json(payload)))
            for evidence_id, kind, payload in entries
        )


def validate_evidence_plan_binding(bundle: GuildEvidenceBundle, plan: ExecutionPlan) -> None:
    commands = {command.step: command.payload for command in plan.commands}
    world_command = commands[PipelineStep.VALIDATE_WORLD]
    run_command = commands[PipelineStep.RUN_EPISODE]
    summary_command = commands[PipelineStep.SUMMARIZE_TELEMETRY]
    graph_command = commands[PipelineStep.QUERY_GRAPH_MEMORY]
    curriculum_command = commands[PipelineStep.SELECT_CURRICULUM]
    training_command = commands[PipelineStep.TRAIN_CANDIDATE_POLICY]
    evaluation_command = commands[PipelineStep.EVALUATE_CANDIDATE_POLICY]
    final_command = commands[PipelineStep.PROMOTE_OR_ROLL_BACK]

    world = bundle.world.world_evidence
    uncertain = any(
        obstacle.property_origin in {"agent_proposed", "uncertain"}
        for obstacle in world.obstacles
    )
    if (
        world_command.get("world_id") != world.world_id
        or run_command.get("world_id") != world.world_id
    ):
        raise ContractViolationError("world evidence does not match the plan world")
    if world_command.get("uncertain_physical_properties") is not uncertain:
        raise ContractViolationError("world evidence uncertainty does not match the plan")

    curriculum = bundle.failure_curriculum.failure_curriculum_evidence
    if (
        curriculum_command.get("curriculum_change_requested")
        is not curriculum.curriculum_change_requested
    ):
        raise ContractViolationError("curriculum evidence does not match the plan")
    if curriculum.source_policy_id != evaluation_command.get("baseline_policy_id"):
        raise ContractViolationError("curriculum evidence policy does not match the baseline")

    episode_ids = (
        run_command.get("episode_id"),
        summary_command.get("episode_id"),
        graph_command.get("episode_id"),
        curriculum_command.get("episode_id"),
    )
    if (
        not isinstance(episode_ids[0], str)
        or not episode_ids[0]
        or any(value != episode_ids[0] for value in episode_ids[1:])
    ):
        raise ContractViolationError("workflow episode identity changes across fixed steps")

    evaluation = bundle.evaluation.evaluation_evidence
    expected = {
        "baseline_policy_id": evaluation.baseline.policy_id,
        "candidate_policy_id": evaluation.candidate.policy_id,
        "heldout_world_set_id": evaluation.heldout_world_set_id,
    }
    if evaluation_command != expected:
        raise ContractViolationError("evaluation evidence identifiers do not match the plan")
    if final_command.get("action") != evaluation.proposed_action:
        raise ContractViolationError("evaluation evidence action does not match the plan")
    if final_command.get("candidate_policy_id") != evaluation.candidate.policy_id:
        raise ContractViolationError("final action candidate does not match evaluation evidence")
    if training_command.get("candidate_policy_id") != evaluation.candidate.policy_id:
        raise ContractViolationError("training candidate does not match evaluation evidence")


__all__ = [
    "CandidateEvaluationEvidence",
    "CurriculumProposalEvidence",
    "EvaluationEvidence",
    "EvaluationEvidenceReference",
    "EvaluationPolicyEvidence",
    "FailureCurriculumEvidence",
    "FailureCurriculumEvidenceReference",
    "FailurePatternEvidence",
    "GuildEvidenceBundle",
    "WorldEvidence",
    "WorldEvidenceReference",
    "WorldObstacleEvidence",
    "WorldValidationEvidence",
    "validate_evidence_plan_binding",
]
