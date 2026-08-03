"""Coordinator-backed role isolation for Guild specialist evidence."""

from __future__ import annotations

from collections.abc import Mapping

from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.orchestration.contracts import ExecutionPlan, GuildRole


class CoordinatorGuildEvidenceSource:
    def __init__(self, coordinator: CoordinatorStore) -> None:
        self._coordinator = coordinator

    def evidence_for(
        self,
        plan: ExecutionPlan,
        role: GuildRole,
    ) -> Mapping[str, object] | None:
        bundle = self._coordinator.workflow_guild_evidence(plan.run_id)
        if bundle is None:
            return None
        durable_plan = self._coordinator.workflow_plan(plan.run_id)
        if durable_plan != plan:
            return None
        if role is GuildRole.WORLD_AND_PHYSICS:
            return {
                "world_evidence": bundle.world.world_evidence.model_dump(mode="json")
            }
        if role is GuildRole.FAILURE_AND_CURRICULUM:
            return {
                "failure_curriculum_evidence": (
                    bundle.failure_curriculum.failure_curriculum_evidence.model_dump(mode="json")
                )
            }
        return {
            "evaluation_evidence": bundle.evaluation.evaluation_evidence.model_dump(mode="json")
        }


__all__ = ["CoordinatorGuildEvidenceSource"]
