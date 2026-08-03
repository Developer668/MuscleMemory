"""Separation boundary between Guild reviews and RocketRide execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from muscle_memory.orchestration.contracts import (
    ContractViolationError,
    ExecutionPlan,
    GuildReviewSet,
)
from muscle_memory.orchestration.guild import GuildCoordinator
from muscle_memory.orchestration.rocketride import PipelineRun, ReviewBlockedError


class PipelineExecutor(Protocol):
    async def execute(
        self,
        plan: ExecutionPlan,
        prior_run: PipelineRun | None = None,
    ) -> PipelineRun: ...


@dataclass(frozen=True, slots=True)
class ReviewedExecution:
    plan: ExecutionPlan
    guild_reviews: GuildReviewSet

    def __post_init__(self) -> None:
        if self.plan.digest != self.guild_reviews.plan_digest:
            raise ContractViolationError(
                "Guild review set belongs to a different execution plan"
            )


class SponsorOrchestrator:
    """Guild reasons first; RocketRide receives only a reviewed immutable plan."""

    def __init__(self, guild: GuildCoordinator, rocketride: PipelineExecutor) -> None:
        self._guild = guild
        self._rocketride = rocketride

    async def review(self, plan: ExecutionPlan) -> ReviewedExecution:
        return ReviewedExecution(plan=plan, guild_reviews=await self._guild.review_plan(plan))

    async def execute(
        self,
        reviewed: ReviewedExecution,
        prior_run: PipelineRun | None = None,
    ) -> PipelineRun:
        if not reviewed.guild_reviews.executable:
            raise ReviewBlockedError(
                "all Guild specialists must recommend proceed before execution"
            )
        return await self._rocketride.execute(reviewed.plan, prior_run)
