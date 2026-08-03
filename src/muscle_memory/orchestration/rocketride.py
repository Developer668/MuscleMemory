"""Fixed RocketRide execution with approval gates and honest fallback states."""

from __future__ import annotations

import importlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from muscle_memory.orchestration.approvals import ApprovalLedger, HumanVerdict
from muscle_memory.orchestration.contracts import (
    FIXED_PIPELINE,
    ApprovalRequirement,
    ContractViolationError,
    ExecutionPlan,
    FallbackRecord,
    HealthState,
    PipelineCommand,
    PipelineStep,
    ProviderMode,
    ProviderName,
    ProviderStatus,
    canonical_json,
    sha256_text,
)


class RocketRideUnavailableError(RuntimeError):
    """The configured RocketRide execution surface is unavailable."""


class ReviewBlockedError(RuntimeError):
    """Guild review blocked a plan before RocketRide received it."""


class RunState(StrEnum):
    AWAITING_HUMAN_APPROVAL = "awaiting_human_approval"
    BLOCKED = "blocked"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    CACHED = "cached"


@dataclass(frozen=True, slots=True)
class StepExecutionResult:
    step: PipelineStep
    output_json: str
    output_sha256: str

    def __post_init__(self) -> None:
        if sha256_text(self.output_json) != self.output_sha256:
            raise ContractViolationError("step output checksum mismatch")
        try:
            decoded = json.loads(self.output_json)
        except json.JSONDecodeError as exc:
            raise ContractViolationError("step output is not valid JSON") from exc
        if not isinstance(decoded, dict) or canonical_json(decoded) != self.output_json:
            raise ContractViolationError("step output must be a canonical JSON object")

    @classmethod
    def create(
        cls,
        step: PipelineStep,
        output: Mapping[str, object],
    ) -> StepExecutionResult:
        output_json = canonical_json(output)
        return cls(step=step, output_json=output_json, output_sha256=sha256_text(output_json))

    @property
    def output(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.output_json))


@dataclass(frozen=True, slots=True)
class PipelineRun:
    run_id: str
    plan_digest: str
    state: RunState
    completed_steps: tuple[StepExecutionResult, ...]
    provider_status: ProviderStatus
    blocked_requirement: ApprovalRequirement | None = None
    failure: str | None = None
    fallback: FallbackRecord | None = None

    def __post_init__(self) -> None:
        expected_prefix = FIXED_PIPELINE[: len(self.completed_steps)]
        if tuple(result.step for result in self.completed_steps) != expected_prefix:
            raise ContractViolationError(
                "completed RocketRide steps are not a fixed-pipeline prefix"
            )
        if self.state is RunState.AWAITING_HUMAN_APPROVAL and self.blocked_requirement is None:
            raise ContractViolationError("approval-waiting run must name its blocked requirement")
        if self.state is RunState.CACHED and self.fallback is None:
            raise ContractViolationError("cached run must carry a fallback record")
        if self.state in {RunState.COMPLETED, RunState.CACHED} and len(
            self.completed_steps
        ) != len(FIXED_PIPELINE):
            raise ContractViolationError("completed runs must contain all fixed pipeline steps")


class StepTransport(Protocol):
    @property
    def status(self) -> ProviderStatus: ...

    async def execute(
        self,
        plan: ExecutionPlan,
        command: PipelineCommand,
    ) -> Mapping[str, object]: ...


class FixedPipelineExecutor:
    """Executes only the fixed commands; it contains no agent reasoning."""

    def __init__(self, transport: StepTransport, approvals: ApprovalLedger) -> None:
        self._transport = transport
        self._approvals = approvals

    @property
    def status(self) -> ProviderStatus:
        return self._transport.status

    async def execute(
        self,
        plan: ExecutionPlan,
        prior_run: PipelineRun | None = None,
    ) -> PipelineRun:
        completed = self._resume_prefix(plan, prior_run)
        requirements = {requirement.step: requirement for requirement in plan.approval_requirements}

        for command in plan.commands[len(completed) :]:
            requirement = requirements.get(command.step)
            if requirement is not None:
                decision = self._approvals.decision_for(requirement.requirement_id)
                if decision is None:
                    return PipelineRun(
                        run_id=plan.run_id,
                        plan_digest=plan.digest,
                        state=RunState.AWAITING_HUMAN_APPROVAL,
                        completed_steps=tuple(completed),
                        provider_status=self.status,
                        blocked_requirement=requirement,
                    )
                if decision.plan_digest != plan.digest:
                    raise ContractViolationError(
                        "approval decision belongs to a different plan"
                    )
                if decision.verdict is HumanVerdict.REJECT:
                    return PipelineRun(
                        run_id=plan.run_id,
                        plan_digest=plan.digest,
                        state=RunState.BLOCKED,
                        completed_steps=tuple(completed),
                        provider_status=self.status,
                        blocked_requirement=requirement,
                        failure="human approval was rejected",
                    )

            try:
                output = await self._transport.execute(plan, command)
                result = StepExecutionResult.create(command.step, output)
            except RocketRideUnavailableError:
                raise
            except Exception as exc:
                return PipelineRun(
                    run_id=plan.run_id,
                    plan_digest=plan.digest,
                    state=RunState.FAILED,
                    completed_steps=tuple(completed),
                    provider_status=self.status,
                    failure=f"{command.step.value} failed: {exc}",
                )
            completed.append(result)
            if (
                command.step is PipelineStep.VALIDATE_WORLD
                and result.output.get("world_valid") is not True
            ):
                return PipelineRun(
                    run_id=plan.run_id,
                    plan_digest=plan.digest,
                    state=RunState.FAILED,
                    completed_steps=tuple(completed),
                    provider_status=self.status,
                    failure="world validation did not pass",
                )

        return PipelineRun(
            run_id=plan.run_id,
            plan_digest=plan.digest,
            state=RunState.COMPLETED,
            completed_steps=tuple(completed),
            provider_status=self.status,
        )

    @staticmethod
    def _resume_prefix(
        plan: ExecutionPlan,
        prior_run: PipelineRun | None,
    ) -> list[StepExecutionResult]:
        if prior_run is None:
            return []
        if prior_run.run_id != plan.run_id or prior_run.plan_digest != plan.digest:
            raise ContractViolationError(
                "cannot resume a run from a different execution plan"
            )
        if prior_run.state not in {
            RunState.AWAITING_HUMAN_APPROVAL,
            RunState.BLOCKED,
            RunState.FAILED,
        }:
            raise ContractViolationError("only an incomplete run may be resumed")
        if prior_run.state is RunState.FAILED and prior_run.completed_steps:
            last_result = prior_run.completed_steps[-1]
            if (
                last_result.step is PipelineStep.VALIDATE_WORLD
                and last_result.output.get("world_valid") is not True
            ):
                raise ContractViolationError("a failed world validation cannot be resumed")
        return list(prior_run.completed_steps)


StepHandler = Callable[[PipelineCommand], Awaitable[Mapping[str, object]]]


class SimulatedStepTransport:
    """Injected local tools for tests/demo, always labeled as simulation."""

    def __init__(self, handlers: Mapping[PipelineStep, StepHandler]) -> None:
        if set(handlers) != set(FIXED_PIPELINE):
            raise ContractViolationError(
                "simulated transport requires exactly one handler per fixed step"
            )
        self._handlers = dict(handlers)
        self._status = ProviderStatus(
            provider=ProviderName.ROCKETRIDE,
            mode=ProviderMode.SIMULATION,
            health=HealthState.HEALTHY,
            detail="local injected execution tools; no RocketRide request was sent",
            checked_at=datetime.now(UTC),
        )

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def execute(
        self,
        plan: ExecutionPlan,
        command: PipelineCommand,
    ) -> Mapping[str, object]:
        del plan
        return await self._handlers[command.step](command)


class UnconfiguredRocketRideTransport:
    def __init__(self, detail: str = "RocketRide URI, API key, or pipeline is missing") -> None:
        self._status = ProviderStatus.unconfigured(ProviderName.ROCKETRIDE, detail)

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def execute(
        self,
        plan: ExecutionPlan,
        command: PipelineCommand,
    ) -> Mapping[str, object]:
        del plan, command
        raise RocketRideUnavailableError(self._status.detail)


@dataclass(frozen=True, slots=True)
class RocketRideSdkConfig:
    uri: str
    api_key: str
    pipeline_path: Path
    pipeline_sha256: str
    request_timeout_ms: float = 120_000.0

    def __post_init__(self) -> None:
        if not self.uri.startswith(("https://", "wss://")):
            raise ContractViolationError("live RocketRide Cloud must use HTTPS or WSS")
        if not self.api_key:
            raise ContractViolationError("RocketRide API key must not be empty")
        if not self.pipeline_path.is_file():
            raise ContractViolationError("RocketRide pipeline file does not exist")
        actual = sha256_text(self.pipeline_path.read_text(encoding="utf-8"))
        if actual != self.pipeline_sha256:
            raise ContractViolationError("RocketRide pipeline checksum mismatch")
        if self.request_timeout_ms <= 0:
            raise ContractViolationError("RocketRide request timeout must be positive")


class RocketRideSdkTransport:
    """Optional live adapter around the official async Python SDK."""

    def __init__(
        self,
        config: RocketRideSdkConfig,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._status = ProviderStatus(
            provider=ProviderName.ROCKETRIDE,
            mode=ProviderMode.LIVE,
            health=HealthState.CONFIGURED,
            detail="RocketRide SDK configured; no successful task yet",
            checked_at=datetime.now(UTC),
        )

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def _factory(self) -> Callable[..., Any]:
        if self._client_factory is not None:
            return self._client_factory
        try:
            module = importlib.import_module("rocketride")
        except ImportError as exc:
            raise RocketRideUnavailableError(
                "official rocketride package is not installed"
            ) from exc
        return cast(Callable[..., Any], module.RocketRideClient)

    async def execute(
        self,
        plan: ExecutionPlan,
        command: PipelineCommand,
    ) -> Mapping[str, object]:
        try:
            factory = self._factory()
            client = factory(
                uri=self._config.uri,
                auth=self._config.api_key,
                request_timeout=self._config.request_timeout_ms,
                persist=False,
                module="muscle-memory",
            )
            async with client:
                task = await client.use(filepath=str(self._config.pipeline_path))
                token_value = task.get("token") if isinstance(task, Mapping) else None
                if not isinstance(token_value, str) or not token_value:
                    raise RocketRideUnavailableError("RocketRide use() returned no task token")
                envelope = canonical_json(
                    {
                        "contract_version": 1,
                        "run_id": plan.run_id,
                        "plan_digest": plan.digest,
                        "step": command.step.value,
                        "payload": command.payload,
                    }
                )
                try:
                    response = await client.send(
                        token_value,
                        envelope,
                        objinfo={"name": f"{command.step.value}.json"},
                        mimetype="application/json",
                    )
                finally:
                    await client.terminate(token_value)
                if not isinstance(response, Mapping):
                    response = {"result": response}
                self._status = ProviderStatus(
                    provider=ProviderName.ROCKETRIDE,
                    mode=ProviderMode.LIVE,
                    health=HealthState.HEALTHY,
                    detail=f"RocketRide completed {command.step.value}",
                    checked_at=datetime.now(UTC),
                )
                return dict(response)
        except RocketRideUnavailableError:
            self._mark_unhealthy("RocketRide task contract failed")
            raise
        except Exception as exc:
            self._mark_unhealthy(str(exc))
            raise RocketRideUnavailableError(f"RocketRide execution failed: {exc}") from exc

    def _mark_unhealthy(self, detail: str) -> None:
        self._status = ProviderStatus(
            provider=ProviderName.ROCKETRIDE,
            mode=ProviderMode.LIVE,
            health=HealthState.UNHEALTHY,
            detail=detail,
            checked_at=datetime.now(UTC),
        )


class InMemoryPipelineRunCache:
    def __init__(self) -> None:
        self._runs: dict[str, PipelineRun] = {}

    def put(self, run: PipelineRun) -> None:
        if run.state is not RunState.COMPLETED:
            raise ContractViolationError("only completed pipeline runs may be cached")
        self._runs.setdefault(run.plan_digest, run)

    def get(self, plan_digest: str) -> PipelineRun | None:
        return self._runs.get(plan_digest)


class ResilientPipelineExecutor:
    """Returns only an exact-plan cached result when live RocketRide is down."""

    def __init__(
        self,
        live: FixedPipelineExecutor,
        cache: InMemoryPipelineRunCache,
    ) -> None:
        self._live = live
        self._cache = cache
        self._status = live.status

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def execute(
        self,
        plan: ExecutionPlan,
        prior_run: PipelineRun | None = None,
    ) -> PipelineRun:
        try:
            run = await self._live.execute(plan, prior_run)
        except RocketRideUnavailableError as exc:
            cached = self._cache.get(plan.digest)
            if cached is None:
                raise
            status = ProviderStatus(
                provider=ProviderName.ROCKETRIDE,
                mode=ProviderMode.CACHED,
                health=HealthState.DEGRADED,
                detail="live RocketRide unavailable; returning exact-plan cached run",
                checked_at=datetime.now(UTC),
            )
            self._status = status
            fallback = FallbackRecord(
                provider=ProviderName.ROCKETRIDE,
                mode=ProviderMode.CACHED,
                reason=str(exc),
                source_digest=plan.digest,
            )
            return PipelineRun(
                run_id=cached.run_id,
                plan_digest=cached.plan_digest,
                state=RunState.CACHED,
                completed_steps=cached.completed_steps,
                provider_status=status,
                fallback=fallback,
            )
        if run.state is RunState.COMPLETED:
            self._cache.put(run)
        self._status = run.provider_status
        return run
