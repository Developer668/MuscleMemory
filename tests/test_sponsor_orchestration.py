"""Safety and provider-contract tests for sponsor-backed orchestration."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import TypeAdapter

from muscle_memory.api.adapters import pipeline_run_view, reviewed_execution_view
from muscle_memory.orchestration import (
    EXACT_GUILD_ROLES,
    FIXED_PIPELINE,
    ApprovalKind,
    ContractViolationError,
    ExecutionPlan,
    FixedPipelineExecutor,
    GuildApiConfig,
    GuildApiCoordinator,
    GuildRole,
    GuildRoleEndpoint,
    GuildRoster,
    GuildUnavailableError,
    HealthState,
    HumanDecision,
    HumanVerdict,
    InMemoryApprovalLedger,
    InMemoryGuildReviewCache,
    InMemoryPipelineRunCache,
    PipelineCommand,
    PipelineRun,
    PipelineStep,
    ProviderMode,
    ResilientGuildCoordinator,
    ResilientPipelineExecutor,
    ReviewBlockedError,
    ReviewRecommendation,
    RocketRideSdkConfig,
    RocketRideSdkTransport,
    RocketRideUnavailableError,
    RunState,
    SimulatedGuildCoordinator,
    SimulatedStepTransport,
    SponsorOrchestrator,
    StepExecutionResult,
    UnconfiguredGuildCoordinator,
    UnconfiguredRocketRideTransport,
)
from muscle_memory.orchestration.contracts import canonical_json, sha256_text
from muscle_memory.orchestration.service import ReviewedExecution


def _plan(
    run_id: str = "run-001",
    *,
    uncertain_physical_properties: bool = False,
    curriculum_change_requested: bool = False,
    reward_change_requested: bool = False,
    action: str = "promote",
) -> ExecutionPlan:
    commands = (
        PipelineCommand.create(
            PipelineStep.VALIDATE_WORLD,
            {
                "uncertain_physical_properties": uncertain_physical_properties,
                "world_id": "world-001",
            },
        ),
        PipelineCommand.create(
            PipelineStep.RUN_EPISODE,
            {"episode_id": "episode-001", "world_id": "world-001"},
        ),
        PipelineCommand.create(
            PipelineStep.SUMMARIZE_TELEMETRY,
            {"episode_id": "episode-001"},
        ),
        PipelineCommand.create(
            PipelineStep.QUERY_GRAPH_MEMORY,
            {"episode_id": "episode-001"},
        ),
        PipelineCommand.create(
            PipelineStep.SELECT_CURRICULUM,
            {
                "curriculum_change_requested": curriculum_change_requested,
                "episode_id": "episode-001",
            },
        ),
        PipelineCommand.create(
            PipelineStep.TRAIN_CANDIDATE_POLICY,
            {
                "reward_change_requested": reward_change_requested,
                "candidate_policy_id": "candidate-001",
            },
        ),
        PipelineCommand.create(
            PipelineStep.EVALUATE_CANDIDATE_POLICY,
            {
                "baseline_policy_id": "baseline-001",
                "candidate_policy_id": "candidate-001",
                "heldout_world_set_id": "heldout-v1",
            },
        ),
        PipelineCommand.create(
            PipelineStep.PROMOTE_OR_ROLL_BACK,
            {"action": action, "candidate_policy_id": "candidate-001"},
        ),
    )
    return ExecutionPlan.create(run_id, commands)


def _simulated_transport(
    calls: list[PipelineStep],
    *,
    world_valid: bool = True,
) -> SimulatedStepTransport:
    async def execute(command: PipelineCommand) -> Mapping[str, object]:
        calls.append(command.step)
        if command.step is PipelineStep.VALIDATE_WORLD:
            return {"world_valid": world_valid}
        return {"ok": True, "step": command.step.value}

    return SimulatedStepTransport({step: execute for step in FIXED_PIPELINE})


def _approve_all(plan: ExecutionPlan, ledger: InMemoryApprovalLedger) -> None:
    for requirement in plan.approval_requirements:
        ledger.record(
            HumanDecision.create(
                requirement,
                human_subject="operator@example.test",
                verdict=HumanVerdict.APPROVE,
            )
        )


def _all_proceed() -> dict[GuildRole, ReviewRecommendation]:
    return {role: ReviewRecommendation.PROCEED for role in EXACT_GUILD_ROLES}


def test_roster_and_execution_plan_require_exact_order() -> None:
    assert GuildRoster().roles == EXACT_GUILD_ROLES
    assert tuple(command.step for command in _plan().commands) == FIXED_PIPELINE

    with pytest.raises(ContractViolationError, match="exact three"):
        GuildRoster(roles=tuple(reversed(EXACT_GUILD_ROLES)))

    plan = _plan()
    with pytest.raises(ContractViolationError, match="fixed eight-step"):
        ExecutionPlan.create(plan.run_id, tuple(reversed(plan.commands)))


def test_sponsor_credentials_are_excluded_from_configuration_reprs(
    tmp_path: Path,
) -> None:
    guild_secret = "guild-key-id:guild-secret-value"
    endpoint = GuildRoleEndpoint(EXACT_GUILD_ROLES[0], guild_secret)
    pipeline = tmp_path / "pipeline.pipe"
    pipeline.write_text("reviewed pipeline", encoding="utf-8")
    rocketride_secret = "rocketride-secret-value"
    config = RocketRideSdkConfig(
        uri="https://cloud.rocketride.example.test",
        api_key=rocketride_secret,
        pipeline_path=pipeline,
        pipeline_sha256=sha256_text("reviewed pipeline"),
        callback_environment={
            "ROCKETRIDE_MM_COORDINATOR_URL": "https://callback.example.test",
            "ROCKETRIDE_MM_COORDINATOR_TOKEN": "x" * 32,
        },
    )

    assert guild_secret not in repr(endpoint)
    assert rocketride_secret not in repr(config)


def test_evaluation_command_rejects_teacher_or_extra_runtime_inputs() -> None:
    with pytest.raises(ContractViolationError, match="only policy ids"):
        PipelineCommand.create(
            PipelineStep.EVALUATE_CANDIDATE_POLICY,
            {
                "baseline_policy_id": "baseline-001",
                "candidate_policy_id": "candidate-001",
                "heldout_world_set_id": "heldout-v1",
                "expert_path": [[0.0, 0.0]],
            },
        )

    with pytest.raises(ContractViolationError, match="only policy ids"):
        PipelineCommand.create(
            PipelineStep.EVALUATE_CANDIDATE_POLICY,
            {
                "candidate_policy_id": "candidate-001",
                "heldout_world_set_id": "heldout-v1",
            },
        )


def test_required_human_gates_are_derived_from_commands() -> None:
    plan = _plan(
        uncertain_physical_properties=True,
        curriculum_change_requested=True,
        reward_change_requested=True,
    )

    assert tuple(requirement.kind for requirement in plan.approval_requirements) == (
        ApprovalKind.UNCERTAIN_PHYSICAL_PROPERTIES,
        ApprovalKind.CURRICULUM_CHANGE,
        ApprovalKind.REWARD_CHANGE,
        ApprovalKind.POLICY_PROMOTION,
    )
    assert _plan(action="roll_back").approval_requirements[-1].kind is (
        ApprovalKind.POLICY_ROLLBACK
    )


def test_executor_pauses_before_tools_and_resumes_only_after_approval() -> None:
    plan = _plan(uncertain_physical_properties=True)
    calls: list[PipelineStep] = []
    ledger = InMemoryApprovalLedger()
    executor = FixedPipelineExecutor(_simulated_transport(calls), ledger)

    waiting = asyncio.run(executor.execute(plan))

    assert waiting.state is RunState.AWAITING_HUMAN_APPROVAL
    assert waiting.blocked_requirement is not None
    assert waiting.blocked_requirement.kind is ApprovalKind.UNCERTAIN_PHYSICAL_PROPERTIES
    assert calls == []

    _approve_all(plan, ledger)
    complete = asyncio.run(executor.execute(plan, waiting))

    assert complete.state is RunState.COMPLETED
    assert tuple(result.step for result in complete.completed_steps) == FIXED_PIPELINE
    assert calls == list(FIXED_PIPELINE)
    assert complete.provider_status.mode is ProviderMode.SIMULATION


def test_rejected_human_gate_blocks_before_execution() -> None:
    plan = _plan(uncertain_physical_properties=True)
    calls: list[PipelineStep] = []
    ledger = InMemoryApprovalLedger()
    requirement = plan.approval_requirements[0]
    ledger.record(
        HumanDecision.create(
            requirement,
            human_subject="operator@example.test",
            verdict=HumanVerdict.REJECT,
        )
    )

    run = asyncio.run(FixedPipelineExecutor(_simulated_transport(calls), ledger).execute(plan))

    assert run.state is RunState.BLOCKED
    assert calls == []


def test_final_promotion_always_waits_for_human_approval() -> None:
    plan = _plan()
    calls: list[PipelineStep] = []
    run = asyncio.run(
        FixedPipelineExecutor(
            _simulated_transport(calls),
            InMemoryApprovalLedger(),
        ).execute(plan)
    )

    assert run.state is RunState.AWAITING_HUMAN_APPROVAL
    assert run.blocked_requirement is not None
    assert run.blocked_requirement.kind is ApprovalKind.POLICY_PROMOTION
    assert calls == list(FIXED_PIPELINE[:-1])


def test_failed_world_validation_halts_and_cannot_be_resumed() -> None:
    plan = _plan()
    calls: list[PipelineStep] = []
    executor = FixedPipelineExecutor(
        _simulated_transport(calls, world_valid=False),
        InMemoryApprovalLedger(),
    )

    failed = asyncio.run(executor.execute(plan))

    assert failed.state is RunState.FAILED
    assert calls == [PipelineStep.VALIDATE_WORLD]
    with pytest.raises(ContractViolationError, match="cannot be resumed"):
        asyncio.run(executor.execute(plan, failed))


def test_step_failure_does_not_expose_handler_exception_text() -> None:
    plan = _plan()
    secret = "provider-secret-should-never-be-public"

    async def fail(_command: PipelineCommand) -> Mapping[str, object]:
        raise RuntimeError(secret)

    transport = SimulatedStepTransport({step: fail for step in FIXED_PIPELINE})
    run = asyncio.run(FixedPipelineExecutor(transport, InMemoryApprovalLedger()).execute(plan))

    assert run.state is RunState.FAILED
    assert run.failure == "validate_world failed (RuntimeError)"
    assert secret not in run.failure


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        plan: ExecutionPlan,
        prior_run: PipelineRun | None = None,
    ) -> PipelineRun:
        del plan, prior_run
        self.calls += 1
        raise AssertionError("the recording executor should not run")


@pytest.mark.parametrize(
    "recommendation",
    [ReviewRecommendation.REVISE, ReviewRecommendation.BLOCK],
)
def test_non_proceed_guild_review_never_reaches_rocketride(
    recommendation: ReviewRecommendation,
) -> None:
    recommendations = _all_proceed()
    recommendations[GuildRole.SAFETY_AND_EVALUATION] = recommendation
    executor = _RecordingExecutor()
    orchestrator = SponsorOrchestrator(SimulatedGuildCoordinator(recommendations), executor)
    reviewed = asyncio.run(orchestrator.review(_plan()))

    with pytest.raises(ReviewBlockedError, match="all Guild specialists"):
        asyncio.run(orchestrator.execute(reviewed))
    assert executor.calls == 0


def test_guild_exact_plan_cache_is_explicit_and_does_not_generalize() -> None:
    plan = _plan()
    reviews = asyncio.run(SimulatedGuildCoordinator(_all_proceed()).review_plan(plan))
    cache = InMemoryGuildReviewCache()
    cache.put(reviews)
    cache.put(reviews)
    resilient = ResilientGuildCoordinator(UnconfiguredGuildCoordinator("offline"), cache)

    cached = asyncio.run(resilient.review_plan(plan))

    assert cached.provider_status.mode is ProviderMode.CACHED
    assert cached.fallback is not None
    assert cached.fallback.source_digest == plan.digest
    assert resilient.status.mode is ProviderMode.CACHED
    with pytest.raises(GuildUnavailableError, match="offline"):
        asyncio.run(resilient.review_plan(_plan("different-run")))


def test_rocketride_exact_plan_cache_is_explicit_and_does_not_generalize() -> None:
    plan = _plan()
    ledger = InMemoryApprovalLedger()
    _approve_all(plan, ledger)
    complete = asyncio.run(FixedPipelineExecutor(_simulated_transport([]), ledger).execute(plan))
    cache = InMemoryPipelineRunCache()
    cache.put(complete)
    cache.put(complete)
    unavailable = FixedPipelineExecutor(
        UnconfiguredRocketRideTransport("offline"),
        ledger,
    )
    resilient = ResilientPipelineExecutor(unavailable, cache)

    cached = asyncio.run(resilient.execute(plan))

    assert cached.state is RunState.CACHED
    assert cached.provider_status.mode is ProviderMode.CACHED
    assert cached.fallback is not None
    assert cached.fallback.source_digest == plan.digest
    assert resilient.status.mode is ProviderMode.CACHED
    with pytest.raises(RocketRideUnavailableError, match="offline"):
        asyncio.run(resilient.execute(_plan("different-run")))


class _FakeGuildHttpTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Mapping[str, str], bytes | None]] = []
        self.sessions: dict[str, Mapping[str, object]] = {}

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> object:
        assert timeout_seconds > 0
        self.requests.append((method, url, headers, body))
        if method == "POST":
            assert body is not None
            payload = json.loads(body)
            session_id = f"session-{len(self.sessions) + 1}"
            self.sessions[session_id] = payload["agent_input"]
            return {"id": session_id}
        session_id = url.rsplit("/", 1)[-1]
        agent_input = self.sessions[session_id]
        return {
            "state": "completed",
            "result": {
                "plan_digest": agent_input["plan_digest"],
                "recommendation": "proceed",
                "requested_approvals": [],
                "role": agent_input["role"],
                "summary": "provider reviewed fixed plan",
            },
        }


class _FixtureGuildEvidenceSource:
    _FIELDS: ClassVar[dict[GuildRole, tuple[str, str]]] = {
        GuildRole.WORLD_AND_PHYSICS: ("world-and-physics", "world_evidence"),
        GuildRole.FAILURE_AND_CURRICULUM: (
            "failure-and-curriculum",
            "failure_curriculum_evidence",
        ),
        GuildRole.SAFETY_AND_EVALUATION: (
            "safety-and-evaluation",
            "evaluation_evidence",
        ),
    }

    def evidence_for(
        self,
        plan: ExecutionPlan,
        role: GuildRole,
    ) -> Mapping[str, object]:
        del plan
        directory, field = self._FIELDS[role]
        fixture = json.loads(
            Path(f"integrations/guild/{directory}/fixtures/valid-input.json").read_text(
                encoding="utf-8"
            )
        )
        return {field: fixture[field]}


class _CurrentGuildHttpTransport(_FakeGuildHttpTransport):
    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> object:
        if method == "POST":
            return await super().request_json(
                method,
                url,
                headers=headers,
                body=body,
                timeout_seconds=timeout_seconds,
            )
        self.requests.append((method, url, headers, body))
        session_id = url.split("/sessions/", 1)[1].split("/", 1)[0]
        agent_input = self.sessions[session_id]
        if "/events?" not in url:
            return {"root_task": {"status": "DONE"}}
        return {
            "items": [
                {
                    "content": {
                        "plan_digest": agent_input["plan_digest"],
                        "recommendation": "proceed",
                        "requested_approvals": [],
                        "role": agent_input["role"],
                        "summary": "provider reviewed fixed plan",
                    }
                }
            ]
        }


def test_guild_api_adapter_calls_each_exact_role_with_basic_auth() -> None:
    transport = _FakeGuildHttpTransport()
    endpoints = tuple(
        GuildRoleEndpoint(role, f"key-{index}:secret-{index}")
        for index, role in enumerate(EXACT_GUILD_ROLES, start=1)
    )
    coordinator = GuildApiCoordinator(
        GuildApiConfig(
            owner="owner",
            workspace="workspace",
            endpoints=endpoints,
            poll_interval_seconds=0.001,
        ),
        transport,
        _FixtureGuildEvidenceSource(),
    )

    reviews = asyncio.run(coordinator.review_plan(_plan()))

    assert reviews.executable
    assert tuple(review.role for review in reviews.reviews) == EXACT_GUILD_ROLES
    assert reviews.provider_status.health is HealthState.END_TO_END_VERIFIED
    assert tuple(review.provider_session_id for review in reviews.reviews) == (
        "session-1",
        "session-2",
        "session-3",
    )
    durable_review = TypeAdapter(ReviewedExecution).validate_json(
        TypeAdapter(ReviewedExecution).dump_json(
            ReviewedExecution(plan=_plan(), guild_reviews=reviews)
        )
    )
    public_review = reviewed_execution_view(durable_review)
    assert tuple(item.provider_session_id for item in public_review.reviews) == (
        "session-1",
        "session-2",
        "session-3",
    )
    assert "secret-" not in public_review.model_dump_json()
    posts = [request for request in transport.requests if request[0] == "POST"]
    assert len(posts) == 3
    for request, endpoint in zip(posts, endpoints, strict=True):
        _, url, headers, body = request
        expected = base64.b64encode(endpoint.basic_credentials.encode()).decode()
        assert url == "https://app.guild.ai/api/workspaces/owner/workspace/sessions"
        assert headers["Authorization"] == f"Basic {expected}"
        assert body is not None
        agent_input = json.loads(body)["agent_input"]
        assert agent_input["role"] == endpoint.role.value
        assert tuple(agent_input["pipeline_steps"]) == tuple(step.value for step in FIXED_PIPELINE)
        _, field = _FixtureGuildEvidenceSource._FIELDS[endpoint.role]
        assert field in agent_input
        assert not (
            {
                "world_evidence",
                "failure_curriculum_evidence",
                "evaluation_evidence",
            }
            - {field}
        ).intersection(agent_input)


def test_guild_api_adapter_reads_current_nested_done_status_and_events() -> None:
    transport = _CurrentGuildHttpTransport()
    endpoints = tuple(
        GuildRoleEndpoint(role, f"key-{index}:secret-{index}")
        for index, role in enumerate(EXACT_GUILD_ROLES, start=1)
    )
    coordinator = GuildApiCoordinator(
        GuildApiConfig(
            owner="owner",
            workspace="workspace",
            endpoints=endpoints,
            poll_interval_seconds=0.001,
        ),
        transport,
        _FixtureGuildEvidenceSource(),
    )

    reviews = asyncio.run(coordinator.review_plan(_plan()))

    assert reviews.executable
    assert tuple(review.provider_session_id for review in reviews.reviews) == (
        "session-1",
        "session-2",
        "session-3",
    )
    event_requests = [
        url for method, url, _, _ in transport.requests if method == "GET" and "/events?" in url
    ]
    assert len(event_requests) == 3


class _FakeRocketRideClient:
    def __init__(self, events: list[object], **kwargs: object) -> None:
        self.events = events
        self.events.append(("init", kwargs))
        self.active = False

    async def __aenter__(self) -> _FakeRocketRideClient:
        self.active = True
        self.events.append("enter")
        return self

    async def __aexit__(self, *args: object) -> None:
        self.active = False
        self.events.append("exit")

    async def use(self, *, filepath: str) -> Mapping[str, object]:
        assert self.active
        self.events.append(("use", filepath))
        return {"token": "task-token"}

    async def send(
        self,
        token: str,
        payload: str,
        *,
        objinfo: Mapping[str, object],
        mimetype: str,
    ) -> Mapping[str, object]:
        assert self.active
        envelope = json.loads(payload)
        self.events.append(("send", token, envelope, objinfo, mimetype))
        output = {"accepted": True, "world_valid": True}
        return {
            "contract_version": 1,
            "output": output,
            "output_sha256": sha256_text(canonical_json(output)),
            "plan_digest": envelope["plan_digest"],
            "request_sha256": sha256_text(payload),
            "run_id": envelope["run_id"],
            "status": "completed",
            "step": envelope["step"],
        }

    async def terminate(self, token: str) -> None:
        assert self.active
        self.events.append(("terminate", token))


def test_rocketride_sdk_adapter_uses_official_async_contract(tmp_path: Path) -> None:
    pipeline = tmp_path / "fixed.pipe"
    pipeline.write_text("{}\n", encoding="utf-8")
    events: list[object] = []

    def factory(**kwargs: object) -> _FakeRocketRideClient:
        return _FakeRocketRideClient(events, **kwargs)

    transport = RocketRideSdkTransport(
        RocketRideSdkConfig(
            uri="https://cloud.rocketride.ai",
            api_key="secret",
            pipeline_path=pipeline,
            pipeline_sha256=sha256_text(pipeline.read_text(encoding="utf-8")),
            callback_environment={
                "ROCKETRIDE_MM_COORDINATOR_URL": "https://callback.example.test",
                "ROCKETRIDE_MM_COORDINATOR_TOKEN": "x" * 32,
            },
        ),
        InMemoryApprovalLedger(),
        client_factory=factory,
    )
    plan = _plan()

    output = asyncio.run(transport.execute(plan, plan.commands[0]))

    assert output["accepted"] is True
    assert transport.status.mode is ProviderMode.LIVE
    assert transport.status.health is HealthState.HEALTHY
    assert output["rocketride_execution"] == {
        "contract_version": 1,
        "run_id": plan.run_id,
        "task_token_sha256": sha256_text("task-token"),
    }
    durable_step = TypeAdapter(StepExecutionResult).validate_json(
        TypeAdapter(StepExecutionResult).dump_json(
            StepExecutionResult.create(plan.commands[0].step, output)
        )
    )
    public_run = pipeline_run_view(
        PipelineRun(
            run_id=plan.run_id,
            plan_digest=plan.digest,
            state=RunState.RUNNING,
            completed_steps=(durable_step,),
            provider_status=transport.status,
        )
    )
    assert public_run.completed_steps[0].provider_task_receipt_sha256 == sha256_text("task-token")
    assert public_run.completed_steps[0].provider_run_id == plan.run_id
    assert "task-token" not in durable_step.output_json
    assert "secret" not in public_run.model_dump_json()
    assert events[0] == (
        "init",
        {
            "uri": "https://cloud.rocketride.ai",
            "auth": "secret",
            "request_timeout": 120_000.0,
            "persist": False,
            "module": "muscle-memory",
            "env": {
                "ROCKETRIDE_MM_COORDINATOR_URL": "https://callback.example.test",
                "ROCKETRIDE_MM_COORDINATOR_TOKEN": "x" * 32,
            },
        },
    )
    assert events[-2:] == [("terminate", "task-token"), "exit"]


def test_live_provider_configs_reject_insecure_urls(tmp_path: Path) -> None:
    endpoints = tuple(GuildRoleEndpoint(role, "key:secret") for role in EXACT_GUILD_ROLES)
    with pytest.raises(ContractViolationError, match="HTTPS"):
        GuildApiConfig(
            owner="owner",
            workspace="workspace",
            endpoints=endpoints,
            base_url="http://app.guild.ai",
        )

    pipeline = tmp_path / "fixed.pipe"
    pipeline.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractViolationError, match="HTTPS or WSS"):
        RocketRideSdkConfig(
            uri="http://cloud.rocketride.ai",
            api_key="secret",
            pipeline_path=pipeline,
            pipeline_sha256=sha256_text(pipeline.read_text(encoding="utf-8")),
            callback_environment={
                "ROCKETRIDE_MM_COORDINATOR_URL": "https://callback.example.test",
                "ROCKETRIDE_MM_COORDINATOR_TOKEN": "x" * 32,
            },
        )
