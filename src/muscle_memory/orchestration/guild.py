"""Guild.ai review adapters with explicit live, simulation, and cached states."""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from muscle_memory.orchestration.contracts import (
    EXACT_GUILD_ROLES,
    ApprovalKind,
    ContractViolationError,
    ExecutionPlan,
    FallbackRecord,
    GuildReview,
    GuildReviewSet,
    GuildRole,
    HealthState,
    PipelineStep,
    ProviderMode,
    ProviderName,
    ProviderStatus,
    ReviewRecommendation,
    sha256_text,
)


class GuildUnavailableError(RuntimeError):
    """The live Guild provider could not produce a validated review set."""


class GuildCoordinator(Protocol):
    @property
    def status(self) -> ProviderStatus: ...

    async def review_plan(self, plan: ExecutionPlan) -> GuildReviewSet: ...


class UnconfiguredGuildCoordinator:
    def __init__(self, detail: str = "Guild API trigger credentials are missing") -> None:
        self._status = ProviderStatus.unconfigured(ProviderName.GUILD, detail)

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def review_plan(self, plan: ExecutionPlan) -> GuildReviewSet:
        del plan
        raise GuildUnavailableError(self._status.detail)


class GuildHttpTransport(Protocol):
    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> object: ...


class UrllibGuildTransport:
    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> object:
        return await asyncio.to_thread(
            self._request_json,
            method,
            url,
            headers,
            body,
            timeout_seconds,
        )

    @staticmethod
    def _request_json(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> object:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise GuildUnavailableError(f"Guild request failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class GuildRoleEndpoint:
    role: GuildRole
    basic_credentials: str = field(repr=False)

    def __post_init__(self) -> None:
        if ":" not in self.basic_credentials:
            raise ContractViolationError("Guild credentials must be api_key_id:api_key_secret")


@dataclass(frozen=True, slots=True)
class GuildApiConfig:
    owner: str
    workspace: str
    endpoints: tuple[GuildRoleEndpoint, ...]
    base_url: str = "https://app.guild.ai"
    timeout_seconds: float = 120.0
    poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith("https://"):
            raise ContractViolationError("live Guild credentials require an HTTPS base URL")
        if tuple(endpoint.role for endpoint in self.endpoints) != EXACT_GUILD_ROLES:
            raise ContractViolationError(
                "Guild API config requires one endpoint for each exact role"
            )
        if not self.owner or not self.workspace:
            raise ContractViolationError("Guild owner and workspace must not be empty")
        if self.timeout_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ContractViolationError("Guild timeouts must be positive")


class GuildApiCoordinator:
    """Calls one API-trigger agent per role and validates structured reviews."""

    def __init__(
        self,
        config: GuildApiConfig,
        transport: GuildHttpTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibGuildTransport()
        self._status = ProviderStatus(
            provider=ProviderName.GUILD,
            mode=ProviderMode.LIVE,
            health=HealthState.CONFIGURED,
            detail="three Guild API triggers configured; no successful session yet",
            checked_at=datetime.now(UTC),
        )

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def review_plan(self, plan: ExecutionPlan) -> GuildReviewSet:
        reviews: list[GuildReview] = []
        try:
            for endpoint in self._config.endpoints:
                reviews.append(await self._review_role(endpoint, plan))
        except (ContractViolationError, GuildUnavailableError) as exc:
            self._status = ProviderStatus(
                provider=ProviderName.GUILD,
                mode=ProviderMode.LIVE,
                health=HealthState.UNHEALTHY,
                detail=str(exc),
                checked_at=datetime.now(UTC),
            )
            if isinstance(exc, GuildUnavailableError):
                raise
            raise GuildUnavailableError(f"Guild returned an invalid review: {exc}") from exc

        self._status = ProviderStatus(
            provider=ProviderName.GUILD,
            mode=ProviderMode.LIVE,
            health=HealthState.HEALTHY,
            detail="all three Guild role sessions completed with validated output",
            checked_at=datetime.now(UTC),
        )
        return GuildReviewSet(
            plan_digest=plan.digest,
            reviews=tuple(reviews),
            provider_status=self._status,
        )

    async def _review_role(
        self,
        endpoint: GuildRoleEndpoint,
        plan: ExecutionPlan,
    ) -> GuildReview:
        auth = base64.b64encode(endpoint.basic_credentials.encode("utf-8")).decode("ascii")
        headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
        session_url = (
            f"{self._config.base_url.rstrip('/')}/api/workspaces/"
            f"{self._config.owner}/{self._config.workspace}/sessions"
        )
        body = json.dumps(
            {
                "session_type": "api_trigger",
                "agent_input": {
                    "contract_version": 1,
                    "role": endpoint.role.value,
                    "plan_digest": plan.digest,
                    "run_id": plan.run_id,
                    "pipeline_steps": [step.value for step in plan_steps(plan)],
                    "requested_output": {
                        "recommendation": "proceed | revise | block",
                        "summary": "non-empty string",
                        "requested_approvals": [kind.value for kind in ApprovalKind],
                    },
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        created = await self._transport.request_json(
            "POST",
            session_url,
            headers=headers,
            body=body,
            timeout_seconds=self._config.timeout_seconds,
        )
        session_id = _find_string(created, ("id", "session_id"))
        if session_id is None:
            raise GuildUnavailableError("Guild session response did not contain a session id")

        deadline = asyncio.get_running_loop().time() + self._config.timeout_seconds
        detail_url = f"{self._config.base_url.rstrip('/')}/api/sessions/{session_id}"
        while True:
            detail = await self._transport.request_json(
                "GET",
                detail_url,
                headers=headers,
                body=None,
                timeout_seconds=self._config.timeout_seconds,
            )
            state = (_find_string(detail, ("state", "status")) or "").lower()
            if state in {"failed", "error", "cancelled", "canceled"}:
                raise GuildUnavailableError(
                    f"Guild {endpoint.role.value} session ended with state {state}"
                )
            candidate = _find_review_mapping(detail)
            if candidate is not None:
                return _parse_review(candidate, endpoint.role, plan.digest)
            if state in {"completed", "complete", "finished", "done"}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise GuildUnavailableError(f"Guild {endpoint.role.value} session timed out")
            await asyncio.sleep(self._config.poll_interval_seconds)

        events = await self._transport.request_json(
            "GET",
            f"{detail_url}/events?limit=100",
            headers=headers,
            body=None,
            timeout_seconds=self._config.timeout_seconds,
        )
        candidate = _find_review_mapping(events)
        if candidate is None:
            raise GuildUnavailableError(
                f"Guild {endpoint.role.value} completed without a structured review"
            )
        return _parse_review(candidate, endpoint.role, plan.digest)


def plan_steps(plan: ExecutionPlan) -> tuple[PipelineStep, ...]:
    return tuple(command.step for command in plan.commands)


def _find_string(value: object, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _find_review_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, str):
        try:
            return _find_review_mapping(json.loads(value))
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        for item in reversed(value):
            candidate = _find_review_mapping(item)
            if candidate is not None:
                return candidate
        return None
    if not isinstance(value, Mapping):
        return None
    if {"recommendation", "summary"}.issubset(value):
        return value
    for key in ("review", "result", "output", "content", "data", "events", "items"):
        if key in value:
            candidate = _find_review_mapping(value[key])
            if candidate is not None:
                return candidate
    return None


def _parse_review(
    value: Mapping[str, object],
    role: GuildRole,
    plan_digest: str,
) -> GuildReview:
    try:
        recommendation = ReviewRecommendation(str(value["recommendation"]))
    except (KeyError, ValueError) as exc:
        raise ContractViolationError("Guild recommendation is missing or invalid") from exc
    summary = value.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ContractViolationError("Guild review summary must be a non-empty string")
    raw_approvals = value.get("requested_approvals", [])
    if not isinstance(raw_approvals, list):
        raise ContractViolationError("Guild requested_approvals must be a list")
    try:
        requested = tuple(ApprovalKind(str(item)) for item in raw_approvals)
    except ValueError as exc:
        raise ContractViolationError("Guild requested an unknown approval kind") from exc
    returned_digest = value.get("plan_digest", plan_digest)
    if returned_digest != plan_digest:
        raise ContractViolationError("Guild review references a different plan digest")
    returned_role = value.get("role", role.value)
    if returned_role != role.value:
        raise ContractViolationError("Guild review references a different specialist role")
    return GuildReview(
        role=role,
        plan_digest=plan_digest,
        recommendation=recommendation,
        summary=summary.strip(),
        requested_approvals=requested,
    )


class InMemoryGuildReviewCache:
    def __init__(self) -> None:
        self._reviews: dict[str, GuildReviewSet] = {}

    def put(self, review_set: GuildReviewSet) -> None:
        self._reviews.setdefault(review_set.plan_digest, review_set)

    def get(self, plan_digest: str) -> GuildReviewSet | None:
        return self._reviews.get(plan_digest)


class ResilientGuildCoordinator:
    """Uses live Guild or an exact-plan cache; never invents a local review."""

    def __init__(
        self,
        live: GuildCoordinator,
        cache: InMemoryGuildReviewCache,
    ) -> None:
        self._live = live
        self._cache = cache
        self._status = live.status

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def review_plan(self, plan: ExecutionPlan) -> GuildReviewSet:
        try:
            result = await self._live.review_plan(plan)
        except GuildUnavailableError as exc:
            cached = self._cache.get(plan.digest)
            if cached is None:
                self._status = self._live.status
                raise
            fallback = FallbackRecord(
                provider=ProviderName.GUILD,
                mode=ProviderMode.CACHED,
                reason=str(exc),
                source_digest=plan.digest,
            )
            self._status = ProviderStatus(
                provider=ProviderName.GUILD,
                mode=ProviderMode.CACHED,
                health=HealthState.DEGRADED,
                detail="live Guild unavailable; replaying exact-plan cached reviews",
                checked_at=datetime.now(UTC),
            )
            return GuildReviewSet(
                plan_digest=cached.plan_digest,
                reviews=cached.reviews,
                provider_status=self._status,
                fallback=fallback,
            )
        self._cache.put(result)
        self._status = result.provider_status
        return result


class SimulatedGuildCoordinator:
    """Deterministic test/demo coordinator that is always labeled simulation."""

    def __init__(self, reviews: Mapping[GuildRole, ReviewRecommendation]) -> None:
        if tuple(reviews) != EXACT_GUILD_ROLES:
            raise ContractViolationError("simulated Guild requires the exact three roles")
        self._reviews = dict(reviews)
        self._status = ProviderStatus(
            provider=ProviderName.GUILD,
            mode=ProviderMode.SIMULATION,
            health=HealthState.HEALTHY,
            detail="deterministic local Guild simulation; no sponsor request was sent",
            checked_at=datetime.now(UTC),
        )

    @property
    def status(self) -> ProviderStatus:
        return self._status

    async def review_plan(self, plan: ExecutionPlan) -> GuildReviewSet:
        reviews = tuple(
            GuildReview(
                role=role,
                plan_digest=plan.digest,
                recommendation=self._reviews[role],
                summary=f"simulated {role.value} review",
            )
            for role in EXACT_GUILD_ROLES
        )
        fallback = FallbackRecord(
            provider=ProviderName.GUILD,
            mode=ProviderMode.SIMULATION,
            reason="explicit simulation coordinator selected",
            source_digest=sha256_text("simulated-guild-v1"),
        )
        return GuildReviewSet(
            plan_digest=plan.digest,
            reviews=reviews,
            provider_status=self._status,
            fallback=fallback,
        )
