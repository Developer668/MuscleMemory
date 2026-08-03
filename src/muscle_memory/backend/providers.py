"""Provider construction and truthful aggregate readiness."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from muscle_memory.api.adapters import (
    asset_provider_view,
    graph_provider_view,
    laserdata_provider_view,
    orchestration_provider_view,
)
from muscle_memory.api.models import (
    ProviderHealth,
    ProviderOperationalState,
    ServiceHealth,
)
from muscle_memory.api.redaction import redact_sensitive_text
from muscle_memory.assets import (
    AssetApprovalLedger,
    AssetGenerationPipeline,
    ContentAddressedAssetCache,
    ReferenceImageHttpAdapter,
    TrellisHttpAdapter,
)
from muscle_memory.backend.config import BackendConfig
from muscle_memory.backend.guild_evidence import CoordinatorGuildEvidenceSource
from muscle_memory.backend.rocketride_artifact import (
    ReviewedPipelineArtifact,
    ReviewedPipelineError,
)
from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.graph_memory import (
    FalkorDBSettings,
    ResilientGraphMemory,
    build_graph_memory,
    settings_from_env,
)
from muscle_memory.orchestration.approvals import ApprovalLedger, InMemoryApprovalLedger
from muscle_memory.orchestration.contracts import (
    EXACT_GUILD_ROLES,
    GuildRole,
    ProviderName,
)
from muscle_memory.orchestration.guild import (
    GuildApiConfig,
    GuildApiCoordinator,
    GuildCoordinator,
    GuildEvidenceSource,
    GuildRoleEndpoint,
    InMemoryGuildReviewCache,
    ResilientGuildCoordinator,
    UnconfiguredGuildCoordinator,
)
from muscle_memory.orchestration.rocketride import (
    FixedPipelineExecutor,
    InMemoryPipelineRunCache,
    ResilientPipelineExecutor,
    RocketRideSdkConfig,
    RocketRideSdkTransport,
    StepTransport,
    UnconfiguredRocketRideTransport,
)
from muscle_memory.telemetry import LaserDataConfig, LaserDataTelemetryBackend


class ProviderDeployment(StrEnum):
    UNCONFIGURED = "unconfigured"
    SELF_HOSTED = "self_hosted"
    CLOUD = "cloud"


@dataclass(frozen=True, slots=True)
class ProviderEvidence:
    provider: str
    evidence_id: str
    deployment: ProviderDeployment
    operation: str
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class ProviderReadiness:
    provider: str
    state: ProviderOperationalState
    deployment: ProviderDeployment
    detail: str
    checked_at: datetime
    evidence_id: str | None

    def public(self) -> ProviderHealth:
        deployment = self.deployment.value.replace("_", "-")
        detail = f"{deployment}: {self.detail}"
        return ProviderHealth(
            provider=self.provider,
            state=self.state,
            detail=redact_sensitive_text(detail),
            checked_at=self.checked_at,
            evidence_id=self.evidence_id,
        )


class ProviderRegistry:
    """Read provider state without equating a fallback with provider success."""

    def __init__(
        self,
        *,
        laserdata: LaserDataTelemetryBackend,
        graph_memory: ResilientGraphMemory,
        guild: GuildCoordinator,
        rocketride: ResilientPipelineExecutor,
        assets: AssetGenerationPipeline,
        deployments: Mapping[str, ProviderDeployment],
        evidence: tuple[ProviderEvidence, ...] = (),
    ) -> None:
        self.laserdata = laserdata
        self.graph_memory = graph_memory
        self.guild = guild
        self.rocketride = rocketride
        self.assets = assets
        self._deployments = dict(deployments)
        self._evidence = {item.provider: item for item in evidence}

    def snapshots(self) -> tuple[ProviderReadiness, ...]:
        now = datetime.now(UTC)
        laser = laserdata_provider_view(self.laserdata.health, checked_at=now)
        graph = graph_provider_view(self.graph_memory.health())
        guild = orchestration_provider_view(self.guild.status)
        rocketride = orchestration_provider_view(self.rocketride.status)
        asset_views = tuple(
            asset_provider_view(snapshot, checked_at=now)
            for snapshot in self.assets.provider_snapshots
        )
        views = (laser, graph, guild, rocketride, *asset_views)
        return tuple(self._readiness(view) for view in views)

    def health(self) -> ServiceHealth:
        snapshots = self.snapshots()
        required_names = {"LaserData", "FalkorDB", "guild.ai", "rocketride.ai"}
        required = tuple(item for item in snapshots if item.provider in required_names)
        states = {item.state for item in required}
        if states == {ProviderOperationalState.UNCONFIGURED}:
            aggregate = ProviderOperationalState.UNCONFIGURED
        elif any(state is ProviderOperationalState.DEGRADED for state in states):
            aggregate = ProviderOperationalState.DEGRADED
        elif required and all(
            item.state is ProviderOperationalState.END_TO_END_VERIFIED for item in required
        ):
            aggregate = ProviderOperationalState.END_TO_END_VERIFIED
        elif required and all(
            item.state
            in {
                ProviderOperationalState.HEALTHY,
                ProviderOperationalState.END_TO_END_VERIFIED,
            }
            for item in required
        ):
            aggregate = ProviderOperationalState.HEALTHY
        else:
            aggregate = ProviderOperationalState.CONFIGURED
        return ServiceHealth(
            state=aggregate,
            providers=tuple(item.public() for item in snapshots),
            checked_at=datetime.now(UTC),
        )

    def _readiness(self, view: ProviderHealth) -> ProviderReadiness:
        evidence = self._evidence.get(view.provider)
        state = view.state
        detail = view.detail
        if state is ProviderOperationalState.END_TO_END_VERIFIED and evidence is None:
            state = ProviderOperationalState.HEALTHY
            detail = f"{detail}; provider verification has no coordinator evidence reference"
        deployment = self._deployments.get(
            view.provider,
            ProviderDeployment.UNCONFIGURED,
        )
        return ProviderReadiness(
            provider=view.provider,
            state=state,
            deployment=deployment,
            detail=detail,
            checked_at=view.checked_at,
            evidence_id=(evidence.evidence_id if evidence is not None else None),
        )


def _deployment(endpoint: str | None) -> ProviderDeployment:
    if not endpoint:
        return ProviderDeployment.UNCONFIGURED
    value = endpoint.strip().lower()
    if any(marker in value for marker in ("localhost", "127.0.0.1", "[::1]")):
        return ProviderDeployment.SELF_HOSTED
    parsed = urlparse(value if "://" in value else f"scheme://{value}")
    hostname = parsed.hostname
    if hostname is not None:
        if "." not in hostname:
            return ProviderDeployment.SELF_HOSTED
        try:
            if ipaddress.ip_address(hostname).is_private:
                return ProviderDeployment.SELF_HOSTED
        except ValueError:
            pass
    return ProviderDeployment.CLOUD


_GUILD_CREDENTIAL_ENV = {
    GuildRole.WORLD_AND_PHYSICS: "MUSCLE_MEMORY_GUILD_WORLD_AND_PHYSICS_CREDENTIALS",
    GuildRole.FAILURE_AND_CURRICULUM: ("MUSCLE_MEMORY_GUILD_FAILURE_AND_CURRICULUM_CREDENTIALS"),
    GuildRole.SAFETY_AND_EVALUATION: ("MUSCLE_MEMORY_GUILD_SAFETY_AND_EVALUATION_CREDENTIALS"),
}


def _build_guild(
    values: Mapping[str, str],
    evidence_source: GuildEvidenceSource | None,
) -> ResilientGuildCoordinator:
    owner = values.get("MUSCLE_MEMORY_GUILD_OWNER", "").strip()
    workspace = values.get("MUSCLE_MEMORY_GUILD_WORKSPACE", "").strip()
    credentials = tuple(
        values.get(_GUILD_CREDENTIAL_ENV[role], "").strip() for role in EXACT_GUILD_ROLES
    )
    if not owner or not workspace or not all(credentials):
        live: GuildCoordinator = UnconfiguredGuildCoordinator(
            "Guild owner, workspace, and all three role credentials are required"
        )
    else:
        live = GuildApiCoordinator(
            GuildApiConfig(
                owner=owner,
                workspace=workspace,
                endpoints=tuple(
                    GuildRoleEndpoint(role=role, basic_credentials=credential)
                    for role, credential in zip(EXACT_GUILD_ROLES, credentials, strict=True)
                ),
                base_url=values.get("MUSCLE_MEMORY_GUILD_BASE_URL", "https://app.guild.ai"),
            ),
            evidence_source=evidence_source,
        )
    return ResilientGuildCoordinator(live, InMemoryGuildReviewCache())


def _build_rocketride(
    values: Mapping[str, str],
    approval_ledger: ApprovalLedger,
) -> ResilientPipelineExecutor:
    uri = values.get("ROCKETRIDE_URI", "").strip()
    api_key = values.get("ROCKETRIDE_APIKEY", "").strip()
    try:
        artifact = ReviewedPipelineArtifact.from_env(values)
    except ReviewedPipelineError:
        artifact = None
    transport: StepTransport
    if not uri or not api_key or artifact is None:
        transport = UnconfiguredRocketRideTransport(
            "RocketRide URI, API key, reviewed pipeline, or callback configuration is missing"
        )
    else:
        transport = RocketRideSdkTransport(
            RocketRideSdkConfig(
                uri=uri,
                api_key=api_key,
                pipeline_path=artifact.pipeline_path,
                pipeline_sha256=artifact.pipeline_sha256,
                callback_environment=artifact.sdk_environment,
            ),
            approval_ledger,
        )
    executor = FixedPipelineExecutor(transport, approval_ledger)
    return ResilientPipelineExecutor(executor, InMemoryPipelineRunCache())


@dataclass(slots=True)
class ProviderBundle:
    registry: ProviderRegistry
    laserdata: LaserDataTelemetryBackend
    graph_memory: ResilientGraphMemory
    guild: ResilientGuildCoordinator
    rocketride: ResilientPipelineExecutor
    assets: AssetGenerationPipeline


def build_provider_bundle(
    config: BackendConfig,
    *,
    approval_ledger: ApprovalLedger | None = None,
    coordinator: CoordinatorStore | None = None,
) -> ProviderBundle:
    values = config.environ
    laser_config = LaserDataConfig.from_env(values)
    laserdata = LaserDataTelemetryBackend(laser_config)

    graph_settings = settings_from_env(values)
    if not graph_settings.cache_path.is_absolute():
        graph_settings = FalkorDBSettings(
            url=graph_settings.url,
            graph_name=graph_settings.graph_name,
            query_timeout_ms=graph_settings.query_timeout_ms,
            cache_path=Path.cwd() / graph_settings.cache_path,
        )
    graph_memory = build_graph_memory(graph_settings)
    guild = _build_guild(
        values,
        None if coordinator is None else CoordinatorGuildEvidenceSource(coordinator),
    )
    rocketride = _build_rocketride(values, approval_ledger or InMemoryApprovalLedger())
    assets = AssetGenerationPipeline(
        cache=ContentAddressedAssetCache(config.asset_cache_path),
        approvals=AssetApprovalLedger(config.asset_approval_path),
        reference_provider=ReferenceImageHttpAdapter(
            endpoint=config.reference_endpoint,
            api_key=config.reference_api_key,
            timeout_seconds=config.reference_timeout_seconds,
        ),
        mesh_provider=TrellisHttpAdapter(
            endpoint=config.trellis_endpoint,
            api_key=config.trellis_api_key,
            timeout_seconds=config.trellis_timeout_seconds,
        ),
    )
    deployments = {
        "LaserData": _deployment(laser_config.connection_string),
        "FalkorDB": _deployment(
            graph_settings.url.get_secret_value() if graph_settings.url is not None else None
        ),
        ProviderName.GUILD.value: (
            ProviderDeployment.CLOUD
            if guild.status.mode.value != "unconfigured"
            else ProviderDeployment.UNCONFIGURED
        ),
        ProviderName.ROCKETRIDE.value: _deployment(values.get("ROCKETRIDE_URI")),
        "reference-image-http": _deployment(config.reference_endpoint),
        "trellis-http": _deployment(config.trellis_endpoint),
    }
    registry = ProviderRegistry(
        laserdata=laserdata,
        graph_memory=graph_memory,
        guild=guild,
        rocketride=rocketride,
        assets=assets,
        deployments=deployments,
    )
    return ProviderBundle(
        registry=registry,
        laserdata=laserdata,
        graph_memory=graph_memory,
        guild=guild,
        rocketride=rocketride,
        assets=assets,
    )


__all__ = [
    "ProviderBundle",
    "ProviderDeployment",
    "ProviderEvidence",
    "ProviderReadiness",
    "ProviderRegistry",
    "build_provider_bundle",
]
