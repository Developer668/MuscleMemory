"""Configuration and honest failover for FalkorDB-backed graph memory."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

from muscle_memory.graph_memory.cache import AppendOnlyGraphCache
from muscle_memory.graph_memory.falkordb import FalkorGraph, FalkorGraphMemory
from muscle_memory.graph_memory.models import (
    ContentAddressedRecord,
    CorrectionMemoryRecord,
    CurriculumQuery,
    CurriculumResult,
    EpisodeMemoryRecord,
    EvaluatedPolicyVersion,
    FailureMemoryRecord,
    FalkorDBSettings,
    GraphMemoryHealth,
    GraphStorage,
    GraphWriteReceipt,
    LessonMemoryRecord,
    ObstacleMemoryRecord,
    PolicyComparisonRecord,
    PolicyEvaluationRecord,
    PolicyTrainingRecord,
    ProviderState,
    WorldMemoryRecord,
)
from muscle_memory.graph_memory.protocol import (
    GraphMemory,
    GraphProviderUnavailableError,
)


class FalkorClient(Protocol):
    def select_graph(self, graph_id: str) -> FalkorGraph: ...


class FalkorFromURL(Protocol):
    def __call__(self, url: str, **kwargs: object) -> FalkorClient: ...


def settings_from_env(environ: Mapping[str, str] | None = None) -> FalkorDBSettings:
    """Load service settings without treating absent credentials as configured."""

    source = os.environ if environ is None else environ
    try:
        timeout = int(source.get("MUSCLE_MEMORY_FALKORDB_QUERY_TIMEOUT_MS", "2000"))
    except ValueError as exc:
        raise ValueError("MUSCLE_MEMORY_FALKORDB_QUERY_TIMEOUT_MS must be an integer") from exc
    return FalkorDBSettings.model_validate(
        {
            "url": source.get("MUSCLE_MEMORY_FALKORDB_URL"),
            "graph_name": source.get("MUSCLE_MEMORY_FALKORDB_GRAPH", "muscle_memory"),
            "query_timeout_ms": timeout,
            "cache_path": Path(
                source.get(
                    "MUSCLE_MEMORY_FALKORDB_CACHE_PATH",
                    ".cache/muscle-memory/falkordb-events.jsonl",
                )
            ),
        }
    )


def _connect_official_client(settings: FalkorDBSettings) -> FalkorGraphMemory:
    if settings.url is None:
        raise ValueError("FalkorDB is unconfigured")
    try:
        module = import_module("falkordb")
        client_class = module.FalkorDB
        from_url = cast(FalkorFromURL, client_class.from_url)
    except (AttributeError, ImportError) as exc:
        raise GraphProviderUnavailableError(
            "official FalkorDB Python client is not installed"
        ) from exc

    timeout_seconds = settings.query_timeout_ms / 1_000.0
    try:
        client = from_url(
            settings.url.get_secret_value(),
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            health_check_interval=30,
            client_name="muscle-memory",
        )
        graph = client.select_graph(settings.graph_name)
    except Exception as exc:
        raise GraphProviderUnavailableError(
            f"official FalkorDB client initialization failed: {type(exc).__name__}"
        ) from exc
    return FalkorGraphMemory(
        graph,
        graph_name=settings.graph_name,
        query_timeout_ms=settings.query_timeout_ms,
    )


class ResilientGraphMemory:
    """Mirror facts locally and expose fallback state on every read or write."""

    def __init__(
        self,
        *,
        settings: FalkorDBSettings,
        cache: AppendOnlyGraphCache,
        remote: GraphMemory | None,
        initialization_error: str | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._remote = remote
        self._initialization_error = initialization_error
        self._reconciled_event_count = 0
        self._reconciliation_error: str | None = None
        self._reconciliation_lock = RLock()

    def configuration_health(self) -> GraphMemoryHealth:
        if self._settings.url is None:
            state = ProviderState.UNCONFIGURED
            detail = "MUSCLE_MEMORY_FALKORDB_URL is not set"
        else:
            state = ProviderState.CONFIGURED
            detail = "FalkorDB credentials are configured but not yet health-checked"
        return GraphMemoryHealth(
            provider_state=state,
            graph_name=self._settings.graph_name,
            detail=detail,
            checked_at=datetime.now(UTC),
        )

    def health(self) -> GraphMemoryHealth:
        if self._settings.url is None:
            return GraphMemoryHealth(
                provider_state=ProviderState.UNCONFIGURED,
                graph_name=self._settings.graph_name,
                detail="FalkorDB is unconfigured; append-only local fallback remains available",
                checked_at=datetime.now(UTC),
            )
        if self._remote is None:
            error = self._initialization_error or "provider client is unavailable"
            return GraphMemoryHealth(
                provider_state=ProviderState.UNAVAILABLE,
                graph_name=self._settings.graph_name,
                detail=f"FalkorDB unavailable: {error}; local fallback remains available",
                checked_at=datetime.now(UTC),
            )
        remote_health = self._remote.health()
        if remote_health.provider_state not in {
            ProviderState.HEALTHY,
            ProviderState.END_TO_END_VERIFIED,
        }:
            pending = max(0, len(self._cache.events) - self._reconciled_event_count)
            return GraphMemoryHealth(
                provider_state=remote_health.provider_state,
                graph_name=remote_health.graph_name,
                detail=f"{remote_health.detail}; {pending} cached facts await reconciliation",
                checked_at=remote_health.checked_at,
            )
        if not self._reconcile_if_needed():
            return GraphMemoryHealth(
                provider_state=ProviderState.UNAVAILABLE,
                graph_name=self._settings.graph_name,
                detail=(
                    "FalkorDB recovered but append-only cache reconciliation failed: "
                    f"{self._reconciliation_error}"
                ),
                checked_at=datetime.now(UTC),
            )
        return GraphMemoryHealth(
            provider_state=remote_health.provider_state,
            graph_name=remote_health.graph_name,
            detail=(
                f"{remote_health.detail}; {self._reconciled_event_count} cached facts "
                "are reconciled"
            ),
            checked_at=remote_health.checked_at,
        )

    def record_world(self, record: WorldMemoryRecord) -> GraphWriteReceipt:
        return self._write(record, self._cache.record_world)

    def record_obstacle(self, record: ObstacleMemoryRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_obstacle,
        )

    def record_evaluated_policy(self, record: EvaluatedPolicyVersion) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_evaluated_policy,
        )

    def record_episode(self, record: EpisodeMemoryRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_episode,
        )

    def record_failure(self, record: FailureMemoryRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_failure,
        )

    def record_correction(self, record: CorrectionMemoryRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_correction,
        )

    def record_lesson(self, record: LessonMemoryRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_lesson,
        )

    def record_policy_training(self, record: PolicyTrainingRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_policy_training,
        )

    def record_policy_evaluation(self, record: PolicyEvaluationRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_policy_evaluation,
        )

    def record_outperformance(self, record: PolicyComparisonRecord) -> GraphWriteReceipt:
        return self._write(
            record,
            self._cache.record_outperformance,
        )

    def query_curriculum(self, query: CurriculumQuery) -> CurriculumResult:
        if self._remote is not None:
            if not self._reconcile_if_needed():
                return self._fallback_query(
                    query,
                    "cached facts are not reconciled to the recovered provider",
                )
            try:
                return self._remote.query_curriculum(query)
            except GraphProviderUnavailableError as exc:
                return self._fallback_query(query, str(exc))
        detail = self._initialization_error or "provider is unconfigured"
        return self._fallback_query(query, detail)

    def synchronize_local_cache(self) -> int:
        """Replay all immutable cached facts idempotently after provider recovery."""

        if self._remote is None:
            raise GraphProviderUnavailableError("cannot synchronize without FalkorDB")
        with self._reconciliation_lock:
            try:
                # FalkorDB receives the immutable local backlog after recovery, keeping
                # explicit experience queryable without rewriting earlier graph facts.
                replayed, event_cursor = self._cache.replay_to_with_cursor(self._remote)
            except GraphProviderUnavailableError as exc:
                self._reconciliation_error = type(exc).__name__
                raise
            self._reconciled_event_count = event_cursor
            self._reconciliation_error = None
            return replayed

    def _write[T: ContentAddressedRecord](
        self,
        record: T,
        cache_write: Callable[[T], GraphWriteReceipt],
    ) -> GraphWriteReceipt:
        local_receipt = cache_write(record)
        if self._remote is None:
            state = (
                ProviderState.UNCONFIGURED
                if self._settings.url is None
                else ProviderState.UNAVAILABLE
            )
            detail = self._initialization_error or "FalkorDB is unconfigured"
            return self._fallback_receipt(local_receipt, state, detail)
        if not self._reconcile_if_needed():
            return self._fallback_receipt(
                local_receipt,
                ProviderState.UNAVAILABLE,
                self._reconciliation_error or "cache reconciliation failed",
            )
        return GraphWriteReceipt(
            record_kind=local_receipt.record_kind,
            record_id=local_receipt.record_id,
            content_hash=local_receipt.content_hash,
            storage=GraphStorage.FALKORDB,
            provider_state=ProviderState.HEALTHY,
            mirrored_to_local_cache=True,
            detail="stored in FalkorDB and mirrored to append-only local cache",
        )

    def _reconcile_if_needed(self) -> bool:
        if self._remote is None:
            return False
        with self._reconciliation_lock:
            if len(self._cache.events) == self._reconciled_event_count:
                return True
            try:
                _replayed, event_cursor = self._cache.replay_to_with_cursor(self._remote)
            except GraphProviderUnavailableError as exc:
                self._reconciliation_error = type(exc).__name__
                return False
            self._reconciled_event_count = event_cursor
            self._reconciliation_error = None
            return True

    def _fallback_query(self, query: CurriculumQuery, reason: str) -> CurriculumResult:
        cached = self._cache.query_curriculum(query)
        state = (
            ProviderState.UNCONFIGURED
            if self._settings.url is None
            else ProviderState.UNAVAILABLE
        )
        return CurriculumResult(
            lessons=cached.lessons,
            storage=GraphStorage.LOCAL_CACHE,
            provider_state=state,
            detail=f"FalkorDB not used ({reason}); results came from local cache",
        )

    @staticmethod
    def _fallback_receipt(
        local: GraphWriteReceipt,
        state: ProviderState,
        reason: str,
    ) -> GraphWriteReceipt:
        return GraphWriteReceipt(
            record_kind=local.record_kind,
            record_id=local.record_id,
            content_hash=local.content_hash,
            storage=GraphStorage.LOCAL_CACHE,
            provider_state=state,
            mirrored_to_local_cache=True,
            detail=f"FalkorDB not used ({reason}); fact is retained only in local cache",
        )


def build_graph_memory(settings: FalkorDBSettings | None = None) -> ResilientGraphMemory:
    """Construct the provider and durable cache without hiding initialization failures."""

    resolved = settings_from_env() if settings is None else settings
    cache = AppendOnlyGraphCache(resolved.cache_path)
    if resolved.url is None:
        return ResilientGraphMemory(settings=resolved, cache=cache, remote=None)
    try:
        remote = _connect_official_client(resolved)
    except GraphProviderUnavailableError as exc:
        return ResilientGraphMemory(
            settings=resolved,
            cache=cache,
            remote=None,
            initialization_error=str(exc),
        )
    return ResilientGraphMemory(settings=resolved, cache=cache, remote=remote)
