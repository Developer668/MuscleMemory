"""FastAPI application factory for the versioned Muscle Memory backend API."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Path,
    Query,
    Request,
    Security,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from muscle_memory.api.auth import (
    APPROVAL_WRITE_SCOPE,
    CORRECTION_WRITE_SCOPE,
    EPISODE_WRITE_SCOPE,
    WORKFLOW_WRITE_SCOPE,
)
from muscle_memory.api.contracts import (
    ApiBackend,
    ApiBackendError,
    AuthenticatedPrincipal,
    Authenticator,
    LiveEpisodeControl,
)
from muscle_memory.api.models import (
    API_VERSION,
    NUMERIC_TELEMETRY_HZ,
    ApiError,
    ApiErrorResponse,
    ApprovalDecisionView,
    AssetStatus,
    AssetStatusList,
    CorrectionRequest,
    CorrectionView,
    DecisionRequest,
    EpisodeDetail,
    EpisodeList,
    LiveEpisodeOptionsView,
    LiveEpisodeStartRequest,
    LiveEpisodeStatusView,
    LivePolicyOptionView,
    MemoryGraphSnapshot,
    PendingApprovalList,
    PolicySummaryList,
    PromotionEligibility,
    ReplayPage,
    ServiceHealth,
    TelemetryPage,
    WorkflowReview,
    WorkflowReviewRequest,
    WorkflowRun,
)
from muscle_memory.api.redaction import redact_sensitive_mapping, redact_sensitive_text
from muscle_memory.api.streaming import LiveTelemetryHub
from muscle_memory.backend.rocketride_callback import (
    CALLBACK_PATH,
    MAX_CALLBACK_BODY_BYTES,
    CallbackApprovalError,
    CallbackContractError,
    CallbackHandlerError,
    CallbackSequenceError,
    CallbackUnauthorizedError,
)
from muscle_memory.live.controller import (
    LiveEpisodeConflictError,
    LiveEpisodeNotFoundError,
    LiveEpisodeSelectionError,
)
from muscle_memory.live.models import LiveEpisodeStatus, VideoProduct
from muscle_memory.live.video import MJPEG_BOUNDARY

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_bearer = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    description="Bearer token verified against a server-side SHA-256 digest.",
)

ErrorResponses = dict[int | str, dict[str, Any]]
COMMON_ERROR_RESPONSES: ErrorResponses = {
    401: {"model": ApiErrorResponse, "description": "Authentication required"},
    403: {"model": ApiErrorResponse, "description": "Scope denied"},
    404: {"model": ApiErrorResponse, "description": "Resource not found"},
    409: {"model": ApiErrorResponse, "description": "Immutable state conflict"},
    422: {"model": ApiErrorResponse, "description": "Request validation failed"},
    503: {"model": ApiErrorResponse, "description": "Required provider unavailable"},
}


class _CallbackBodyTooLargeError(RuntimeError):
    pass


async def _bounded_callback_body(request: Request) -> bytes:
    """Consume the ASGI stream without ever accumulating more than the limit."""

    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_CALLBACK_BODY_BYTES:
                raise _CallbackBodyTooLargeError
        except ValueError as exc:
            raise CallbackContractError("callback content length is invalid") from exc
    body = bytearray()
    async for chunk in request.stream():
        remaining = MAX_CALLBACK_BODY_BYTES - len(body)
        if len(chunk) > remaining:
            raise _CallbackBodyTooLargeError
        body.extend(chunk)
    if not body:
        raise CallbackContractError("callback body is empty")
    return bytes(body)


def _frontend_dist() -> FsPath:
    configured = os.environ.get("MM_FRONTEND_DIST")
    if configured:
        return FsPath(configured).expanduser().resolve()
    return FsPath(__file__).resolve().parents[3] / "frontend" / "dist"


@dataclass(frozen=True, slots=True)
class ApiRuntime:
    backend: ApiBackend
    authenticator: Authenticator
    live_hub: LiveTelemetryHub
    live_episodes: LiveEpisodeControl | None


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else uuid.uuid4().hex


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    payload = ApiErrorResponse(
        error=ApiError(
            code=code,
            message=redact_sensitive_text(message),
            request_id=request_id,
            details=(redact_sensitive_mapping(details) if details is not None else None),
        )
    )
    headers = {"X-Request-ID": request_id}
    if status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def _runtime_from_request(request: Request) -> ApiRuntime:
    runtime = getattr(request.app.state, "api_runtime", None)
    if not isinstance(runtime, ApiRuntime):
        raise RuntimeError("API runtime is not configured")
    return runtime


def _principal_dependency(
    required_scope: str,
) -> Callable[..., Awaitable[AuthenticatedPrincipal]]:
    async def require_principal(
        request: Request,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Security(_bearer),
        ] = None,
    ) -> AuthenticatedPrincipal:
        runtime = _runtime_from_request(request)
        if not runtime.authenticator.configured:
            raise ApiBackendError(
                status.HTTP_401_UNAUTHORIZED,
                "authentication_unconfigured",
                "mutation authentication is not configured",
            )
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise ApiBackendError(
                status.HTTP_401_UNAUTHORIZED,
                "authentication_required",
                "a valid bearer credential is required",
            )
        principal = runtime.authenticator.authenticate(credentials.credentials)
        if principal is None:
            raise ApiBackendError(
                status.HTTP_401_UNAUTHORIZED,
                "invalid_credential",
                "a valid bearer credential is required",
            )
        if required_scope not in principal.scopes:
            raise ApiBackendError(
                status.HTTP_403_FORBIDDEN,
                "insufficient_scope",
                "the authenticated principal lacks the required scope",
                details={"required_scope": required_scope},
            )
        return principal

    return require_principal


RequireApprovalPrincipal = Annotated[
    AuthenticatedPrincipal,
    Depends(_principal_dependency(APPROVAL_WRITE_SCOPE)),
]
RequireWorkflowPrincipal = Annotated[
    AuthenticatedPrincipal,
    Depends(_principal_dependency(WORKFLOW_WRITE_SCOPE)),
]
RequireCorrectionPrincipal = Annotated[
    AuthenticatedPrincipal,
    Depends(_principal_dependency(CORRECTION_WRITE_SCOPE)),
]
RequireEpisodePrincipal = Annotated[
    AuthenticatedPrincipal,
    Depends(_principal_dependency(EPISODE_WRITE_SCOPE)),
]


def _live_control(request: Request) -> LiveEpisodeControl:
    control = _runtime_from_request(request).live_episodes
    if control is None:
        raise ApiBackendError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "live_runtime_unconfigured",
            "the live simulator has no admitted world and policy configuration",
        )
    return control


def _live_status_view(value: LiveEpisodeStatus) -> LiveEpisodeStatusView:
    encoded_episode_id = quote(value.episode_id, safe="")
    payload = value.as_json_value()
    payload["detail"] = (
        None if value.detail is None else redact_sensitive_text(value.detail)[:500]
    )
    payload["video_streams"] = {
        product.value: (
            f"/api/{API_VERSION}/episodes/{encoded_episode_id}/video/"
            f"{product.value}.mjpeg"
        )
        for product in VideoProduct
    }
    return LiveEpisodeStatusView.model_validate(payload)


def _not_found(resource: str, identifier: str) -> ApiBackendError:
    return ApiBackendError(
        status.HTTP_404_NOT_FOUND,
        f"{resource}_not_found",
        f"{resource.replace('_', ' ')} was not found",
        details={f"{resource}_id": identifier},
    )


def _build_router() -> APIRouter:
    router = APIRouter(prefix=f"/api/{API_VERSION}")

    @router.get(
        "/health",
        response_model=ServiceHealth,
        summary="Read service and provider health",
    )
    async def health(request: Request) -> ServiceHealth:
        return await _runtime_from_request(request).backend.health()

    @router.get(
        "/memory/graph",
        response_model=MemoryGraphSnapshot,
        summary="Read the operational FalkorDB memory graph",
    )
    async def memory_graph(request: Request) -> MemoryGraphSnapshot:
        return await _runtime_from_request(request).backend.memory_graph()

    @router.get(
        "/episodes",
        response_model=EpisodeList,
        responses=COMMON_ERROR_RESPONSES,
        summary="List operational episodes",
    )
    async def episodes(
        request: Request,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> EpisodeList:
        return await _runtime_from_request(request).backend.list_episodes(
            cursor=cursor,
            limit=limit,
        )

    @router.get(
        "/episodes/{episode_id}",
        response_model=EpisodeDetail,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read one operational episode",
    )
    async def episode(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
    ) -> EpisodeDetail:
        result = await _runtime_from_request(request).backend.episode(episode_id)
        if result is None:
            raise _not_found("episode", episode_id)
        return result

    @router.get(
        "/episodes/{episode_id}/telemetry",
        response_model=TelemetryPage,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read append-only episode telemetry",
    )
    async def telemetry(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
        after_sequence: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=2_000)] = 400,
    ) -> TelemetryPage:
        result = await _runtime_from_request(request).backend.telemetry(
            episode_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        if result is None:
            raise _not_found("episode", episode_id)
        return result

    @router.get(
        "/episodes/{episode_id}/replay",
        response_model=ReplayPage,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read an ordered episode replay",
    )
    async def replay(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
        after_sequence: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=2_000)] = 400,
    ) -> ReplayPage:
        result = await _runtime_from_request(request).backend.replay(
            episode_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        if result is None:
            raise _not_found("episode", episode_id)
        return result

    @router.get(
        "/live/options",
        response_model=LiveEpisodeOptionsView,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read admitted live simulator options",
    )
    async def live_options(request: Request) -> LiveEpisodeOptionsView:
        control = _runtime_from_request(request).live_episodes
        if control is None:
            return LiveEpisodeOptionsView(
                enabled=False,
                unavailable_reason=(
                    "No immutable evaluated checkpoint and live-world catalog are "
                    "admitted in this deployment."
                ),
                seeds=(),
                policies=(),
                default_policy_id=None,
                video_products=(),
            )
        options = control.options()
        return LiveEpisodeOptionsView(
            enabled=True,
            catalog_id=options.catalog_id,
            catalog_sha256=options.catalog_sha256,
            seeds=options.seeds,
            policies=tuple(
                LivePolicyOptionView(
                    policy_id=policy.policy_id,
                    policy_hash=policy.policy_hash,
                    evaluated_episode_count=policy.evaluated_episode_count,
                    promotable=policy.promotable,
                    deployment_status=policy.deployment_status,  # type: ignore[arg-type]
                    is_default=policy.is_default,
                )
                for policy in options.policies
            ),
            default_policy_id=options.default_policy_id,
            video_products=options.video_products,  # type: ignore[arg-type]
            maximum_duration_seconds=options.maximum_duration_seconds,
        )

    @router.post(
        "/live/episodes",
        response_model=LiveEpisodeStatusView,
        status_code=status.HTTP_202_ACCEPTED,
        responses=COMMON_ERROR_RESPONSES,
        summary="Start one validation-gated real simulator episode",
    )
    async def start_live_episode(
        request: Request,
        body: LiveEpisodeStartRequest,
        _principal: RequireEpisodePrincipal,
    ) -> LiveEpisodeStatusView:
        control = _live_control(request)
        try:
            started = await asyncio.to_thread(
                control.start,
                seed=body.seed,
                policy_id=body.policy_id,
            )
        except LiveEpisodeSelectionError as exc:
            raise ApiBackendError(422, "live_selection_not_admitted", str(exc)) from exc
        except LiveEpisodeConflictError as exc:
            raise ApiBackendError(409, "live_episode_conflict", str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise ApiBackendError(
                503,
                "live_runtime_integrity_failure",
                "the admitted live runtime failed its integrity check",
            ) from exc
        return _live_status_view(started)

    @router.get(
        "/live/episodes/{episode_id}",
        response_model=LiveEpisodeStatusView,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read one live simulator worker status",
    )
    async def live_episode_status(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
    ) -> LiveEpisodeStatusView:
        try:
            value = _live_control(request).status(episode_id)
        except LiveEpisodeNotFoundError as exc:
            raise _not_found("live_episode", episode_id) from exc
        return _live_status_view(value)

    @router.post(
        "/live/episodes/{episode_id}/cancel",
        response_model=LiveEpisodeStatusView,
        status_code=status.HTTP_202_ACCEPTED,
        responses=COMMON_ERROR_RESPONSES,
        summary="Request cancellation on the next 20 Hz tick",
    )
    async def cancel_live_episode(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
        _principal: RequireEpisodePrincipal,
    ) -> LiveEpisodeStatusView:
        try:
            value = _live_control(request).cancel(episode_id)
        except LiveEpisodeNotFoundError as exc:
            raise _not_found("live_episode", episode_id) from exc
        return _live_status_view(value)

    @router.get(
        "/episodes/{episode_id}/video/{product}.mjpeg",
        responses=COMMON_ERROR_RESPONSES,
        response_class=StreamingResponse,
        summary="Stream one direct simulator video product",
    )
    async def live_video_stream(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
        product: VideoProduct,
        after_frame_index: Annotated[int, Query(ge=-1)] = -1,
    ) -> StreamingResponse:
        try:
            chunks = _live_control(request).iter_mjpeg(
                episode_id,
                product,
                after_frame_index=after_frame_index,
            )
        except LiveEpisodeNotFoundError as exc:
            raise _not_found("live_episode", episode_id) from exc
        return StreamingResponse(
            chunks,
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "X-Frame-Join-Key": "frame_id"},
        )

    @router.get(
        "/episodes/{episode_id}/video/{product}/frames/{frame_index}",
        responses=COMMON_ERROR_RESPONSES,
        response_class=Response,
        summary="Read one exact buffered simulator frame",
    )
    async def live_video_frame(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
        product: VideoProduct,
        frame_index: Annotated[int, Path(ge=0)],
    ) -> Response:
        try:
            frame_set = _live_control(request).video_frame(episode_id, frame_index)
        except LiveEpisodeNotFoundError as exc:
            raise _not_found("live_episode", episode_id) from exc
        if frame_set is None:
            raise ApiBackendError(
                404,
                "video_frame_not_found",
                "the frame is unavailable or has left the bounded direct-video buffer",
                details={"episode_id": episode_id, "frame_index": frame_index},
            )
        frame = frame_set.product(product)
        return Response(
            content=frame.data,
            media_type=frame.mime_type,
            headers={
                "Cache-Control": "no-store",
                "ETag": f'"{frame.sha256}"',
                "X-Frame-ID": frame_set.metadata.frame_id,
                "X-Frame-Join-Key": "frame_id",
            },
        )

    @router.get(
        "/approvals/pending",
        response_model=PendingApprovalList,
        responses=COMMON_ERROR_RESPONSES,
        summary="List blocking human decisions",
    )
    async def pending_approvals(request: Request) -> PendingApprovalList:
        return await _runtime_from_request(request).backend.pending_approvals()

    @router.post(
        "/approvals/{requirement_id}/decision",
        response_model=ApprovalDecisionView,
        responses=COMMON_ERROR_RESPONSES,
        summary="Record one immutable authenticated human decision",
    )
    async def submit_approval_decision(
        request: Request,
        requirement_id: Annotated[str, Path(min_length=1, max_length=256)],
        body: DecisionRequest,
        principal: RequireApprovalPrincipal,
    ) -> ApprovalDecisionView:
        return await _runtime_from_request(request).backend.submit_approval_decision(
            requirement_id,
            body,
            principal,
        )

    @router.post(
        "/workflows/review",
        response_model=WorkflowReview,
        status_code=status.HTTP_201_CREATED,
        responses=COMMON_ERROR_RESPONSES,
        summary="Request the three specialist reviews",
    )
    async def review_workflow(
        request: Request,
        body: WorkflowReviewRequest,
        principal: RequireWorkflowPrincipal,
    ) -> WorkflowReview:
        return await _runtime_from_request(request).backend.review_workflow(body, principal)

    @router.get(
        "/workflows/{run_id}",
        response_model=WorkflowRun,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read a workflow run state",
    )
    async def workflow(
        request: Request,
        run_id: Annotated[str, Path(min_length=1, max_length=128)],
    ) -> WorkflowRun:
        result = await _runtime_from_request(request).backend.workflow(run_id)
        if result is None:
            raise _not_found("workflow", run_id)
        return result

    @router.post(
        "/workflows/{run_id}/execute",
        response_model=WorkflowRun,
        responses=COMMON_ERROR_RESPONSES,
        summary="Execute a reviewed fixed workflow",
    )
    async def execute_workflow(
        request: Request,
        run_id: Annotated[str, Path(min_length=1, max_length=128)],
        principal: RequireWorkflowPrincipal,
    ) -> WorkflowRun:
        return await _runtime_from_request(request).backend.execute_workflow(
            run_id,
            principal,
        )

    @router.post(
        "/workflows/{run_id}/resume",
        response_model=WorkflowRun,
        responses=COMMON_ERROR_RESPONSES,
        summary="Resume an incomplete workflow after its blocking decision",
    )
    async def resume_workflow(
        request: Request,
        run_id: Annotated[str, Path(min_length=1, max_length=128)],
        principal: RequireWorkflowPrincipal,
    ) -> WorkflowRun:
        return await _runtime_from_request(request).backend.resume_workflow(
            run_id,
            principal,
        )

    @router.post(
        "/episodes/{episode_id}/corrections",
        response_model=CorrectionView,
        status_code=status.HTTP_201_CREATED,
        responses=COMMON_ERROR_RESPONSES,
        summary="Submit a human route or keep-out correction",
    )
    async def submit_correction(
        request: Request,
        episode_id: Annotated[str, Path(min_length=1, max_length=128)],
        body: CorrectionRequest,
        principal: RequireCorrectionPrincipal,
    ) -> CorrectionView:
        return await _runtime_from_request(request).backend.submit_correction(
            episode_id,
            body,
            principal,
        )

    @router.post(
        "/corrections/{correction_id}/decision",
        response_model=CorrectionView,
        responses=COMMON_ERROR_RESPONSES,
        summary="Approve or reject a pending correction",
    )
    async def decide_correction(
        request: Request,
        correction_id: Annotated[str, Path(min_length=1, max_length=256)],
        body: DecisionRequest,
        principal: RequireCorrectionPrincipal,
    ) -> CorrectionView:
        return await _runtime_from_request(request).backend.decide_correction(
            correction_id,
            body,
            principal,
        )

    @router.get(
        "/policies",
        response_model=PolicySummaryList,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read policy evaluation summaries",
    )
    async def policies(request: Request) -> PolicySummaryList:
        return await _runtime_from_request(request).backend.policy_summaries()

    @router.get(
        "/policies/promotion-eligibility",
        response_model=PromotionEligibility,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read measured promotion-gate eligibility",
    )
    async def promotion_eligibility(
        request: Request,
        baseline_policy_id: Annotated[str, Query(min_length=1, max_length=128)],
        candidate_policy_id: Annotated[str, Query(min_length=1, max_length=128)],
    ) -> PromotionEligibility:
        return await _runtime_from_request(request).backend.promotion_eligibility(
            baseline_policy_id=baseline_policy_id,
            candidate_policy_id=candidate_policy_id,
        )

    @router.get(
        "/assets",
        response_model=AssetStatusList,
        responses=COMMON_ERROR_RESPONSES,
        summary="List generated-asset admission status",
    )
    async def assets(request: Request) -> AssetStatusList:
        items = await _runtime_from_request(request).backend.asset_statuses()
        return AssetStatusList(items=items)

    @router.get(
        "/assets/{asset_id}",
        response_model=AssetStatus,
        responses=COMMON_ERROR_RESPONSES,
        summary="Read one generated-asset admission status",
    )
    async def asset(
        request: Request,
        asset_id: Annotated[str, Path(min_length=1, max_length=256)],
    ) -> AssetStatus:
        result = await _runtime_from_request(request).backend.asset_status(asset_id)
        if result is None:
            raise _not_found("asset", asset_id)
        return result

    @router.websocket("/episodes/{episode_id}/live")
    async def live_episode(websocket: WebSocket, episode_id: str) -> None:
        runtime = websocket.app.state.api_runtime
        if not isinstance(runtime, ApiRuntime):
            await websocket.close(code=1011, reason="API runtime unavailable")
            return
        try:
            existing = await runtime.backend.episode(episode_id)
        except Exception:
            await websocket.close(code=1011, reason="episode lookup failed")
            return
        if existing is None:
            await websocket.close(code=4404, reason="episode not found")
            return
        await websocket.accept()
        minimum_interval = 1.0 / NUMERIC_TELEMETRY_HZ
        last_send = 0.0
        try:
            async with runtime.live_hub.subscribe(episode_id) as subscription:
                while True:
                    message = await subscription.receive()
                    if message is None:
                        break
                    now = asyncio.get_running_loop().time()
                    delay = minimum_interval - (now - last_send)
                    if delay > 0.0:
                        await asyncio.sleep(delay)
                    await websocket.send_text(message.model_dump_json())
                    last_send = asyncio.get_running_loop().time()
        except WebSocketDisconnect:
            return

    return router


def create_app(
    *,
    backend: ApiBackend,
    authenticator: Authenticator,
    live_hub: LiveTelemetryHub | None = None,
    live_episodes: LiveEpisodeControl | None = None,
) -> FastAPI:
    """Build an app around explicit domain services and authentication."""

    hub = live_hub or LiveTelemetryHub()
    backend.bind_live_publisher(hub)
    control = live_episodes
    if control is None:
        control = getattr(backend, "live_episode_controller", None)
    runtime = ApiRuntime(
        backend=backend,
        authenticator=authenticator,
        live_hub=hub,
        live_episodes=control,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        hub.bind_running_loop()
        await backend.startup()
        try:
            yield
        finally:
            try:
                if control is not None:
                    await asyncio.to_thread(control.shutdown)
            finally:
                await hub.close()
                await backend.shutdown()

    app = FastAPI(
        title="Muscle Memory API",
        version="1.0.0",
        description=(
            "Operational API for immutable episodes, sponsor workflow state, "
            "human decisions, and measured policy evidence."
        ),
        lifespan=lifespan,
    )
    app.state.api_runtime = runtime

    @app.post(CALLBACK_PATH, include_in_schema=False)
    async def rocketride_callback(request: Request) -> JSONResponse:
        authenticate = getattr(backend, "authenticate_rocketride_callback", None)
        if not callable(authenticate):
            return JSONResponse(
                status_code=503,
                content={"error": "rocketride_callback_unconfigured"},
            )
        try:
            authenticate(request.headers.get("authorization", ""))
        except CallbackUnauthorizedError:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        except ApiBackendError:
            return JSONResponse(
                status_code=503,
                content={"error": "rocketride_callback_unconfigured"},
            )
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            return JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={"error": "content_type_must_be_application_json"},
            )
        try:
            body = await _bounded_callback_body(request)
        except _CallbackBodyTooLargeError:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"error": "body_too_large"},
            )
        except CallbackContractError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "contract_violation", "detail": str(exc)},
            )
        try:
            wrapper = json.loads(body)
            if not isinstance(wrapper, dict) or set(wrapper) != {"data"}:
                raise CallbackContractError("callback body must contain only the data field")
            encoded = wrapper["data"]
            if not isinstance(encoded, str):
                raise CallbackContractError("callback data must be a string")
            dispatch = getattr(backend, "dispatch_rocketride_callback", None)
            if not callable(dispatch):
                raise ApiBackendError(
                    503,
                    "rocketride_callback_unconfigured",
                    "RocketRide callback configuration is unavailable",
                )
            result = await asyncio.to_thread(
                dispatch,
                encoded,
                request.headers.get("authorization", ""),
            )
        except CallbackUnauthorizedError:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        except CallbackApprovalError as exc:
            return JSONResponse(
                status_code=403,
                content={"error": "approval_rejected", "detail": str(exc)},
            )
        except CallbackSequenceError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": "sequence_violation", "detail": str(exc)},
            )
        except CallbackContractError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": "contract_violation", "detail": str(exc)},
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(status_code=400, content={"error": "invalid_json"})
        except CallbackHandlerError as exc:
            return JSONResponse(
                status_code=502,
                content={"error": "handler_failed", "detail": str(exc)},
            )
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})

    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiBackendError)
    async def backend_error_handler(
        request: Request,
        exc: ApiBackendError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        issues = [
            {
                "location": [str(part) for part in error["loc"]],
                "message": str(error["msg"]),
                "type": str(error["type"]),
            }
            for error in exc.errors()
        ]
        return _error_response(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="request_validation_failed",
            message="request validation failed",
            details={"issues": issues},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="the request could not be completed",
        )

    app.include_router(_build_router())
    frontend_dist = _frontend_dist()
    index_file = frontend_dist / "index.html"
    assets_dir = frontend_dist / "assets"
    if index_file.is_file() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        @app.get("/about", include_in_schema=False)
        @app.get("/console", include_in_schema=False)
        @app.get("/app", include_in_schema=False)
        async def frontend() -> FileResponse:
            return FileResponse(index_file, headers={"Cache-Control": "no-cache"})

    return app


__all__ = ["ApiRuntime", "create_app"]
