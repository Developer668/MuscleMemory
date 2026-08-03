"""Bounded HTTP adapters for reference-image and TRELLIS generation services."""

from __future__ import annotations

import base64
import binascii
import json
import struct
import urllib.error
import urllib.request
from collections.abc import Mapping
from threading import RLock
from typing import Any, Protocol, cast

from muscle_memory.assets.models import (
    ArtifactRole,
    AssetRequest,
    GeneratedArtifact,
    ProviderSnapshot,
    ProviderState,
    VisualMeshFormat,
)

MAX_PROVIDER_TIMEOUT_SECONDS = 30.0
MAX_JSON_RESPONSE_BYTES = 140 * 1024 * 1024
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024


class AssetProviderError(RuntimeError):
    """A provider failed without producing a cacheable artifact."""


class AssetProviderUnconfiguredError(AssetProviderError):
    """A provider endpoint was not configured."""


class JsonHttpTransport(Protocol):
    """Minimal injectable transport used by both HTTP adapters."""

    def post_json(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, object]: ...


class UrllibJsonTransport:
    """Standard-library JSON transport with a strict response-size ceiling."""

    def post_json(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        body = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", **dict(headers)},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise AssetProviderError("provider HTTP request failed") from exc
        if len(raw) > MAX_JSON_RESPONSE_BYTES:
            raise AssetProviderError("provider response exceeded the configured size limit")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssetProviderError("provider returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise AssetProviderError("provider JSON response must be an object")
        return cast(dict[str, object], decoded)


def _decode_artifact(response: Mapping[str, object]) -> tuple[bytes, str]:
    encoded = response.get("data_base64")
    media_type = response.get("media_type")
    if not isinstance(encoded, str) or not isinstance(media_type, str) or not media_type:
        raise AssetProviderError("provider response requires data_base64 and media_type")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AssetProviderError("provider artifact is not valid base64") from exc
    if not data:
        raise AssetProviderError("provider returned an empty artifact")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise AssetProviderError("provider artifact exceeded the configured size limit")
    return data, media_type


def _validate_visual_mesh(data: bytes, mesh_format: VisualMeshFormat) -> None:
    if mesh_format is VisualMeshFormat.GLB:
        if len(data) < 20 or data[:4] != b"glTF":
            raise AssetProviderError("TRELLIS response is not a GLB container")
        version, declared_length = struct.unpack_from("<II", data, 4)
        if version != 2 or declared_length != len(data):
            raise AssetProviderError("TRELLIS GLB header failed validation")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssetProviderError("TRELLIS OBJ response is not UTF-8 text") from exc
    lines = tuple(line.lstrip() for line in text.splitlines())
    if not any(line.startswith("v ") for line in lines):
        raise AssetProviderError("TRELLIS OBJ response has no vertices")
    if not any(line.startswith("f ") for line in lines):
        raise AssetProviderError("TRELLIS OBJ response has no faces")


class _HttpProviderState:
    def __init__(self, *, provider_name: str, endpoint: str | None) -> None:
        self.provider_name = provider_name
        self.endpoint = endpoint.strip() if endpoint else None
        self._lock = RLock()
        if self.endpoint is None:
            self._snapshot = ProviderSnapshot(
                provider=provider_name,
                state=ProviderState.UNCONFIGURED,
                detail="endpoint is not configured",
            )
        else:
            self._snapshot = ProviderSnapshot(
                provider=provider_name,
                state=ProviderState.CONFIGURED,
                detail="endpoint is configured but has not produced a verified artifact",
            )

    @property
    def snapshot(self) -> ProviderSnapshot:
        with self._lock:
            return self._snapshot

    def require_endpoint(self) -> str:
        if self.endpoint is None:
            raise AssetProviderUnconfiguredError(f"{self.provider_name} endpoint is unconfigured")
        return self.endpoint

    def mark_healthy(self, detail: str) -> None:
        with self._lock:
            self._snapshot = ProviderSnapshot(
                provider=self.provider_name,
                state=ProviderState.HEALTHY,
                detail=detail,
            )

    def mark_degraded(self, detail: str) -> None:
        with self._lock:
            self._snapshot = ProviderSnapshot(
                provider=self.provider_name,
                state=ProviderState.DEGRADED,
                detail=detail,
            )


def _validate_timeout(timeout_seconds: float) -> float:
    if not 0.0 < timeout_seconds <= MAX_PROVIDER_TIMEOUT_SECONDS:
        raise ValueError(
            f"provider timeout must be within (0, {MAX_PROVIDER_TIMEOUT_SECONDS}] seconds"
        )
    return timeout_seconds


def _authorization_headers(api_key: str | None) -> dict[str, str]:
    if api_key is None or not api_key.strip():
        return {}
    return {"Authorization": f"Bearer {api_key.strip()}"}


class ReferenceImageHttpAdapter:
    """Provider-neutral image endpoint using a small documented JSON contract."""

    provider_name = "reference-image-http"

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self._state = _HttpProviderState(provider_name=self.provider_name, endpoint=endpoint)
        self._api_key = api_key
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._transport = transport or UrllibJsonTransport()

    @property
    def snapshot(self) -> ProviderSnapshot:
        return self._state.snapshot

    def generate(self, request: AssetRequest) -> GeneratedArtifact:
        endpoint = self._state.require_endpoint()
        try:
            response = self._transport.post_json(
                endpoint=endpoint,
                payload={
                    "prompt": request.prompt,
                    "request_id": request.request_id,
                    "response_format": "base64",
                },
                headers=_authorization_headers(self._api_key),
                timeout_seconds=self._timeout_seconds,
            )
            data, media_type = _decode_artifact(response)
            if not media_type.startswith("image/"):
                raise AssetProviderError("reference provider returned a non-image media type")
        except AssetProviderUnconfiguredError:
            raise
        except (AssetProviderError, OSError, TimeoutError) as exc:
            self._state.mark_degraded("last reference-image request failed")
            if isinstance(exc, AssetProviderError):
                raise
            raise AssetProviderError("reference-image provider failed") from exc
        self._state.mark_healthy("last request produced a validated reference image")
        return GeneratedArtifact(
            data=data,
            media_type=media_type,
            role=ArtifactRole.REFERENCE_INPUT,
        )


class TrellisHttpAdapter:
    """TRELLIS HTTP adapter whose output is always tagged as rendering-only."""

    provider_name = "trellis-http"

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self._state = _HttpProviderState(provider_name=self.provider_name, endpoint=endpoint)
        self._api_key = api_key
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._transport = transport or UrllibJsonTransport()

    @property
    def snapshot(self) -> ProviderSnapshot:
        return self._state.snapshot

    def generate(
        self,
        reference_image: GeneratedArtifact,
        *,
        mesh_format: VisualMeshFormat,
    ) -> GeneratedArtifact:
        if reference_image.role is not ArtifactRole.REFERENCE_INPUT:
            raise ValueError("TRELLIS requires a reference-image artifact")
        endpoint = self._state.require_endpoint()
        try:
            response = self._transport.post_json(
                endpoint=endpoint,
                payload={
                    "image_base64": base64.b64encode(reference_image.data).decode("ascii"),
                    "image_media_type": reference_image.media_type,
                    "output_format": mesh_format.value,
                },
                headers=_authorization_headers(self._api_key),
                timeout_seconds=self._timeout_seconds,
            )
            data, media_type = _decode_artifact(response)
            response_format = response.get("format", mesh_format.value)
            if response_format != mesh_format.value:
                raise AssetProviderError("TRELLIS returned an unexpected mesh format")
            _validate_visual_mesh(data, mesh_format)
        except AssetProviderUnconfiguredError:
            raise
        except (AssetProviderError, OSError, TimeoutError) as exc:
            self._state.mark_degraded("last TRELLIS request failed")
            if isinstance(exc, AssetProviderError):
                raise
            raise AssetProviderError("TRELLIS provider failed") from exc
        self._state.mark_healthy("last request produced a validated rendering mesh")
        return GeneratedArtifact(
            data=data,
            media_type=media_type,
            role=ArtifactRole.RENDERING_ONLY,
            mesh_format=mesh_format,
        )


def parse_provider_json(value: bytes) -> Mapping[str, object]:
    """Narrow JSON parser exposed for transport conformance tests."""
    decoded: Any = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("provider payload must be a JSON object")
    return cast(dict[str, object], decoded)
