"""Packaged validator for the reviewed external RocketRide pipe bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from muscle_memory.paths import REPOSITORY_ROOT

COORDINATOR_URL_ENV = "ROCKETRIDE_MM_COORDINATOR_URL"
COORDINATOR_TOKEN_ENV = "ROCKETRIDE_MM_COORDINATOR_TOKEN"
CALLBACK_PATH = "/webhook/muscle-memory-fixed-step"
DEFAULT_BUNDLE_ROOT = REPOSITORY_ROOT / "integrations" / "rocketride"


class ReviewedPipelineError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedPipelineArtifact:
    pipeline_path: Path
    pipeline_sha256: str
    coordinator_url: str
    coordinator_token: str = field(repr=False)

    @property
    def sdk_environment(self) -> dict[str, str]:
        return {
            COORDINATOR_URL_ENV: self.coordinator_url,
            COORDINATOR_TOKEN_ENV: self.coordinator_token,
        }

    @property
    def callback_endpoint(self) -> str:
        return f"{self.coordinator_url}{CALLBACK_PATH}"

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str],
        *,
        bundle_root: Path = DEFAULT_BUNDLE_ROOT,
    ) -> ReviewedPipelineArtifact:
        url = values.get(COORDINATOR_URL_ENV, "").strip()
        token = values.get(COORDINATOR_TOKEN_ENV, "").strip()
        if not url or not token:
            raise ReviewedPipelineError("RocketRide callback configuration is incomplete")
        parsed = urlparse(url)
        loopback_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
        if parsed.scheme != "https" and not loopback_http:
            raise ReviewedPipelineError("RocketRide callback must use HTTPS or loopback HTTP")
        if not parsed.netloc or parsed.path not in {"", "/"}:
            raise ReviewedPipelineError("RocketRide callback must be an origin")
        if len(token) < 32:
            raise ReviewedPipelineError("RocketRide callback token is too short")

        manifest_path = bundle_root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ReviewedPipelineError("RocketRide bundle manifest is invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("algorithm") != "sha256":
            raise ReviewedPipelineError("RocketRide bundle manifest algorithm is invalid")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ReviewedPipelineError("RocketRide bundle manifest has no artifacts")
        for relative, expected in artifacts.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ReviewedPipelineError("RocketRide artifact record is invalid")
            path = bundle_root / relative
            try:
                path.resolve().relative_to(bundle_root.resolve())
            except ValueError as exc:
                raise ReviewedPipelineError("RocketRide artifact escapes its bundle") from exc
            if not path.is_file() or _sha256(path) != expected:
                raise ReviewedPipelineError(
                    f"RocketRide reviewed artifact failed verification: {relative}"
                )
        pipeline_digest = artifacts.get("fixed-step.pipe")
        if not isinstance(pipeline_digest, str):
            raise ReviewedPipelineError("RocketRide manifest has no reviewed pipe")
        return cls(
            pipeline_path=bundle_root / "fixed-step.pipe",
            pipeline_sha256=pipeline_digest,
            coordinator_url=url.rstrip("/"),
            coordinator_token=token,
        )


__all__ = ["ReviewedPipelineArtifact", "ReviewedPipelineError"]
