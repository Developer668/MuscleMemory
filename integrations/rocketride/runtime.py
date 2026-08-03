"""Canonical handoff from backend composition to the reviewed RocketRide artifact."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from integrations.rocketride.protocol import ContractError

COORDINATOR_URL_ENV = "ROCKETRIDE_MM_COORDINATOR_URL"
COORDINATOR_TOKEN_ENV = "ROCKETRIDE_MM_COORDINATOR_TOKEN"
BUNDLE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class ReviewedPipelineArtifact:
    pipeline_path: Path
    pipeline_sha256: str
    coordinator_url: str
    coordinator_token: str = field(repr=False)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        bundle_root: Path = BUNDLE_ROOT,
    ) -> ReviewedPipelineArtifact:
        values = os.environ if environ is None else environ
        url = values.get(COORDINATOR_URL_ENV, "").strip()
        token = values.get(COORDINATOR_TOKEN_ENV, "").strip()
        if not url or not token:
            raise ContractError(f"{COORDINATOR_URL_ENV} and {COORDINATOR_TOKEN_ENV} are required")
        return cls.from_values(url, token, bundle_root=bundle_root)

    @classmethod
    def from_values(
        cls,
        coordinator_url: str,
        coordinator_token: str,
        *,
        bundle_root: Path = BUNDLE_ROOT,
    ) -> ReviewedPipelineArtifact:
        parsed = urlparse(coordinator_url)
        local_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
        if parsed.scheme != "https" and not local_http:
            raise ContractError("coordinator URL must use HTTPS or loopback HTTP")
        if not parsed.netloc or parsed.path not in {"", "/"}:
            raise ContractError("coordinator URL must be an origin without a path")
        if len(coordinator_token) < 32:
            raise ContractError("coordinator callback token must contain at least 32 characters")
        from integrations.rocketride.validator import validate_bundle

        bundle = validate_bundle(bundle_root)
        pipeline = bundle["pipeline"]
        if not isinstance(pipeline, dict):
            raise ContractError("validated bundle did not return pipeline evidence")
        digest = pipeline.get("pipeline_sha256")
        if not isinstance(digest, str):
            raise ContractError("validated bundle did not return a pipeline checksum")
        return cls(
            pipeline_path=bundle_root / "fixed-step.pipe",
            pipeline_sha256=digest,
            coordinator_url=coordinator_url.rstrip("/"),
            coordinator_token=coordinator_token,
        )

    @property
    def sdk_environment(self) -> dict[str, str]:
        return {
            COORDINATOR_URL_ENV: self.coordinator_url,
            COORDINATOR_TOKEN_ENV: self.coordinator_token,
        }

    @property
    def public_evidence(self) -> dict[str, str]:
        return {
            "pipeline_path": str(self.pipeline_path),
            "pipeline_sha256": self.pipeline_sha256,
            "coordinator_origin": self.coordinator_url,
        }
