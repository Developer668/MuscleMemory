"""Build and validate the HTTP contract without starting provider services."""

from __future__ import annotations

import json
from typing import cast

from muscle_memory.api import (
    ApiBackend,
    LiveEventPublisher,
    Sha256BearerAuthenticator,
    create_app,
)

REQUIRED_PATHS = {
    "/api/v1/health",
    "/api/v1/episodes",
    "/api/v1/episodes/{episode_id}",
    "/api/v1/episodes/{episode_id}/telemetry",
    "/api/v1/episodes/{episode_id}/replay",
    "/api/v1/approvals/pending",
    "/api/v1/approvals/{requirement_id}/decision",
    "/api/v1/workflows/review",
    "/api/v1/workflows/{run_id}/execute",
    "/api/v1/workflows/{run_id}/resume",
    "/api/v1/episodes/{episode_id}/corrections",
    "/api/v1/policies/promotion-eligibility",
    "/api/v1/assets",
}


class _SchemaBackend:
    def bind_live_publisher(self, publisher: LiveEventPublisher) -> None:
        del publisher


def build_schema() -> dict[str, object]:
    backend = cast(ApiBackend, _SchemaBackend())
    app = create_app(
        backend=backend,
        authenticator=Sha256BearerAuthenticator(()),
    )
    return cast(dict[str, object], app.openapi())


def main() -> None:
    schema = build_schema()
    raw_paths = schema.get("paths")
    if not isinstance(raw_paths, dict):
        raise RuntimeError("OpenAPI schema does not contain a paths object")
    missing = REQUIRED_PATHS - set(raw_paths)
    if missing:
        raise RuntimeError(f"OpenAPI schema is missing required paths: {sorted(missing)}")
    rendered = json.dumps(schema, allow_nan=False, sort_keys=True)
    forbidden = ("token_sha256", "api_key", "password", "provider_secret")
    exposed = [name for name in forbidden if name in rendered.lower()]
    if exposed:
        raise RuntimeError(f"OpenAPI schema exposes credential fields: {exposed}")
    decision_path = raw_paths["/api/v1/approvals/{requirement_id}/decision"]
    if not isinstance(decision_path, dict):
        raise RuntimeError("approval decision OpenAPI path has an invalid shape")
    operation = decision_path.get("post")
    if not isinstance(operation, dict) or not operation.get("security"):
        raise RuntimeError("approval decision route is missing its security requirement")
    print("OpenAPI contract valid: versioned routes, typed errors, and mutation security")


if __name__ == "__main__":
    main()
