"""Production HTTP and live-stream API surfaces."""

from muscle_memory.api.app import ApiRuntime, create_app
from muscle_memory.api.auth import (
    ALL_MUTATION_SCOPES,
    HashedBearerCredential,
    Sha256BearerAuthenticator,
    authenticator_from_env,
)
from muscle_memory.api.contracts import (
    ApiBackend,
    ApiBackendError,
    AuthenticatedPrincipal,
    Authenticator,
    LiveEventPublisher,
)
from muscle_memory.api.streaming import LiveTelemetryHub

__all__ = [
    "ALL_MUTATION_SCOPES",
    "ApiBackend",
    "ApiBackendError",
    "ApiRuntime",
    "AuthenticatedPrincipal",
    "Authenticator",
    "HashedBearerCredential",
    "LiveEventPublisher",
    "LiveTelemetryHub",
    "Sha256BearerAuthenticator",
    "authenticator_from_env",
    "create_app",
]
