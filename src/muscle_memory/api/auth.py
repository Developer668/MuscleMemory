"""Fail-closed bearer authentication using only configured token digests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from muscle_memory.api.contracts import AuthenticatedPrincipal, Authenticator

APPROVAL_WRITE_SCOPE = "approvals:write"
WORKFLOW_WRITE_SCOPE = "workflows:write"
CORRECTION_WRITE_SCOPE = "corrections:write"
ALL_MUTATION_SCOPES = frozenset(
    {APPROVAL_WRITE_SCOPE, WORKFLOW_WRITE_SCOPE, CORRECTION_WRITE_SCOPE}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME = "MM_API_AUTH_CREDENTIALS_JSON"


@dataclass(frozen=True, slots=True)
class HashedBearerCredential:
    subject: str
    token_sha256: str
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("credential subject must not be blank")
        if _SHA256.fullmatch(self.token_sha256) is None:
            raise ValueError("credential token_sha256 must be a lowercase SHA-256 digest")
        if not self.scopes or any(not scope.strip() for scope in self.scopes):
            raise ValueError("credentials require at least one non-empty scope")

    @classmethod
    def from_plaintext(
        cls,
        *,
        subject: str,
        token: str,
        scopes: frozenset[str] = ALL_MUTATION_SCOPES,
    ) -> HashedBearerCredential:
        """Hash a token before storage; useful for secure provisioning and tests."""

        if not token:
            raise ValueError("bearer token must not be empty")
        return cls(
            subject=subject,
            token_sha256=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            scopes=scopes,
        )


class Sha256BearerAuthenticator:
    """Authenticate by constant-time comparison against non-reversible digests."""

    def __init__(self, credentials: tuple[HashedBearerCredential, ...]) -> None:
        subjects = tuple(credential.subject for credential in credentials)
        if len(subjects) != len(set(subjects)):
            raise ValueError("credential subjects must be unique")
        token_digests = tuple(credential.token_sha256 for credential in credentials)
        if len(token_digests) != len(set(token_digests)):
            raise ValueError("credential token digests must be unique")
        self._credentials = credentials

    @property
    def configured(self) -> bool:
        return bool(self._credentials)

    def authenticate(self, token: str) -> AuthenticatedPrincipal | None:
        if not token or not self._credentials:
            return None
        supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
        matched: HashedBearerCredential | None = None
        for credential in self._credentials:
            if hmac.compare_digest(supplied, credential.token_sha256):
                matched = credential
        if matched is None:
            return None
        return AuthenticatedPrincipal(
            subject=matched.subject,
            authentication_method="bearer_sha256",
            scopes=matched.scopes,
        )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> Sha256BearerAuthenticator:
        values = os.environ if environ is None else environ
        raw = values.get(_ENV_NAME, "").strip()
        if not raw:
            return cls(())
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{_ENV_NAME} must contain valid JSON") from exc
        if not isinstance(decoded, list):
            raise ValueError(f"{_ENV_NAME} must be a JSON list")
        credentials: list[HashedBearerCredential] = []
        for item in decoded:
            if not isinstance(item, dict) or set(item) != {
                "subject",
                "token_sha256",
                "scopes",
            }:
                raise ValueError(f"{_ENV_NAME} entries require subject, token_sha256, and scopes")
            subject = item["subject"]
            digest = item["token_sha256"]
            scopes = item["scopes"]
            if (
                not isinstance(subject, str)
                or not isinstance(digest, str)
                or not isinstance(scopes, list)
                or not all(isinstance(scope, str) for scope in scopes)
            ):
                raise ValueError(f"{_ENV_NAME} entry values have invalid types")
            credentials.append(
                HashedBearerCredential(
                    subject=subject,
                    token_sha256=digest,
                    scopes=frozenset(scopes),
                )
            )
        return cls(tuple(credentials))


def authenticator_from_env(
    environ: Mapping[str, str] | None = None,
) -> Authenticator:
    return Sha256BearerAuthenticator.from_env(environ)


__all__ = [
    "ALL_MUTATION_SCOPES",
    "APPROVAL_WRITE_SCOPE",
    "CORRECTION_WRITE_SCOPE",
    "WORKFLOW_WRITE_SCOPE",
    "HashedBearerCredential",
    "Sha256BearerAuthenticator",
    "authenticator_from_env",
]
