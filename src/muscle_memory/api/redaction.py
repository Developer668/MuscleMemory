"""Credential redaction shared by public response and error contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping

_URL_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@",
    re.IGNORECASE,
)
_NAMED_SECRET = re.compile(
    r"(?i)\b(api[_ -]?key|token|secret|password|authorization)\b\s*[:=]\s*[^,;\s]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "connection_string",
        "credential",
        "password",
        "secret",
        "token",
    }
)


def redact_sensitive_text(value: str) -> str:
    redacted = _URL_CREDENTIALS.sub(r"\g<scheme>[redacted]@", value)
    redacted = _NAMED_SECRET.sub(
        lambda match: f"{match.group(1)}=[redacted]",
        redacted,
    )
    return _BEARER_SECRET.sub("Bearer [redacted]", redacted)


def redact_sensitive_object(value: object) -> object:
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if any(part in normalized for part in _SECRET_KEY_PARTS):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact_sensitive_object(item)
        return redacted
    if isinstance(value, tuple):
        return tuple(redact_sensitive_object(item) for item in value)
    if isinstance(value, list):
        return [redact_sensitive_object(item) for item in value]
    return value


def redact_sensitive_mapping(value: dict[str, object]) -> dict[str, object]:
    redacted = redact_sensitive_object(value)
    if not isinstance(redacted, dict):
        raise RuntimeError("mapping redaction returned an invalid shape")
    return redacted


__all__ = [
    "redact_sensitive_mapping",
    "redact_sensitive_object",
    "redact_sensitive_text",
]

