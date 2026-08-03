"""Smoke-test a running API and, optionally, require ready provider adapters."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

KNOWN_STATES = frozenset(
    {
        "unconfigured",
        "configured",
        "healthy",
        "degraded",
        "end_to_end_verified",
        "simulation",
        "cached",
    }
)
READY_PROVIDER_STATES = frozenset({"healthy", "end_to_end_verified"})


class SmokeError(RuntimeError):
    """The backend did not satisfy the requested smoke-test boundary."""


def validate_health(
    payload: object,
    *,
    required_providers: Sequence[str] = (),
) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise SmokeError("health payload is not an object")
    service_state = payload.get("state")
    if service_state not in KNOWN_STATES:
        raise SmokeError("health payload has an unknown service state")
    raw_providers = payload.get("providers")
    if not isinstance(raw_providers, list):
        raise SmokeError("health payload has no provider list")
    states: dict[str, str] = {}
    for provider in raw_providers:
        if not isinstance(provider, Mapping):
            raise SmokeError("health payload contains an invalid provider item")
        name = provider.get("provider")
        state = provider.get("state")
        if not isinstance(name, str) or not name:
            raise SmokeError("provider health item has no name")
        if not isinstance(state, str) or state not in KNOWN_STATES:
            raise SmokeError(f"provider {name} has an unknown state")
        states[name] = state
    missing = sorted(set(required_providers) - states.keys())
    if missing:
        raise SmokeError(f"health payload omitted providers: {', '.join(missing)}")
    unavailable = sorted(
        name for name in required_providers if states[name] not in READY_PROVIDER_STATES
    )
    if unavailable:
        details = ", ".join(f"{name}={states[name]}" for name in unavailable)
        raise SmokeError(f"required providers are not ready: {details}")
    return states


def fetch_health(url: str, *, timeout_seconds: float) -> object:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise SmokeError(f"health endpoint returned HTTP {response.status}")
        return json.load(response)


def wait_for_health(
    url: str,
    *,
    deadline_seconds: float,
    required_providers: Sequence[str],
) -> dict[str, str]:
    deadline = time.monotonic() + deadline_seconds
    last_error = "no request attempted"
    while time.monotonic() < deadline:
        try:
            payload = fetch_health(url, timeout_seconds=min(3.0, deadline_seconds))
            return validate_health(payload, required_providers=required_providers)
        except (OSError, ValueError, SmokeError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise SmokeError(f"backend smoke test timed out: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Muscle Memory API.")
    parser.add_argument(
        "--url",
        default=os.environ.get("MM_API_HEALTH_URL", "http://127.0.0.1:8000/api/v1/health"),
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--require-provider",
        action="append",
        default=[],
        help="Provider name that must report healthy or end_to_end_verified.",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        raise SmokeError("timeout must be positive")
    states = wait_for_health(
        args.url,
        deadline_seconds=args.timeout,
        required_providers=args.require_provider,
    )
    rendered_states = ", ".join(f"{name}={state}" for name, state in sorted(states.items()))
    print(f"backend API smoke passed: {args.url}; {rendered_states or 'no providers'}")


if __name__ == "__main__":
    main()
