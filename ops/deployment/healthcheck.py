"""Container-level API liveness probe using only the Python standard library."""

from __future__ import annotations

import json
import os
import urllib.request

_KNOWN_STATES = {
    "unconfigured",
    "configured",
    "healthy",
    "degraded",
    "end_to_end_verified",
    "simulation",
    "cached",
}


def main() -> None:
    port = os.environ.get("MM_API_PORT", "8000")
    url = f"http://127.0.0.1:{port}/api/v1/health"
    with urllib.request.urlopen(url, timeout=3.0) as response:
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("state") not in _KNOWN_STATES:
        raise RuntimeError("health endpoint returned an invalid service state")
    providers = payload.get("providers")
    if not isinstance(providers, list):
        raise RuntimeError("health endpoint omitted provider state")


if __name__ == "__main__":
    main()
