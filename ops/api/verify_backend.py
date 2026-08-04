"""Initialize the production composition and print redacted readiness evidence."""

from __future__ import annotations

import asyncio
import json

from muscle_memory.api.redaction import redact_sensitive_text
from muscle_memory.graph_memory import GraphMemoryIntegrityError
from muscle_memory.runtime import create_api_backend


async def _verify() -> int:
    backend = None
    try:
        backend = create_api_backend()
        await backend.startup()
        health = await backend.health()
        payload = health.model_dump(mode="json")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        detail = redact_sensitive_text(str(exc)).strip()
        error: dict[str, str] = {
            "type": type(exc).__name__,
            "detail": detail[:512] or "backend readiness preflight failed",
        }
        if isinstance(exc, GraphMemoryIntegrityError):
            for key in ("record_kind", "record_id", "expected_hash", "actual_hash"):
                value = getattr(exc, key, None)
                if isinstance(value, str) and value:
                    error[key] = value
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error": error,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    finally:
        if backend is not None:
            await backend.shutdown()


def main() -> None:
    raise SystemExit(asyncio.run(_verify()))


if __name__ == "__main__":
    main()
