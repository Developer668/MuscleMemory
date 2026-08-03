"""Initialize the production composition and print redacted readiness evidence."""

from __future__ import annotations

import asyncio
import json

from muscle_memory.runtime import create_api_backend


async def _verify() -> int:
    backend = create_api_backend()
    await backend.startup()
    try:
        health = await backend.health()
        payload = health.model_dump(mode="json")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        await backend.shutdown()


def main() -> None:
    raise SystemExit(asyncio.run(_verify()))


if __name__ == "__main__":
    main()
