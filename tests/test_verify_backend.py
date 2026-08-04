from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from muscle_memory.graph_memory import GraphMemoryIntegrityError  # noqa: E402
from ops.api import verify_backend  # noqa: E402


class _StartupFailureBackend:
    def __init__(self) -> None:
        self.shutdowns = 0

    async def startup(self) -> None:
        raise GraphMemoryIntegrityError(
            "graph identity conflict redis://user:password@graph.example.test:6379 token=secret",
            record_kind="evaluated_policy",
            record_id="delivery-v1-bc",
            expected_hash="a" * 64,
            actual_hash="b" * 64,
        )

    async def shutdown(self) -> None:
        self.shutdowns += 1


def test_verify_backend_emits_redacted_blocker_and_closes_partial_backend(
    monkeypatch,
    capsys,
) -> None:
    backend = _StartupFailureBackend()
    monkeypatch.setattr(verify_backend, "create_api_backend", lambda: backend)

    exit_code = asyncio.run(verify_backend._verify())

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["state"] == "blocked"
    assert payload["error"]["type"] == "GraphMemoryIntegrityError"
    assert payload["error"]["record_kind"] == "evaluated_policy"
    assert payload["error"]["record_id"] == "delivery-v1-bc"
    assert payload["error"]["expected_hash"] == "a" * 64
    assert payload["error"]["actual_hash"] == "b" * 64
    assert "password" not in payload["error"]["detail"]
    assert "secret" not in payload["error"]["detail"]
    assert backend.shutdowns == 1
