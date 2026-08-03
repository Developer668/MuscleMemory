"""Secret-safe environment configuration for the production composition root."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from muscle_memory.paths import REPOSITORY_ROOT


def _path(
    values: Mapping[str, str],
    name: str,
    default: str,
) -> Path:
    candidate = Path(values.get(name, default)).expanduser()
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _seconds(
    values: Mapping[str, str],
    name: str,
    default: float,
    *,
    maximum: float,
) -> float:
    try:
        value = float(values.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not 0.0 < value <= maximum:
        raise ValueError(f"{name} must be within (0, {maximum:g}]")
    return value


@dataclass(frozen=True, slots=True)
class BackendConfig:
    coordinator_path: Path
    asset_cache_path: Path
    asset_approval_path: Path
    reference_endpoint: str | None
    reference_api_key: str | None = field(default=None, repr=False)
    reference_timeout_seconds: float = 10.0
    trellis_endpoint: str | None = None
    trellis_api_key: str | None = field(default=None, repr=False)
    trellis_timeout_seconds: float = 30.0
    environ: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BackendConfig:
        values = dict(os.environ if environ is None else environ)
        return cls(
            coordinator_path=_path(
                values,
                "MUSCLE_MEMORY_COORDINATOR_DB_PATH",
                ".cache/muscle-memory/coordinator.sqlite3",
            ),
            asset_cache_path=_path(
                values,
                "MM_ASSET_CACHE_DIR",
                "artifacts/assets/cache",
            ),
            asset_approval_path=_path(
                values,
                "MM_ASSET_APPROVAL_LEDGER_DIR",
                "artifacts/assets/approvals",
            ),
            reference_endpoint=(values.get("MM_ASSET_REFERENCE_ENDPOINT", "").strip() or None),
            reference_api_key=(values.get("MM_ASSET_REFERENCE_API_KEY", "").strip() or None),
            reference_timeout_seconds=_seconds(
                values,
                "MM_ASSET_REFERENCE_TIMEOUT_SECONDS",
                10.0,
                maximum=30.0,
            ),
            trellis_endpoint=(values.get("MM_ASSET_TRELLIS_ENDPOINT", "").strip() or None),
            trellis_api_key=(values.get("MM_ASSET_TRELLIS_API_KEY", "").strip() or None),
            trellis_timeout_seconds=_seconds(
                values,
                "MM_ASSET_TRELLIS_TIMEOUT_SECONDS",
                30.0,
                maximum=30.0,
            ),
            environ=values,
        )


__all__ = ["BackendConfig"]
