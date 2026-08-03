"""Content-addressed live worlds admitted by the offline validation gate."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from muscle_memory.paths import REPOSITORY_ROOT
from muscle_memory.worlds.models import TrainingWorld

LIVE_WORLD_CATALOG_PATH = REPOSITORY_ROOT / "config" / "worlds" / "live-training-v1.json"
LIVE_WORLD_CATALOG_SHA256 = "c44bc36365ec9107a060556dc4e82689a02911e4f2ef26186cc1ffffc1ffbe75"
MAXIMUM_CATALOG_BYTES = 2 * 1024 * 1024
REQUIRED_VALIDATION_CHECKS = frozenset(
    {
        "approved_colliders",
        "baseline_path_exists",
        "minimum_passage_clearance",
        "no_overlapping_objects",
        "physical_parameters_within_bounds",
        "start_destination_connected",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class LiveWorldCatalogError(ValueError):
    """A live-world artifact is absent, changed, or lacks validation evidence."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate catalog key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ValidatedRuntimeWorld:
    """A training world bound to an immutable offline validation receipt."""

    world: TrainingWorld
    world_sha256: str
    baseline_path_sha256: str
    validation_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.world_sha256) is None:
            raise LiveWorldCatalogError("live world hash is malformed")
        if _SHA256_PATTERN.fullmatch(self.baseline_path_sha256) is None:
            raise LiveWorldCatalogError("baseline validation hash is malformed")
        if _sha256_json(self.world.model_dump(mode="json")) != self.world_sha256:
            raise LiveWorldCatalogError("live world bytes do not match validation evidence")
        if frozenset(self.validation_checks) != REQUIRED_VALIDATION_CHECKS:
            raise LiveWorldCatalogError("live world is missing a mandatory validation check")
        expected_world_id = (
            f"train-v{self.world.generation_version}-{self.world.seed:016x}"
        )
        if self.world.world_id != expected_world_id:
            raise LiveWorldCatalogError("live world identity does not match its seed")


@dataclass(frozen=True, slots=True)
class LiveWorldCatalog:
    catalog_id: str
    artifact_sha256: str
    validated_at: datetime
    worlds: tuple[ValidatedRuntimeWorld, ...]

    def __post_init__(self) -> None:
        if not self.catalog_id or _SHA256_PATTERN.fullmatch(self.artifact_sha256) is None:
            raise LiveWorldCatalogError("live world catalog identity is malformed")
        if self.validated_at.tzinfo is None or self.validated_at.utcoffset() is None:
            raise LiveWorldCatalogError("live world catalog validation time must be timezone-aware")
        if not self.worlds:
            raise LiveWorldCatalogError("live world catalog must not be empty")
        seeds = tuple(item.world.seed for item in self.worlds)
        world_ids = tuple(item.world.world_id for item in self.worlds)
        if len(seeds) != len(set(seeds)) or len(world_ids) != len(set(world_ids)):
            raise LiveWorldCatalogError("live world catalog identities must be unique")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(item.world.seed for item in self.worlds))

    def world_for_seed(self, seed: int) -> ValidatedRuntimeWorld:
        for admitted in self.worlds:
            if admitted.world.seed == seed:
                return admitted
        raise KeyError(seed)

    @classmethod
    def load(
        cls,
        path: Path = LIVE_WORLD_CATALOG_PATH,
        *,
        expected_sha256: str = LIVE_WORLD_CATALOG_SHA256,
    ) -> LiveWorldCatalog:
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise LiveWorldCatalogError("expected live world catalog hash is malformed")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise LiveWorldCatalogError("live world catalog is unavailable") from exc
        if not 0 < size <= MAXIMUM_CATALOG_BYTES:
            raise LiveWorldCatalogError("live world catalog size is invalid")
        try:
            decoded = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise LiveWorldCatalogError("live world catalog is not valid JSON") from exc
        if not isinstance(decoded, dict) or set(decoded) != {
            "schema_version",
            "catalog_id",
            "validated_at",
            "worlds",
        }:
            raise LiveWorldCatalogError("live world catalog contract is invalid")
        if (
            decoded["schema_version"] != 1
            or not isinstance(decoded["catalog_id"], str)
            or not isinstance(decoded["validated_at"], str)
        ):
            raise LiveWorldCatalogError("live world catalog version is unsupported")
        try:
            validated_at = datetime.fromisoformat(decoded["validated_at"])
        except ValueError as exc:
            raise LiveWorldCatalogError(
                "live world catalog validation time is malformed"
            ) from exc
        if validated_at.tzinfo is None or validated_at.utcoffset() is None:
            raise LiveWorldCatalogError(
                "live world catalog validation time must be timezone-aware"
            )
        artifact_sha256 = _sha256_json(decoded)
        if artifact_sha256 != expected_sha256:
            raise LiveWorldCatalogError("live world catalog does not match its pinned hash")
        raw_worlds = decoded["worlds"]
        if not isinstance(raw_worlds, list):
            raise LiveWorldCatalogError("live world catalog worlds must be a list")
        admitted: list[ValidatedRuntimeWorld] = []
        for raw in raw_worlds:
            if not isinstance(raw, dict) or set(raw) != {
                "world",
                "world_sha256",
                "baseline_path_sha256",
                "validation_checks",
            }:
                raise LiveWorldCatalogError("live world validation receipt is malformed")
            checks = raw["validation_checks"]
            if not isinstance(checks, list) or not all(
                isinstance(item, str) for item in checks
            ):
                raise LiveWorldCatalogError("live world validation checks are malformed")
            try:
                world = TrainingWorld.model_validate(raw["world"])
                admitted.append(
                    ValidatedRuntimeWorld(
                        world=world,
                        world_sha256=cast(str, raw["world_sha256"]),
                        baseline_path_sha256=cast(str, raw["baseline_path_sha256"]),
                        validation_checks=tuple(checks),
                    )
                )
            except (ValidationError, TypeError, ValueError) as exc:
                raise LiveWorldCatalogError("live world validation receipt is invalid") from exc
        return cls(
            catalog_id=decoded["catalog_id"],
            artifact_sha256=artifact_sha256,
            validated_at=validated_at,
            worlds=tuple(admitted),
        )


__all__ = [
    "LIVE_WORLD_CATALOG_PATH",
    "LIVE_WORLD_CATALOG_SHA256",
    "LiveWorldCatalog",
    "LiveWorldCatalogError",
    "ValidatedRuntimeWorld",
]
