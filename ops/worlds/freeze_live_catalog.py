"""Freeze validation-gated training worlds for the teacher-free live runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from muscle_memory.live.catalog import (
    LIVE_WORLD_CATALOG_PATH,
    REQUIRED_VALIDATION_CHECKS,
)
from muscle_memory.worlds.generation import generate_training_world


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def freeze_live_catalog(path: Path, seeds: range) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable catalog: {path}")
    worlds: list[dict[str, object]] = []
    for seed in seeds:
        validated = generate_training_world(seed)
        world = validated.world.model_dump(mode="json")
        baseline_path = [point.model_dump(mode="json") for point in validated.baseline_path]
        worlds.append(
            {
                "world": world,
                "world_sha256": _sha256(world),
                "baseline_path_sha256": _sha256(baseline_path),
                "validation_checks": sorted(REQUIRED_VALIDATION_CHECKS),
            }
        )
    payload = {
        "schema_version": 1,
        "catalog_id": "live-training-v1",
        "worlds": worlds,
    }
    artifact_hash = _sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artifact_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=LIVE_WORLD_CATALOG_PATH)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.seed_start < 0 or not 1 <= args.count <= 200:
        raise ValueError("seed start and catalog count are outside safe bounds")
    artifact_hash = freeze_live_catalog(
        args.output,
        range(args.seed_start, args.seed_start + args.count),
    )
    print(f"frozen {args.count} worlds at {args.output} sha256={artifact_hash}")


if __name__ == "__main__":
    main()
