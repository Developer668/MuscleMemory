"""Collect on-policy corrections from disjoint validated training worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muscle_memory.paths import (
    DAGGER_DATASET_V1,
    DAGGER_DATASET_V1_METADATA,
    EXPERT_DATASET_V1,
    POLICY_V1_CHECKPOINT,
)
from muscle_memory.policy.network import BehaviorClonedPolicy
from muscle_memory.training.dataset import (
    DAGGER_SEED_SEARCH_START,
    DEFAULT_DAGGER_EPISODES,
    build_dagger_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=POLICY_V1_CHECKPOINT)
    parser.add_argument("--base-dataset", type=Path, default=EXPERT_DATASET_V1)
    parser.add_argument("--output", type=Path, default=DAGGER_DATASET_V1)
    parser.add_argument("--metadata", type=Path, default=DAGGER_DATASET_V1_METADATA)
    parser.add_argument("--episodes", type=int, default=DEFAULT_DAGGER_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DAGGER_SEED_SEARCH_START)
    return parser


def main() -> int:
    args = _parser().parse_args()
    metadata = build_dagger_dataset(
        BehaviorClonedPolicy.load(args.checkpoint),
        base_dataset_path=args.base_dataset,
        output_path=args.output,
        metadata_path=args.metadata,
        episode_count=args.episodes,
        seed_start=args.seed_start,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
