"""CLI for recording the native expert demonstration dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muscle_memory.paths import EXPERT_DATASET_V1, EXPERT_DATASET_V1_METADATA
from muscle_memory.training.dataset import (
    DEFAULT_TRAINING_EPISODES,
    TRAINING_SEED_SEARCH_START,
    build_expert_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERT_DATASET_V1)
    parser.add_argument("--metadata", type=Path, default=EXPERT_DATASET_V1_METADATA)
    parser.add_argument("--episodes", type=int, default=DEFAULT_TRAINING_EPISODES)
    parser.add_argument("--seed-start", type=int, default=TRAINING_SEED_SEARCH_START)
    return parser


def main() -> int:
    args = _parser().parse_args()
    metadata = build_expert_dataset(
        output_path=args.output,
        metadata_path=args.metadata,
        episode_count=args.episodes,
        seed_start=args.seed_start,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
