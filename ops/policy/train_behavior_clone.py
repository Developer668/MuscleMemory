"""CLI for training the immutable delivery-v1 behavior-cloned checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from muscle_memory.paths import (
    EXPERT_DATASET_V1,
    POLICY_V1_CHECKPOINT,
    POLICY_V1_TRAINING_EVIDENCE,
)
from muscle_memory.training.behavior_clone import BehaviorCloneConfig, train_behavior_clone


def _parser() -> argparse.ArgumentParser:
    defaults = BehaviorCloneConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=EXPERT_DATASET_V1)
    parser.add_argument("--output", type=Path, default=POLICY_V1_CHECKPOINT)
    parser.add_argument("--evidence", type=Path, default=POLICY_V1_TRAINING_EVIDENCE)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--policy-id", default=defaults.policy_id)
    return parser


def main() -> int:
    args = _parser().parse_args()
    defaults = BehaviorCloneConfig()
    result = train_behavior_clone(
        dataset_path=args.dataset,
        output_path=args.output,
        evidence_path=args.evidence,
        config=BehaviorCloneConfig(
            hidden_1=defaults.hidden_1,
            hidden_2=defaults.hidden_2,
            epochs=args.epochs,
            batch_size=defaults.batch_size,
            learning_rate=defaults.learning_rate,
            validation_episode_fraction=defaults.validation_episode_fraction,
            seed=args.seed,
            condition_on_previous_action=defaults.condition_on_previous_action,
            mirror_training_fraction=defaults.mirror_training_fraction,
            policy_id=args.policy_id,
        ),
    )
    print(json.dumps(asdict(result), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
