"""Train the immutable delivery-v2 sensor-fusion candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from muscle_memory.paths import (
    EXPERT_DATASET_V1,
    POLICY_V2_CHECKPOINT,
    POLICY_V2_TRAINING_EVIDENCE,
)
from muscle_memory.policy.network import SENSOR_FUSION_INFERENCE_STRATEGY
from muscle_memory.training.behavior_clone import BehaviorCloneConfig, train_behavior_clone


def _parser() -> argparse.ArgumentParser:
    defaults = BehaviorCloneConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=EXPERT_DATASET_V1)
    parser.add_argument("--output", type=Path, default=POLICY_V2_CHECKPOINT)
    parser.add_argument("--evidence", type=Path, default=POLICY_V2_TRAINING_EVIDENCE)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--avoidance-distance-m", type=float, default=2.6)
    parser.add_argument("--avoidance-gain", type=float, default=1.35)
    parser.add_argument("--avoidance-exponent", type=float, default=2.0)
    parser.add_argument("--avoidance-activation", type=float, default=0.2)
    parser.add_argument("--avoidance-docking-suppression-m", type=float, default=1.0)
    parser.add_argument("--learned-turn-blend", type=float, default=0.0)
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
            condition_on_previous_action=False,
            mirror_training_fraction=defaults.mirror_training_fraction,
            policy_id="delivery-v2-sensor-fusion",
            inference_strategy=SENSOR_FUSION_INFERENCE_STRATEGY,
            avoidance_distance_m=args.avoidance_distance_m,
            avoidance_gain=args.avoidance_gain,
            avoidance_exponent=args.avoidance_exponent,
            avoidance_activation=args.avoidance_activation,
            avoidance_docking_suppression_m=args.avoidance_docking_suppression_m,
            learned_turn_blend=args.learned_turn_blend,
        ),
    )
    print(json.dumps(asdict(result), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
