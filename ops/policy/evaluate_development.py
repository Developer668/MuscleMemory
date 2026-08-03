"""Evaluate V0 and V1 on disposable, non-held-out development worlds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult, run_policy_episode
from muscle_memory.paths import POLICY_V1_CHECKPOINT
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.policy.network import BehaviorClonedPolicy
from muscle_memory.training.expert import (
    direct_route_requires_avoidance,
    plan_expert_path,
)
from muscle_memory.worlds.generation import generate_training_world
from muscle_memory.worlds.generation.models import ValidatedTrainingWorld
from muscle_memory.worlds.rules import load_world_rules

DEVELOPMENT_SEED_START = 500_000_000
DEFAULT_DEVELOPMENT_WORLDS = 12
MAXIMUM_EXPERT_PATH_LENGTH_M = 8.2
DEFAULT_OUTPUT = Path("artifacts/policy/development-evaluation.json")


def _development_worlds(
    count: int,
    seed_start: int,
) -> tuple[ValidatedTrainingWorld, ...]:
    rules = load_world_rules()
    selected: list[ValidatedTrainingWorld] = []
    seed = seed_start
    while len(selected) < count:
        validated = generate_training_world(seed, rules)
        expert = plan_expert_path(validated.world, rules)
        if (
            expert is not None
            and expert.length_m <= MAXIMUM_EXPERT_PATH_LENGTH_M
            and direct_route_requires_avoidance(validated.world, rules)
        ):
            selected.append(validated)
        seed += 1
    return tuple(selected)


def _evaluate(
    worlds: tuple[ValidatedTrainingWorld, ...],
    policy: DirectGoalPolicy | BehaviorClonedPolicy,
) -> tuple[PolicyEpisodeResult, ...]:
    results: list[PolicyEpisodeResult] = []
    for index, world in enumerate(worlds):
        result = run_policy_episode(
            world,
            policy,
            episode_id=f"development-{policy.policy_id}-{index:02d}",
        )
        results.append(result)
        print(
            f"{policy.policy_id} {index + 1:02d}/{len(worlds)} "
            f"seed={world.world.seed} success={result.success} "
            f"collisions={result.body_collisions} "
            f"clearance={result.minimum_obstacle_clearance_m:.3f}",
            flush=True,
        )
    return tuple(results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=DEFAULT_DEVELOPMENT_WORLDS)
    parser.add_argument("--seed-start", type=int, default=DEVELOPMENT_SEED_START)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=POLICY_V1_CHECKPOINT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    worlds = _development_worlds(args.worlds, args.seed_start)
    baseline = _evaluate(worlds, DirectGoalPolicy())
    candidate = _evaluate(worlds, BehaviorClonedPolicy.load(args.checkpoint))
    decision = evaluate_promotion(baseline, candidate)
    payload = {
        "schema_version": 1,
        "purpose": "development_only_not_held_out",
        "world_ids": [world.world.world_id for world in worlds],
        "baseline_results": [asdict(result) for result in baseline],
        "candidate_results": [asdict(result) for result in candidate],
        "promotion_preview": asdict(decision),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(decision), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
