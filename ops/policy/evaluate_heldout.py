"""Run the one-shot paired held-out evaluation with no teacher imports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from muscle_memory.evaluation.heldout_trust import (
    DELIVERY_V2_HELDOUT_TRUST_ROOT,
    consume_checked_in_heldout_access,
)
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult, run_policy_episode
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.policy.network import BehaviorClonedPolicy
from muscle_memory.simulation.world_scene import ValidatedWorldEnvelope


def _assert_pre_gate_import_isolation() -> None:
    forbidden = tuple(
        name
        for name in sys.modules
        if name == "muscle_memory.evaluation.heldout"
        or name.startswith("muscle_memory.training")
        or name.startswith("muscle_memory.worlds.generation")
    )
    if forbidden:
        raise RuntimeError(f"pre-gate evaluation loaded forbidden modules: {forbidden}")


def _evaluate(
    worlds: tuple[ValidatedWorldEnvelope, ...],
    policy: DirectGoalPolicy | BehaviorClonedPolicy,
    *,
    evaluation_scope: str,
    role: str,
) -> tuple[PolicyEpisodeResult, ...]:
    results: list[PolicyEpisodeResult] = []
    for index, world in enumerate(worlds):
        result = run_policy_episode(
            world,
            policy,
            episode_id=f"heldout-{evaluation_scope}-{role}-{index:02d}",
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
    return argparse.ArgumentParser(
        description=(
            "Run the reviewed delivery-v2 one-shot held-out evaluation. "
            "All inputs and outputs are fixed by the committed trust root."
        )
    )


def main() -> int:
    _parser().parse_args()
    _assert_pre_gate_import_isolation()
    trusted = consume_checked_in_heldout_access()
    from muscle_memory.evaluation.heldout import load_heldout_worlds

    worlds = load_heldout_worlds()
    candidate_policy = BehaviorClonedPolicy.load(
        DELIVERY_V2_HELDOUT_TRUST_ROOT.candidate_checkpoint_path
    )
    if candidate_policy.policy_hash != trusted.candidate_checkpoint_sha256:
        raise RuntimeError("candidate identity changed after held-out access was consumed")
    evaluation_scope = candidate_policy.policy_hash[:16]
    baseline = _evaluate(
        worlds,
        DirectGoalPolicy(),
        evaluation_scope=evaluation_scope,
        role="baseline",
    )
    candidate = _evaluate(
        worlds,
        candidate_policy,
        evaluation_scope=evaluation_scope,
        role="candidate",
    )
    decision = evaluate_promotion(baseline, candidate)
    payload = {
        "schema_version": 1,
        "heldout_bundle_sha256": trusted.heldout_bundle_sha256,
        "candidate_checkpoint_sha256": candidate_policy.policy_hash,
        "baseline_results": [asdict(result) for result in baseline],
        "candidate_results": [asdict(result) for result in candidate],
        "promotion_decision": asdict(decision),
    }
    trusted.output_path.parent.mkdir(parents=True, exist_ok=True)
    with trusted.output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(asdict(decision), indent=2), flush=True)
    return 0 if decision.promotable else 2


if __name__ == "__main__":
    raise SystemExit(main())
