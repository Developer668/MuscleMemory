"""Run the one-shot paired held-out evaluation with no teacher imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from muscle_memory.evaluation.heldout import load_heldout_worlds
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult, run_policy_episode
from muscle_memory.paths import (
    HELDOUT_WORLDS_BUNDLE,
    POLICY_V2_CHECKPOINT,
    POLICY_V2_HELDOUT_EVIDENCE,
)
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.policy.network import BehaviorClonedPolicy

DEFAULT_OUTPUT = POLICY_V2_HELDOUT_EVIDENCE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_evaluation_import_isolation() -> None:
    forbidden = tuple(
        name
        for name in sys.modules
        if name.startswith("muscle_memory.training")
        or name.startswith("muscle_memory.worlds.generation")
    )
    if forbidden:
        raise RuntimeError(f"held-out evaluation loaded forbidden modules: {forbidden}")


def _evaluate(
    policy: DirectGoalPolicy | BehaviorClonedPolicy,
    *,
    evaluation_scope: str,
    role: str,
) -> tuple[PolicyEpisodeResult, ...]:
    worlds = load_heldout_worlds()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=POLICY_V2_CHECKPOINT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable held-out evidence already exists: {args.output}")
    _assert_evaluation_import_isolation()
    candidate_policy = BehaviorClonedPolicy.load(args.checkpoint)
    evaluation_scope = candidate_policy.policy_hash[:16]
    baseline = _evaluate(
        DirectGoalPolicy(),
        evaluation_scope=evaluation_scope,
        role="baseline",
    )
    candidate = _evaluate(
        candidate_policy,
        evaluation_scope=evaluation_scope,
        role="candidate",
    )
    decision = evaluate_promotion(baseline, candidate)
    payload = {
        "schema_version": 1,
        "heldout_bundle_sha256": _sha256_file(HELDOUT_WORLDS_BUNDLE),
        "candidate_checkpoint_sha256": candidate_policy.policy_hash,
        "baseline_results": [asdict(result) for result in baseline],
        "candidate_results": [asdict(result) for result in candidate],
        "promotion_decision": asdict(decision),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(decision), indent=2), flush=True)
    return 0 if decision.promotable else 2


if __name__ == "__main__":
    raise SystemExit(main())
