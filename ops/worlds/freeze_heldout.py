"""Generate and physically qualify the immutable held-out evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from muscle_memory.evaluation.heldout import (
    HELDOUT_SPLIT_ID,
    HELDOUT_VALIDATION_CHECKS,
    HELDOUT_WORLD_COUNT,
    HeldOutValidationCertificate,
    HeldOutWorldBundle,
    HeldOutWorldRecord,
    PhysicalWorldQualification,
    heldout_bundle_aggregate,
    world_sha256,
)
from muscle_memory.evaluation.success import (
    EpisodeOutcome,
    Vector2,
    evaluate_safe_delivery,
)
from muscle_memory.paths import HELDOUT_WORLDS_BUNDLE
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.simulation.metrics import EpisodeMetricsTracker
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.world_scene import assemble_episode_scene
from muscle_memory.training.expert import (
    ExpertNavigator,
    ExpertPath,
    direct_route_requires_avoidance,
    plan_expert_path,
)
from muscle_memory.worlds.generation import generate_training_world
from muscle_memory.worlds.models import HeldOutWorld, TrainingWorld, Vec2
from muscle_memory.worlds.rules import DEFAULT_RULES_PATH, load_world_rules

HELDOUT_SEED_SEARCH_START = 900_000_000
MAXIMUM_EXPERT_PATH_LENGTH_M = 8.2
QUALIFICATION_SECONDS = 30.0
METRIC_SAMPLE_INTERVAL_STEPS = 5


@dataclass(frozen=True, slots=True)
class _ValidatedHeldOutCandidate:
    world: HeldOutWorld


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_sha256(points: tuple[Vec2, ...]) -> str:
    payload = [point.model_dump(mode="json") for point in points]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _heldout_world(training_world: TrainingWorld) -> HeldOutWorld:
    payload = training_world.model_dump(mode="json", exclude={"world_id", "split"})
    return HeldOutWorld(
        **payload,
        world_id=(
            f"heldout-v{training_world.generation_version}-{training_world.seed:016x}"
        ),
        split="held_out",
    )


def _yaw(data: object) -> float:
    body = data.body("pelvis")  # type: ignore[attr-defined]
    rotation = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _physical_qualification(
    world: HeldOutWorld,
    path: ExpertPath,
) -> PhysicalWorldQualification | None:
    scene = assemble_episode_scene(_ValidatedHeldOutCandidate(world))
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    metrics = EpisodeMetricsTracker(scene, simulation.data)
    current_metrics = metrics.observe(simulation.data)
    navigator = ExpertNavigator(path)
    reached_at: float | None = None
    fall_count = 0
    was_fallen = False

    def teacher(_time: float):  # type: ignore[no-untyped-def]
        return navigator.command(
            Vec2(x=float(simulation.data.qpos[0]), y=float(simulation.data.qpos[1])),
            _yaw(simulation.data),
        )

    for step in range(round(QUALIFICATION_SECONDS * 500)):
        simulation.step(teacher)
        if step % METRIC_SAMPLE_INTERVAL_STEPS == 0:
            current_metrics = metrics.observe(simulation.data)
        fallen = (
            float(simulation.data.qpos[2]) <= 0.35
            or float(simulation.data.sensor("upvector_torso").data[2]) <= 0.0
        )
        if fallen and not was_fallen:
            fall_count += 1
        was_fallen = fallen
        distance = math.hypot(
            float(world.destination.x) - float(simulation.data.qpos[0]),
            float(world.destination.y) - float(simulation.data.qpos[1]),
        )
        if reached_at is None and distance <= 0.5:
            reached_at = float(simulation.data.time)
        speed = float(np.linalg.norm(simulation.data.qvel[:2]))
        if reached_at is not None and speed <= 0.05 and simulation.data.time >= reached_at + 1.0:
            break
        if fall_count or current_metrics.body_collisions:
            return None

    current_metrics = metrics.observe(simulation.data)
    rotation = np.asarray(simulation.data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    speed = float(np.linalg.norm(simulation.data.qvel[:2]))
    outcome = EpisodeOutcome(
        time_to_resident_seconds=reached_at,
        robot_stop_position=Vector2(
            x=float(simulation.data.qpos[0]),
            y=float(simulation.data.qpos[1]),
        ),
        resident_position=Vector2(
            x=float(world.destination.x),
            y=float(world.destination.y),
        ),
        robot_forward=Vector2(x=float(rotation[0, 0]), y=float(rotation[1, 0])),
        stopped=speed <= 0.05,
        falls=fall_count,
        body_collisions=current_metrics.body_collisions,
        minimum_obstacle_clearance_metres=current_metrics.minimum_obstacle_clearance_m,
        maximum_tray_tilt_degrees=current_metrics.maximum_tray_tilt_degrees,
        package_slipped=current_metrics.package_slipped,
        human_interventions=0,
    )
    evaluation = evaluate_safe_delivery(outcome)
    if not evaluation.success or reached_at is None or evaluation.facing_error_degrees is None:
        return None
    return PhysicalWorldQualification(
        success=True,
        time_to_resident_seconds=reached_at,
        stop_distance_metres=evaluation.stop_distance_metres,
        facing_error_degrees=evaluation.facing_error_degrees,
        stopped_speed_metres_per_second=speed,
        falls=fall_count,
        body_collisions=current_metrics.body_collisions,
        minimum_obstacle_clearance_metres=current_metrics.minimum_obstacle_clearance_m,
        maximum_tray_tilt_degrees=current_metrics.maximum_tray_tilt_degrees,
        package_slipped=current_metrics.package_slipped,
        human_interventions=0,
    )


def build_heldout_bundle() -> HeldOutWorldBundle:
    """Select the first 20 deterministic worlds passing every pre-training gate."""
    rules = load_world_rules()
    rules_hash = _sha256_file(DEFAULT_RULES_PATH)
    robot = verify_mm01_bundle()
    records: list[HeldOutWorldRecord] = []
    seed = HELDOUT_SEED_SEARCH_START
    while len(records) < HELDOUT_WORLD_COUNT:
        validated_training = generate_training_world(seed, rules)
        expert_path = plan_expert_path(validated_training.world, rules)
        if (
            expert_path is not None
            and expert_path.length_m <= MAXIMUM_EXPERT_PATH_LENGTH_M
            and direct_route_requires_avoidance(validated_training.world, rules)
        ):
            heldout = _heldout_world(validated_training.world)
            physical = _physical_qualification(heldout, expert_path)
            if physical is not None:
                record = HeldOutWorldRecord(
                    world=heldout,
                    certificate=HeldOutValidationCertificate(
                        schema_version=1,
                        robot_checksum=robot.robot_checksum,
                        world_sha256=world_sha256(heldout),
                        rules_sha256=rules_hash,
                        baseline_path_sha256=_path_sha256(
                            validated_training.baseline_path
                        ),
                        robust_expert_path_sha256=_path_sha256(expert_path.waypoints),
                        validation_checks=HELDOUT_VALIDATION_CHECKS,
                        physical_qualification=physical,
                    ),
                )
                records.append(record)
                print(
                    f"qualified {len(records):02d}/{HELDOUT_WORLD_COUNT}: "
                    f"seed={seed} time={physical.time_to_resident_seconds:.3f}s",
                    flush=True,
                )
        seed += 1
    bundle = HeldOutWorldBundle(
        schema_version=1,
        split_id=HELDOUT_SPLIT_ID,
        generation_version=int(rules.generation_version),
        records=tuple(records),
        aggregate_sha256="0" * 64,
    )
    return bundle.model_copy(
        update={"aggregate_sha256": heldout_bundle_aggregate(bundle)}
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HELDOUT_WORLDS_BUNDLE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    bundle = build_heldout_bundle()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} aggregate={bundle.aggregate_sha256}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
