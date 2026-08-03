"""Record physical A* demonstrations without exposing held-out worlds to training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from muscle_memory.evaluation.success import EpisodeOutcome, Vector2, evaluate_safe_delivery
from muscle_memory.paths import (
    DAGGER_DATASET_V1,
    DAGGER_DATASET_V1_METADATA,
    EXPERT_DATASET_V1,
    EXPERT_DATASET_V1_METADATA,
)
from muscle_memory.policy.actions import PolicyAction
from muscle_memory.policy.observation import (
    NAVIGATION_OBSERVATION_SIZE,
    NavigationObservation,
    navigation_observation,
)
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import PHYSICS_HZ, TASK_POLICY_HZ, verify_mm01_bundle
from muscle_memory.simulation.metrics import EpisodeMetricsTracker
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.sensors import EpisodeSensorExtractor
from muscle_memory.simulation.world_scene import assemble_episode_scene
from muscle_memory.training.expert import (
    ExpertNavigator,
    ExpertPath,
    direct_route_requires_avoidance,
    plan_expert_path,
)
from muscle_memory.worlds.generation import generate_training_world
from muscle_memory.worlds.generation.models import ValidatedTrainingWorld
from muscle_memory.worlds.models import TrainingWorld, Vec2
from muscle_memory.worlds.rules import DEFAULT_RULES_PATH, load_world_rules

DATASET_SCHEMA_VERSION = 2
LEGACY_CLASSIFICATION_DATASET_SCHEMA_VERSION = 1
TRAINING_SEED_SEARCH_START = 100_000_000
DEFAULT_TRAINING_EPISODES = 64
MAXIMUM_EXPERT_PATH_LENGTH_M = 8.2
EPISODE_LIMIT_SECONDS = 30.0
METRIC_SAMPLE_HZ = 20
POLICY_RENDER_WIDTH = 64
POLICY_RENDER_HEIGHT = 48
STOPPED_SPEED_MPS = 0.05
STOP_SETTLE_SECONDS = 1.0
DAGGER_SEED_SEARCH_START = 200_000_000
DEFAULT_DAGGER_EPISODES = 24


class RolloutPolicy(Protocol):
    policy_id: str
    policy_hash: str

    def command(self, observation: NavigationObservation) -> TaskCommand: ...


@dataclass(frozen=True, slots=True)
class RecordedExpertEpisode:
    """One successful native-controller demonstration."""

    world_id: str
    world_seed: int
    world_hash: str
    observations: npt.NDArray[np.float32]
    actions: npt.NDArray[np.int64]
    commands: npt.NDArray[np.float32]
    time_to_resident_seconds: float
    minimum_obstacle_clearance_m: float
    maximum_tray_tilt_degrees: float


@dataclass(frozen=True, slots=True)
class ReplayedExpertCommands:
    """Continuous commands recovered from an exact native expert replay."""

    world_id: str
    world_hash: str
    commands: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RecordedDaggerEpisode:
    """Teacher labels on states visited by a candidate policy."""

    world_id: str
    world_seed: int
    world_hash: str
    observations: npt.NDArray[np.float32]
    actions: npt.NDArray[np.int64]
    commands: npt.NDArray[np.float32]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaw(data: mujoco.MjData) -> float:
    rotation = np.asarray(data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _action_label(command: TaskCommand) -> PolicyAction:
    if command.stop_requested:
        return PolicyAction.STOP
    if command.forward_speed_mps <= 0.01:
        return (
            PolicyAction.TURN_LEFT
            if command.turning_rate_rad_s >= 0.0
            else PolicyAction.TURN_RIGHT
        )
    if command.turning_rate_rad_s > 0.05:
        return PolicyAction.DRIVE_LEFT
    if command.turning_rate_rad_s < -0.05:
        return PolicyAction.DRIVE_RIGHT
    return PolicyAction.DRIVE_STRAIGHT


def record_expert_episode(
    validated_world: ValidatedTrainingWorld,
    path: ExpertPath,
) -> RecordedExpertEpisode | None:
    """Record one demonstration only when the real physical episode passes every gate."""
    if not isinstance(validated_world, ValidatedTrainingWorld) or not isinstance(
        validated_world.world, TrainingWorld
    ):
        raise TypeError("expert recording accepts only a ValidatedTrainingWorld")
    scene = assemble_episode_scene(validated_world)
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    metrics_tracker = EpisodeMetricsTracker(scene, simulation.data)
    current_metrics = metrics_tracker.observe(simulation.data)
    sensor_extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
    navigator = ExpertNavigator(path)
    observations: list[npt.NDArray[np.float32]] = []
    actions: list[int] = []
    commands: list[tuple[float, float, float]] = []
    previous_action = TaskCommand(0.0, 0.0, 1.0)
    reached_at: float | None = None
    fall_count = 0
    was_fallen = False
    metric_interval = PHYSICS_HZ // METRIC_SAMPLE_HZ

    with mujoco.Renderer(
        simulation.model,
        height=POLICY_RENDER_HEIGHT,
        width=POLICY_RENDER_WIDTH,
    ) as renderer:

        def teacher(_time: float) -> TaskCommand:
            nonlocal previous_action
            command = navigator.command(
                Vec2(
                    x=float(simulation.data.qpos[0]),
                    y=float(simulation.data.qpos[1]),
                ),
                _yaw(simulation.data),
            )
            frame_id = f"expert-{scene.world.world_id}:{simulation.task_policy_updates:08d}"
            stereo = sensor_extractor.capture_policy_stereo(renderer, frame_id)
            observation = navigation_observation(
                simulation.data,
                scene.world.destination,
                stereo.derived_depth_sectors,
                current_metrics,
                previous_action,
            )
            observations.append(observation.values.copy())
            actions.append(int(_action_label(command)))
            commands.append(
                (
                    command.forward_speed_mps,
                    command.turning_rate_rad_s,
                    command.stop_probability,
                )
            )
            previous_action = command
            return command

        maximum_steps = round(EPISODE_LIMIT_SECONDS * PHYSICS_HZ)
        for step in range(maximum_steps):
            simulation.step(teacher)
            if step % metric_interval == 0:
                current_metrics = metrics_tracker.observe(simulation.data)
            fallen = bool(
                float(simulation.data.qpos[2]) <= 0.35
                or float(simulation.data.sensor("upvector_torso").data[2]) <= 0.0
            )
            if fallen and not was_fallen:
                fall_count += 1
            was_fallen = fallen
            distance = math.hypot(
                float(scene.world.destination.x) - float(simulation.data.qpos[0]),
                float(scene.world.destination.y) - float(simulation.data.qpos[1]),
            )
            if reached_at is None and distance <= 0.5:
                reached_at = float(simulation.data.time)
            speed = float(np.linalg.norm(simulation.data.qvel[:2]))
            if (
                reached_at is not None
                and speed <= STOPPED_SPEED_MPS
                and simulation.data.time >= reached_at + STOP_SETTLE_SECONDS
            ):
                break
            if fall_count or current_metrics.body_collisions:
                return None

    current_metrics = metrics_tracker.observe(simulation.data)
    rotation = np.asarray(simulation.data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    stopped_speed = float(np.linalg.norm(simulation.data.qvel[:2]))
    outcome = EpisodeOutcome(
        time_to_resident_seconds=reached_at,
        robot_stop_position=Vector2(
            x=float(simulation.data.qpos[0]),
            y=float(simulation.data.qpos[1]),
        ),
        resident_position=Vector2(
            x=float(scene.world.destination.x),
            y=float(scene.world.destination.y),
        ),
        robot_forward=Vector2(x=float(rotation[0, 0]), y=float(rotation[1, 0])),
        stopped=stopped_speed <= STOPPED_SPEED_MPS,
        falls=fall_count,
        body_collisions=current_metrics.body_collisions,
        minimum_obstacle_clearance_metres=current_metrics.minimum_obstacle_clearance_m,
        maximum_tray_tilt_degrees=current_metrics.maximum_tray_tilt_degrees,
        package_slipped=current_metrics.package_slipped,
        human_interventions=0,
    )
    evaluation = evaluate_safe_delivery(outcome)
    if not evaluation.success or reached_at is None or not observations:
        return None
    return RecordedExpertEpisode(
        world_id=scene.world.world_id,
        world_seed=scene.world.seed,
        world_hash=scene.world_hash,
        observations=np.stack(observations).astype(np.float32, copy=False),
        actions=np.asarray(actions, dtype=np.int64),
        commands=np.asarray(commands, dtype=np.float32),
        time_to_resident_seconds=reached_at,
        minimum_obstacle_clearance_m=current_metrics.minimum_obstacle_clearance_m,
        maximum_tray_tilt_degrees=current_metrics.maximum_tray_tilt_degrees,
    )


def replay_expert_commands(
    validated_world: ValidatedTrainingWorld,
    path: ExpertPath,
) -> ReplayedExpertCommands | None:
    """Replay an expert without rendering and recover its exact continuous targets."""
    if not isinstance(validated_world, ValidatedTrainingWorld) or not isinstance(
        validated_world.world, TrainingWorld
    ):
        raise TypeError("expert replay accepts only a ValidatedTrainingWorld")
    scene = assemble_episode_scene(validated_world)
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    metrics_tracker = EpisodeMetricsTracker(scene, simulation.data)
    current_metrics = metrics_tracker.observe(simulation.data)
    navigator = ExpertNavigator(path)
    commands: list[tuple[float, float, float]] = []
    reached_at: float | None = None
    fall_count = 0
    was_fallen = False
    metric_interval = PHYSICS_HZ // METRIC_SAMPLE_HZ

    def teacher(_time: float) -> TaskCommand:
        command = navigator.command(
            Vec2(
                x=float(simulation.data.qpos[0]),
                y=float(simulation.data.qpos[1]),
            ),
            _yaw(simulation.data),
        )
        commands.append(
            (
                command.forward_speed_mps,
                command.turning_rate_rad_s,
                command.stop_probability,
            )
        )
        return command

    maximum_steps = round(EPISODE_LIMIT_SECONDS * PHYSICS_HZ)
    for step in range(maximum_steps):
        simulation.step(teacher)
        if step % metric_interval == 0:
            current_metrics = metrics_tracker.observe(simulation.data)
        fallen = bool(
            float(simulation.data.qpos[2]) <= 0.35
            or float(simulation.data.sensor("upvector_torso").data[2]) <= 0.0
        )
        if fallen and not was_fallen:
            fall_count += 1
        was_fallen = fallen
        distance = math.hypot(
            float(scene.world.destination.x) - float(simulation.data.qpos[0]),
            float(scene.world.destination.y) - float(simulation.data.qpos[1]),
        )
        if reached_at is None and distance <= 0.5:
            reached_at = float(simulation.data.time)
        speed = float(np.linalg.norm(simulation.data.qvel[:2]))
        if (
            reached_at is not None
            and speed <= STOPPED_SPEED_MPS
            and simulation.data.time >= reached_at + STOP_SETTLE_SECONDS
        ):
            break
        if fall_count or current_metrics.body_collisions:
            return None

    current_metrics = metrics_tracker.observe(simulation.data)
    rotation = np.asarray(simulation.data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    stopped_speed = float(np.linalg.norm(simulation.data.qvel[:2]))
    evaluation = evaluate_safe_delivery(
        EpisodeOutcome(
            time_to_resident_seconds=reached_at,
            robot_stop_position=Vector2(
                x=float(simulation.data.qpos[0]),
                y=float(simulation.data.qpos[1]),
            ),
            resident_position=Vector2(
                x=float(scene.world.destination.x),
                y=float(scene.world.destination.y),
            ),
            robot_forward=Vector2(x=float(rotation[0, 0]), y=float(rotation[1, 0])),
            stopped=stopped_speed <= STOPPED_SPEED_MPS,
            falls=fall_count,
            body_collisions=current_metrics.body_collisions,
            minimum_obstacle_clearance_metres=current_metrics.minimum_obstacle_clearance_m,
            maximum_tray_tilt_degrees=current_metrics.maximum_tray_tilt_degrees,
            package_slipped=current_metrics.package_slipped,
            human_interventions=0,
        )
    )
    if not evaluation.success or not commands:
        return None
    return ReplayedExpertCommands(
        world_id=scene.world.world_id,
        world_hash=scene.world_hash,
        commands=np.asarray(commands, dtype=np.float32),
    )


def upgrade_expert_dataset_with_commands(
    *,
    dataset_path: Path = EXPERT_DATASET_V1,
    metadata_path: Path = EXPERT_DATASET_V1_METADATA,
) -> dict[str, object]:
    """Upgrade a verified classification corpus with exactly aligned continuous targets."""
    with np.load(dataset_path, allow_pickle=False) as dataset:
        schema_version = int(dataset["schema_version"])
        if schema_version == DATASET_SCHEMA_VERSION and "commands" in dataset.files:
            return json.loads(metadata_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        if schema_version != LEGACY_CLASSIFICATION_DATASET_SCHEMA_VERSION:
            raise RuntimeError("expert dataset cannot be upgraded from this schema")
        robot_checksum = str(dataset["robot_checksum"])
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.int64)
        episode_indices = np.asarray(dataset["episode_indices"], dtype=np.int32)
        world_ids = np.asarray(dataset["world_ids"])
        world_seeds = np.asarray(dataset["world_seeds"], dtype=np.int64)
        world_hashes = np.asarray(dataset["world_hashes"])
    robot = verify_mm01_bundle()
    if robot_checksum != robot.robot_checksum:
        raise RuntimeError("expert dataset belongs to a different robot")
    episode_count = world_seeds.size
    if not (
        world_ids.shape == world_hashes.shape == world_seeds.shape == (episode_count,)
        and np.array_equal(np.unique(episode_indices), np.arange(episode_count))
    ):
        raise RuntimeError("expert dataset episode provenance is invalid")

    rules = load_world_rules()
    command_batches: list[npt.NDArray[np.float32]] = []
    for episode_index, seed_value in enumerate(world_seeds):
        validated = generate_training_world(int(seed_value), rules)
        path = plan_expert_path(validated.world, rules)
        if path is None:
            raise RuntimeError("expert replay path unexpectedly disappeared")
        replay = replay_expert_commands(validated, path)
        if replay is None:
            raise RuntimeError("expert replay no longer passes the physical gate")
        if replay.world_id != str(world_ids[episode_index]) or replay.world_hash != str(
            world_hashes[episode_index]
        ):
            raise RuntimeError("expert replay world provenance changed")
        expected_actions = actions[episode_indices == episode_index]
        replay_actions = np.asarray(
            [
                int(
                    _action_label(
                        TaskCommand(
                            forward_speed_mps=float(command[0]),
                            turning_rate_rad_s=float(command[1]),
                            stop_probability=float(command[2]),
                        )
                    )
                )
                for command in replay.commands
            ],
            dtype=np.int64,
        )
        if not np.array_equal(replay_actions, expected_actions):
            raise RuntimeError("expert replay commands do not align with recorded observations")
        command_batches.append(replay.commands)
        print(
            f"replayed {episode_index + 1:02d}/{episode_count}: "
            f"seed={int(seed_value)} samples={replay.commands.shape[0]}",
            flush=True,
        )

    commands = np.concatenate(command_batches, axis=0)
    if commands.shape != (observations.shape[0], 3):
        raise RuntimeError("continuous command targets have the wrong shape")
    temporary_path = dataset_path.with_name(f"{dataset_path.stem}.upgraded.npz")
    np.savez_compressed(
        temporary_path,
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
        robot_checksum=np.asarray(robot_checksum),
        observations=observations,
        actions=actions,
        commands=commands,
        episode_indices=episode_indices,
        world_ids=world_ids,
        world_seeds=world_seeds,
        world_hashes=world_hashes,
    )
    temporary_path.replace(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_sha256": _sha256_file(dataset_path),
            "command_targets": [
                "forward_speed_mps",
                "turning_rate_rad_s",
                "stop_probability",
            ],
            "command_replay_verified_against_legacy_labels": True,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata  # type: ignore[no-any-return]


def record_dagger_episode(
    validated_world: ValidatedTrainingWorld,
    path: ExpertPath,
    policy: RolloutPolicy,
) -> RecordedDaggerEpisode | None:
    """Execute the student and label each visited sensor state with the expert."""
    if not isinstance(validated_world, ValidatedTrainingWorld) or not isinstance(
        validated_world.world, TrainingWorld
    ):
        raise TypeError("DAgger recording accepts only a ValidatedTrainingWorld")
    scene = assemble_episode_scene(validated_world)
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    metrics_tracker = EpisodeMetricsTracker(scene, simulation.data)
    current_metrics = metrics_tracker.observe(simulation.data)
    sensor_extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
    navigator = ExpertNavigator(path)
    observations: list[npt.NDArray[np.float32]] = []
    actions: list[int] = []
    commands: list[tuple[float, float, float]] = []
    previous_action = TaskCommand(0.0, 0.0, 1.0)
    reached_at: float | None = None
    metric_interval = PHYSICS_HZ // METRIC_SAMPLE_HZ

    with mujoco.Renderer(
        simulation.model,
        height=POLICY_RENDER_HEIGHT,
        width=POLICY_RENDER_WIDTH,
    ) as renderer:

        def student(_time: float) -> TaskCommand:
            nonlocal previous_action
            frame_id = f"dagger-{scene.world.world_id}:{simulation.task_policy_updates:08d}"
            stereo = sensor_extractor.capture_policy_stereo(renderer, frame_id)
            observation = navigation_observation(
                simulation.data,
                scene.world.destination,
                stereo.derived_depth_sectors,
                current_metrics,
                previous_action,
            )
            teacher_command = navigator.command(
                Vec2(
                    x=float(simulation.data.qpos[0]),
                    y=float(simulation.data.qpos[1]),
                ),
                _yaw(simulation.data),
            )
            student_command = policy.command(observation)
            observations.append(observation.values.copy())
            actions.append(int(_action_label(teacher_command)))
            commands.append(
                (
                    teacher_command.forward_speed_mps,
                    teacher_command.turning_rate_rad_s,
                    teacher_command.stop_probability,
                )
            )
            previous_action = student_command
            return student_command

        maximum_steps = round(EPISODE_LIMIT_SECONDS * PHYSICS_HZ)
        for step in range(maximum_steps):
            simulation.step(student)
            if step % metric_interval == 0:
                current_metrics = metrics_tracker.observe(simulation.data)
            distance = math.hypot(
                float(scene.world.destination.x) - float(simulation.data.qpos[0]),
                float(scene.world.destination.y) - float(simulation.data.qpos[1]),
            )
            if reached_at is None and distance <= 0.5:
                reached_at = float(simulation.data.time)
            speed = float(np.linalg.norm(simulation.data.qvel[:2]))
            if (
                reached_at is not None
                and speed <= STOPPED_SPEED_MPS
                and simulation.data.time >= reached_at + STOP_SETTLE_SECONDS
            ):
                break
            fallen = bool(
                float(simulation.data.qpos[2]) <= 0.35
                or float(simulation.data.sensor("upvector_torso").data[2]) <= 0.0
            )
            if fallen or current_metrics.body_collisions:
                break

    if not observations:
        return None
    return RecordedDaggerEpisode(
        world_id=scene.world.world_id,
        world_seed=scene.world.seed,
        world_hash=scene.world_hash,
        observations=np.stack(observations).astype(np.float32, copy=False),
        actions=np.asarray(actions, dtype=np.int64),
        commands=np.asarray(commands, dtype=np.float32),
    )


def build_dagger_dataset(
    policy: RolloutPolicy,
    *,
    base_dataset_path: Path = EXPERT_DATASET_V1,
    output_path: Path = DAGGER_DATASET_V1,
    metadata_path: Path = DAGGER_DATASET_V1_METADATA,
    episode_count: int = DEFAULT_DAGGER_EPISODES,
    seed_start: int = DAGGER_SEED_SEARCH_START,
) -> dict[str, object]:
    """Aggregate student-visited training states from a disjoint seed range."""
    if episode_count < 1:
        raise ValueError("DAgger dataset needs at least one rollout episode")
    base_sha256 = _sha256_file(base_dataset_path)
    with np.load(base_dataset_path, allow_pickle=False) as base:
        if int(base["schema_version"]) != DATASET_SCHEMA_VERSION:
            raise RuntimeError("DAgger requires the continuous expert dataset")
        robot_checksum = str(base["robot_checksum"])
        base_observations = np.asarray(base["observations"], dtype=np.float32)
        base_actions = np.asarray(base["actions"], dtype=np.int64)
        base_commands = np.asarray(base["commands"], dtype=np.float32)
        base_episode_indices = np.asarray(base["episode_indices"], dtype=np.int32)
        base_world_ids = np.asarray(base["world_ids"])
        base_world_seeds = np.asarray(base["world_seeds"], dtype=np.int64)
        base_world_hashes = np.asarray(base["world_hashes"])
    robot = verify_mm01_bundle()
    if robot_checksum != robot.robot_checksum:
        raise RuntimeError("DAgger base dataset belongs to a different robot")

    rules = load_world_rules()
    episodes: list[RecordedDaggerEpisode] = []
    seed = seed_start
    while len(episodes) < episode_count:
        validated = generate_training_world(seed, rules)
        path = plan_expert_path(validated.world, rules)
        if (
            path is not None
            and path.length_m <= MAXIMUM_EXPERT_PATH_LENGTH_M
            and direct_route_requires_avoidance(validated.world, rules)
        ):
            recorded = record_dagger_episode(validated, path, policy)
            if recorded is not None:
                episodes.append(recorded)
                print(
                    f"aggregated {len(episodes):02d}/{episode_count}: "
                    f"seed={seed} samples={recorded.commands.shape[0]}",
                    flush=True,
                )
        seed += 1

    observations = np.concatenate(
        (base_observations, *(episode.observations for episode in episodes)),
        axis=0,
    )
    actions = np.concatenate(
        (base_actions, *(episode.actions for episode in episodes)),
        axis=0,
    )
    commands = np.concatenate(
        (base_commands, *(episode.commands for episode in episodes)),
        axis=0,
    )
    base_episode_count = base_world_seeds.size
    episode_indices = np.concatenate(
        (
            base_episode_indices,
            *(
                np.full(
                    episode.commands.shape[0],
                    base_episode_count + index,
                    dtype=np.int32,
                )
                for index, episode in enumerate(episodes)
            ),
        )
    )
    world_ids = np.concatenate(
        (base_world_ids, np.asarray([episode.world_id for episode in episodes]))
    )
    world_seeds = np.concatenate(
        (
            base_world_seeds,
            np.asarray([episode.world_seed for episode in episodes], dtype=np.int64),
        )
    )
    world_hashes = np.concatenate(
        (base_world_hashes, np.asarray([episode.world_hash for episode in episodes]))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
        robot_checksum=np.asarray(robot_checksum),
        observations=observations,
        actions=actions,
        commands=commands,
        episode_indices=episode_indices,
        world_ids=world_ids,
        world_seeds=world_seeds,
        world_hashes=world_hashes,
    )
    metadata: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "robot_checksum": robot_checksum,
        "dataset_sha256": _sha256_file(output_path),
        "base_dataset_sha256": base_sha256,
        "base_episode_count": int(base_episode_count),
        "dagger_episode_count": len(episodes),
        "total_episode_count": int(world_seeds.size),
        "base_sample_count": int(base_observations.shape[0]),
        "dagger_sample_count": int(observations.shape[0] - base_observations.shape[0]),
        "total_sample_count": int(observations.shape[0]),
        "rollout_policy_id": policy.policy_id,
        "rollout_policy_sha256": policy.policy_hash,
        "seed_search_start": seed_start,
        "seed_search_exclusive_end": seed,
        "dagger_world_ids": [episode.world_id for episode in episodes],
        "dagger_world_seeds": [episode.world_seed for episode in episodes],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_expert_dataset(
    *,
    output_path: Path = EXPERT_DATASET_V1,
    metadata_path: Path = EXPERT_DATASET_V1_METADATA,
    episode_count: int = DEFAULT_TRAINING_EPISODES,
    seed_start: int = TRAINING_SEED_SEARCH_START,
) -> dict[str, object]:
    """Build a deterministic physical dataset from training worlds only."""
    if episode_count < 2:
        raise ValueError("expert dataset needs at least two episodes")
    rules = load_world_rules()
    robot = verify_mm01_bundle()
    episodes: list[RecordedExpertEpisode] = []
    seed = seed_start
    while len(episodes) < episode_count:
        validated = generate_training_world(seed, rules)
        path = plan_expert_path(validated.world, rules)
        if (
            path is not None
            and path.length_m <= MAXIMUM_EXPERT_PATH_LENGTH_M
            and direct_route_requires_avoidance(validated.world, rules)
        ):
            recorded = record_expert_episode(validated, path)
            if recorded is not None:
                episodes.append(recorded)
                print(
                    f"recorded {len(episodes):02d}/{episode_count}: "
                    f"seed={seed} samples={recorded.actions.size} "
                    f"time={recorded.time_to_resident_seconds:.3f}s",
                    flush=True,
                )
        seed += 1

    observation_batches = [episode.observations for episode in episodes]
    action_batches = [episode.actions for episode in episodes]
    command_batches = [episode.commands for episode in episodes]
    observations = np.concatenate(observation_batches, axis=0)
    actions = np.concatenate(action_batches, axis=0)
    commands = np.concatenate(command_batches, axis=0)
    episode_indices = np.concatenate(
        [
            np.full(episode.actions.shape, index, dtype=np.int32)
            for index, episode in enumerate(episodes)
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
        robot_checksum=np.asarray(robot.robot_checksum),
        observations=observations,
        actions=actions,
        commands=commands,
        episode_indices=episode_indices,
        world_ids=np.asarray([episode.world_id for episode in episodes]),
        world_seeds=np.asarray([episode.world_seed for episode in episodes], dtype=np.int64),
        world_hashes=np.asarray([episode.world_hash for episode in episodes]),
    )
    label_counts = np.bincount(actions, minlength=len(PolicyAction)).tolist()
    metadata: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "robot_checksum": robot.robot_checksum,
        "dataset_sha256": _sha256_file(output_path),
        "episode_count": len(episodes),
        "sample_count": int(observations.shape[0]),
        "observation_size": NAVIGATION_OBSERVATION_SIZE,
        "action_count": len(PolicyAction),
        "action_labels": [action.name for action in PolicyAction],
        "action_sample_counts": label_counts,
        "command_targets": [
            "forward_speed_mps",
            "turning_rate_rad_s",
            "stop_probability",
        ],
        "task_policy_hz": TASK_POLICY_HZ,
        "physics_hz": PHYSICS_HZ,
        "stereo_render_size": [POLICY_RENDER_WIDTH, POLICY_RENDER_HEIGHT],
        "seed_search_start": seed_start,
        "seed_search_exclusive_end": seed,
        "world_ids": [episode.world_id for episode in episodes],
        "world_seeds": [episode.world_seed for episode in episodes],
        "world_hashes": [episode.world_hash for episode in episodes],
        "minimum_physical_clearance_m": min(
            episode.minimum_obstacle_clearance_m for episode in episodes
        ),
        "maximum_physical_tray_tilt_degrees": max(
            episode.maximum_tray_tilt_degrees for episode in episodes
        ),
        "rules_sha256": _sha256_file(DEFAULT_RULES_PATH),
        "teacher_source_sha256": _sha256_file(Path(__file__).with_name("expert.py")),
        "observation_source_sha256": _sha256_file(
            Path(__file__).parents[1] / "policy" / "observation.py"
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
