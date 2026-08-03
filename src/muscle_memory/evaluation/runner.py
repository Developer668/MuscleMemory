"""Evaluation-only native episode runner with no path-teacher dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import mujoco  # type: ignore[import-untyped]
import numpy as np

from muscle_memory.evaluation.success import (
    EpisodeOutcome,
    Vector2,
    evaluate_safe_delivery,
)
from muscle_memory.policy.observation import NavigationObservation, navigation_observation
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import PHYSICS_HZ
from muscle_memory.simulation.metrics import EpisodeMetricsTracker
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.sensors import EpisodeSensorExtractor
from muscle_memory.simulation.world_scene import (
    ValidatedWorldEnvelope,
    assemble_episode_scene,
)

EPISODE_LIMIT_SECONDS = 30.0
METRIC_SAMPLE_HZ = 20
POLICY_RENDER_WIDTH = 64
POLICY_RENDER_HEIGHT = 48
STOPPED_SPEED_MPS = 0.05
STOP_SETTLE_SECONDS = 1.0


class TaskPolicy(Protocol):
    """The complete policy surface reachable from evaluation."""

    policy_id: str
    policy_hash: str

    def command(self, observation: NavigationObservation) -> TaskCommand: ...


@dataclass(frozen=True, slots=True)
class PolicyStep:
    """One 10 Hz policy decision, suitable for append-only telemetry."""

    frame_id: str
    simulation_time_seconds: float
    position_x_m: float
    position_y_m: float
    yaw_radians: float
    action: TaskCommand
    depth_sectors_m: tuple[float, ...]
    obstacle_clearance_m: float
    tray_tilt_degrees: float


@dataclass(frozen=True, slots=True)
class PolicyEpisodeResult:
    """Measured evaluation result for one immutable policy/world pairing."""

    episode_id: str
    world_id: str
    world_seed: int
    world_split: str
    world_hash: str
    robot_checksum: str
    policy_id: str
    policy_hash: str
    success: bool
    failed_reasons: tuple[str, ...]
    time_to_resident_seconds: float | None
    simulated_duration_seconds: float
    stop_distance_m: float
    facing_error_degrees: float | None
    stopped_speed_mps: float
    falls: int
    body_collisions: int
    minimum_obstacle_clearance_m: float
    maximum_tray_tilt_degrees: float
    package_slipped: bool
    human_interventions: int
    direct_distance_m: float
    path_length_m: float
    path_efficiency: float
    energy_joules: float
    task_policy_updates: int
    trace: tuple[PolicyStep, ...]


def _yaw(data: mujoco.MjData) -> float:
    rotation = np.asarray(data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _fallen(data: mujoco.MjData) -> bool:
    return bool(
        float(data.qpos[2]) <= 0.35
        or float(data.sensor("upvector_torso").data[2]) <= 0.0
    )


def run_policy_episode(
    envelope: ValidatedWorldEnvelope,
    policy: TaskPolicy,
    *,
    episode_id: str,
    record_trace: bool = False,
) -> PolicyEpisodeResult:
    """Run a policy from sensor input alone for at most 30 simulated seconds."""
    if not episode_id:
        raise ValueError("episode_id must not be empty")
    scene = assemble_episode_scene(envelope)
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    metrics_tracker = EpisodeMetricsTracker(scene, simulation.data)
    current_metrics = metrics_tracker.observe(simulation.data)
    sensor_extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
    previous_action = TaskCommand(0.0, 0.0, 1.0)
    reached_at: float | None = None
    fall_count = 0
    was_fallen = False
    trace: list[PolicyStep] = []
    path_length = 0.0
    energy_joules = 0.0
    actuator_joint_ids = np.asarray(simulation.model.actuator_trnid[:, 0], dtype=np.int32)
    actuator_dof_addresses = np.asarray(
        simulation.model.jnt_dofadr[actuator_joint_ids],
        dtype=np.int32,
    )
    previous_position = np.asarray(simulation.data.qpos[:2], dtype=np.float64).copy()
    metric_interval = PHYSICS_HZ // METRIC_SAMPLE_HZ

    with mujoco.Renderer(
        simulation.model,
        height=POLICY_RENDER_HEIGHT,
        width=POLICY_RENDER_WIDTH,
    ) as renderer:

        def decide(_time: float) -> TaskCommand:
            nonlocal previous_action
            frame_id = f"{episode_id}:{simulation.task_policy_updates:08d}"
            stereo = sensor_extractor.capture_policy_stereo(renderer, frame_id)
            observation = navigation_observation(
                simulation.data,
                scene.world.destination,
                stereo.derived_depth_sectors,
                current_metrics,
                previous_action,
            )
            action = policy.command(observation)
            if not isinstance(action, TaskCommand):
                raise TypeError("task policy must return TaskCommand")
            if record_trace:
                trace.append(
                    PolicyStep(
                        frame_id=frame_id,
                        simulation_time_seconds=float(simulation.data.time),
                        position_x_m=float(simulation.data.qpos[0]),
                        position_y_m=float(simulation.data.qpos[1]),
                        yaw_radians=_yaw(simulation.data),
                        action=action,
                        depth_sectors_m=tuple(
                            float(value) for value in stereo.derived_depth_sectors
                        ),
                        obstacle_clearance_m=current_metrics.current_obstacle_clearance_m,
                        tray_tilt_degrees=current_metrics.current_tray_tilt_degrees,
                    )
                )
            previous_action = action
            return action

        maximum_steps = round(EPISODE_LIMIT_SECONDS * PHYSICS_HZ)
        for step in range(maximum_steps):
            power_watts = float(
                np.sum(
                    np.abs(
                        simulation.data.actuator_force
                        * simulation.data.qvel[actuator_dof_addresses]
                    )
                )
            )
            simulation.step(decide)
            energy_joules += power_watts / PHYSICS_HZ
            position = np.asarray(simulation.data.qpos[:2], dtype=np.float64)
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position.copy()
            if step % metric_interval == 0:
                current_metrics = metrics_tracker.observe(simulation.data)
            fallen = _fallen(simulation.data)
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
    direct_distance = math.dist(
        (float(scene.world.start.x), float(scene.world.start.y)),
        (float(scene.world.destination.x), float(scene.world.destination.y)),
    )
    path_efficiency = min(1.0, direct_distance / path_length) if path_length > 0.0 else 0.0
    return PolicyEpisodeResult(
        episode_id=episode_id,
        world_id=scene.world.world_id,
        world_seed=scene.world.seed,
        world_split=scene.world.split,
        world_hash=scene.world_hash,
        robot_checksum=scene.robot_checksum,
        policy_id=policy.policy_id,
        policy_hash=policy.policy_hash,
        success=evaluation.success,
        failed_reasons=tuple(reason.name for reason in evaluation.failed_reasons),
        time_to_resident_seconds=reached_at,
        simulated_duration_seconds=float(simulation.data.time),
        stop_distance_m=evaluation.stop_distance_metres,
        facing_error_degrees=evaluation.facing_error_degrees,
        stopped_speed_mps=stopped_speed,
        falls=fall_count,
        body_collisions=current_metrics.body_collisions,
        minimum_obstacle_clearance_m=current_metrics.minimum_obstacle_clearance_m,
        maximum_tray_tilt_degrees=current_metrics.maximum_tray_tilt_degrees,
        package_slipped=current_metrics.package_slipped,
        human_interventions=0,
        direct_distance_m=direct_distance,
        path_length_m=path_length,
        path_efficiency=path_efficiency,
        energy_joules=energy_joules,
        task_policy_updates=simulation.task_policy_updates,
        trace=tuple(trace),
    )
