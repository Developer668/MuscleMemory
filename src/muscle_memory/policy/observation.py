"""Build the fixed task-policy observation from existing MM-01 signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from muscle_memory.robot.command import TaskCommand
from muscle_memory.simulation.metrics import EpisodeMetrics
from muscle_memory.simulation.sensors import (
    DEFAULT_DEPTH_SECTORS,
    MAXIMUM_STEREO_DEPTH_M,
    MINIMUM_STEREO_DEPTH_M,
)
from muscle_memory.worlds.models import Vec2

NAVIGATION_OBSERVATION_SIZE = 69


@dataclass(frozen=True, slots=True)
class NavigationObservation:
    """One immutable task-policy input with both raw task context and model vector."""

    values: npt.NDArray[np.float32]
    destination_distance_m: float
    destination_bearing_rad: float

    def __post_init__(self) -> None:
        if self.values.shape != (NAVIGATION_OBSERVATION_SIZE,):
            raise ValueError("navigation observation has the wrong shape")
        if self.values.dtype != np.float32 or not np.isfinite(self.values).all():
            raise ValueError("navigation observation must be finite float32")
        if not math.isfinite(self.destination_distance_m) or self.destination_distance_m < 0:
            raise ValueError("destination distance must be finite and non-negative")
        if not math.isfinite(self.destination_bearing_rad):
            raise ValueError("destination bearing must be finite")


def _yaw(data: mujoco.MjData) -> float:
    rotation = np.asarray(data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def navigation_observation(
    data: mujoco.MjData,
    destination: Vec2,
    derived_depth_sectors: npt.NDArray[np.float64],
    payload_metrics: EpisodeMetrics,
    previous_action: TaskCommand,
) -> NavigationObservation:
    """Normalize the declared policy inputs without adding hidden world state."""
    depth = np.asarray(derived_depth_sectors, dtype=np.float64)
    if depth.shape != (DEFAULT_DEPTH_SECTORS,) or not np.isfinite(depth).all():
        raise ValueError("policy requires exactly 48 finite stereo-derived depth sectors")
    position_x = float(data.qpos[0])
    position_y = float(data.qpos[1])
    delta_x = float(destination.x) - position_x
    delta_y = float(destination.y) - position_y
    distance = math.hypot(delta_x, delta_y)
    bearing = _wrap_angle(math.atan2(delta_y, delta_x) - _yaw(data))
    torso_orientation = np.asarray(
        data.sensor("orientation_torso").data,
        dtype=np.float64,
    )
    torso_gyro = np.asarray(data.sensor("gyro_torso").data, dtype=np.float64)
    base_velocity = np.asarray(
        data.sensor("local_linvel_pelvis").data[:2],
        dtype=np.float64,
    )
    effort = np.abs(np.asarray(data.actuator_force, dtype=np.float64))
    foot_contacts = np.asarray(
        [
            float(data.sensor("left_foot_floor_found").data[0] > 0.5),
            float(data.sensor("right_foot_floor_found").data[0] > 0.5),
        ],
        dtype=np.float64,
    )
    normalized_depth = np.clip(
        depth,
        MINIMUM_STEREO_DEPTH_M,
        MAXIMUM_STEREO_DEPTH_M,
    ) / MAXIMUM_STEREO_DEPTH_M
    values = np.hstack(
        (
            normalized_depth,
            min(distance, 10.0) / 10.0,
            math.sin(bearing),
            math.cos(bearing),
            torso_orientation,
            np.clip(torso_gyro / 5.0, -1.0, 1.0),
            np.clip(base_velocity / 2.0, -1.0, 1.0),
            min(float(np.mean(effort)), 100.0) / 100.0,
            min(float(np.max(effort)), 100.0) / 100.0,
            foot_contacts,
            min(payload_metrics.current_tray_tilt_degrees, 12.0) / 12.0,
            float(payload_metrics.package_slipped),
            previous_action.forward_speed_mps / 0.4,
            previous_action.turning_rate_rad_s / 0.5,
            previous_action.stop_probability,
        )
    ).astype(np.float32)
    return NavigationObservation(
        values=values,
        destination_distance_m=distance,
        destination_bearing_rad=bearing,
    )
