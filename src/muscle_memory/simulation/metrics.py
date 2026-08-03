"""Episode metrics measured directly from composed MuJoCo state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco  # type: ignore[import-untyped]
import numpy as np

from muscle_memory.simulation.world_scene import EpisodeScene
from muscle_memory.telemetry.models import SignalUseLabel

PACKAGE_SLIP_THRESHOLD_M = 0.02
GEOM_DISTANCE_LIMIT_M = 20.0


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Cumulative safety metrics and current payload state."""

    body_collisions: int
    minimum_obstacle_clearance_m: float
    current_obstacle_clearance_m: float
    maximum_tray_tilt_degrees: float
    current_tray_tilt_degrees: float
    package_slipped: bool
    signal_use: SignalUseLabel = SignalUseLabel.SIMULATOR_GROUND_TRUTH


class EpisodeMetricsTracker:
    """Tracks contact transitions, exact geom distance, and payload pose."""

    def __init__(self, scene: EpisodeScene, data: mujoco.MjData) -> None:
        self._scene = scene
        self._active_collision_pairs: set[tuple[int, int]] = set()
        self._body_collisions = 0
        self._minimum_clearance = math.inf
        self._maximum_tray_tilt = 0.0
        self._package_slipped = False
        tray = data.body(scene.tray_body_id)
        package = data.body(scene.package_body_id)
        tray_rotation = np.asarray(tray.xmat, dtype=np.float64).reshape(3, 3)
        self._expected_package_offset = tray_rotation.T @ (
            np.asarray(package.xpos, dtype=np.float64)
            - np.asarray(tray.xpos, dtype=np.float64)
        )

    def _collision_pairs(self, data: mujoco.MjData) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            if (
                first in self._scene.robot_geom_ids
                and second in self._scene.obstacle_geom_ids
            ) or (
                second in self._scene.robot_geom_ids
                and first in self._scene.obstacle_geom_ids
            ):
                pairs.add((min(first, second), max(first, second)))
        return pairs

    def _clearance(self, data: mujoco.MjData) -> float:
        closest = GEOM_DISTANCE_LIMIT_M
        segment = np.zeros(6, dtype=np.float64)
        for robot_geom in self._scene.robot_geom_ids:
            for obstacle_geom in self._scene.obstacle_geom_ids:
                signed_distance = float(
                    mujoco.mj_geomDistance(
                        self._scene.model,
                        data,
                        robot_geom,
                        obstacle_geom,
                        GEOM_DISTANCE_LIMIT_M,
                        segment,
                    )
                )
                # MuJoCo can report zero for a separated pair while still returning
                # the valid closest-point segment. Preserve negative penetration,
                # otherwise use the geometric segment length.
                distance = (
                    signed_distance
                    if signed_distance < 0.0
                    or signed_distance >= GEOM_DISTANCE_LIMIT_M
                    else float(np.linalg.norm(segment[3:] - segment[:3]))
                )
                closest = min(closest, distance)
        return closest

    def observe(self, data: mujoco.MjData) -> EpisodeMetrics:
        """Sample the current state and update cumulative extrema/events."""
        active_pairs = self._collision_pairs(data)
        self._body_collisions += len(active_pairs - self._active_collision_pairs)
        self._active_collision_pairs = active_pairs

        clearance = self._clearance(data)
        self._minimum_clearance = min(self._minimum_clearance, clearance)

        tray = data.body(self._scene.tray_body_id)
        package = data.body(self._scene.package_body_id)
        tray_rotation = np.asarray(tray.xmat, dtype=np.float64).reshape(3, 3)
        tray_up_z = float(np.clip(tray_rotation[2, 2], -1.0, 1.0))
        tray_tilt = math.degrees(math.acos(tray_up_z))
        self._maximum_tray_tilt = max(self._maximum_tray_tilt, tray_tilt)

        package_offset = tray_rotation.T @ (
            np.asarray(package.xpos, dtype=np.float64)
            - np.asarray(tray.xpos, dtype=np.float64)
        )
        self._package_slipped = self._package_slipped or bool(
            np.linalg.norm(package_offset - self._expected_package_offset)
            > PACKAGE_SLIP_THRESHOLD_M
        )
        return EpisodeMetrics(
            body_collisions=self._body_collisions,
            minimum_obstacle_clearance_m=self._minimum_clearance,
            current_obstacle_clearance_m=clearance,
            maximum_tray_tilt_degrees=self._maximum_tray_tilt,
            current_tray_tilt_degrees=tray_tilt,
            package_slipped=self._package_slipped,
        )
