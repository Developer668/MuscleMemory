"""Synchronized virtual stereo capture and truthful eight-category sensor rail."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from muscle_memory.simulation.metrics import EpisodeMetrics
from muscle_memory.telemetry.models import (
    SensorCategory,
    SensorReading,
    SensorSnapshot,
    SignalUseLabel,
)

LEFT_EYE_LOCAL_POSITION = np.array([0.2, 0.035, 0.2], dtype=np.float64)
RIGHT_EYE_LOCAL_POSITION = np.array([0.2, -0.035, 0.2], dtype=np.float64)
MINIMUM_DEPTH_SECTORS = 32
MAXIMUM_DEPTH_SECTORS = 64
DEFAULT_DEPTH_SECTORS = 48
STEREO_BASELINE_M = 0.07
MINIMUM_STEREO_DEPTH_M = 0.2
MAXIMUM_STEREO_DEPTH_M = 8.0

SENSOR_USE_LABELS = {
    SensorCategory.STEREO_VISION_AND_DEPTH: SignalUseLabel.USED_BY_POLICY,
    SensorCategory.LINKWISE_IMUS: SignalUseLabel.USED_BY_POLICY,
    SensorCategory.JOINT_POSITION_AND_EFFORT: SignalUseLabel.USED_BY_POLICY,
    SensorCategory.FOOT_CONTACTS: SignalUseLabel.USED_BY_POLICY,
    SensorCategory.WRIST_FORCE_AND_TRAY_BALANCE: SignalUseLabel.USED_BY_POLICY,
    SensorCategory.HAND_PRESSURE_AND_SLIP: SignalUseLabel.USED_BY_POLICY,
    SensorCategory.MICROPHONE_ACTIVITY: SignalUseLabel.LOGGED_ONLY,
    SensorCategory.BATTERY_AND_ENERGY: SignalUseLabel.LOGGED_ONLY,
}


@dataclass(frozen=True, slots=True)
class StereoFrameBundle:
    """All visual products synchronized by one and only one frame ID."""

    frame_id: str
    left_eye_rgb: npt.NDArray[np.uint8]
    right_eye_rgb: npt.NDArray[np.uint8]
    stereo_composite: npt.NDArray[np.uint8]
    derived_depth_sectors: npt.NDArray[np.float64] | None
    simulator_debug_depth: npt.NDArray[np.float32]
    simulator_debug_segmentation: npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class PolicyStereoFrame:
    """Only the synchronized stereo products reachable by the task policy."""

    frame_id: str
    left_eye_rgb: npt.NDArray[np.uint8]
    right_eye_rgb: npt.NDArray[np.uint8]
    derived_depth_sectors: npt.NDArray[np.float64]


class StereoDepthEstimator:
    """Derive navigation sectors from rectified stereo RGB using OpenCV SGBM."""

    def __init__(
        self,
        sector_count: int = DEFAULT_DEPTH_SECTORS,
        *,
        baseline_m: float = STEREO_BASELINE_M,
        minimum_depth_m: float = MINIMUM_STEREO_DEPTH_M,
        maximum_depth_m: float = MAXIMUM_STEREO_DEPTH_M,
    ) -> None:
        if not MINIMUM_DEPTH_SECTORS <= sector_count <= MAXIMUM_DEPTH_SECTORS:
            raise ValueError("stereo-derived depth must contain 32 to 64 sectors")
        if not math.isfinite(baseline_m) or baseline_m <= 0.0:
            raise ValueError("stereo baseline must be positive and finite")
        if (
            not math.isfinite(minimum_depth_m)
            or not math.isfinite(maximum_depth_m)
            or minimum_depth_m <= 0.0
            or maximum_depth_m <= minimum_depth_m
        ):
            raise ValueError("stereo depth bounds must be finite and increasing")
        self.sector_count = sector_count
        self.baseline_m = baseline_m
        self.minimum_depth_m = minimum_depth_m
        self.maximum_depth_m = maximum_depth_m
        self._matchers: dict[int, cv2.StereoSGBM] = {}

    def _matcher(self, width: int) -> cv2.StereoSGBM:
        matcher = self._matchers.get(width)
        if matcher is not None:
            return matcher
        maximum_disparity = min(128, max(16, (width // 4 // 16) * 16))
        matcher = cast(
            cv2.StereoSGBM,
            cv2.StereoSGBM_create(  # type: ignore[attr-defined]
                minDisparity=0,
                numDisparities=maximum_disparity,
                blockSize=5,
                P1=8 * 5 * 5,
                P2=32 * 5 * 5,
                disp12MaxDiff=1,
                uniquenessRatio=8,
                speckleWindowSize=50,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            ),
        )
        self._matchers[width] = matcher
        return matcher

    def estimate(
        self,
        left_rgb: npt.NDArray[np.uint8],
        right_rgb: npt.NDArray[np.uint8],
        *,
        vertical_fov_degrees: float,
    ) -> npt.NDArray[np.float64]:
        """Return nearest robust stereo depth per horizontal navigation sector."""
        if (
            left_rgb.shape != right_rgb.shape
            or left_rgb.ndim != 3
            or left_rgb.shape[2] != 3
            or left_rgb.dtype != np.uint8
            or right_rgb.dtype != np.uint8
        ):
            raise ValueError("stereo inputs must be matching uint8 RGB images")
        if not 0.0 < vertical_fov_degrees < 180.0:
            raise ValueError("vertical field of view must be between 0 and 180 degrees")

        height, width, _ = left_rgb.shape
        if width < 64 or height < 32:
            raise ValueError("stereo inputs must be at least 64 by 32 pixels")
        left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        disparity = self._matcher(width).compute(left_gray, right_gray).astype(np.float32) / 16.0
        focal_px = (height / 2.0) / math.tan(math.radians(vertical_fov_degrees) / 2.0)
        valid = disparity > 0.5
        depth = np.full(disparity.shape, np.nan, dtype=np.float64)
        depth[valid] = focal_px * self.baseline_m / disparity[valid]

        upper = max(0, round(height * 0.15))
        lower = min(height, round(height * 0.9))
        sectors = np.full(self.sector_count, self.maximum_depth_m, dtype=np.float64)
        column_edges = np.linspace(0, width, self.sector_count + 1, dtype=np.int32)
        for index in range(self.sector_count):
            sector_depth = depth[upper:lower, column_edges[index] : column_edges[index + 1]]
            usable = sector_depth[
                np.isfinite(sector_depth)
                & (sector_depth >= self.minimum_depth_m)
                & (sector_depth <= self.maximum_depth_m)
            ]
            if usable.size:
                sectors[index] = float(np.percentile(usable, 20.0))
        return sectors


def _virtual_camera(
    data: mujoco.MjData, local_position: npt.NDArray[np.float64]
) -> mujoco.MjvCamera:
    torso = data.body("torso_link")
    rotation = np.asarray(torso.xmat, dtype=np.float64).reshape(3, 3)
    eye = np.asarray(torso.xpos, dtype=np.float64) + rotation @ local_position
    forward = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    horizontal = math.hypot(float(forward[0]), float(forward[1]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = eye + forward
    camera.distance = 1.0
    camera.azimuth = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
    camera.elevation = -math.degrees(math.atan2(float(forward[2]), horizontal))
    return camera


def third_person_camera(
    data: mujoco.MjData,
    *,
    world_width_m: float,
    world_depth_m: float,
) -> mujoco.MjvCamera:
    """Frame the robot from the room interior so boundary walls cannot occlude it."""
    if world_width_m <= 0.0 or world_depth_m <= 0.0:
        raise ValueError("world dimensions must be positive")
    target = np.asarray(data.qpos[:3], dtype=np.float64).copy()
    target[2] = max(0.8, float(target[2]))
    toward_centre = np.array(
        [world_width_m / 2.0 - target[0], world_depth_m / 2.0 - target[1]],
        dtype=np.float64,
    )
    centre_distance = float(np.linalg.norm(toward_centre))
    if centre_distance < 0.25:
        toward_centre[:] = [1.0, 1.0]
        centre_distance = math.sqrt(2.0)
    look_direction = -toward_centre / centre_distance
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = min(3.0, max(2.2, centre_distance * 0.7))
    camera.azimuth = math.degrees(
        math.atan2(float(look_direction[1]), float(look_direction[0]))
    )
    camera.elevation = -28.0
    return camera


def overview_camera(
    *,
    world_width_m: float,
    world_depth_m: float,
) -> mujoco.MjvCamera:
    """Frame the complete validated room from a deterministic elevated viewpoint."""
    if world_width_m <= 0.0 or world_depth_m <= 0.0:
        raise ValueError("world dimensions must be positive")
    room_diagonal = math.hypot(world_width_m, world_depth_m)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [world_width_m / 2.0, world_depth_m / 2.0, 0.35]
    camera.distance = max(4.0, room_diagonal)
    camera.azimuth = 225.0
    camera.elevation = -72.0
    return camera


class EpisodeSensorExtractor:
    """Extract existing candidate signals without mutating its frozen sensor set."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model = model
        self._data = data
        actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
        self._joint_qpos_addresses = np.asarray(
            model.jnt_qposadr[actuator_joint_ids], dtype=np.int32
        )
        self._joint_dof_addresses = np.asarray(
            model.jnt_dofadr[actuator_joint_ids], dtype=np.int32
        )
        self._energy_joules = 0.0
        self._last_time = float(data.time)
        self._last_power_watts = self._instantaneous_power_watts()
        self._depth_estimators: dict[int, StereoDepthEstimator] = {}

    def _instantaneous_power_watts(self) -> float:
        return float(
            np.sum(
                np.abs(
                    self._data.actuator_force
                    * self._data.qvel[self._joint_dof_addresses]
                )
            )
        )

    def _rgb(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera) -> npt.NDArray[np.uint8]:
        renderer.update_scene(self._data, camera=camera)
        return np.asarray(renderer.render(), dtype=np.uint8).copy()

    def capture_stereo(
        self,
        renderer: mujoco.Renderer,
        frame_id: str,
        *,
        depth_sector_count: int = DEFAULT_DEPTH_SECTORS,
    ) -> StereoFrameBundle:
        """Capture RGB and debug products with a shared frame identity.

        Simulator depth is deliberately kept separate from stereo-derived depth.
        """
        policy_frame = self.capture_policy_stereo(
            renderer,
            frame_id,
            depth_sector_count=depth_sector_count,
        )
        left_camera = _virtual_camera(self._data, LEFT_EYE_LOCAL_POSITION)

        renderer.enable_depth_rendering()
        try:
            renderer.update_scene(self._data, camera=left_camera)
            debug_depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        finally:
            renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(self._data, camera=left_camera)
            debug_segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
        finally:
            renderer.disable_segmentation_rendering()

        return StereoFrameBundle(
            frame_id=frame_id,
            left_eye_rgb=policy_frame.left_eye_rgb,
            right_eye_rgb=policy_frame.right_eye_rgb,
            stereo_composite=np.concatenate(
                (policy_frame.left_eye_rgb, policy_frame.right_eye_rgb), axis=1
            ),
            derived_depth_sectors=policy_frame.derived_depth_sectors,
            simulator_debug_depth=debug_depth,
            simulator_debug_segmentation=debug_segmentation,
        )

    def capture_policy_stereo(
        self,
        renderer: mujoco.Renderer,
        frame_id: str,
        *,
        depth_sector_count: int = DEFAULT_DEPTH_SECTORS,
    ) -> PolicyStereoFrame:
        """Capture only RGB and stereo-derived depth available to the task policy."""
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        if not MINIMUM_DEPTH_SECTORS <= depth_sector_count <= MAXIMUM_DEPTH_SECTORS:
            raise ValueError("stereo-derived depth must contain 32 to 64 sectors")
        left_camera = _virtual_camera(self._data, LEFT_EYE_LOCAL_POSITION)
        right_camera = _virtual_camera(self._data, RIGHT_EYE_LOCAL_POSITION)
        left_rgb = self._rgb(renderer, left_camera)
        right_rgb = self._rgb(renderer, right_camera)
        estimator = self._depth_estimators.setdefault(
            depth_sector_count, StereoDepthEstimator(depth_sector_count)
        )
        derived_depth_sectors = estimator.estimate(
            left_rgb,
            right_rgb,
            vertical_fov_degrees=float(self._model.vis.global_.fovy),
        )
        return PolicyStereoFrame(
            frame_id=frame_id,
            left_eye_rgb=left_rgb,
            right_eye_rgb=right_rgb,
            derived_depth_sectors=derived_depth_sectors,
        )

    def snapshot(
        self,
        frame_id: str,
        derived_depth_sectors: npt.NDArray[np.float64] | None = None,
        payload_metrics: EpisodeMetrics | None = None,
    ) -> SensorSnapshot:
        """Build the complete sensor rail for the same frame join key."""
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        if derived_depth_sectors is not None and not (
            MINIMUM_DEPTH_SECTORS <= derived_depth_sectors.size <= MAXIMUM_DEPTH_SECTORS
        ):
            raise ValueError("stereo-derived depth must contain 32 to 64 sectors")

        power_watts = self._instantaneous_power_watts()
        now = float(self._data.time)
        elapsed = max(0.0, now - self._last_time)
        self._energy_joules += elapsed * (power_watts + self._last_power_watts) / 2.0
        self._last_time = now
        self._last_power_watts = power_watts

        stereo_values = {
            "frame_id": frame_id,
            "left_eye_rgb": "video_stream",
            "right_eye_rgb": "video_stream",
            "stereo_composite": "video_stream",
            "derived_depth_sectors": (
                None
                if derived_depth_sectors is None
                else np.asarray(derived_depth_sectors, dtype=np.float64).tolist()
            ),
        }
        imu_values = {
            "pelvis": {
                "accelerometer_mps2": self._data.sensor("accelerometer_pelvis").data.tolist(),
                "angular_velocity_rad_s": self._data.sensor("gyro_pelvis").data.tolist(),
                "orientation_wxyz": self._data.sensor("orientation_pelvis").data.tolist(),
            },
            "torso": {
                "accelerometer_mps2": self._data.sensor("accelerometer_torso").data.tolist(),
                "angular_velocity_rad_s": self._data.sensor("gyro_torso").data.tolist(),
                "orientation_wxyz": self._data.sensor("orientation_torso").data.tolist(),
            },
            "head": None,
            "selected_limbs": None,
        }
        joint_values = {
            "position_rad": self._data.qpos[self._joint_qpos_addresses].tolist(),
            "velocity_rad_s": self._data.qvel[self._joint_dof_addresses].tolist(),
            "actuator_effort": self._data.actuator_force.tolist(),
        }
        foot_values = {
            "left_force_n": self._data.sensor("left_foot_force").data.tolist(),
            "right_force_n": self._data.sensor("right_foot_force").data.tolist(),
            "left_floor_contact": bool(self._data.sensor("left_foot_floor_found").data[0]),
            "right_floor_contact": bool(
                self._data.sensor("right_foot_floor_found").data[0]
            ),
            "centre_of_pressure": None,
        }
        return SensorSnapshot(
            stereo_vision_and_depth=SensorReading.available_reading(
                SensorCategory.STEREO_VISION_AND_DEPTH,
                SENSOR_USE_LABELS[SensorCategory.STEREO_VISION_AND_DEPTH],
                stereo_values,
            ),
            linkwise_imus=SensorReading.available_reading(
                SensorCategory.LINKWISE_IMUS,
                SENSOR_USE_LABELS[SensorCategory.LINKWISE_IMUS],
                imu_values,
            ),
            joint_position_and_effort=SensorReading.available_reading(
                SensorCategory.JOINT_POSITION_AND_EFFORT,
                SENSOR_USE_LABELS[SensorCategory.JOINT_POSITION_AND_EFFORT],
                joint_values,
            ),
            foot_contacts=SensorReading.available_reading(
                SensorCategory.FOOT_CONTACTS,
                SENSOR_USE_LABELS[SensorCategory.FOOT_CONTACTS],
                foot_values,
            ),
            wrist_force_and_tray_balance=(
                SensorReading.unavailable(
                    SensorCategory.WRIST_FORCE_AND_TRAY_BALANCE,
                    SENSOR_USE_LABELS[SensorCategory.WRIST_FORCE_AND_TRAY_BALANCE],
                )
                if payload_metrics is None
                else SensorReading.available_reading(
                    SensorCategory.WRIST_FORCE_AND_TRAY_BALANCE,
                    SENSOR_USE_LABELS[SensorCategory.WRIST_FORCE_AND_TRAY_BALANCE],
                    {
                        "six_axis_force_torque": None,
                        "tray_tilt_degrees": payload_metrics.current_tray_tilt_degrees,
                        "maximum_tray_tilt_degrees": (
                            payload_metrics.maximum_tray_tilt_degrees
                        ),
                    },
                )
            ),
            hand_pressure_and_slip=(
                SensorReading.unavailable(
                    SensorCategory.HAND_PRESSURE_AND_SLIP,
                    SENSOR_USE_LABELS[SensorCategory.HAND_PRESSURE_AND_SLIP],
                )
                if payload_metrics is None
                else SensorReading.available_reading(
                    SensorCategory.HAND_PRESSURE_AND_SLIP,
                    SENSOR_USE_LABELS[SensorCategory.HAND_PRESSURE_AND_SLIP],
                    {
                        "pressure": None,
                        "approximate_shear": None,
                        "package_slipped": payload_metrics.package_slipped,
                    },
                )
            ),
            microphone_activity=SensorReading.unavailable(
                SensorCategory.MICROPHONE_ACTIVITY,
                SENSOR_USE_LABELS[SensorCategory.MICROPHONE_ACTIVITY],
            ),
            battery_and_energy=SensorReading.available_reading(
                SensorCategory.BATTERY_AND_ENERGY,
                SENSOR_USE_LABELS[SensorCategory.BATTERY_AND_ENERGY],
                {
                    "charge_percent": None,
                    "estimated_energy_joules": self._energy_joules,
                    "power_draw_watts": power_watts,
                },
            ),
        )
