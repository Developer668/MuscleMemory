"""Real MuJoCo live episode runner with separated telemetry and video paths."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import cv2
import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from muscle_memory.episodes.models import (
    EpisodeAbort,
    EpisodeAppendReceipt,
    EpisodeClosure,
    EpisodeIdentity,
)
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.evaluation.success import (
    EpisodeOutcome,
    Vector2,
    evaluate_safe_delivery,
)
from muscle_memory.graph_memory import WorldSplit
from muscle_memory.live.models import (
    BatteryEvent,
    CollisionEvent,
    CompletionEvent,
    ContactEvent,
    EncodedVideoProduct,
    EvaluatedPolicySelection,
    ImuEvent,
    JointEffortEvent,
    LiveEpisodeConfig,
    LiveEpisodeHealth,
    LiveTelemetryPayload,
    PolicyActionEvent,
    RewardProgressEvent,
    SimulatorPoseEvent,
    TactileSlipEvent,
    TrayStateEvent,
    ValidatedTrainingWorldEnvelope,
    VideoFrameMetadata,
    VideoFrameSet,
    VideoProduct,
    require_validated_training_world,
)
from muscle_memory.live.video import BoundedVideoService
from muscle_memory.policy.observation import navigation_observation
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import PHYSICS_HZ
from muscle_memory.simulation.metrics import EpisodeMetrics, EpisodeMetricsTracker
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.sensors import (
    MAXIMUM_STEREO_DEPTH_M,
    MINIMUM_STEREO_DEPTH_M,
    EpisodeSensorExtractor,
    third_person_camera,
)
from muscle_memory.simulation.world_scene import assemble_episode_scene
from muscle_memory.telemetry import EpisodeTelemetryRecord, SignalUseLabel

STOPPED_SPEED_MPS = 0.05
STOP_SETTLE_SECONDS = 1.0
MINIMUM_CLEARANCE_M = 0.25
MAXIMUM_TRAY_TILT_DEGREES = 12.0
_EPSILON = 1e-9


class LiveEpisodeLifecycle(Protocol):
    """Existing durable episode boundary consumed by the worker."""

    async def open_episode(self, identity: EpisodeIdentity) -> EpisodeIdentity: ...

    async def append_telemetry(self, record: EpisodeTelemetryRecord) -> EpisodeAppendReceipt: ...

    async def close_episode(
        self,
        result: PolicyEpisodeResult,
        *,
        closed_at: datetime | None = None,
    ) -> EpisodeClosure: ...

    async def abort_episode(
        self,
        episode_id: str,
        *,
        error_type: str,
        aborted_at: datetime | None = None,
    ) -> EpisodeAbort: ...


@dataclass(frozen=True, slots=True)
class LiveRunProgress:
    simulation_time_seconds: float
    wall_elapsed_seconds: float
    wall_clock_lag_seconds: float
    telemetry_records: int
    video_frames: int
    dropped_video_frames: int
    last_frame_id: str | None
    provider_state: str | None
    health: LiveEpisodeHealth


@dataclass(frozen=True, slots=True)
class LiveRunResult:
    result: PolicyEpisodeResult
    closure: EpisodeClosure
    completion_reason: str
    progress: LiveRunProgress


def _float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise RuntimeError("sensor payload was not a numeric list")
    return tuple(float(item) for item in value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("sensor payload was not an object")
    return cast(Mapping[str, Any], value)


def _fallen(data: mujoco.MjData) -> bool:
    return bool(
        float(data.qpos[2]) <= 0.35
        or float(data.sensor("upvector_torso").data[2]) <= 0.0
    )


def _encode_jpeg(
    product: VideoProduct,
    rgb: npt.NDArray[np.uint8],
    *,
    quality: int,
) -> EncodedVideoProduct:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError("video product must be uint8 RGB")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError(f"could not encode {product.value} frame")
    data = encoded.tobytes()
    return EncodedVideoProduct(
        product=product,
        mime_type="image/jpeg",
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def _depth_visual(
    sectors: npt.NDArray[np.float64],
    *,
    width: int,
    height: int,
) -> npt.NDArray[np.uint8]:
    columns = np.interp(
        np.linspace(0.0, float(sectors.size - 1), width),
        np.arange(sectors.size, dtype=np.float64),
        sectors,
    )
    normalized = np.clip(
        (columns - MINIMUM_STEREO_DEPTH_M)
        / (MAXIMUM_STEREO_DEPTH_M - MINIMUM_STEREO_DEPTH_M),
        0.0,
        1.0,
    )
    intensity = np.tile(np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8), (height, 1))
    bgr = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _segmentation_visual(
    segmentation: npt.NDArray[np.int32],
) -> npt.NDArray[np.uint8]:
    object_ids = segmentation[:, :, 0].astype(np.int64)
    object_types = segmentation[:, :, 1].astype(np.int64)
    visible = object_ids >= 0
    rgb = np.zeros((*object_ids.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.where(visible, (object_ids * 53 + 47) % 255, 14)
    rgb[:, :, 1] = np.where(visible, (object_ids * 97 + object_types * 29 + 31) % 255, 18)
    rgb[:, :, 2] = np.where(visible, (object_ids * 193 + 79) % 255, 22)
    return rgb


def _safety_markers(
    metrics: EpisodeMetrics,
    *,
    new_collisions: int,
    new_falls: int,
) -> tuple[str, ...]:
    markers: list[str] = []
    if new_falls:
        markers.append("fell")
    if new_collisions:
        markers.append("body_collision")
    if metrics.current_obstacle_clearance_m < MINIMUM_CLEARANCE_M:
        markers.append("insufficient_obstacle_clearance")
    if metrics.current_tray_tilt_degrees >= MAXIMUM_TRAY_TILT_DEGREES:
        markers.append("excessive_tray_tilt")
    if metrics.package_slipped:
        markers.append("package_slipped")
    return tuple(markers)


class LiveEpisodeRunner:
    """Run one validated training episode without any teacher or graph control access."""

    def __init__(
        self,
        *,
        lifecycle: LiveEpisodeLifecycle,
        video: BoundedVideoService,
        config: LiveEpisodeConfig,
    ) -> None:
        self._lifecycle = lifecycle
        self._video = video
        self._config = config

    def run(
        self,
        *,
        episode_id: str,
        world: ValidatedTrainingWorldEnvelope,
        selection: EvaluatedPolicySelection,
        cancel_requested: Callable[[], bool],
        on_progress: Callable[[LiveRunProgress], None],
    ) -> LiveRunResult:
        return asyncio.run(
            self._run_async(
                episode_id=episode_id,
                world=world,
                selection=selection,
                cancel_requested=cancel_requested,
                on_progress=on_progress,
            )
        )

    async def _run_async(
        self,
        *,
        episode_id: str,
        world: ValidatedTrainingWorldEnvelope,
        selection: EvaluatedPolicySelection,
        cancel_requested: Callable[[], bool],
        on_progress: Callable[[LiveRunProgress], None],
    ) -> LiveRunResult:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        training_world = require_validated_training_world(world)
        selection.verify_integrity()

        scene = assemble_episode_scene(world)
        simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
        metrics_tracker = EpisodeMetricsTracker(scene, simulation.data)
        metrics = metrics_tracker.observe(simulation.data)
        sensor_extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
        identity = EpisodeIdentity(
            episode_id=episode_id,
            robot_checksum=scene.robot_checksum,
            world_id=training_world.world_id,
            world_hash=scene.world_hash,
            world_split=WorldSplit.TRAINING,
            policy_id=selection.policy.policy_id,
            policy_hash=selection.policy.policy_hash,
            opened_at=datetime.now(UTC),
        )
        await self._lifecycle.open_episode(identity)

        wall_started = time.monotonic()
        path_length = 0.0
        energy_joules = 0.0
        fall_count = 0
        was_fallen = False
        reached_at: float | None = None
        previous_position = np.asarray(simulation.data.qpos[:2], dtype=np.float64).copy()
        previous_tick_distance = math.dist(
            (float(previous_position[0]), float(previous_position[1])),
            (float(scene.world.destination.x), float(scene.world.destination.y)),
        )
        previous_collision_count = 0
        previous_fall_count = 0
        cumulative_reward = 0.0
        current_action = TaskCommand(0.0, 0.0, 1.0)
        latest_depth: npt.NDArray[np.float64] | None = None
        latest_frame_id: str | None = None
        pending_video: dict[int, list[VideoFrameMetadata]] = {}
        telemetry_sequence = 0
        video_index = 0
        termination_reason: str | None = None
        last_provider_state: str | None = None
        actuator_joint_ids = np.asarray(simulation.model.actuator_trnid[:, 0], dtype=np.int32)
        actuator_dof_addresses = np.asarray(
            simulation.model.jnt_dofadr[actuator_joint_ids], dtype=np.int32
        )

        with mujoco.Renderer(
            simulation.model,
            height=self._config.render_height,
            width=self._config.render_width,
        ) as renderer:

            def capture_video(index: int) -> None:
                nonlocal latest_depth, latest_frame_id
                scheduled_time = index / 30.0
                frame_id = f"{episode_id}:video:{index:08d}"
                stereo = sensor_extractor.capture_stereo(renderer, frame_id)
                camera = third_person_camera(
                    simulation.data,
                    world_width_m=float(scene.world.template.width_m),
                    world_depth_m=float(scene.world.template.depth_m),
                )
                renderer.update_scene(simulation.data, camera=camera)
                third_person = np.asarray(renderer.render(), dtype=np.uint8).copy()
                if stereo.derived_depth_sectors is None:
                    raise RuntimeError("policy stereo capture did not derive depth sectors")
                latest_depth = stereo.derived_depth_sectors.copy()
                latest_frame_id = frame_id
                products = (
                    _encode_jpeg(
                        VideoProduct.THIRD_PERSON,
                        third_person,
                        quality=self._config.jpeg_quality,
                    ),
                    _encode_jpeg(
                        VideoProduct.LEFT_EYE_RGB,
                        stereo.left_eye_rgb,
                        quality=self._config.jpeg_quality,
                    ),
                    _encode_jpeg(
                        VideoProduct.RIGHT_EYE_RGB,
                        stereo.right_eye_rgb,
                        quality=self._config.jpeg_quality,
                    ),
                    _encode_jpeg(
                        VideoProduct.STEREO_COMPOSITE,
                        stereo.stereo_composite,
                        quality=self._config.jpeg_quality,
                    ),
                    _encode_jpeg(
                        VideoProduct.DERIVED_DEPTH,
                        _depth_visual(
                            latest_depth,
                            width=self._config.render_width,
                            height=self._config.render_height,
                        ),
                        quality=self._config.jpeg_quality,
                    ),
                    _encode_jpeg(
                        VideoProduct.SIMULATOR_DEBUG_SEGMENTATION,
                        _segmentation_visual(stereo.simulator_debug_segmentation),
                        quality=self._config.jpeg_quality,
                    ),
                )
                assigned_sequence = max(
                    0,
                    math.ceil(scheduled_time * 20.0 - _EPSILON),
                )
                metadata = VideoFrameMetadata(
                    frame_id=frame_id,
                    frame_index=index,
                    scheduled_time_seconds=scheduled_time,
                    captured_time_seconds=float(simulation.data.time),
                    telemetry_sequence=assigned_sequence,
                    products=tuple(
                        item.metadata(episode_id=episode_id, frame_index=index)
                        for item in products
                    ),
                )
                self._video.append(
                    episode_id,
                    VideoFrameSet(metadata=metadata, products=products),
                )
                pending_video.setdefault(assigned_sequence, []).append(metadata)

            capture_video(video_index)
            video_index += 1
            assert latest_depth is not None
            current_action = selection.policy.command(
                navigation_observation(
                    simulation.data,
                    scene.world.destination,
                    latest_depth,
                    metrics,
                    current_action,
                )
            )
            if not isinstance(current_action, TaskCommand):
                raise TypeError("task policy must return TaskCommand")

            def decide(_time: float) -> TaskCommand:
                nonlocal current_action
                if termination_reason is not None:
                    current_action = TaskCommand(0.0, 0.0, 1.0)
                    return current_action
                if simulation.task_policy_updates == 0:
                    return current_action
                if latest_depth is None:
                    raise RuntimeError("task policy has no synchronized stereo frame")
                current_action = selection.policy.command(
                    navigation_observation(
                        simulation.data,
                        scene.world.destination,
                        latest_depth,
                        metrics,
                        current_action,
                    )
                )
                if not isinstance(current_action, TaskCommand):
                    raise TypeError("task policy must return TaskCommand")
                return current_action

            while True:
                simulation_time = float(simulation.data.time)
                if cancel_requested() and termination_reason is None:
                    termination_reason = "cancelled"
                    current_action = TaskCommand(0.0, 0.0, 1.0)

                while video_index / 30.0 <= simulation_time + _EPSILON:
                    capture_video(video_index)
                    video_index += 1

                expected_telemetry_time = telemetry_sequence / 20.0
                if expected_telemetry_time <= simulation_time + _EPSILON:
                    new_collisions = metrics.body_collisions - previous_collision_count
                    new_falls = fall_count - previous_fall_count
                    distance = math.dist(
                        (float(simulation.data.qpos[0]), float(simulation.data.qpos[1])),
                        (float(scene.world.destination.x), float(scene.world.destination.y)),
                    )
                    tick_progress = previous_tick_distance - distance
                    tick_reward = (
                        tick_progress
                        - float(new_collisions)
                        - 5.0 * float(new_falls)
                        - float(metrics.package_slipped)
                        - max(0.0, MINIMUM_CLEARANCE_M - metrics.current_obstacle_clearance_m)
                    )
                    cumulative_reward += tick_reward
                    markers = _safety_markers(
                        metrics,
                        new_collisions=new_collisions,
                        new_falls=new_falls,
                    )
                    if termination_reason is None and (
                        simulation_time + _EPSILON >= self._config.maximum_duration_seconds
                    ):
                        termination_reason = "time_limit"
                    frame_metadata = tuple(pending_video.pop(telemetry_sequence, ()))
                    if latest_frame_id is None:
                        raise RuntimeError("telemetry has no direct-video frame join")
                    sensors = sensor_extractor.snapshot(
                        latest_frame_id,
                        latest_depth,
                        payload_metrics=metrics,
                    )
                    imu = _mapping(sensors.linkwise_imus.values)
                    pelvis = _mapping(imu["pelvis"])
                    torso = _mapping(imu["torso"])
                    joints = _mapping(sensors.joint_position_and_effort.values)
                    contacts = _mapping(sensors.foot_contacts.values)
                    tray = _mapping(sensors.wrist_force_and_tray_balance.values)
                    battery = _mapping(sensors.battery_and_energy.values)
                    payload = LiveTelemetryPayload(
                        imu=ImuEvent(
                            pelvis_accelerometer_mps2=_float_tuple(
                                pelvis["accelerometer_mps2"]
                            ),
                            pelvis_angular_velocity_rad_s=_float_tuple(
                                pelvis["angular_velocity_rad_s"]
                            ),
                            pelvis_orientation_wxyz=_float_tuple(pelvis["orientation_wxyz"]),
                            torso_accelerometer_mps2=_float_tuple(
                                torso["accelerometer_mps2"]
                            ),
                            torso_angular_velocity_rad_s=_float_tuple(
                                torso["angular_velocity_rad_s"]
                            ),
                            torso_orientation_wxyz=_float_tuple(torso["orientation_wxyz"]),
                        ),
                        joint_effort=JointEffortEvent(
                            position_rad=_float_tuple(joints["position_rad"]),
                            velocity_rad_s=_float_tuple(joints["velocity_rad_s"]),
                            actuator_effort=_float_tuple(joints["actuator_effort"]),
                        ),
                        contacts=ContactEvent(
                            left_force_n=_float_tuple(contacts["left_force_n"]),
                            right_force_n=_float_tuple(contacts["right_force_n"]),
                            left_floor_contact=bool(contacts["left_floor_contact"]),
                            right_floor_contact=bool(contacts["right_floor_contact"]),
                        ),
                        tray_state=TrayStateEvent(
                            current_tilt_degrees=float(tray["tray_tilt_degrees"]),
                            maximum_tilt_degrees=float(tray["maximum_tray_tilt_degrees"]),
                        ),
                        tactile_slip=TactileSlipEvent(
                            package_slipped=metrics.package_slipped,
                            pressure_available=False,
                        ),
                        battery=BatteryEvent(
                            charge_percent=(
                                None
                                if battery["charge_percent"] is None
                                else float(battery["charge_percent"])
                            ),
                            estimated_energy_joules=float(battery["estimated_energy_joules"]),
                            power_draw_watts=float(battery["power_draw_watts"]),
                        ),
                        policy_action=PolicyActionEvent(
                            forward_speed_mps=current_action.forward_speed_mps,
                            turning_rate_rad_s=current_action.turning_rate_rad_s,
                            stop_probability=current_action.stop_probability,
                        ),
                        collisions=CollisionEvent(
                            body_collisions=metrics.body_collisions,
                            new_body_collisions=new_collisions,
                            current_clearance_m=metrics.current_obstacle_clearance_m,
                            minimum_clearance_m=metrics.minimum_obstacle_clearance_m,
                            falls=fall_count,
                            new_falls=new_falls,
                        ),
                        reward_progress=RewardProgressEvent(
                            destination_distance_m=distance,
                            progress_m=tick_progress,
                            tick_reward=tick_reward,
                            cumulative_reward=cumulative_reward,
                        ),
                        simulator_pose=SimulatorPoseEvent(
                            position_x_m=float(simulation.data.qpos[0]),
                            position_y_m=float(simulation.data.qpos[1]),
                            yaw_radians=float(
                                math.atan2(
                                    float(simulation.data.body("pelvis").xmat[3]),
                                    float(simulation.data.body("pelvis").xmat[0]),
                                )
                            ),
                        ),
                        safety_markers=markers,
                        completion=CompletionEvent(
                            completed=termination_reason is not None,
                            reason=termination_reason,
                        ),
                        video_frames=frame_metadata,
                    )
                    record = EpisodeTelemetryRecord.create(
                        episode_id=episode_id,
                        world_id=scene.world.world_id,
                        policy_id=selection.policy.policy_id,
                        sequence=telemetry_sequence,
                        sim_time_seconds=expected_telemetry_time,
                        robot_checksum=scene.robot_checksum,
                        policy_hash=selection.policy.policy_hash,
                        world_hash=scene.world_hash,
                        signal_use=SignalUseLabel.LOGGED_ONLY,
                        sensors=sensors,
                        payload=payload.as_json_value(),
                        failure_type=markers[0] if markers else None,
                        frame_id=latest_frame_id,
                    )
                    receipt = await self._lifecycle.append_telemetry(record)
                    last_provider_state = receipt.provider_state.value
                    telemetry_sequence += 1
                    previous_tick_distance = distance
                    previous_collision_count = metrics.body_collisions
                    previous_fall_count = fall_count
                    stats = self._video.stats(episode_id)
                    elapsed = time.monotonic() - wall_started
                    lag = max(0.0, elapsed - simulation_time)
                    health = (
                        LiveEpisodeHealth.DEGRADED
                        if (
                            lag > self._config.degraded_lag_seconds
                            or stats.dropped_frames
                            or last_provider_state
                            not in {"healthy", "end_to_end_verified"}
                        )
                        else LiveEpisodeHealth.HEALTHY
                    )
                    on_progress(
                        LiveRunProgress(
                            simulation_time_seconds=simulation_time,
                            wall_elapsed_seconds=elapsed,
                            wall_clock_lag_seconds=lag,
                            telemetry_records=telemetry_sequence,
                            video_frames=stats.appended_frames,
                            dropped_video_frames=stats.dropped_frames,
                            last_frame_id=latest_frame_id,
                            provider_state=last_provider_state,
                            health=health,
                        )
                    )
                    if termination_reason is not None:
                        break

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
                if simulation.step_index % (PHYSICS_HZ // 20) == 0:
                    metrics = metrics_tracker.observe(simulation.data)
                fallen = _fallen(simulation.data)
                if fallen and not was_fallen:
                    fall_count += 1
                was_fallen = fallen
                distance = math.dist(
                    (float(position[0]), float(position[1])),
                    (float(scene.world.destination.x), float(scene.world.destination.y)),
                )
                if reached_at is None and distance <= 0.5:
                    reached_at = float(simulation.data.time)
                speed = float(np.linalg.norm(simulation.data.qvel[:2]))
                if (
                    termination_reason is None
                    and reached_at is not None
                    and speed <= STOPPED_SPEED_MPS
                    and simulation.data.time >= reached_at + STOP_SETTLE_SECONDS
                ):
                    termination_reason = "settled_at_destination"
                if (
                    termination_reason is None
                    and simulation.data.time + _EPSILON
                    >= self._config.maximum_duration_seconds
                ):
                    termination_reason = "time_limit"
                if self._config.realtime:
                    target_elapsed = float(simulation.data.time)
                    remaining = target_elapsed - (time.monotonic() - wall_started)
                    if remaining > 0.0:
                        await asyncio.sleep(remaining)

        metrics = metrics_tracker.observe(simulation.data)
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
            body_collisions=metrics.body_collisions,
            minimum_obstacle_clearance_metres=metrics.minimum_obstacle_clearance_m,
            maximum_tray_tilt_degrees=metrics.maximum_tray_tilt_degrees,
            package_slipped=metrics.package_slipped,
            human_interventions=0,
        )
        evaluation = evaluate_safe_delivery(outcome)
        direct_distance = math.dist(
            (float(scene.world.start.x), float(scene.world.start.y)),
            (float(scene.world.destination.x), float(scene.world.destination.y)),
        )
        result = PolicyEpisodeResult(
            episode_id=episode_id,
            world_id=scene.world.world_id,
            world_seed=scene.world.seed,
            world_split=scene.world.split,
            world_hash=scene.world_hash,
            robot_checksum=scene.robot_checksum,
            policy_id=selection.policy.policy_id,
            policy_hash=selection.policy.policy_hash,
            success=evaluation.success,
            failed_reasons=tuple(reason.name for reason in evaluation.failed_reasons),
            time_to_resident_seconds=reached_at,
            simulated_duration_seconds=float(simulation.data.time),
            stop_distance_m=evaluation.stop_distance_metres,
            facing_error_degrees=evaluation.facing_error_degrees,
            stopped_speed_mps=stopped_speed,
            falls=fall_count,
            body_collisions=metrics.body_collisions,
            minimum_obstacle_clearance_m=metrics.minimum_obstacle_clearance_m,
            maximum_tray_tilt_degrees=metrics.maximum_tray_tilt_degrees,
            package_slipped=metrics.package_slipped,
            human_interventions=0,
            direct_distance_m=direct_distance,
            path_length_m=path_length,
            path_efficiency=(
                min(1.0, direct_distance / path_length) if path_length > 0.0 else 0.0
            ),
            energy_joules=energy_joules,
            task_policy_updates=simulation.task_policy_updates,
            trace=(),
        )
        closure = await self._lifecycle.close_episode(result, closed_at=datetime.now(UTC))
        stats = self._video.finish_episode(episode_id)
        elapsed = time.monotonic() - wall_started
        lag = max(0.0, elapsed - float(simulation.data.time))
        final_progress = LiveRunProgress(
            simulation_time_seconds=float(simulation.data.time),
            wall_elapsed_seconds=elapsed,
            wall_clock_lag_seconds=lag,
            telemetry_records=telemetry_sequence,
            video_frames=stats.appended_frames,
            dropped_video_frames=stats.dropped_frames,
            last_frame_id=latest_frame_id,
            provider_state=last_provider_state,
            health=LiveEpisodeHealth.TERMINAL,
        )
        return LiveRunResult(
            result=result,
            closure=closure,
            completion_reason=termination_reason or "time_limit",
            progress=final_progress,
        )


__all__ = [
    "LiveEpisodeLifecycle",
    "LiveEpisodeRunner",
    "LiveRunProgress",
    "LiveRunResult",
]
