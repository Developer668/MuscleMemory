"""Run fail-closed native MuJoCo qualification for a frozen 100 Hz gait policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import PHYSICS_HZ, robot_bundle_checksum
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.world_scene import assemble_payload_qualification_scene
from ops.controller.contract import (
    CONTROLLER_HZ,
    CTRL_DT_SECONDS,
    GAIT_ACTION_OUTPUTS,
    QualificationEvidence,
    RunMode,
    evaluate_qualification,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FALL_HEIGHT_M = 0.35
FALL_UP_Z = 0.0
PACKAGE_SLIP_THRESHOLD_M = 0.02
NON_FOOT_COLLISION_GEOMS = (
    "left_thigh",
    "left_shin",
    "right_thigh",
    "right_shin",
    "left_hand_collision",
    "right_hand_collision",
)

CommandSchedule = Callable[[float], TaskCommand]


@dataclass(frozen=True, slots=True)
class TrialResult:
    duration_seconds: float
    physics_steps: int
    controller_inferences: int
    controller_supervisor_ticks: int
    finite_state: bool
    fell: bool
    body_collisions: int
    planar_delta_m: tuple[float, float]
    cumulative_yaw_radians: float
    maximum_drift_m: float
    last_second_maximum_speed_mps: float
    maximum_tray_tilt_degrees: float
    package_slipped: bool
    final_qpos: npt.NDArray[np.float64]
    final_qvel: npt.NDArray[np.float64]


def _yaw(data: mujoco.MjData) -> float:
    rotation = np.asarray(data.body("pelvis").xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _collision_ids(model: mujoco.MjModel) -> tuple[frozenset[int], frozenset[int]]:
    robot_ids = frozenset(model.geom(name).id for name in NON_FOOT_COLLISION_GEOMS)
    world_ids = frozenset(
        geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == 0
    )
    return robot_ids, world_ids


def _active_body_collisions(
    data: mujoco.MjData,
    robot_geom_ids: frozenset[int],
    world_geom_ids: frozenset[int],
) -> set[tuple[int, int]]:
    active: set[tuple[int, int]] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if (first in robot_geom_ids and second in world_geom_ids) or (
            second in robot_geom_ids and first in world_geom_ids
        ):
            active.add((min(first, second), max(first, second)))
    return active


def _payload_state(
    data: mujoco.MjData,
    tray_body_id: int,
    package_body_id: int,
    expected_package_offset: npt.NDArray[np.float64],
) -> tuple[float, bool]:
    tray = data.body(tray_body_id)
    package = data.body(package_body_id)
    rotation = np.asarray(tray.xmat, dtype=np.float64).reshape(3, 3)
    tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
    offset = rotation.T @ (
        np.asarray(package.xpos, dtype=np.float64) - np.asarray(tray.xpos, dtype=np.float64)
    )
    slipped = bool(np.linalg.norm(offset - expected_package_offset) > PACKAGE_SLIP_THRESHOLD_M)
    return tilt, slipped


def _run_trial(
    policy_path: Path,
    duration_seconds: float,
    schedule: CommandSchedule,
    *,
    model: mujoco.MjModel | None = None,
    tray_body_id: int | None = None,
    package_body_id: int | None = None,
) -> TrialResult:
    simulation = HeadlessG1Simulation(
        model,
        controller_policy_path=policy_path,
        controller_inference_hz=CONTROLLER_HZ,
    )
    steps = round(duration_seconds * PHYSICS_HZ)
    if not math.isclose(steps / PHYSICS_HZ, duration_seconds):
        raise ValueError("trial duration must align to the 500 Hz physics clock")
    start_position = np.asarray(simulation.data.qpos[:2], dtype=np.float64).copy()
    previous_yaw = _yaw(simulation.data)
    cumulative_yaw = 0.0
    maximum_drift = 0.0
    finite_state = True
    fell = False
    active_collisions: set[tuple[int, int]] = set()
    body_collisions = 0
    robot_geom_ids, world_geom_ids = _collision_ids(simulation.model)
    stop_speeds: list[float] = []
    maximum_tray_tilt = 0.0
    package_slipped = False
    expected_package_offset = np.zeros(3, dtype=np.float64)
    if (tray_body_id is None) != (package_body_id is None):
        raise ValueError("payload body IDs must be provided together")
    if tray_body_id is not None and package_body_id is not None:
        tray = simulation.data.body(tray_body_id)
        package = simulation.data.body(package_body_id)
        rotation = np.asarray(tray.xmat, dtype=np.float64).reshape(3, 3)
        expected_package_offset = rotation.T @ (
            np.asarray(package.xpos, dtype=np.float64)
            - np.asarray(tray.xpos, dtype=np.float64)
        )

    for step_index in range(steps):
        simulation.step(schedule)
        current_yaw = _yaw(simulation.data)
        cumulative_yaw += _wrap_angle(current_yaw - previous_yaw)
        previous_yaw = current_yaw
        position = np.asarray(simulation.data.qpos[:2], dtype=np.float64)
        maximum_drift = max(maximum_drift, float(np.linalg.norm(position - start_position)))
        speed = float(np.linalg.norm(simulation.data.qvel[:2]))
        if step_index >= steps - PHYSICS_HZ:
            stop_speeds.append(speed)
        finite_state = finite_state and bool(
            np.isfinite(simulation.data.qpos).all()
            and np.isfinite(simulation.data.qvel).all()
        )
        pelvis_height = float(simulation.data.qpos[2])
        torso_up_z = float(simulation.data.sensor("upvector_torso").data[2])
        fell = fell or pelvis_height <= FALL_HEIGHT_M or torso_up_z <= FALL_UP_Z
        current_collisions = _active_body_collisions(
            simulation.data, robot_geom_ids, world_geom_ids
        )
        body_collisions += len(current_collisions - active_collisions)
        active_collisions = current_collisions
        if tray_body_id is not None and package_body_id is not None:
            tilt, slipped = _payload_state(
                simulation.data,
                tray_body_id,
                package_body_id,
                expected_package_offset,
            )
            maximum_tray_tilt = max(maximum_tray_tilt, tilt)
            package_slipped = package_slipped or slipped

    final_position = np.asarray(simulation.data.qpos[:2], dtype=np.float64)
    return TrialResult(
        duration_seconds=duration_seconds,
        physics_steps=steps,
        controller_inferences=simulation.controller.inference_count,
        controller_supervisor_ticks=simulation.controller_supervisor_ticks,
        finite_state=finite_state,
        fell=fell,
        body_collisions=body_collisions,
        planar_delta_m=(
            float(final_position[0] - start_position[0]),
            float(final_position[1] - start_position[1]),
        ),
        cumulative_yaw_radians=cumulative_yaw,
        maximum_drift_m=maximum_drift,
        last_second_maximum_speed_mps=max(stop_speeds),
        maximum_tray_tilt_degrees=maximum_tray_tilt,
        package_slipped=package_slipped,
        final_qpos=np.asarray(simulation.data.qpos, dtype=np.float64).copy(),
        final_qvel=np.asarray(simulation.data.qvel, dtype=np.float64).copy(),
    )


def generate_evidence(policy_path: Path) -> tuple[QualificationEvidence, dict[str, object]]:
    """Execute every physical trial and return strict evidence plus raw summaries."""
    forward_command = TaskCommand(0.4, 0.0, 0.0)
    stop_command = TaskCommand(0.0, 0.0, 1.0)
    forward = _run_trial(policy_path, 10.0, lambda _time: forward_command)
    forward_repeat = _run_trial(policy_path, 10.0, lambda _time: forward_command)
    left = _run_trial(policy_path, 4.0, lambda _time: TaskCommand(0.0, 0.5, 0.0))
    right = _run_trial(policy_path, 4.0, lambda _time: TaskCommand(0.0, -0.5, 0.0))
    stop = _run_trial(
        policy_path,
        15.0,
        lambda time: forward_command if time < 5.0 else stop_command,
    )
    standstill = _run_trial(policy_path, 60.0, lambda _time: stop_command)
    payload_scene = assemble_payload_qualification_scene()
    payload = _run_trial(
        policy_path,
        19.0,
        lambda time: (
            TaskCommand(0.3, 0.0, 0.0)
            if time < 5.0
            else TaskCommand(0.0, 0.3, 0.0)
            if time < 7.0
            else TaskCommand(0.0, -0.3, 0.0)
            if time < 9.0
            else stop_command
        ),
        model=payload_scene.model,
        tray_body_id=payload_scene.tray_body_id,
        package_body_id=payload_scene.package_body_id,
    )
    trials = (forward, forward_repeat, left, right, stop, standstill, payload)
    measured_rates_match = all(
        trial.physics_steps == round(trial.duration_seconds * PHYSICS_HZ)
        and trial.controller_inferences == round(trial.duration_seconds * CONTROLLER_HZ)
        and trial.controller_supervisor_ticks == round(trial.duration_seconds * CONTROLLER_HZ)
        for trial in trials
    )
    repeat_delta = max(
        float(np.max(np.abs(forward.final_qpos - forward_repeat.final_qpos))),
        float(np.max(np.abs(forward.final_qvel - forward_repeat.final_qvel))),
    )
    forward_delta = np.asarray(forward.planar_delta_m, dtype=np.float64)
    raw_trials = {
        "schema_version": 1,
        "trials": {
            name: {
                key: value
                for key, value in asdict(trial).items()
                if key not in {"final_qpos", "final_qvel"}
            }
            for name, trial in zip(
                (
                    "forward",
                    "forward_repeat",
                    "left",
                    "right",
                    "stop",
                    "standstill",
                    "payload",
                ),
                trials,
                strict=True,
            )
        },
    }
    raw_trial_bytes = (
        json.dumps(raw_trials, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    evidence = QualificationEvidence(
        controller_hz=CONTROLLER_HZ if measured_rates_match else 0,
        ctrl_dt_seconds=CTRL_DT_SECONDS if measured_rates_match else 0.0,
        physics_hz=PHYSICS_HZ,
        task_command_outputs=3,
        gait_action_outputs=GAIT_ACTION_OUTPUTS,
        finite_state=all(trial.finite_state for trial in trials),
        fall_count=sum(int(trial.fell) for trial in trials),
        body_collision_count=sum(trial.body_collisions for trial in trials),
        forward_progress_metres=float(forward_delta[0]),
        forward_cross_track_error_metres=abs(float(forward_delta[1])),
        forward_heading_error_degrees=abs(math.degrees(forward.cumulative_yaw_radians)),
        left_turn_error_degrees=abs(math.degrees(_wrap_angle(left.cumulative_yaw_radians - 2.0))),
        right_turn_error_degrees=abs(
            math.degrees(_wrap_angle(right.cumulative_yaw_radians + 2.0))
        ),
        stop_speed_metres_per_second=stop.last_second_maximum_speed_mps,
        standstill_duration_seconds=standstill.duration_seconds,
        standstill_drift_metres=standstill.maximum_drift_m,
        payload_stop_speed_metres_per_second=payload.last_second_maximum_speed_mps,
        maximum_payload_tray_tilt_degrees=payload.maximum_tray_tilt_degrees,
        payload_package_slipped=payload.package_slipped,
        deterministic_repeat_max_metric_delta=repeat_delta,
        robot_checksum=robot_bundle_checksum(policy_path),
        controller_onnx_sha256=sha256_file(policy_path),
        qualification_program_sha256=sha256_file(Path(__file__)),
        qualification_trials_sha256=hashlib.sha256(raw_trial_bytes).hexdigest(),
    )
    return evidence, raw_trials


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--mode", type=RunMode, choices=tuple(RunMode), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    policy_path = args.policy.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    evidence, raw_trials = generate_evidence(policy_path)
    result = evaluate_qualification(evidence, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_path = args.output.with_name("qualification-trials.json")
    raw_path.write_text(json.dumps(raw_trials, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_payload = {
        "schema_version": 1,
        "mode": args.mode.value,
        "qualified": result.qualified,
        "failures": list(result.failures),
        "evidence_path": args.output.resolve().as_posix(),
        "trials_path": raw_path.resolve().as_posix(),
    }
    print(json.dumps(result_payload, indent=2, sort_keys=True))
    return 0 if result.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
