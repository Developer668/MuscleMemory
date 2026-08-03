"""Rate-separated native MuJoCo runtime for the frozen MM-01 controller."""

from collections.abc import Callable
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np

from muscle_memory.paths import G1_SCENE_XML, MM01_CONTROLLER_ONNX
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import (
    CONTROLLER_INFERENCE_HZ,
    CONTROLLER_SUPERVISOR_HZ,
    PHYSICS_HZ,
    TASK_POLICY_HZ,
)
from muscle_memory.simulation.controller import FrozenG1Controller

PHYSICS_DT = 1.0 / PHYSICS_HZ
CONTROLLER_SUPERVISOR_INTERVAL_STEPS = PHYSICS_HZ // CONTROLLER_SUPERVISOR_HZ
CONTROLLER_INTERVAL_STEPS = PHYSICS_HZ // CONTROLLER_INFERENCE_HZ
TASK_POLICY_INTERVAL_STEPS = PHYSICS_HZ // TASK_POLICY_HZ
STOP_FORWARD_DECELERATION_MPS2 = 0.75
STOP_TURN_DECELERATION_RAD_S2 = 5.0


def _approach_zero(value: float, maximum_delta: float) -> float:
    if abs(value) <= maximum_delta + 1e-12:
        return 0.0
    return float(np.copysign(abs(value) - maximum_delta, value))


class HeadlessG1Simulation:
    """Runs one real native-MuJoCo MM-01 scene without training hooks."""

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        initialize_data: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
        *,
        controller_policy_path: Path = MM01_CONTROLLER_ONNX,
        controller_inference_hz: int = CONTROLLER_INFERENCE_HZ,
    ) -> None:
        if mujoco.__version__ != "3.6.0":
            raise RuntimeError(f"MM-01 requires MuJoCo 3.6.0, found {mujoco.__version__}")
        if any(
            PHYSICS_HZ % rate
            for rate in (TASK_POLICY_HZ, CONTROLLER_SUPERVISOR_HZ, controller_inference_hz)
        ):
            raise RuntimeError("MM-01 rates must divide the physics rate exactly")
        self.model = model or mujoco.MjModel.from_xml_path(G1_SCENE_XML.as_posix())
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent"
        )
        if keyframe_id < 0:
            raise RuntimeError("MM-01 model is missing the knees_bent keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        if initialize_data is not None:
            initialize_data(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.controller = FrozenG1Controller(
            self.model,
            controller_policy_path,
            controller_hz=controller_inference_hz,
        )
        self.controller_inference_hz = controller_inference_hz
        self._controller_interval_steps = PHYSICS_HZ // controller_inference_hz
        self.step_index = 0
        self.task_policy_updates = 0
        self.controller_supervisor_ticks = 0
        self._pending_task_command = TaskCommand(0.0, 0.0, 1.0)
        self._controller_command = self._pending_task_command

    @property
    def controller_command(self) -> TaskCommand:
        return self._controller_command

    def _supervise_command(self) -> TaskCommand:
        pending = self._pending_task_command
        if not pending.stop_requested:
            return pending

        forward = _approach_zero(
            self._controller_command.forward_speed_mps,
            STOP_FORWARD_DECELERATION_MPS2 / CONTROLLER_SUPERVISOR_HZ,
        )
        turning = _approach_zero(
            self._controller_command.turning_rate_rad_s,
            STOP_TURN_DECELERATION_RAD_S2 / CONTROLLER_SUPERVISOR_HZ,
        )
        stopped = forward == 0.0 and turning == 0.0
        return TaskCommand(forward, turning, 1.0 if stopped else 0.0)

    def step(self, task_policy: Callable[[float], TaskCommand]) -> None:
        """Advance exactly one 500 Hz physics step with separated rate gates."""
        if self.step_index % TASK_POLICY_INTERVAL_STEPS == 0:
            command = task_policy(self.data.time)
            if not isinstance(command, TaskCommand):
                raise TypeError("task policy must return TaskCommand")
            self._pending_task_command = command
            self.task_policy_updates += 1
        if self.step_index % CONTROLLER_SUPERVISOR_INTERVAL_STEPS == 0:
            self._controller_command = self._supervise_command()
            self.controller_supervisor_ticks += 1
        if self.step_index % self._controller_interval_steps == 0:
            self.controller.infer(self.model, self.data, self._controller_command)
        mujoco.mj_step(self.model, self.data)
        self.step_index += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo state became non-finite")
