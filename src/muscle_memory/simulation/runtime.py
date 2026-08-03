"""Rate-separated native MuJoCo runtime for the candidate controller."""

from collections.abc import Callable

import mujoco  # type: ignore[import-untyped]
import numpy as np

from muscle_memory.paths import G1_POLICY_ONNX, G1_SCENE_XML
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


class HeadlessG1Simulation:
    """Runs one real native-MuJoCo candidate scene without training hooks."""

    def __init__(self) -> None:
        if mujoco.__version__ != "3.6.0":
            raise RuntimeError(f"candidate requires MuJoCo 3.6.0, found {mujoco.__version__}")
        if any(
            PHYSICS_HZ % rate
            for rate in (TASK_POLICY_HZ, CONTROLLER_SUPERVISOR_HZ, CONTROLLER_INFERENCE_HZ)
        ):
            raise RuntimeError("candidate rates must divide the physics rate exactly")
        self.model = mujoco.MjModel.from_xml_path(G1_SCENE_XML.as_posix())
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent"
        )
        if keyframe_id < 0:
            raise RuntimeError("candidate model is missing the knees_bent keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        self.controller = FrozenG1Controller(self.model, G1_POLICY_ONNX)
        self.step_index = 0
        self.task_policy_updates = 0
        self.controller_supervisor_ticks = 0
        self._pending_task_command = TaskCommand(0.0, 0.0, 1.0)
        self._controller_command = self._pending_task_command

    @property
    def controller_command(self) -> TaskCommand:
        return self._controller_command

    def step(self, task_policy: Callable[[float], TaskCommand]) -> None:
        """Advance exactly one 500 Hz physics step with separated rate gates."""
        if self.step_index % TASK_POLICY_INTERVAL_STEPS == 0:
            command = task_policy(self.data.time)
            if not isinstance(command, TaskCommand):
                raise TypeError("task policy must return TaskCommand")
            self._pending_task_command = command
            self.task_policy_updates += 1
        if self.step_index % CONTROLLER_SUPERVISOR_INTERVAL_STEPS == 0:
            # The 100 Hz supervisor validates and transfers the held policy command.
            self._controller_command = TaskCommand(
                self._pending_task_command.forward_speed_mps,
                self._pending_task_command.turning_rate_rad_s,
                self._pending_task_command.stop_probability,
            )
            self.controller_supervisor_ticks += 1
        if self.step_index % CONTROLLER_INTERVAL_STEPS == 0:
            self.controller.infer(self.model, self.data, self._controller_command)
        mujoco.mj_step(self.model, self.data)
        self.step_index += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo state became non-finite")
