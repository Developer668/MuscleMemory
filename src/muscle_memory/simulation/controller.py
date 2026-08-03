"""Inference-only wrapper for the pinned Playground G1 ONNX policy."""

import math
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

from muscle_memory.robot.command import TaskCommand

OBSERVATION_SIZE = 103
ACTION_SIZE = 29
ACTION_SCALE = 0.5
GAIT_FREQUENCY_HZ = 1.5
CONTROLLER_DT = 0.02


class FrozenG1Controller:
    """Runs the immutable gait policy with NumPy and ONNX Runtime only."""

    def __init__(self, model: mujoco.MjModel, policy_path: Path) -> None:
        if ort.__version__ != "1.28.0":
            raise RuntimeError(f"candidate requires ONNX Runtime 1.28.0, found {ort.__version__}")
        self._session = ort.InferenceSession(
            policy_path.as_posix(), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "obs"
            or inputs[0].shape != [1, 103]
            or inputs[0].type != "tensor(float)"
        ):
            raise RuntimeError("candidate ONNX input contract is not float32[1, 103]")
        if (
            len(outputs) != 1
            or outputs[0].name != "continuous_actions"
            or outputs[0].shape != [1, 29]
            or outputs[0].type != "tensor(float)"
        ):
            raise RuntimeError("candidate ONNX output contract is not float32[1, 29]")
        if model.nu != ACTION_SIZE:
            raise RuntimeError(f"candidate model has {model.nu} actuators, expected {ACTION_SIZE}")

        self._default_angles = np.array(model.keyframe("knees_bent").qpos[7:], dtype=np.float64)
        self._last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._phase = np.array([0.0, np.pi], dtype=np.float64)
        self._phase_delta = 2.0 * math.pi * GAIT_FREQUENCY_HZ * CONTROLLER_DT
        self.inference_count = 0

    @property
    def last_action(self) -> npt.NDArray[np.float32]:
        """Return a copy so callers cannot mutate controller state."""
        return self._last_action.copy()

    def _observation(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        command: TaskCommand,
    ) -> npt.NDArray[np.float32]:
        linvel = data.sensor("local_linvel_pelvis").data
        gyro = data.sensor("gyro_pelvis").data
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        velocity_command = np.asarray(command.frozen_controller_command(), dtype=np.float64)
        observation = np.hstack(
            (
                linvel,
                gyro,
                gravity,
                velocity_command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise RuntimeError("candidate controller observation is invalid")
        return observation

    def infer(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        command: TaskCommand,
    ) -> None:
        """Apply one fixed 50 Hz inference result to MuJoCo position actuators."""
        observation = self._observation(model, data, command)
        result = self._session.run(
            ["continuous_actions"], {"obs": observation.reshape(1, OBSERVATION_SIZE)}
        )[0][0]
        action = np.asarray(result, dtype=np.float32)
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
            raise RuntimeError("candidate controller returned an invalid action")
        self._last_action = action.copy()
        data.ctrl[:] = action * ACTION_SCALE + self._default_angles
        self._phase = np.fmod(self._phase + self._phase_delta + np.pi, 2 * np.pi) - np.pi
        self.inference_count += 1
