"""NumPy-only inference for immutable behavior-cloned policy checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

from muscle_memory.paths import POLICY_V1_CHECKPOINT
from muscle_memory.policy.observation import NAVIGATION_OBSERVATION_SIZE, NavigationObservation
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import verify_mm01_bundle

POLICY_CHECKPOINT_SCHEMA_VERSION = 2
POLICY_OUTPUT_COUNT = 3
POLICY_MAXIMUM_FORWARD_SPEED_MPS = 0.3
POLICY_MAXIMUM_TURNING_RATE_RAD_S = 0.5
DOCKING_ENTRY_DISTANCE_M = 0.65
DOCKING_STOP_DISTANCE_M = 0.45
DOCKING_FORWARD_SPEED_MPS = 0.18


def policy_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BehaviorClonedPolicy:
    """Run a continuous three-output regressor with no teacher or graph access."""

    def __init__(
        self,
        *,
        policy_id: str,
        policy_hash: str,
        input_mean: npt.NDArray[np.float32],
        input_std: npt.NDArray[np.float32],
        weight_1: npt.NDArray[np.float32],
        bias_1: npt.NDArray[np.float32],
        weight_2: npt.NDArray[np.float32],
        bias_2: npt.NDArray[np.float32],
        weight_output: npt.NDArray[np.float32],
        bias_output: npt.NDArray[np.float32],
    ) -> None:
        self.policy_id = policy_id
        self.policy_hash = policy_hash
        self._input_mean = input_mean
        self._input_std = input_std
        self._weight_1 = weight_1
        self._bias_1 = bias_1
        self._weight_2 = weight_2
        self._bias_2 = bias_2
        self._weight_output = weight_output
        self._bias_output = bias_output

    @classmethod
    def load(cls, path: Path = POLICY_V1_CHECKPOINT) -> BehaviorClonedPolicy:
        """Load and validate a checkpoint bound to the current frozen robot."""
        robot = verify_mm01_bundle()
        try:
            with np.load(path, allow_pickle=False) as checkpoint:
                schema_version = int(checkpoint["schema_version"])
                policy_id = str(checkpoint["policy_id"])
                robot_checksum = str(checkpoint["robot_checksum"])
                arrays = {
                    name: np.asarray(checkpoint[name], dtype=np.float32)
                    for name in (
                        "input_mean",
                        "input_std",
                        "weight_1",
                        "bias_1",
                        "weight_2",
                        "bias_2",
                        "weight_output",
                        "bias_output",
                    )
                }
        except Exception as error:
            raise RuntimeError(f"invalid task-policy checkpoint: {path}") from error
        if schema_version != POLICY_CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("task-policy checkpoint schema changed")
        if not policy_id:
            raise RuntimeError("task-policy checkpoint has an empty policy ID")
        if robot_checksum != robot.robot_checksum:
            raise RuntimeError("task-policy checkpoint belongs to a different robot")
        hidden_1 = arrays["bias_1"].shape[0]
        hidden_2 = arrays["bias_2"].shape[0]
        if hidden_1 <= 0 or hidden_2 <= 0:
            raise RuntimeError("task-policy checkpoint hidden layers must not be empty")
        expected_shapes = {
            "input_mean": (NAVIGATION_OBSERVATION_SIZE,),
            "input_std": (NAVIGATION_OBSERVATION_SIZE,),
            "weight_1": (NAVIGATION_OBSERVATION_SIZE, hidden_1),
            "bias_1": (hidden_1,),
            "weight_2": (hidden_1, hidden_2),
            "bias_2": (hidden_2,),
            "weight_output": (hidden_2, POLICY_OUTPUT_COUNT),
            "bias_output": (POLICY_OUTPUT_COUNT,),
        }
        for name, expected in expected_shapes.items():
            array = arrays[name]
            if array.shape != expected or not np.isfinite(array).all():
                raise RuntimeError(f"task-policy checkpoint tensor is invalid: {name}")
        if np.any(arrays["input_std"] <= 0.0):
            raise RuntimeError("task-policy input standard deviation must be positive")
        return cls(
            policy_id=policy_id,
            policy_hash=policy_file_sha256(path),
            **arrays,
        )

    def normalized_outputs(
        self,
        observation: NavigationObservation,
    ) -> npt.NDArray[np.float32]:
        """Return forward [0, 1], turn [-1, 1], and stop [0, 1]."""
        normalized = (observation.values - self._input_mean) / self._input_std
        hidden_1 = np.tanh(normalized @ self._weight_1 + self._bias_1)
        hidden_2 = np.tanh(hidden_1 @ self._weight_2 + self._bias_2)
        raw = hidden_2 @ self._weight_output + self._bias_output
        return np.asarray(
            (
                0.5 * (np.tanh(raw[0]) + 1.0),
                np.tanh(raw[1]),
                1.0 / (1.0 + np.exp(-raw[2])),
            ),
            dtype=np.float32,
        )

    def command(self, observation: NavigationObservation) -> TaskCommand:
        distance = observation.destination_distance_m
        if distance <= DOCKING_STOP_DISTANCE_M:
            return TaskCommand(0.0, 0.0, 1.0)
        outputs = self.normalized_outputs(observation)
        if distance < DOCKING_ENTRY_DISTANCE_M:
            return TaskCommand(
                max(
                    DOCKING_FORWARD_SPEED_MPS,
                    float(outputs[0] * POLICY_MAXIMUM_FORWARD_SPEED_MPS),
                ),
                max(
                    -0.3,
                    min(0.3, 1.4 * observation.destination_bearing_rad),
                ),
                0.0,
            )
        return TaskCommand(
            forward_speed_mps=float(outputs[0] * POLICY_MAXIMUM_FORWARD_SPEED_MPS),
            turning_rate_rad_s=float(outputs[1] * POLICY_MAXIMUM_TURNING_RATE_RAD_S),
            stop_probability=float(outputs[2]),
        )
