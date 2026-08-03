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
SENSOR_FUSION_CHECKPOINT_SCHEMA_VERSION = 3
CONTINUOUS_INFERENCE_STRATEGY = "continuous_v1"
SENSOR_FUSION_INFERENCE_STRATEGY = "sensor_fusion_v2"
POLICY_OUTPUT_COUNT = 3
POLICY_MAXIMUM_FORWARD_SPEED_MPS = 0.3
POLICY_MAXIMUM_TURNING_RATE_RAD_S = 0.5
DOCKING_ENTRY_DISTANCE_M = 0.65
DOCKING_STOP_DISTANCE_M = 0.45
DOCKING_FORWARD_SPEED_MPS = 0.18
STEREO_HORIZONTAL_HALF_FOV_RAD = 0.52


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
        inference_strategy: str = CONTINUOUS_INFERENCE_STRATEGY,
        avoidance_distance_m: float = 2.6,
        avoidance_gain: float = 1.35,
        avoidance_exponent: float = 2.0,
        avoidance_activation: float = 0.2,
        avoidance_docking_suppression_m: float = 1.0,
        learned_turn_blend: float = 0.0,
        avoidance_release: float = 0.06,
        avoidance_reversal: float = 0.18,
        avoidance_release_ticks: int = 5,
        held_repulsion: float = 0.16,
        turn_slew_per_update: float = 0.1,
        sparse_risk_threshold: float = 0.2,
        sparse_turn_slew_per_update: float = 0.15,
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
        self.inference_strategy = inference_strategy
        self._avoidance_distance_m = avoidance_distance_m
        self._avoidance_gain = avoidance_gain
        self._avoidance_exponent = avoidance_exponent
        self._avoidance_activation = avoidance_activation
        self._avoidance_docking_suppression_m = avoidance_docking_suppression_m
        self._learned_turn_blend = learned_turn_blend
        self._avoidance_release = avoidance_release
        self._avoidance_reversal = avoidance_reversal
        self._avoidance_release_ticks = avoidance_release_ticks
        self._held_repulsion = held_repulsion
        self._turn_slew_per_update = turn_slew_per_update
        self._sparse_risk_threshold = sparse_risk_threshold
        self._sparse_turn_slew_per_update = sparse_turn_slew_per_update
        self._avoidance_side = 0
        self._avoidance_clear_ticks = 0
        self._previous_turn = 0.0

    @classmethod
    def load(cls, path: Path = POLICY_V1_CHECKPOINT) -> BehaviorClonedPolicy:
        """Load and validate a checkpoint bound to the current frozen robot."""
        robot = verify_mm01_bundle()
        try:
            with np.load(path, allow_pickle=False) as checkpoint:
                schema_version = int(checkpoint["schema_version"])
                policy_id = str(checkpoint["policy_id"])
                robot_checksum = str(checkpoint["robot_checksum"])
                inference_strategy = (
                    str(checkpoint["inference_strategy"])
                    if "inference_strategy" in checkpoint.files
                    else CONTINUOUS_INFERENCE_STRATEGY
                )
                fusion_parameters = {
                    "avoidance_distance_m": (
                        float(checkpoint["avoidance_distance_m"])
                        if "avoidance_distance_m" in checkpoint.files
                        else 2.6
                    ),
                    "avoidance_gain": (
                        float(checkpoint["avoidance_gain"])
                        if "avoidance_gain" in checkpoint.files
                        else 1.35
                    ),
                    "avoidance_exponent": (
                        float(checkpoint["avoidance_exponent"])
                        if "avoidance_exponent" in checkpoint.files
                        else 2.0
                    ),
                    "learned_turn_blend": (
                        float(checkpoint["learned_turn_blend"])
                        if "learned_turn_blend" in checkpoint.files
                        else 0.0
                    ),
                    "avoidance_activation": (
                        float(checkpoint["avoidance_activation"])
                        if "avoidance_activation" in checkpoint.files
                        else 0.0
                    ),
                    "avoidance_docking_suppression_m": (
                        float(checkpoint["avoidance_docking_suppression_m"])
                        if "avoidance_docking_suppression_m" in checkpoint.files
                        else 1.0
                    ),
                    "avoidance_release": (
                        float(checkpoint["avoidance_release"])
                        if "avoidance_release" in checkpoint.files
                        else 0.06
                    ),
                    "avoidance_reversal": (
                        float(checkpoint["avoidance_reversal"])
                        if "avoidance_reversal" in checkpoint.files
                        else 0.18
                    ),
                    "avoidance_release_ticks": (
                        int(checkpoint["avoidance_release_ticks"])
                        if "avoidance_release_ticks" in checkpoint.files
                        else 5
                    ),
                    "held_repulsion": (
                        float(checkpoint["held_repulsion"])
                        if "held_repulsion" in checkpoint.files
                        else 0.16
                    ),
                    "turn_slew_per_update": (
                        float(checkpoint["turn_slew_per_update"])
                        if "turn_slew_per_update" in checkpoint.files
                        else 0.1
                    ),
                    "sparse_risk_threshold": (
                        float(checkpoint["sparse_risk_threshold"])
                        if "sparse_risk_threshold" in checkpoint.files
                        else 0.2
                    ),
                    "sparse_turn_slew_per_update": (
                        float(checkpoint["sparse_turn_slew_per_update"])
                        if "sparse_turn_slew_per_update" in checkpoint.files
                        else 0.15
                    ),
                }
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
        if schema_version not in (
            POLICY_CHECKPOINT_SCHEMA_VERSION,
            SENSOR_FUSION_CHECKPOINT_SCHEMA_VERSION,
        ):
            raise RuntimeError("task-policy checkpoint schema changed")
        if schema_version == POLICY_CHECKPOINT_SCHEMA_VERSION and (
            inference_strategy != CONTINUOUS_INFERENCE_STRATEGY
        ):
            raise RuntimeError("schema-2 checkpoint must use continuous inference")
        if schema_version == SENSOR_FUSION_CHECKPOINT_SCHEMA_VERSION and (
            inference_strategy != SENSOR_FUSION_INFERENCE_STRATEGY
        ):
            raise RuntimeError("schema-3 checkpoint must use sensor-fusion inference")
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
        if not all(np.isfinite(float(value)) for value in fusion_parameters.values()):
            raise RuntimeError("task-policy sensor-fusion parameters must be finite")
        if not (
            0.2 < fusion_parameters["avoidance_distance_m"] <= 8.0
            and 0.0 <= fusion_parameters["avoidance_gain"] <= 4.0
            and 1.0 <= fusion_parameters["avoidance_exponent"] <= 4.0
            and 0.0 <= fusion_parameters["avoidance_activation"] <= 0.52
            and 0.45
            <= fusion_parameters["avoidance_docking_suppression_m"]
            <= 2.0
            and 0.0 <= fusion_parameters["learned_turn_blend"] <= 1.0
            and 0.0 <= fusion_parameters["avoidance_release"] < 0.52
            and 0.0 < fusion_parameters["avoidance_reversal"] <= 0.52
            and 1 <= fusion_parameters["avoidance_release_ticks"] <= 100
            and 0.0 <= fusion_parameters["held_repulsion"] <= 0.52
            and 0.0 < fusion_parameters["turn_slew_per_update"] <= 0.5
            and 0.0 <= fusion_parameters["sparse_risk_threshold"] <= 48.0
            and 0.0
            < fusion_parameters["sparse_turn_slew_per_update"]
            <= 0.5
        ):
            raise RuntimeError("task-policy sensor-fusion parameters are out of bounds")
        return cls(
            policy_id=policy_id,
            policy_hash=policy_file_sha256(path),
            inference_strategy=inference_strategy,
            avoidance_distance_m=fusion_parameters["avoidance_distance_m"],
            avoidance_gain=fusion_parameters["avoidance_gain"],
            avoidance_exponent=fusion_parameters["avoidance_exponent"],
            avoidance_activation=fusion_parameters["avoidance_activation"],
            avoidance_docking_suppression_m=fusion_parameters[
                "avoidance_docking_suppression_m"
            ],
            learned_turn_blend=fusion_parameters["learned_turn_blend"],
            avoidance_release=fusion_parameters["avoidance_release"],
            avoidance_reversal=fusion_parameters["avoidance_reversal"],
            avoidance_release_ticks=int(fusion_parameters["avoidance_release_ticks"]),
            held_repulsion=fusion_parameters["held_repulsion"],
            turn_slew_per_update=fusion_parameters["turn_slew_per_update"],
            sparse_risk_threshold=fusion_parameters["sparse_risk_threshold"],
            sparse_turn_slew_per_update=fusion_parameters[
                "sparse_turn_slew_per_update"
            ],
            input_mean=arrays["input_mean"],
            input_std=arrays["input_std"],
            weight_1=arrays["weight_1"],
            bias_1=arrays["bias_1"],
            weight_2=arrays["weight_2"],
            bias_2=arrays["bias_2"],
            weight_output=arrays["weight_output"],
            bias_output=arrays["bias_output"],
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

    def _continuous_command(self, observation: NavigationObservation) -> TaskCommand:
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

    def _sensor_fusion_command(self, observation: NavigationObservation) -> TaskCommand:
        distance = observation.destination_distance_m
        if observation.values[-1] >= 0.5 and distance > 1.0:
            self._avoidance_side = 0
            self._avoidance_clear_ticks = 0
            self._previous_turn = 0.0
        if distance <= DOCKING_STOP_DISTANCE_M:
            return TaskCommand(0.0, 0.0, 1.0)

        outputs = self.normalized_outputs(observation)
        depths_m = np.asarray(observation.values[:48], dtype=np.float64) * 8.0
        sector_angles = np.linspace(
            STEREO_HORIZONTAL_HALF_FOV_RAD,
            -STEREO_HORIZONTAL_HALF_FOV_RAD,
            depths_m.size,
            dtype=np.float64,
        )
        risk = np.clip(
            (self._avoidance_distance_m - depths_m)
            / (self._avoidance_distance_m - 0.2),
            0.0,
            1.0,
        ) ** self._avoidance_exponent
        total_risk = float(np.sum(risk))
        repulsion = float(np.sum(risk * -sector_angles) / max(total_risk, 0.05))
        front_depth = float(np.percentile(depths_m[17:31], 20.0))
        if distance <= self._avoidance_docking_suppression_m:
            self._avoidance_side = 0
        elif self._avoidance_side:
            if repulsion * self._avoidance_side < -self._avoidance_reversal:
                self._avoidance_side = 1 if repulsion > 0.0 else -1
                self._avoidance_clear_ticks = 0
            elif abs(repulsion) < self._avoidance_release:
                self._avoidance_clear_ticks += 1
                if self._avoidance_clear_ticks >= self._avoidance_release_ticks:
                    self._avoidance_side = 0
            else:
                self._avoidance_clear_ticks = 0
        elif abs(repulsion) >= self._avoidance_activation:
            self._avoidance_side = 1 if repulsion > 0.0 else -1

        avoidance_active = bool(self._avoidance_side)
        active_repulsion = 0.0
        if avoidance_active:
            active_repulsion = (
                repulsion
                if repulsion * self._avoidance_side > 0.0
                else self._held_repulsion * self._avoidance_side
            )
        direct_turn = 1.4 * observation.destination_bearing_rad
        learned_turn = float(outputs[1] * POLICY_MAXIMUM_TURNING_RATE_RAD_S)
        turning_delta = 0.0
        if avoidance_active:
            turning_delta = self._avoidance_gain * active_repulsion
            turning_delta += self._learned_turn_blend * (learned_turn - direct_turn)
        desired_turn = max(
            -POLICY_MAXIMUM_TURNING_RATE_RAD_S,
            min(
                POLICY_MAXIMUM_TURNING_RATE_RAD_S,
                direct_turn + turning_delta,
            ),
        )
        turn_slew = (
            self._sparse_turn_slew_per_update
            if total_risk < self._sparse_risk_threshold
            else self._turn_slew_per_update
        )
        turning = max(
            self._previous_turn - turn_slew,
            min(self._previous_turn + turn_slew, desired_turn),
        )
        self._previous_turn = turning
        target_heading = turning / 1.4
        if abs(target_heading) > 0.22:
            forward = 0.0
        elif avoidance_active and front_depth < 1.0:
            forward = 0.12
        else:
            forward = DOCKING_FORWARD_SPEED_MPS if distance < 0.8 else 0.3
        return TaskCommand(forward, turning, 0.0)

    def command(self, observation: NavigationObservation) -> TaskCommand:
        if self.inference_strategy == SENSOR_FUSION_INFERENCE_STRATEGY:
            return self._sensor_fusion_command(observation)
        return self._continuous_command(observation)
