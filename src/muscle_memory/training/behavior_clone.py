"""Deterministic NumPy behavior cloning for the high-level delivery policy."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from muscle_memory.paths import (
    EXPERT_DATASET_V1,
    POLICY_V1_CHECKPOINT,
    POLICY_V1_TRAINING_EVIDENCE,
)
from muscle_memory.policy.network import (
    CONTINUOUS_INFERENCE_STRATEGY,
    POLICY_CHECKPOINT_SCHEMA_VERSION,
    POLICY_MAXIMUM_FORWARD_SPEED_MPS,
    POLICY_MAXIMUM_TURNING_RATE_RAD_S,
    POLICY_OUTPUT_COUNT,
    SENSOR_FUSION_CHECKPOINT_SCHEMA_VERSION,
    SENSOR_FUSION_INFERENCE_STRATEGY,
)
from muscle_memory.policy.observation import NAVIGATION_OBSERVATION_SIZE
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.training.dataset import DATASET_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class BehaviorCloneConfig:
    hidden_1: int = 96
    hidden_2: int = 64
    epochs: int = 180
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_episode_fraction: float = 0.2
    seed: int = 668
    condition_on_previous_action: bool = False
    mirror_training_fraction: float = 0.5
    policy_id: str = "delivery-v1-bc"
    inference_strategy: str = CONTINUOUS_INFERENCE_STRATEGY
    avoidance_distance_m: float = 2.6
    avoidance_gain: float = 1.8
    avoidance_exponent: float = 2.0
    avoidance_activation: float = 0.1
    avoidance_docking_suppression_m: float = 1.0
    learned_turn_blend: float = 0.0
    avoidance_release: float = 0.06
    avoidance_reversal: float = 0.18
    avoidance_release_ticks: int = 5
    held_repulsion: float = 0.16
    turn_slew_per_update: float = 0.1
    sparse_risk_threshold: float = 0.2
    sparse_turn_slew_per_update: float = 0.15


DEFAULT_BEHAVIOR_CLONE_CONFIG = BehaviorCloneConfig()


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy_id: str
    policy_sha256: str
    dataset_sha256: str
    training_episode_count: int
    validation_episode_count: int
    training_sample_count: int
    validation_sample_count: int
    best_epoch: int
    training_command_accuracy: float
    validation_command_accuracy: float
    validation_loss: float
    validation_forward_mae_mps: float
    validation_turning_mae_rad_s: float
    validation_stop_mae: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forward(
    inputs: npt.NDArray[np.float32],
    parameters: dict[str, npt.NDArray[np.float32]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    hidden_1 = np.tanh(inputs @ parameters["weight_1"] + parameters["bias_1"])
    hidden_2 = np.tanh(hidden_1 @ parameters["weight_2"] + parameters["bias_2"])
    logits = hidden_2 @ parameters["weight_output"] + parameters["bias_output"]
    return hidden_1, hidden_2, np.asarray(logits, dtype=np.float32)


def _normalized_outputs(raw: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    forward = 0.5 * (np.tanh(raw[:, 0]) + 1.0)
    turning = np.tanh(raw[:, 1])
    stop = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 2], -30.0, 30.0)))
    return np.column_stack((forward, turning, stop)).astype(np.float32)


def mirror_navigation_samples(
    observations: npt.NDArray[np.float32],
    targets: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Mirror sensor/task samples across the robot's sagittal plane."""
    if observations.ndim != 2 or observations.shape[1] != NAVIGATION_OBSERVATION_SIZE:
        raise ValueError("mirror augmentation requires navigation observations")
    if targets.shape != (observations.shape[0], POLICY_OUTPUT_COUNT):
        raise ValueError("mirror augmentation requires continuous policy targets")
    mirrored_observations = observations.copy()
    mirrored_targets = targets.copy()
    mirrored_observations[:, :48] = observations[:, :48][:, ::-1]
    mirrored_observations[:, 49] *= -1.0
    mirrored_observations[:, (52, 54)] *= -1.0
    mirrored_observations[:, (55, 57)] *= -1.0
    mirrored_observations[:, 59] *= -1.0
    mirrored_observations[:, (62, 63)] = observations[:, (63, 62)]
    mirrored_observations[:, 67] *= -1.0
    mirrored_targets[:, 1] *= -1.0
    return mirrored_observations, mirrored_targets


def _metrics(
    inputs: npt.NDArray[np.float32],
    targets: npt.NDArray[np.float32],
    parameters: dict[str, npt.NDArray[np.float32]],
) -> tuple[float, float, npt.NDArray[np.float32]]:
    predictions = _normalized_outputs(_forward(inputs, parameters)[2])
    component_weights = np.asarray((1.0, 1.5, 2.0), dtype=np.float32)
    sample_weights = np.where(targets[:, 2] >= 0.5, 4.0, 1.0).astype(np.float32)
    squared_error = (predictions - targets) ** 2
    loss = float(
        np.sum(squared_error * component_weights[None, :] * sample_weights[:, None])
        / np.sum(component_weights * np.sum(sample_weights))
    )
    within_tolerance = (
        (np.abs(predictions[:, 0] - targets[:, 0]) <= 0.1)
        & (np.abs(predictions[:, 1] - targets[:, 1]) <= 0.1)
        & ((predictions[:, 2] >= 0.5) == (targets[:, 2] >= 0.5))
    )
    mae = np.mean(np.abs(predictions - targets), axis=0).astype(np.float32)
    return loss, float(np.mean(within_tolerance)), mae


def _initial_parameters(
    config: BehaviorCloneConfig,
    rng: np.random.Generator,
) -> dict[str, npt.NDArray[np.float32]]:
    def weight(rows: int, columns: int) -> npt.NDArray[np.float32]:
        limit = np.sqrt(6.0 / (rows + columns))
        return rng.uniform(-limit, limit, size=(rows, columns)).astype(np.float32)

    return {
        "weight_1": weight(NAVIGATION_OBSERVATION_SIZE, config.hidden_1),
        "bias_1": np.zeros(config.hidden_1, dtype=np.float32),
        "weight_2": weight(config.hidden_1, config.hidden_2),
        "bias_2": np.zeros(config.hidden_2, dtype=np.float32),
        "weight_output": weight(config.hidden_2, POLICY_OUTPUT_COUNT),
        "bias_output": np.zeros(POLICY_OUTPUT_COUNT, dtype=np.float32),
    }


def train_behavior_clone(
    *,
    dataset_path: Path = EXPERT_DATASET_V1,
    output_path: Path = POLICY_V1_CHECKPOINT,
    evidence_path: Path = POLICY_V1_TRAINING_EVIDENCE,
    config: BehaviorCloneConfig = DEFAULT_BEHAVIOR_CLONE_CONFIG,
) -> TrainingResult:
    """Train and serialize one immutable candidate from a world-grouped split."""
    if output_path.exists():
        raise FileExistsError(f"immutable task-policy checkpoint already exists: {output_path}")
    if evidence_path.exists():
        raise FileExistsError(f"immutable training evidence already exists: {evidence_path}")
    robot = verify_mm01_bundle()
    dataset_sha256 = _sha256_file(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        schema_version = int(dataset["schema_version"])
        dataset_robot_checksum = str(dataset["robot_checksum"])
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        commands = np.asarray(dataset["commands"], dtype=np.float32)
        episode_indices = np.asarray(dataset["episode_indices"], dtype=np.int32)
    if schema_version != DATASET_SCHEMA_VERSION:
        raise RuntimeError("expert dataset schema changed")
    if dataset_robot_checksum != robot.robot_checksum:
        raise RuntimeError("expert dataset belongs to a different robot")
    if observations.ndim != 2 or observations.shape[1] != NAVIGATION_OBSERVATION_SIZE:
        raise RuntimeError("expert dataset observation tensor is invalid")
    if commands.shape != (observations.shape[0], POLICY_OUTPUT_COUNT) or (
        episode_indices.shape != (observations.shape[0],)
    ):
        raise RuntimeError("expert dataset commands or episode membership are invalid")
    if not np.isfinite(observations).all() or not np.isfinite(commands).all():
        raise RuntimeError("expert dataset contains invalid values")
    if (
        np.any(commands[:, 0] < 0.0)
        or np.any(commands[:, 0] > POLICY_MAXIMUM_FORWARD_SPEED_MPS)
        or np.any(np.abs(commands[:, 1]) > POLICY_MAXIMUM_TURNING_RATE_RAD_S)
        or np.any(commands[:, 2] < 0.0)
        or np.any(commands[:, 2] > 1.0)
    ):
        raise RuntimeError("expert dataset commands exceed the task-policy contract")
    normalized_targets = np.column_stack(
        (
            commands[:, 0] / POLICY_MAXIMUM_FORWARD_SPEED_MPS,
            commands[:, 1] / POLICY_MAXIMUM_TURNING_RATE_RAD_S,
            commands[:, 2],
        )
    ).astype(np.float32)

    episode_ids = np.unique(episode_indices)
    if episode_ids.size < 2:
        raise RuntimeError("expert dataset needs at least two distinct episodes")
    rng = np.random.default_rng(config.seed)
    shuffled_episodes = rng.permutation(episode_ids)
    validation_count = max(
        1,
        round(float(episode_ids.size) * config.validation_episode_fraction),
    )
    validation_episodes = shuffled_episodes[:validation_count]
    training_episodes = shuffled_episodes[validation_count:]
    validation_mask = np.isin(episode_indices, validation_episodes)
    training_mask = np.isin(episode_indices, training_episodes)
    train_x_raw = observations[training_mask].copy()
    train_targets = normalized_targets[training_mask]
    validation_x_raw = observations[validation_mask].copy()
    validation_targets = normalized_targets[validation_mask]
    if not 0.0 <= config.mirror_training_fraction <= 1.0:
        raise ValueError("mirror training fraction must be within [0, 1]")
    mirror_count = round(train_x_raw.shape[0] * config.mirror_training_fraction)
    if mirror_count:
        mirror_indices = rng.permutation(train_x_raw.shape[0])[:mirror_count]
        mirrored_x, mirrored_targets = mirror_navigation_samples(
            train_x_raw[mirror_indices],
            train_targets[mirror_indices],
        )
        train_x_raw = np.concatenate((train_x_raw, mirrored_x), axis=0)
        train_targets = np.concatenate((train_targets, mirrored_targets), axis=0)
    if not config.condition_on_previous_action:
        train_x_raw[:, -3:] = 0.0
        validation_x_raw[:, -3:] = 0.0
    input_mean = np.mean(train_x_raw, axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.std(train_x_raw, axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(input_std, np.float32(1e-4))
    if not config.condition_on_previous_action:
        input_std[-3:] = 1.0
    train_x = np.asarray((train_x_raw - input_mean) / input_std, dtype=np.float32)
    validation_x = np.asarray(
        (validation_x_raw - input_mean) / input_std,
        dtype=np.float32,
    )
    parameters = _initial_parameters(config, rng)
    if not config.condition_on_previous_action:
        parameters["weight_1"][-3:, :] = 0.0
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    best_parameters = copy.deepcopy(parameters)
    best_epoch = 0
    best_validation_loss = float("inf")
    adam_step = 0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    component_weights = np.asarray((1.0, 1.5, 2.0), dtype=np.float32)

    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(train_targets.shape[0])
        for start in range(0, train_targets.shape[0], config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_x = train_x[batch_indices]
            batch_targets = train_targets[batch_indices]
            hidden_1, hidden_2, raw_outputs = _forward(batch_x, parameters)
            predictions = _normalized_outputs(raw_outputs)
            sample_weights = np.where(
                batch_targets[:, 2] >= 0.5,
                4.0,
                1.0,
            ).astype(np.float32)
            output_derivatives = np.column_stack(
                (
                    2.0 * predictions[:, 0] * (1.0 - predictions[:, 0]),
                    1.0 - predictions[:, 1] ** 2,
                    predictions[:, 2] * (1.0 - predictions[:, 2]),
                )
            ).astype(np.float32)
            output_gradient = (
                2.0
                * (predictions - batch_targets)
                * component_weights[None, :]
                * sample_weights[:, None]
                * output_derivatives
                / float(np.sum(sample_weights) * np.sum(component_weights))
            )
            gradients = {
                "weight_output": hidden_2.T @ output_gradient,
                "bias_output": np.sum(output_gradient, axis=0),
            }
            hidden_2_gradient = (
                output_gradient @ parameters["weight_output"].T
            ) * (1.0 - hidden_2 * hidden_2)
            gradients["weight_2"] = hidden_1.T @ hidden_2_gradient
            gradients["bias_2"] = np.sum(hidden_2_gradient, axis=0)
            hidden_1_gradient = (
                hidden_2_gradient @ parameters["weight_2"].T
            ) * (1.0 - hidden_1 * hidden_1)
            gradients["weight_1"] = batch_x.T @ hidden_1_gradient
            gradients["bias_1"] = np.sum(hidden_1_gradient, axis=0)
            adam_step += 1
            for name, parameter in parameters.items():
                gradient = np.asarray(gradients[name], dtype=np.float32)
                first_moment[name] = beta_1 * first_moment[name] + (1.0 - beta_1) * gradient
                second_moment[name] = beta_2 * second_moment[name] + (
                    1.0 - beta_2
                ) * gradient * gradient
                corrected_first = first_moment[name] / (1.0 - beta_1**adam_step)
                corrected_second = second_moment[name] / (1.0 - beta_2**adam_step)
                parameter -= config.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + epsilon
                )

        validation_loss, validation_accuracy, validation_mae = _metrics(
            validation_x,
            validation_targets,
            parameters,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_parameters = copy.deepcopy(parameters)
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            _, training_accuracy, _ = _metrics(train_x, train_targets, parameters)
            print(
                f"epoch={epoch:03d} train_acc={training_accuracy:.4f} "
                f"validation_acc={validation_accuracy:.4f} "
                f"validation_loss={validation_loss:.5f} "
                f"mae=({validation_mae[0]:.4f},"
                f"{validation_mae[1]:.4f},{validation_mae[2]:.4f})",
                flush=True,
            )

    training_loss, training_accuracy, _ = _metrics(
        train_x,
        train_targets,
        best_parameters,
    )
    validation_loss, validation_accuracy, validation_mae = _metrics(
        validation_x,
        validation_targets,
        best_parameters,
    )
    policy_id = config.policy_id
    if not policy_id:
        raise ValueError("policy ID must not be empty")
    if config.inference_strategy not in (
        CONTINUOUS_INFERENCE_STRATEGY,
        SENSOR_FUSION_INFERENCE_STRATEGY,
    ):
        raise ValueError("unsupported inference strategy")
    schema_version = (
        SENSOR_FUSION_CHECKPOINT_SCHEMA_VERSION
        if config.inference_strategy == SENSOR_FUSION_INFERENCE_STRATEGY
        else POLICY_CHECKPOINT_SCHEMA_VERSION
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(schema_version, dtype=np.int64),
        policy_id=np.asarray(policy_id),
        robot_checksum=np.asarray(robot.robot_checksum),
        dataset_sha256=np.asarray(dataset_sha256),
        inference_strategy=np.asarray(config.inference_strategy),
        avoidance_distance_m=np.asarray(config.avoidance_distance_m, dtype=np.float64),
        avoidance_gain=np.asarray(config.avoidance_gain, dtype=np.float64),
        avoidance_exponent=np.asarray(config.avoidance_exponent, dtype=np.float64),
        avoidance_activation=np.asarray(config.avoidance_activation, dtype=np.float64),
        avoidance_docking_suppression_m=np.asarray(
            config.avoidance_docking_suppression_m,
            dtype=np.float64,
        ),
        learned_turn_blend=np.asarray(config.learned_turn_blend, dtype=np.float64),
        avoidance_release=np.asarray(config.avoidance_release, dtype=np.float64),
        avoidance_reversal=np.asarray(config.avoidance_reversal, dtype=np.float64),
        avoidance_release_ticks=np.asarray(
            config.avoidance_release_ticks,
            dtype=np.int64,
        ),
        held_repulsion=np.asarray(config.held_repulsion, dtype=np.float64),
        turn_slew_per_update=np.asarray(config.turn_slew_per_update, dtype=np.float64),
        sparse_risk_threshold=np.asarray(config.sparse_risk_threshold, dtype=np.float64),
        sparse_turn_slew_per_update=np.asarray(
            config.sparse_turn_slew_per_update,
            dtype=np.float64,
        ),
        input_mean=input_mean,
        input_std=input_std,
        weight_1=best_parameters["weight_1"],
        bias_1=best_parameters["bias_1"],
        weight_2=best_parameters["weight_2"],
        bias_2=best_parameters["bias_2"],
        weight_output=best_parameters["weight_output"],
        bias_output=best_parameters["bias_output"],
    )
    result = TrainingResult(
        policy_id=policy_id,
        policy_sha256=_sha256_file(output_path),
        dataset_sha256=dataset_sha256,
        training_episode_count=int(training_episodes.size),
        validation_episode_count=int(validation_episodes.size),
        training_sample_count=int(train_targets.shape[0]),
        validation_sample_count=int(validation_targets.shape[0]),
        best_epoch=best_epoch,
        training_command_accuracy=training_accuracy,
        validation_command_accuracy=validation_accuracy,
        validation_loss=validation_loss,
        validation_forward_mae_mps=(
            float(validation_mae[0]) * POLICY_MAXIMUM_FORWARD_SPEED_MPS
        ),
        validation_turning_mae_rad_s=(
            float(validation_mae[1]) * POLICY_MAXIMUM_TURNING_RATE_RAD_S
        ),
        validation_stop_mae=float(validation_mae[2]),
    )
    evidence = {
        "schema_version": 1,
        "robot_checksum": robot.robot_checksum,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "config": asdict(config),
        "training_episode_indices": [int(value) for value in training_episodes],
        "validation_episode_indices": [int(value) for value in validation_episodes],
        "training_loss": training_loss,
        **asdict(result),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return result
