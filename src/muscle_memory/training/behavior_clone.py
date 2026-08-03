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
from muscle_memory.policy.actions import POLICY_ACTION_COUNT
from muscle_memory.policy.network import POLICY_CHECKPOINT_SCHEMA_VERSION
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
    training_accuracy: float
    validation_accuracy: float
    validation_loss: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(exponential / np.sum(exponential, axis=1, keepdims=True), dtype=np.float32)


def _forward(
    inputs: npt.NDArray[np.float32],
    parameters: dict[str, npt.NDArray[np.float32]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    hidden_1 = np.tanh(inputs @ parameters["weight_1"] + parameters["bias_1"])
    hidden_2 = np.tanh(hidden_1 @ parameters["weight_2"] + parameters["bias_2"])
    logits = hidden_2 @ parameters["weight_output"] + parameters["bias_output"]
    return hidden_1, hidden_2, np.asarray(logits, dtype=np.float32)


def _metrics(
    inputs: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    parameters: dict[str, npt.NDArray[np.float32]],
) -> tuple[float, float]:
    probabilities = _softmax(_forward(inputs, parameters)[2])
    loss = float(-np.mean(np.log(probabilities[np.arange(labels.size), labels] + 1e-8)))
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == labels))
    return loss, accuracy


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
        "weight_output": weight(config.hidden_2, POLICY_ACTION_COUNT),
        "bias_output": np.zeros(POLICY_ACTION_COUNT, dtype=np.float32),
    }


def train_behavior_clone(
    *,
    dataset_path: Path = EXPERT_DATASET_V1,
    output_path: Path = POLICY_V1_CHECKPOINT,
    evidence_path: Path = POLICY_V1_TRAINING_EVIDENCE,
    config: BehaviorCloneConfig = DEFAULT_BEHAVIOR_CLONE_CONFIG,
) -> TrainingResult:
    """Train and serialize one immutable candidate from a world-grouped split."""
    robot = verify_mm01_bundle()
    dataset_sha256 = _sha256_file(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        schema_version = int(dataset["schema_version"])
        dataset_robot_checksum = str(dataset["robot_checksum"])
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        labels = np.asarray(dataset["actions"], dtype=np.int64)
        episode_indices = np.asarray(dataset["episode_indices"], dtype=np.int32)
    if schema_version != DATASET_SCHEMA_VERSION:
        raise RuntimeError("expert dataset schema changed")
    if dataset_robot_checksum != robot.robot_checksum:
        raise RuntimeError("expert dataset belongs to a different robot")
    if observations.ndim != 2 or observations.shape[1] != NAVIGATION_OBSERVATION_SIZE:
        raise RuntimeError("expert dataset observation tensor is invalid")
    if labels.shape != (observations.shape[0],) or episode_indices.shape != labels.shape:
        raise RuntimeError("expert dataset labels or episode membership are invalid")
    if not np.isfinite(observations).all() or np.any(labels < 0) or np.any(
        labels >= POLICY_ACTION_COUNT
    ):
        raise RuntimeError("expert dataset contains invalid values")

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
    train_y = labels[training_mask]
    validation_x_raw = observations[validation_mask].copy()
    validation_y = labels[validation_mask]
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
    class_counts = np.bincount(train_y, minlength=POLICY_ACTION_COUNT).astype(np.float32)
    if np.any(class_counts == 0):
        raise RuntimeError("expert dataset does not cover the complete action vocabulary")
    class_weights = np.sqrt(float(train_y.size) / (POLICY_ACTION_COUNT * class_counts))
    class_weights /= np.mean(class_weights)

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

    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(train_y.size)
        for start in range(0, train_y.size, config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            hidden_1, hidden_2, logits = _forward(batch_x, parameters)
            probabilities = _softmax(logits)
            sample_weights = class_weights[batch_y]
            output_gradient = probabilities
            output_gradient[np.arange(batch_y.size), batch_y] -= 1.0
            output_gradient *= sample_weights[:, None] / float(batch_y.size)
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

        validation_loss, validation_accuracy = _metrics(
            validation_x,
            validation_y,
            parameters,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_parameters = copy.deepcopy(parameters)
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            _, training_accuracy = _metrics(train_x, train_y, parameters)
            print(
                f"epoch={epoch:03d} train_acc={training_accuracy:.4f} "
                f"validation_acc={validation_accuracy:.4f} "
                f"validation_loss={validation_loss:.5f}",
                flush=True,
            )

    training_loss, training_accuracy = _metrics(train_x, train_y, best_parameters)
    validation_loss, validation_accuracy = _metrics(
        validation_x,
        validation_y,
        best_parameters,
    )
    policy_id = "delivery-v1-bc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(POLICY_CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
        policy_id=np.asarray(policy_id),
        robot_checksum=np.asarray(robot.robot_checksum),
        dataset_sha256=np.asarray(dataset_sha256),
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
        training_sample_count=int(train_y.size),
        validation_sample_count=int(validation_y.size),
        best_epoch=best_epoch,
        training_accuracy=training_accuracy,
        validation_accuracy=validation_accuracy,
        validation_loss=validation_loss,
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
