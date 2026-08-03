"""High-level policy isolation, observation, and checkpoint tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from muscle_memory.evaluation.development import assert_development_gate
from muscle_memory.evaluation.heldout import load_heldout_worlds
from muscle_memory.evaluation.promotion import evaluate_promotion
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.paths import (
    POLICY_V1_CHECKPOINT,
    POLICY_V2_CHECKPOINT,
    POLICY_V2_DEVELOPMENT_EVIDENCE,
    POLICY_V2_TRAINING_EVIDENCE,
)
from muscle_memory.policy.actions import POLICY_ACTION_COMMANDS, PolicyAction
from muscle_memory.policy.baseline import DirectGoalPolicy
from muscle_memory.policy.network import (
    POLICY_OUTPUT_COUNT,
    SENSOR_FUSION_INFERENCE_STRATEGY,
    BehaviorClonedPolicy,
    policy_file_sha256,
)
from muscle_memory.policy.observation import (
    NAVIGATION_OBSERVATION_SIZE,
    NavigationObservation,
    navigation_observation,
)
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.simulation.metrics import EpisodeMetricsTracker
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.world_scene import assemble_episode_scene
from muscle_memory.training.behavior_clone import (
    BehaviorCloneConfig,
    mirror_navigation_samples,
    train_behavior_clone,
)
from muscle_memory.training.dataset import DATASET_SCHEMA_VERSION, record_expert_episode
from muscle_memory.training.expert import ExpertPath
from muscle_memory.worlds.generation import generate_training_world
from muscle_memory.worlds.generation.models import ValidatedTrainingWorld


def _write_checkpoint(path: Path, *, robot_checksum: str) -> None:
    rng = np.random.default_rng(4)
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int64),
        policy_id=np.asarray("test-policy"),
        robot_checksum=np.asarray(robot_checksum),
        input_mean=np.zeros(NAVIGATION_OBSERVATION_SIZE, dtype=np.float32),
        input_std=np.ones(NAVIGATION_OBSERVATION_SIZE, dtype=np.float32),
        weight_1=rng.normal(size=(NAVIGATION_OBSERVATION_SIZE, 8)).astype(np.float32),
        bias_1=np.zeros(8, dtype=np.float32),
        weight_2=rng.normal(size=(8, 4)).astype(np.float32),
        bias_2=np.zeros(4, dtype=np.float32),
        weight_output=rng.normal(size=(4, POLICY_OUTPUT_COUNT)).astype(np.float32),
        bias_output=np.zeros(POLICY_OUTPUT_COUNT, dtype=np.float32),
    )


def _observation_for_seed(seed: int, depth_m: float):  # type: ignore[no-untyped-def]
    validated = generate_training_world(seed)
    scene = assemble_episode_scene(validated)
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    metrics = EpisodeMetricsTracker(scene, simulation.data).observe(simulation.data)
    return navigation_observation(
        simulation.data,
        scene.world.destination,
        np.full(48, depth_m, dtype=np.float64),
        metrics,
        TaskCommand(0.0, 0.0, 1.0),
    )


def _episode_result(
    index: int,
    *,
    policy_id: str,
    success: bool,
    collisions: int,
    path_efficiency: float,
) -> PolicyEpisodeResult:
    return PolicyEpisodeResult(
        episode_id=f"{policy_id}-{index}",
        world_id=f"world-{index}",
        world_seed=index,
        world_split="held_out",
        world_hash=f"hash-{index}",
        robot_checksum="a" * 64,
        policy_id=policy_id,
        policy_hash=f"hash-{policy_id}",
        success=success,
        failed_reasons=() if success else ("BODY_COLLISION",),
        time_to_resident_seconds=20.0 if success else None,
        simulated_duration_seconds=21.0 if success else 30.0,
        stop_distance_m=0.4 if success else 2.0,
        facing_error_degrees=5.0 if success else 90.0,
        stopped_speed_mps=0.01,
        falls=0,
        body_collisions=collisions,
        minimum_obstacle_clearance_m=0.3,
        maximum_tray_tilt_degrees=8.0,
        package_slipped=False,
        human_interventions=0,
        direct_distance_m=6.0,
        path_length_m=6.0 / path_efficiency,
        path_efficiency=path_efficiency,
        energy_joules=1000.0,
        task_policy_updates=210,
        trace=(),
    )


def test_navigation_observation_uses_fixed_finite_shape() -> None:
    observation = _observation_for_seed(7, 4.0)

    assert observation.values.shape == (NAVIGATION_OBSERVATION_SIZE,)
    assert observation.values.dtype == np.float32
    assert np.isfinite(observation.values).all()
    assert observation.destination_distance_m > 0.0


def test_policy_action_vocabulary_contains_only_three_allowed_outputs() -> None:
    assert set(POLICY_ACTION_COMMANDS) == set(PolicyAction)
    assert all(isinstance(command, TaskCommand) for command in POLICY_ACTION_COMMANDS.values())
    assert all(command.forward_speed_mps <= 0.3 for command in POLICY_ACTION_COMMANDS.values())
    assert all(
        abs(command.turning_rate_rad_s) <= 0.5 for command in POLICY_ACTION_COMMANDS.values()
    )


def test_checkpoint_loads_and_is_bound_to_current_robot(tmp_path: Path) -> None:
    robot = verify_mm01_bundle()
    checkpoint = tmp_path / "policy.npz"
    _write_checkpoint(checkpoint, robot_checksum=robot.robot_checksum)

    policy = BehaviorClonedPolicy.load(checkpoint)
    observation = _observation_for_seed(8, 8.0)

    assert policy.policy_id == "test-policy"
    command = policy.command(observation)
    assert 0.0 <= command.forward_speed_mps <= 0.3
    assert -0.5 <= command.turning_rate_rad_s <= 0.5
    assert 0.0 <= command.stop_probability <= 1.0

    _write_checkpoint(checkpoint, robot_checksum="0" * 64)
    with pytest.raises(RuntimeError, match="different robot"):
        BehaviorClonedPolicy.load(checkpoint)


def test_v1_and_v2_checkpoint_identities_are_distinct_and_evidence_bound() -> None:
    v1 = BehaviorClonedPolicy.load(POLICY_V1_CHECKPOINT)
    v2 = BehaviorClonedPolicy.load(POLICY_V2_CHECKPOINT)
    training = json.loads(POLICY_V2_TRAINING_EVIDENCE.read_text(encoding="utf-8"))
    development = json.loads(POLICY_V2_DEVELOPMENT_EVIDENCE.read_text(encoding="utf-8"))
    lock = json.loads(
        (POLICY_V2_DEVELOPMENT_EVIDENCE.parent / "lock.json").read_text(encoding="utf-8")
    )

    assert v1.policy_id == "delivery-v1-bc"
    assert v2.policy_id == "delivery-v2-sensor-fusion-hysteresis"
    assert v2.inference_strategy == SENSOR_FUSION_INFERENCE_STRATEGY
    assert v1.policy_hash == policy_file_sha256(POLICY_V1_CHECKPOINT)
    assert v2.policy_hash == policy_file_sha256(POLICY_V2_CHECKPOINT)
    assert v1.policy_hash != v2.policy_hash
    assert training["policy_sha256"] == v2.policy_hash
    assert development["candidate_policy_sha256"] == v2.policy_hash
    assert development["selection_status"] == "rejected_before_heldout"
    assert not development["promotion_preview"]["promotable"]
    assert lock["checkpoint_sha256"] == v2.policy_hash
    assert lock["heldout_access"] == "denied_by_development_gate"


def test_sensor_fusion_episode_state_resets_on_initial_stop_observation() -> None:
    policy = BehaviorClonedPolicy.load(POLICY_V2_CHECKPOINT)
    initial_values = np.ones(NAVIGATION_OBSERVATION_SIZE, dtype=np.float32)
    initial_values[-3:] = (0.0, 0.0, 1.0)
    running_values = initial_values.copy()
    running_values[-1] = 0.0
    initial = NavigationObservation(initial_values, 5.0, 0.8)
    running = NavigationObservation(running_values, 5.0, 0.8)

    first = policy.command(initial)
    second = policy.command(running)
    reset = policy.command(initial)

    assert first.turning_rate_rad_s == pytest.approx(0.15)
    assert second.turning_rate_rad_s == pytest.approx(0.3)
    assert reset == first


def test_failed_development_gate_blocks_heldout_evaluation() -> None:
    with pytest.raises(RuntimeError, match="held-out access denied"):
        assert_development_gate(
            POLICY_V2_DEVELOPMENT_EVIDENCE,
            POLICY_V2_CHECKPOINT,
            lock_path=POLICY_V2_DEVELOPMENT_EVIDENCE.with_name("lock.json"),
        )


def test_training_refuses_to_replace_an_immutable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "existing.npz"
    checkpoint.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        train_behavior_clone(
            dataset_path=tmp_path / "missing-dataset.npz",
            output_path=checkpoint,
            evidence_path=tmp_path / "training.json",
        )


def test_evaluation_imports_cannot_reach_teacher_or_world_generator() -> None:
    audit = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import muscle_memory.evaluation.runner; "
            "import muscle_memory.policy; "
            "assert not any(name.startswith('muscle_memory.training') for name in sys.modules); "
            "assert not any(name.startswith('muscle_memory.worlds.generation') "
            "for name in sys.modules)",
        ),
        capture_output=True,
        check=False,
        text=True,
    )

    assert audit.returncode == 0, audit.stderr


def test_expert_recorder_rejects_heldout_world_envelope() -> None:
    heldout = load_heldout_worlds()[0]
    path = ExpertPath(
        waypoints=(heldout.world.start, heldout.world.destination),
        obstacle_clearance_m=0.85,
        length_m=1.0,
    )

    with pytest.raises(TypeError, match="ValidatedTrainingWorld"):
        record_expert_episode(cast(ValidatedTrainingWorld, heldout), path)


def test_behavior_clone_splits_complete_episodes(tmp_path: Path) -> None:
    robot = verify_mm01_bundle()
    rng = np.random.default_rng(12)
    episode_count = 8
    samples_per_episode = 24
    sample_count = episode_count * samples_per_episode
    dataset = tmp_path / "dataset.npz"
    checkpoint = tmp_path / "policy.npz"
    evidence = tmp_path / "training.json"
    labels = np.tile(np.arange(len(PolicyAction), dtype=np.int64), sample_count // 6)
    commands = np.tile(
        np.asarray(
            (
                (0.0, 0.0, 1.0),
                (0.0, 0.5, 0.0),
                (0.0, -0.5, 0.0),
                (0.3, 0.12, 0.0),
                (0.3, 0.0, 0.0),
                (0.3, -0.12, 0.0),
            ),
            dtype=np.float32,
        ),
        (sample_count // 6, 1),
    )
    np.savez_compressed(
        dataset,
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
        robot_checksum=np.asarray(robot.robot_checksum),
        observations=rng.normal(size=(sample_count, NAVIGATION_OBSERVATION_SIZE)).astype(
            np.float32
        ),
        actions=labels,
        commands=commands,
        episode_indices=np.repeat(
            np.arange(episode_count, dtype=np.int32),
            samples_per_episode,
        ),
    )

    result = train_behavior_clone(
        dataset_path=dataset,
        output_path=checkpoint,
        evidence_path=evidence,
        config=BehaviorCloneConfig(
            hidden_1=12,
            hidden_2=8,
            epochs=2,
            batch_size=32,
            learning_rate=0.001,
            validation_episode_fraction=0.25,
            seed=9,
        ),
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))

    assert checkpoint.is_file()
    assert result.training_episode_count == 6
    assert result.validation_episode_count == 2
    assert set(payload["training_episode_indices"]).isdisjoint(
        payload["validation_episode_indices"]
    )
    assert len(payload["training_episode_indices"]) == 6
    assert len(payload["validation_episode_indices"]) == 2
    assert BehaviorClonedPolicy.load(checkpoint).policy_id == "delivery-v1-bc"


def test_direct_goal_baseline_is_obstacle_unaware_but_output_limited() -> None:
    observation = _observation_for_seed(9, 0.2)
    command = DirectGoalPolicy().command(observation)

    assert isinstance(command, TaskCommand)
    assert command.forward_speed_mps <= 0.3
    assert abs(command.turning_rate_rad_s) <= 0.5


def test_navigation_mirror_augmentation_is_an_involution() -> None:
    rng = np.random.default_rng(31)
    observations = rng.normal(size=(5, NAVIGATION_OBSERVATION_SIZE)).astype(np.float32)
    targets = rng.uniform(-1.0, 1.0, size=(5, POLICY_OUTPUT_COUNT)).astype(np.float32)

    mirrored_observations, mirrored_targets = mirror_navigation_samples(
        observations,
        targets,
    )
    restored_observations, restored_targets = mirror_navigation_samples(
        mirrored_observations,
        mirrored_targets,
    )

    assert np.array_equal(restored_observations, observations)
    assert np.array_equal(restored_targets, targets)


def test_promotion_requires_measured_paired_improvement() -> None:
    baseline = tuple(
        _episode_result(
            index,
            policy_id="v0",
            success=index < 4,
            collisions=1 if index < 4 else 0,
            path_efficiency=1.0,
        )
        for index in range(10)
    )
    candidate = tuple(
        _episode_result(
            index,
            policy_id="v1",
            success=index < 8,
            collisions=1 if index == 8 else 0,
            path_efficiency=0.9,
        )
        for index in range(10)
    )

    decision = evaluate_promotion(baseline, candidate)

    assert decision.promotable
    assert decision.success_rate_improvement == pytest.approx(0.4)
    assert decision.collision_rate_reduction == pytest.approx(0.75)
    assert all(decision.checks.values())

    mismatched = (candidate[1], candidate[0], *candidate[2:])
    with pytest.raises(ValueError, match="identical ordered worlds"):
        evaluate_promotion(baseline, mismatched)
