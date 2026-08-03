"""Focused checks for validation-gated Stage 1 episode composition."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest

from muscle_memory.paths import G1_SCENE_XML, REPOSITORY_ROOT
from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.simulation.metrics import EpisodeMetricsTracker
from muscle_memory.simulation.runtime import HeadlessG1Simulation
from muscle_memory.simulation.sensors import (
    MAXIMUM_STEREO_DEPTH_M,
    SENSOR_USE_LABELS,
    EpisodeSensorExtractor,
    StereoDepthEstimator,
    overview_camera,
    third_person_camera,
)
from muscle_memory.simulation.world_scene import (
    PACKAGE_BODY_NAME,
    TRAY_BODY_NAME,
    assemble_episode_scene,
    assemble_payload_qualification_scene,
)
from muscle_memory.telemetry.models import SensorCategory, SignalUseLabel
from muscle_memory.worlds.generation import generate_training_world

SENSOR_PROFILE = REPOSITORY_ROOT / "config" / "robot" / "mm01-sensor-profile.json"


@pytest.fixture(scope="module")
def validated_world():  # type: ignore[no-untyped-def]
    return generate_training_world(7)


@pytest.fixture(scope="module")
def scene(validated_world):  # type: ignore[no-untyped-def]
    return assemble_episode_scene(validated_world)


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def test_episode_import_cannot_reach_training_generator_or_teacher() -> None:
    audit = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import muscle_memory.simulation.world_scene",
                    "import muscle_memory.simulation.metrics",
                    "import muscle_memory.simulation.sensors",
                    "assert not any(name.startswith('muscle_memory.worlds.generation') "
                    "for name in sys.modules)",
                )
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert audit.returncode == 0, audit.stderr


def test_raw_world_cannot_bypass_validation_envelope(validated_world) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TypeError, match="validated training-world envelope"):
        assemble_episode_scene(validated_world.world)  # type: ignore[arg-type]


def test_assembly_is_deterministic_and_preserves_qualified_robot_checksum(
    validated_world, scene  # type: ignore[no-untyped-def]
) -> None:
    second = assemble_episode_scene(validated_world)
    verification = verify_mm01_bundle()

    assert scene.robot_checksum == verification.robot_checksum
    assert second.robot_checksum == scene.robot_checksum
    assert second.world_hash == scene.world_hash
    assert second.assembly_hash == scene.assembly_hash
    assert second.model.nbody == scene.model.nbody
    assert np.array_equal(second.model.body_pos, scene.model.body_pos)
    assert np.array_equal(second.model.geom_size, scene.model.geom_size)


def test_world_uses_only_primitive_physics_colliders(scene) -> None:  # type: ignore[no-untyped-def]
    approved = {
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
    }
    for geom_id in scene.obstacle_geom_ids:
        assert scene.model.geom_type[geom_id] in approved
        assert scene.model.geom_dataid[geom_id] == -1


def test_capsule_composition_preserves_declared_extents_and_mass() -> None:
    validated = generate_training_world(1)
    obstacle = next(
        item for item in validated.world.objects if item.collider.kind.value == "capsule"
    )
    scene = assemble_episode_scene(validated)
    body_id = scene.model.body(f"mm01_obstacle_{obstacle.object_id}_body").id
    geom_ids = np.flatnonzero(scene.model.geom_bodyid == body_id)
    lower = np.full(3, math.inf)
    upper = np.full(3, -math.inf)
    for geom_id in geom_ids:
        radius_or_half_extents = scene.model.geom_size[geom_id]
        position = scene.model.geom_pos[geom_id]
        if scene.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            half_extents = radius_or_half_extents
        else:
            half_extents = np.array(
                [
                    radius_or_half_extents[0],
                    radius_or_half_extents[0],
                    radius_or_half_extents[1],
                ]
            )
        lower = np.minimum(lower, position - half_extents)
        upper = np.maximum(upper, position + half_extents)

    assert upper - lower == pytest.approx(
        [
            obstacle.collider.dimensions.length_m,
            obstacle.collider.dimensions.width_m,
            obstacle.collider.dimensions.height_m,
        ]
    )
    assert scene.model.body_mass[body_id] == pytest.approx(obstacle.physical.mass_kg)


def test_payload_is_world_side_and_robot_sensor_set_is_unchanged(scene) -> None:  # type: ignore[no-untyped-def]
    base = mujoco.MjModel.from_xml_path(G1_SCENE_XML.as_posix())
    model = scene.model

    assert model.nu == base.nu == 29
    assert _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu) == _names(
        base, mujoco.mjtObj.mjOBJ_ACTUATOR, base.nu
    )
    assert _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor) == _names(
        base, mujoco.mjtObj.mjOBJ_SENSOR, base.nsensor
    )
    assert np.array_equal(model.sensor_type, base.sensor_type)
    assert model.ncam == base.ncam
    assert model.body_parentid[model.body(TRAY_BODY_NAME).id] == 0
    assert model.body_parentid[model.body(PACKAGE_BODY_NAME).id] == 0
    assert model.neq == 4
    assert model.nq > base.nq


def test_flat_payload_qualification_fixture_preserves_robot_boundary() -> None:
    base = mujoco.MjModel.from_xml_path(G1_SCENE_XML.as_posix())
    fixture = assemble_payload_qualification_scene()

    assert fixture.model.nu == base.nu == 29
    assert fixture.model.nsensor == base.nsensor
    assert np.array_equal(fixture.model.sensor_type, base.sensor_type)
    assert fixture.model.neq == 4
    assert fixture.model.body(TRAY_BODY_NAME).id == fixture.tray_body_id
    assert fixture.model.body(PACKAGE_BODY_NAME).id == fixture.package_body_id


def test_composed_scene_runs_real_controller_without_world_joint_leakage(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)

    assert np.allclose(simulation.data.qpos[:2], [scene.world.start.x, scene.world.start.y])
    assert np.linalg.norm(simulation.data.efc_pos[:12]) < 1e-10
    for _ in range(20):
        simulation.step(lambda _time: TaskCommand(0.0, 0.0, 1.0))

    assert simulation.controller.inference_count == 4
    assert np.isfinite(simulation.data.qpos).all()
    assert np.isfinite(simulation.data.qvel).all()


def test_metrics_are_finite_and_explicitly_ground_truth(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    tracker = EpisodeMetricsTracker(scene, simulation.data)
    metrics = tracker.observe(simulation.data)

    assert metrics.signal_use is SignalUseLabel.SIMULATOR_GROUND_TRUTH
    assert metrics.body_collisions == 0
    assert math.isfinite(metrics.minimum_obstacle_clearance_m)
    assert math.isfinite(metrics.current_obstacle_clearance_m)
    assert metrics.minimum_obstacle_clearance_m >= 0.25
    assert metrics.current_tray_tilt_degrees < 1e-6
    assert not metrics.package_slipped


def test_sensor_snapshot_covers_all_categories_with_exact_labels(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
    snapshot = extractor.snapshot("episode-01:00000000")

    assert {reading.category for reading in snapshot.readings} == set(SensorCategory)
    assert {
        reading.category: reading.signal_use for reading in snapshot.readings
    } == SENSOR_USE_LABELS
    assert snapshot.stereo_vision_and_depth.values["frame_id"] == (
        "episode-01:00000000"
    )
    assert len(snapshot.joint_position_and_effort.values["position_rad"]) == 29
    assert snapshot.wrist_force_and_tray_balance.available is False
    assert snapshot.hand_pressure_and_slip.available is False
    assert snapshot.microphone_activity.available is False
    assert snapshot.battery_and_energy.available is True
    assert "simulator_debug_segmentation" not in snapshot.stereo_vision_and_depth.values


def test_frozen_sensor_profile_matches_runtime_categories_and_stereo_constants() -> None:
    profile = json.loads(SENSOR_PROFILE.read_text(encoding="utf-8"))
    categories = {item["name"]: item for item in profile["categories"]}

    assert set(categories) == {category.name.lower() for category in SensorCategory}
    assert {item["use"] for item in categories.values()} <= {
        label.value for label in SignalUseLabel
    }
    assert profile["stereo"]["baseline_m"] == 0.07
    assert profile["stereo"]["depth_sector_count"] == 48
    assert categories["microphone_activity"]["status"] == "unavailable"
    assert categories["wrist_force_and_tray_balance"]["status"] == "partial"


def test_payload_metrics_feed_only_available_policy_signals(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
    metrics = EpisodeMetricsTracker(scene, simulation.data).observe(simulation.data)

    snapshot = extractor.snapshot(
        "episode-01:00000004",
        np.ones(48, dtype=np.float64),
        payload_metrics=metrics,
    )

    assert snapshot.wrist_force_and_tray_balance.available
    assert snapshot.wrist_force_and_tray_balance.values["six_axis_force_torque"] is None
    assert snapshot.wrist_force_and_tray_balance.values["tray_tilt_degrees"] == pytest.approx(0.0)
    assert snapshot.hand_pressure_and_slip.available
    assert snapshot.hand_pressure_and_slip.values["pressure"] is None
    assert snapshot.hand_pressure_and_slip.values["package_slipped"] is False


def test_depth_sector_contract_rejects_non_profile_counts(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    extractor = EpisodeSensorExtractor(simulation.model, simulation.data)

    with pytest.raises(ValueError, match="32 to 64"):
        extractor.snapshot("episode-01:00000001", np.ones(31, dtype=np.float64))
    snapshot = extractor.snapshot(
        "episode-01:00000002", np.ones(32, dtype=np.float64)
    )
    assert len(snapshot.stereo_vision_and_depth.values["derived_depth_sectors"]) == 32


def test_stereo_matcher_recovers_shifted_image_depth() -> None:
    rng = np.random.default_rng(3)
    left = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    disparity_pixels = 8
    right = np.zeros_like(left)
    right[:, :-disparity_pixels] = left[:, disparity_pixels:]
    estimator = StereoDepthEstimator(32)

    sectors = estimator.estimate(left, right, vertical_fov_degrees=45.0)

    expected_depth = (96 / 2.0) / math.tan(math.radians(45.0) / 2.0)
    expected_depth *= 0.07 / disparity_pixels
    recovered = sectors[sectors < MAXIMUM_STEREO_DEPTH_M]
    assert recovered.size >= 16
    assert float(np.median(recovered)) == pytest.approx(expected_depth, abs=0.2)


def test_stereo_capture_produces_synchronized_nonblank_views(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    extractor = EpisodeSensorExtractor(simulation.model, simulation.data)
    renderer = mujoco.Renderer(simulation.model, height=48, width=64)
    try:
        bundle = extractor.capture_stereo(
            renderer,
            "episode-01:00000003",
            depth_sector_count=48,
        )
    finally:
        renderer.close()

    assert bundle.frame_id == "episode-01:00000003"
    assert bundle.left_eye_rgb.shape == (48, 64, 3)
    assert bundle.right_eye_rgb.shape == (48, 64, 3)
    assert bundle.stereo_composite.shape == (48, 128, 3)
    assert bundle.simulator_debug_depth.shape == (48, 64)
    assert bundle.simulator_debug_segmentation.shape == (48, 64, 2)
    assert bundle.derived_depth_sectors is not None
    assert bundle.derived_depth_sectors.shape == (48,)
    assert np.isfinite(bundle.derived_depth_sectors).all()
    assert np.all(bundle.derived_depth_sectors >= 0.2)
    assert np.all(bundle.derived_depth_sectors <= MAXIMUM_STEREO_DEPTH_M)
    assert np.any(bundle.left_eye_rgb)
    assert np.any(bundle.right_eye_rgb)
    assert not np.array_equal(bundle.left_eye_rgb, bundle.right_eye_rgb)
    visible_geom_ids = set(
        bundle.simulator_debug_segmentation[
            bundle.simulator_debug_segmentation[:, :, 1]
            == int(mujoco.mjtObj.mjOBJ_GEOM),
            0,
        ].tolist()
    )
    assert visible_geom_ids & scene.obstacle_geom_ids
    assert float(np.std(bundle.left_eye_rgb)) > 10.0


def test_third_person_camera_keeps_robot_visible_inside_boundaries(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    camera = third_person_camera(
        simulation.data,
        world_width_m=float(scene.world.template.width_m),
        world_depth_m=float(scene.world.template.depth_m),
    )
    renderer = mujoco.Renderer(simulation.model, height=120, width=160)
    try:
        renderer.update_scene(simulation.data, camera=camera)
        rgb = np.asarray(renderer.render(), dtype=np.uint8)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(simulation.data, camera=camera)
        segmentation = np.asarray(renderer.render(), dtype=np.int32)
    finally:
        renderer.close()

    visible_geom_ids = set(
        segmentation[
            segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
            0,
        ].tolist()
    )
    last_robot_body_id = scene.model.body("right_wrist_yaw_link").id
    robot_geom_ids = {
        geom_id
        for geom_id in range(scene.model.ngeom)
        if 0 < scene.model.geom_bodyid[geom_id] <= last_robot_body_id
    }
    assert visible_geom_ids & robot_geom_ids
    assert float(np.std(rgb)) > 10.0


def test_overview_camera_keeps_room_robot_and_obstacles_visible(scene) -> None:  # type: ignore[no-untyped-def]
    simulation = HeadlessG1Simulation(scene.model, scene.initialize_data)
    camera = overview_camera(
        world_width_m=float(scene.world.template.width_m),
        world_depth_m=float(scene.world.template.depth_m),
    )
    renderer = mujoco.Renderer(simulation.model, height=240, width=320)
    try:
        renderer.update_scene(simulation.data, camera=camera)
        rgb = np.asarray(renderer.render(), dtype=np.uint8)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(simulation.data, camera=camera)
        segmentation = np.asarray(renderer.render(), dtype=np.int32)
    finally:
        renderer.close()

    visible_geom_ids = set(
        segmentation[
            segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
            0,
        ].tolist()
    )
    last_robot_body_id = scene.model.body("right_wrist_yaw_link").id
    robot_geom_ids = {
        geom_id
        for geom_id in range(scene.model.ngeom)
        if 0 < scene.model.geom_bodyid[geom_id] <= last_robot_body_id
    }
    assert visible_geom_ids & robot_geom_ids
    assert visible_geom_ids & scene.obstacle_geom_ids
    assert float(np.std(rgb)) > 10.0
