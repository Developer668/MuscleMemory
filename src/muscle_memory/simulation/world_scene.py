"""Deterministic, validation-gated episode scene composition."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

import mujoco  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from muscle_memory.paths import G1_SCENE_XML
from muscle_memory.robot.identity import verify_candidate_bundle
from muscle_memory.worlds.models import ColliderKind, TrainingWorld, WorldObject

ASSEMBLY_SCHEMA_VERSION = 1
TRAY_BODY_NAME = "mm01_payload_tray"
TRAY_GEOM_NAME = "mm01_payload_tray_geom"
TRAY_JOINT_NAME = "mm01_payload_tray_free"
PACKAGE_BODY_NAME = "mm01_payload_package"
PACKAGE_GEOM_NAME = "mm01_payload_package_geom"
PACKAGE_JOINT_NAME = "mm01_payload_package_free"
LEFT_EYE_NAME = "mm01_left_eye"
RIGHT_EYE_NAME = "mm01_right_eye"

_ROBOT_COLLISION_GEOMS = (
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_hand_collision",
    "right_hand_collision",
)


class ValidatedWorldEnvelope(Protocol):
    """Narrow runtime boundary supplied by the training-world validation gate."""

    world: TrainingWorld


class WorldAssemblyError(RuntimeError):
    """Raised when an episode scene cannot preserve the frozen robot boundary."""


@dataclass(frozen=True, slots=True)
class EpisodeScene:
    """A compiled episode model plus deterministic initialization metadata."""

    model: mujoco.MjModel
    world: TrainingWorld
    robot_checksum: str
    world_hash: str
    assembly_hash: str
    robot_geom_ids: frozenset[int]
    obstacle_geom_ids: frozenset[int]
    tray_body_id: int
    package_body_id: int

    def initialize_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Translate the robot and both world-side payload bodies to the world start."""
        if model is not self.model:
            raise WorldAssemblyError("episode initializer received a different MuJoCo model")
        translation = np.array([self.world.start.x, self.world.start.y], dtype=np.float64)
        data.qpos[:2] += translation
        for joint_name in (TRAY_JOINT_NAME, PACKAGE_JOINT_NAME):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise WorldAssemblyError(f"assembled model is missing {joint_name}")
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_address : qpos_address + 2] += translation


@dataclass(frozen=True, slots=True)
class PayloadQualificationScene:
    """Flat qualification fixture with only the approved external payload assembly."""

    model: mujoco.MjModel
    robot_checksum: str
    tray_body_id: int
    package_body_id: int


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, object_id)
    return value or ""


def _robot_signature(model: mujoco.MjModel) -> str:
    """Hash only frozen robot-owned model fields, excluding composed world state."""
    body_names = [_name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody)]
    robot_body_count = body_names.index("right_wrist_yaw_link") + 1
    sensor_names = [
        _name(model, mujoco.mjtObj.mjOBJ_SENSOR, index) for index in range(model.nsensor)
    ]
    actuator_names = [
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    ]
    payload = {
        "body_names": body_names[:robot_body_count],
        "body_mass": model.body_mass[:robot_body_count].tolist(),
        "body_inertia": model.body_inertia[:robot_body_count].tolist(),
        "joint_names": [
            _name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(min(30, model.njnt))
        ],
        "joint_type": model.jnt_type[: min(30, model.njnt)].tolist(),
        "joint_axis": model.jnt_axis[: min(30, model.njnt)].tolist(),
        "actuator_names": actuator_names,
        "actuator_trnid": model.actuator_trnid.tolist(),
        "sensor_names": sensor_names,
        "sensor_type": model.sensor_type.tolist(),
        "sensor_dim": model.sensor_dim.tolist(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _relative_pose(
    first_position: npt.NDArray[np.float64],
    first_quaternion: npt.NDArray[np.float64],
    second_position: npt.NDArray[np.float64],
    second_quaternion: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    inverse_position = np.zeros(3, dtype=np.float64)
    inverse_quaternion = np.zeros(4, dtype=np.float64)
    relative_position = np.zeros(3, dtype=np.float64)
    relative_quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_negPose(
        inverse_position, inverse_quaternion, first_position, first_quaternion
    )
    mujoco.mju_mulPose(
        relative_position,
        relative_quaternion,
        inverse_position,
        inverse_quaternion,
        second_position,
        second_quaternion,
    )
    return relative_position, relative_quaternion


def _add_world_geom(
    body: mujoco.MjsBody, obstacle: WorldObject, geom_name: str
) -> tuple[str, ...]:
    dimensions = obstacle.collider.dimensions
    def common(name: str, mass: float) -> dict[str, object]:
        return {
            "name": name,
            "contype": 1,
            "conaffinity": 1,
            "friction": [obstacle.physical.sliding_friction, 0.005, 0.0001],
            "solref": [0.02, max(0.1, 1.0 - obstacle.physical.restitution)],
            "mass": mass,
            "rgba": [0.42, 0.48, 0.52, 1.0],
        }

    if obstacle.collider.kind is ColliderKind.BOX:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                dimensions.length_m / 2.0,
                dimensions.width_m / 2.0,
                dimensions.height_m / 2.0,
            ],
            **common(geom_name, float(obstacle.physical.mass_kg)),
        )
        return (geom_name,)
    elif obstacle.collider.kind is ColliderKind.CYLINDER:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[dimensions.length_m / 2.0, dimensions.height_m / 2.0],
            **common(geom_name, float(obstacle.physical.mass_kg)),
        )
        return (geom_name,)

    radius = min(float(dimensions.length_m), float(dimensions.width_m)) / 2.0
    long_extent = max(float(dimensions.length_m), float(dimensions.width_m))
    half_segment = max(long_extent / 2.0 - radius, 0.0)
    height = float(dimensions.height_m)
    rectangle_area = 4.0 * half_segment * radius
    cap_area = math.pi * radius * radius
    total_area = rectangle_area + cap_area
    total_mass = float(obstacle.physical.mass_kg)
    names: list[str] = []
    if half_segment > 1e-6:
        centre_name = f"{geom_name}_centre"
        centre_size = (
            [half_segment, radius, height / 2.0]
            if dimensions.length_m >= dimensions.width_m
            else [radius, half_segment, height / 2.0]
        )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=centre_size,
            **common(centre_name, total_mass * rectangle_area / total_area),
        )
        names.append(centre_name)
        for suffix, sign in (("end_a", -1.0), ("end_b", 1.0)):
            end_name = f"{geom_name}_{suffix}"
            position = (
                [sign * half_segment, 0.0, 0.0]
                if dimensions.length_m >= dimensions.width_m
                else [0.0, sign * half_segment, 0.0]
            )
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=position,
                size=[radius, height / 2.0],
                **common(end_name, total_mass * cap_area / total_area / 2.0),
            )
            names.append(end_name)
    else:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, height / 2.0],
            **common(geom_name, total_mass),
        )
        names.append(geom_name)
    return tuple(names)


def _add_boundaries(spec: mujoco.MjSpec, width: float, depth: float) -> list[str]:
    wall_height = 1.5
    wall_thickness = 0.08
    walls = (
        (
            "west",
            [-wall_thickness / 2.0, depth / 2.0, wall_height / 2.0],
            [wall_thickness / 2.0, depth / 2.0, wall_height / 2.0],
        ),
        (
            "east",
            [width + wall_thickness / 2.0, depth / 2.0, wall_height / 2.0],
            [wall_thickness / 2.0, depth / 2.0, wall_height / 2.0],
        ),
        (
            "south",
            [width / 2.0, -wall_thickness / 2.0, wall_height / 2.0],
            [width / 2.0, wall_thickness / 2.0, wall_height / 2.0],
        ),
        (
            "north",
            [width / 2.0, depth + wall_thickness / 2.0, wall_height / 2.0],
            [width / 2.0, wall_thickness / 2.0, wall_height / 2.0],
        ),
    )
    geom_names: list[str] = []
    for side, position, size in walls:
        geom_name = f"mm01_boundary_{side}"
        body = spec.worldbody.add_body(name=f"{geom_name}_body", pos=position)
        body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            contype=1,
            conaffinity=1,
            friction=[0.8, 0.005, 0.0001],
            rgba=[0.72, 0.74, 0.76, 1.0],
        )
        geom_names.append(geom_name)
    return geom_names


def _add_payload(
    spec: mujoco.MjSpec,
    base_model: mujoco.MjModel,
    preceding_freejoint_qpos: list[float],
) -> None:
    base_data = mujoco.MjData(base_model)
    key_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(base_model, base_data, key_id)
    mujoco.mj_forward(base_model, base_data)

    palm_position = np.asarray(base_data.site("right_palm").xpos, dtype=np.float64)
    right_wrist = base_data.body("right_wrist_yaw_link")
    left_wrist = base_data.body("left_wrist_yaw_link")
    torso = base_data.body("torso_link")
    tray_position = palm_position + np.array([0.13, 0.0, 0.0], dtype=np.float64)
    tray_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    package_position = tray_position + np.array([0.0, 0.0, 0.055], dtype=np.float64)
    package_quaternion = tray_quaternion.copy()

    tray = spec.worldbody.add_body(name=TRAY_BODY_NAME, pos=tray_position)
    tray.add_freejoint(name=TRAY_JOINT_NAME)
    tray.add_geom(
        name=TRAY_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.19, 0.14, 0.0125],
        mass=0.4,
        contype=1,
        conaffinity=1,
        friction=[0.8, 0.005, 0.0001],
        rgba=[0.35, 0.37, 0.4, 1.0],
    )
    package = spec.worldbody.add_body(name=PACKAGE_BODY_NAME, pos=package_position)
    package.add_freejoint(name=PACKAGE_JOINT_NAME)
    package.add_geom(
        name=PACKAGE_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.06, 0.05, 0.03],
        mass=0.25,
        contype=1,
        conaffinity=1,
        friction=[0.9, 0.005, 0.0001],
        rgba=[0.72, 0.22, 0.16, 1.0],
    )

    wrist_relative_position, wrist_relative_quaternion = _relative_pose(
        tray_position,
        tray_quaternion,
        np.asarray(right_wrist.xpos, dtype=np.float64),
        np.asarray(right_wrist.xquat, dtype=np.float64),
    )
    left_wrist_relative_position, left_wrist_relative_quaternion = _relative_pose(
        tray_position,
        tray_quaternion,
        np.asarray(left_wrist.xpos, dtype=np.float64),
        np.asarray(left_wrist.xquat, dtype=np.float64),
    )
    torso_relative_position, torso_relative_quaternion = _relative_pose(
        tray_position,
        tray_quaternion,
        np.asarray(torso.xpos, dtype=np.float64),
        np.asarray(torso.xquat, dtype=np.float64),
    )
    package_relative_position, package_relative_quaternion = _relative_pose(
        package_position,
        package_quaternion,
        tray_position,
        tray_quaternion,
    )
    spec.add_equality(
        name="mm01_wrist_tray_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=TRAY_BODY_NAME,
        name2="right_wrist_yaw_link",
        data=[
            0.0,
            0.0,
            0.0,
            *wrist_relative_position,
            *wrist_relative_quaternion,
            1.0,
        ],
    )
    spec.add_equality(
        name="mm01_package_tray_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=PACKAGE_BODY_NAME,
        name2=TRAY_BODY_NAME,
        data=[
            0.0,
            0.0,
            0.0,
            *package_relative_position,
            *package_relative_quaternion,
            1.0,
        ],
    )
    spec.add_equality(
        name="mm01_left_wrist_tray_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=TRAY_BODY_NAME,
        name2="left_wrist_yaw_link",
        data=[
            0.0,
            0.0,
            0.0,
            *left_wrist_relative_position,
            *left_wrist_relative_quaternion,
            1.0,
        ],
    )
    spec.add_equality(
        name="mm01_torso_tray_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=TRAY_BODY_NAME,
        name2="torso_link",
        data=[
            0.0,
            0.0,
            0.0,
            *torso_relative_position,
            *torso_relative_quaternion,
            1.0,
        ],
    )
    spec.add_exclude(
        name="mm01_payload_internal_exclude",
        bodyname1=TRAY_BODY_NAME,
        bodyname2=PACKAGE_BODY_NAME,
    )

    payload_qpos = [
        *tray_position,
        *tray_quaternion,
        *package_position,
        *package_quaternion,
    ]
    for key in spec.keys:
        key.qpos = [*key.qpos, *preceding_freejoint_qpos, *payload_qpos]


def assemble_payload_qualification_scene() -> PayloadQualificationScene:
    """Compile the payload fixture without importing a world generator or path teacher."""
    verification_before = verify_candidate_bundle()
    spec = mujoco.MjSpec.from_file(G1_SCENE_XML.as_posix())
    base_model = mujoco.MjModel.from_xml_path(G1_SCENE_XML.as_posix())
    base_signature = _robot_signature(base_model)
    _add_payload(spec, base_model, [])
    model = spec.compile()
    if _robot_signature(model) != base_signature:
        raise WorldAssemblyError("payload assembly changed frozen robot model fields")
    verification_after = verify_candidate_bundle()
    if verification_after.aggregate_sha256 != verification_before.aggregate_sha256:
        raise WorldAssemblyError("payload assembly changed the frozen robot checksum")
    return PayloadQualificationScene(
        model=model,
        robot_checksum=verification_after.aggregate_sha256,
        tray_body_id=model.body(TRAY_BODY_NAME).id,
        package_body_id=model.body(PACKAGE_BODY_NAME).id,
    )


def assemble_episode_scene(envelope: ValidatedWorldEnvelope) -> EpisodeScene:
    """Compile a validated training world around unchanged candidate robot bytes."""
    world = getattr(envelope, "world", None)
    if not isinstance(world, TrainingWorld):
        raise TypeError("episode assembly requires a validated training-world envelope")

    verification_before = verify_candidate_bundle()
    spec = mujoco.MjSpec.from_file(G1_SCENE_XML.as_posix())
    base_model = mujoco.MjModel.from_xml_path(G1_SCENE_XML.as_posix())
    base_signature = _robot_signature(base_model)

    obstacle_geom_names = _add_boundaries(
        spec, float(world.template.width_m), float(world.template.depth_m)
    )
    movable_obstacle_qpos: list[float] = []
    for obstacle in world.objects:
        offset = obstacle.collider.centre_offset
        yaw = math.radians(float(obstacle.yaw_degrees))
        offset_x = math.cos(yaw) * float(offset.x) - math.sin(yaw) * float(offset.y)
        offset_y = math.sin(yaw) * float(offset.x) + math.cos(yaw) * float(offset.y)
        dimensions = obstacle.collider.dimensions
        body = spec.worldbody.add_body(
            name=f"mm01_obstacle_{obstacle.object_id}_body",
            pos=[
                float(obstacle.position.x) + offset_x,
                float(obstacle.position.y) + offset_y,
                float(dimensions.height_m) / 2.0,
            ],
            euler=[0.0, 0.0, yaw],
        )
        if obstacle.physical.movable:
            body.add_freejoint(name=f"mm01_obstacle_{obstacle.object_id}_free")
            movable_obstacle_qpos.extend(
                [
                    float(obstacle.position.x) + offset_x,
                    float(obstacle.position.y) + offset_y,
                    float(dimensions.height_m) / 2.0,
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ]
            )
        geom_name = f"mm01_obstacle_{obstacle.object_id}"
        obstacle_geom_names.extend(_add_world_geom(body, obstacle, geom_name))

    _add_payload(spec, base_model, movable_obstacle_qpos)
    for robot_geom in _ROBOT_COLLISION_GEOMS:
        for world_geom in obstacle_geom_names:
            spec.add_pair(
                name=f"mm01_contact_{robot_geom}_{world_geom}",
                geomname1=robot_geom,
                geomname2=world_geom,
                condim=3,
                friction=[0.8, 0.005, 0.0001, 0.0001, 0.0001],
            )

    model = spec.compile()
    if _robot_signature(model) != base_signature:
        raise WorldAssemblyError("world assembly changed frozen robot model fields")
    verification_after = verify_candidate_bundle()
    if verification_after.aggregate_sha256 != verification_before.aggregate_sha256:
        raise WorldAssemblyError("world assembly changed the frozen robot checksum")

    robot_geom_ids = frozenset(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in _ROBOT_COLLISION_GEOMS
    )
    obstacle_geom_ids = frozenset(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in obstacle_geom_names
    )
    if -1 in robot_geom_ids or -1 in obstacle_geom_ids:
        raise WorldAssemblyError("assembled collision geometry could not be resolved")

    world_json = world.model_dump_json()
    world_hash = hashlib.sha256(world_json.encode("utf-8")).hexdigest()
    assembly_payload = json.dumps(
        {
            "schema_version": ASSEMBLY_SCHEMA_VERSION,
            "robot_checksum": verification_after.aggregate_sha256,
            "world_hash": world_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return EpisodeScene(
        model=model,
        world=world,
        robot_checksum=verification_after.aggregate_sha256,
        world_hash=world_hash,
        assembly_hash=hashlib.sha256(assembly_payload.encode("ascii")).hexdigest(),
        robot_geom_ids=robot_geom_ids,
        obstacle_geom_ids=obstacle_geom_ids,
        tray_body_id=model.body(TRAY_BODY_NAME).id,
        package_body_id=model.body(PACKAGE_BODY_NAME).id,
    )
