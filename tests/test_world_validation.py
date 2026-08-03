"""Mandatory rejection checks for unsafe or unusable worlds."""

import pytest

from muscle_memory.worlds import (
    ColliderKind,
    Dimensions3D,
    ObjectCategory,
    PhysicalProperties,
    PrimitiveCollider,
    Vec2,
    WorldObject,
)
from muscle_memory.worlds.generation import (
    WorldValidationError,
    generate_training_world,
    validate_training_world,
)


def _assert_rejected(world: object, expected_code: str) -> None:
    with pytest.raises(WorldValidationError) as caught:
        validate_training_world(world)  # type: ignore[arg-type]
    assert expected_code in caught.value.codes


def test_rejects_overlapping_objects() -> None:
    world = generate_training_world(101).world
    first, second, *rest = world.objects
    overlapping = second.model_copy(update={"position": first.position})
    candidate = world.model_copy(update={"objects": (first, overlapping, *rest)})

    _assert_rejected(candidate, "objects_overlap")


def test_rejects_object_outside_room() -> None:
    world = generate_training_world(102).world
    first, *rest = world.objects
    outside = first.model_copy(update={"position": Vec2(x=0.0, y=0.0)})
    candidate = world.model_copy(update={"objects": (outside, *rest)})

    _assert_rejected(candidate, "object_out_of_bounds")


def test_rejects_unsafe_physical_parameter() -> None:
    world = generate_training_world(103).world
    first, *rest = world.objects
    unsafe = first.model_copy(
        update={"physical": first.physical.model_copy(update={"mass_kg": 10_000.0})}
    )
    candidate = world.model_copy(update={"objects": (unsafe, *rest)})

    _assert_rejected(candidate, "physical_parameter_out_of_range")


def test_rejects_unapproved_primitive_for_category() -> None:
    world = generate_training_world(104).world
    first, *rest = world.objects
    invalid = first.model_copy(
        update={"collider": first.collider.model_copy(update={"kind": ColliderKind.CAPSULE})}
    )
    candidate = world.model_copy(update={"objects": (invalid, *rest)})

    _assert_rejected(candidate, "collider_not_approved")


def test_rejects_blocked_passage_and_missing_baseline_path() -> None:
    world = generate_training_world(105).world
    wall = tuple(
        WorldObject(
            object_id=f"box-wall-{index}",
            category=ObjectCategory.BOX,
            position=Vec2(x=4.0, y=0.5 + index),
            yaw_degrees=0.0,
            collider=PrimitiveCollider(
                kind=ColliderKind.BOX,
                dimensions=Dimensions3D(length_m=0.6, width_m=0.8, height_m=0.5),
            ),
            physical=PhysicalProperties(
                mass_kg=4.0,
                sliding_friction=0.6,
                restitution=0.1,
                movable=True,
            ),
        )
        for index in range(6)
    )
    candidate = world.model_copy(update={"objects": wall})

    with pytest.raises(WorldValidationError) as caught:
        validate_training_world(candidate)
    assert {
        "start_destination_disconnected",
        "baseline_path_missing",
    } <= caught.value.codes


def test_rejects_obstructed_start() -> None:
    world = generate_training_world(106).world
    first, *rest = world.objects
    blocked = first.model_copy(update={"position": world.start})
    candidate = world.model_copy(update={"objects": (blocked, *rest)})

    _assert_rejected(candidate, "start_obstructed")
