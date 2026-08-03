"""Determinism and breadth checks for the training-world generator."""

import hashlib
import subprocess
import sys

import pytest
from pydantic import ValidationError

from muscle_memory.worlds import (
    Dimensions3D,
    PhysicalProperties,
    PrimitiveCollider,
    Vec2,
    WorldObject,
    WorldTemplate,
)
from muscle_memory.worlds.generation import generate_training_world, validate_training_world
from muscle_memory.worlds.models import ObjectCategory
from muscle_memory.worlds.rules import NumericRange, WorldRules, load_world_rules


def test_generates_and_validates_twenty_training_layouts() -> None:
    rules = load_world_rules()
    layouts = [generate_training_world(seed) for seed in range(20)]

    assert len({layout.world.world_id for layout in layouts}) == 20
    for layout in layouts:
        assert layout.world.template.width_m == 8.0
        assert layout.world.template.depth_m == 6.0
        assert layout.world.split == "training"
        assert set(obstacle.category for obstacle in layout.world.objects) <= set(ObjectCategory)
        assert validate_training_world(layout.world, rules) == layout


def test_same_seed_is_content_stable() -> None:
    first = generate_training_world(8675309)
    second = generate_training_world(8675309)
    serialized = first.model_dump_json()

    assert first == second
    assert serialized == second.model_dump_json()
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "6f44a5df1b30715c73c49309d1b1443cf5894f30488d0b7fc494d6eb9a18ddb2"
    )


def test_different_seeds_change_layout_content() -> None:
    first = generate_training_world(41)
    second = generate_training_world(42)

    assert first.world.objects != second.world.objects


def test_world_records_are_frozen() -> None:
    generated = generate_training_world(7)

    with pytest.raises(ValidationError):
        generated.world.start.x = 2.0


def test_collider_schema_rejects_detailed_mesh_input() -> None:
    with pytest.raises(ValidationError, match="mesh_uri"):
        PrimitiveCollider.model_validate(
            {
                "kind": "box",
                "dimensions": {"length_m": 0.5, "width_m": 0.5, "height_m": 0.5},
                "mesh_uri": "s3://assets/detailed-generated.glb",
            }
        )


def test_base_schema_import_cannot_discover_expert_path() -> None:
    audit = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import muscle_memory.worlds as worlds",
                    "import muscle_memory.worlds.models as models",
                    "assert not hasattr(worlds, 'ValidatedTrainingWorld')",
                    "assert not hasattr(models, 'ValidatedTrainingWorld')",
                    "assert 'baseline_path' not in models.TrainingWorld.model_fields",
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


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_world_models_reject_non_finite_scalars(non_finite: float) -> None:
    valid_dimensions = {"length_m": 0.5, "width_m": 0.5, "height_m": 0.5}
    valid_physics = {
        "mass_kg": 2.0,
        "sliding_friction": 0.5,
        "restitution": 0.1,
        "movable": True,
    }
    object_payload = {
        "object_id": "box-00",
        "category": "box",
        "position": {"x": 2.0, "y": 2.0},
        "yaw_degrees": 0.0,
        "collider": {"kind": "box", "dimensions": valid_dimensions},
        "physical": valid_physics,
    }
    invalid_payloads = (
        (Vec2, {"x": non_finite, "y": 0.0}),
        (Vec2, {"x": 0.0, "y": non_finite}),
        (Dimensions3D, {**valid_dimensions, "length_m": non_finite}),
        (Dimensions3D, {**valid_dimensions, "width_m": non_finite}),
        (Dimensions3D, {**valid_dimensions, "height_m": non_finite}),
        (PhysicalProperties, {**valid_physics, "mass_kg": non_finite}),
        (PhysicalProperties, {**valid_physics, "sliding_friction": non_finite}),
        (PhysicalProperties, {**valid_physics, "restitution": non_finite}),
        (WorldObject, {**object_payload, "yaw_degrees": non_finite}),
        (
            WorldTemplate,
            {
                "template_id": "bounded-apartment-8x6-v1",
                "width_m": non_finite,
                "depth_m": 6.0,
            },
        ),
        (
            WorldTemplate,
            {
                "template_id": "bounded-apartment-8x6-v1",
                "width_m": 8.0,
                "depth_m": non_finite,
            },
        ),
    )
    for model, payload in invalid_payloads:
        with pytest.raises(ValidationError):
            model.model_validate(payload)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_world_rules_reject_non_finite_scalars(non_finite: float) -> None:
    with pytest.raises(ValidationError):
        NumericRange(minimum=non_finite, maximum=1.0)
    with pytest.raises(ValidationError):
        NumericRange(minimum=0.0, maximum=non_finite)

    rule_payload = load_world_rules().model_dump(mode="json")
    continuous_fields = (
        "grid_resolution_m",
        "robot_radius_m",
        "minimum_clearance_m",
        "object_separation_m",
        "object_boundary_clearance_m",
        "minimum_start_goal_distance_m",
    )
    for field_name in continuous_fields:
        invalid = {**rule_payload, field_name: non_finite}
        with pytest.raises(ValidationError):
            WorldRules.model_validate(invalid)
