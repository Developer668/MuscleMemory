"""Strict gate that no generated world may bypass."""

import math

from pydantic import Field

from muscle_memory.worlds.generation._geometry import Aabb, object_aabb
from muscle_memory.worlds.generation.models import ValidatedTrainingWorld
from muscle_memory.worlds.generation.pathfinding import find_baseline_path
from muscle_memory.worlds.models import (
    ColliderKind,
    FrozenModel,
    TrainingWorld,
    Vec2,
)
from muscle_memory.worlds.rules import ObjectRule, WorldRules, load_world_rules


class ValidationIssue(FrozenModel):
    """One deterministic reason a candidate world was rejected."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    object_ids: tuple[str, ...] = ()


class WorldValidationError(ValueError):
    """Raised when a world fails one or more mandatory validation checks."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{issue.code}: {issue.message}" for issue in issues))

    @property
    def codes(self) -> frozenset[str]:
        """Return stable issue codes for callers and tests."""
        return frozenset(issue.code for issue in self.issues)


def _issue(code: str, message: str, *object_ids: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, object_ids=tuple(object_ids))


def _validate_object_physics(
    obstacle_id: str,
    *,
    collider_kind: ColliderKind,
    length_m: float,
    width_m: float,
    height_m: float,
    mass_kg: float,
    sliding_friction: float,
    restitution: float,
    rule: ObjectRule,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if collider_kind not in rule.approved_colliders:
        issues.append(
            _issue(
                "collider_not_approved",
                f"{collider_kind.value} is not approved for this object category",
                obstacle_id,
            )
        )
    if collider_kind is ColliderKind.CYLINDER and not math.isclose(
        length_m, width_m, abs_tol=1e-9
    ):
        issues.append(
            _issue(
                "invalid_cylinder_dimensions",
                "cylinder floor dimensions must have equal diameter",
                obstacle_id,
            )
        )
    values = {
        "length_m": (length_m, rule.length_m),
        "width_m": (width_m, rule.width_m),
        "height_m": (height_m, rule.height_m),
        "mass_kg": (mass_kg, rule.mass_kg),
        "sliding_friction": (sliding_friction, rule.sliding_friction),
        "restitution": (restitution, rule.restitution),
    }
    for name, (value, allowed) in values.items():
        if not allowed.contains(value):
            issues.append(
                _issue(
                    "physical_parameter_out_of_range",
                    f"{name}={value} is outside [{allowed.minimum}, {allowed.maximum}]",
                    obstacle_id,
                )
            )
    return issues


def _inside_room(bounds: Aabb, world: TrainingWorld, margin: float) -> bool:
    return (
        bounds.minimum_x >= margin
        and bounds.minimum_y >= margin
        and bounds.maximum_x <= world.template.width_m - margin
        and bounds.maximum_y <= world.template.depth_m - margin
    )


def _point_inside_navigation_bounds(point: Vec2, world: TrainingWorld, margin: float) -> bool:
    return (
        margin <= point.x <= world.template.width_m - margin
        and margin <= point.y <= world.template.depth_m - margin
    )


def validate_training_world(
    world: TrainingWorld,
    rules: WorldRules | None = None,
) -> ValidatedTrainingWorld:
    """Validate every mandatory invariant and return a gate-marked world."""
    active_rules = rules or load_world_rules()
    issues: list[ValidationIssue] = []

    if world.template != active_rules.template:
        issues.append(_issue("template_mismatch", "world must use the frozen 8 x 6 m template"))
    if world.generation_version != active_rules.generation_version:
        issues.append(
            _issue(
                "generation_version_mismatch",
                "world generation version does not match the active rules",
            )
        )
    if not (
        active_rules.object_count.minimum
        <= len(world.objects)
        <= active_rules.object_count.maximum
    ):
        issues.append(
            _issue(
                "object_count_out_of_range",
                "world object count falls outside the configured generation bounds",
            )
        )

    identifiers = [obstacle.object_id for obstacle in world.objects]
    if len(identifiers) != len(set(identifiers)):
        issues.append(_issue("duplicate_object_id", "object identifiers must be unique"))

    obstacle_bounds: list[tuple[str, Aabb]] = []
    for obstacle in world.objects:
        rule = active_rules.objects[obstacle.category]
        dimensions = obstacle.collider.dimensions
        issues.extend(
            _validate_object_physics(
                obstacle.object_id,
                collider_kind=obstacle.collider.kind,
                length_m=dimensions.length_m,
                width_m=dimensions.width_m,
                height_m=dimensions.height_m,
                mass_kg=obstacle.physical.mass_kg,
                sliding_friction=obstacle.physical.sliding_friction,
                restitution=obstacle.physical.restitution,
                rule=rule,
            )
        )
        bounds = object_aabb(obstacle)
        obstacle_bounds.append((obstacle.object_id, bounds))
        if not _inside_room(bounds, world, active_rules.object_boundary_clearance_m):
            issues.append(
                _issue(
                    "object_out_of_bounds",
                    "primitive collider extends outside the approved room bounds",
                    obstacle.object_id,
                )
            )

    separation_inflation = active_rules.object_separation_m / 2.0
    for index, (left_id, left) in enumerate(obstacle_bounds):
        for right_id, right in obstacle_bounds[index + 1 :]:
            if left.inflated(separation_inflation).overlaps(
                right.inflated(separation_inflation)
            ):
                issues.append(
                    _issue(
                        "objects_overlap",
                        "primitive colliders overlap or violate minimum separation",
                        left_id,
                        right_id,
                    )
                )

    navigation_margin = active_rules.robot_radius_m + active_rules.minimum_clearance_m
    for label, point in (("start", world.start), ("destination", world.destination)):
        if not _point_inside_navigation_bounds(point, world, navigation_margin):
            issues.append(
                _issue(
                    f"{label}_out_of_bounds",
                    f"{label} must preserve robot radius and clearance from room walls",
                )
            )
        if any(bounds.inflated(navigation_margin).contains(point) for _, bounds in obstacle_bounds):
            issues.append(
                _issue(
                    f"{label}_obstructed",
                    f"{label} lacks required clearance from an obstacle",
                )
            )

    if math.dist((world.start.x, world.start.y), (world.destination.x, world.destination.y)) < (
        active_rules.minimum_start_goal_distance_m
    ):
        issues.append(
            _issue(
                "start_destination_too_close",
                "start and destination do not meet the minimum task distance",
            )
        )

    baseline_path = find_baseline_path(world, active_rules)
    if baseline_path is None:
        issues.extend(
            (
                _issue(
                    "start_destination_disconnected",
                    "no clearance-aware connection exists between start and destination",
                ),
                _issue(
                    "baseline_path_missing",
                    "A* could not produce a valid baseline grid path",
                ),
            )
        )

    if issues:
        raise WorldValidationError(tuple(issues))
    assert baseline_path is not None
    return ValidatedTrainingWorld(world=world, baseline_path=baseline_path)
