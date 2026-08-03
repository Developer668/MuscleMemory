"""Deterministic seeded generator for validated training layouts."""

import random

from muscle_memory.worlds.generation._geometry import object_aabb
from muscle_memory.worlds.generation.models import ValidatedTrainingWorld
from muscle_memory.worlds.generation.validation import (
    WorldValidationError,
    validate_training_world,
)
from muscle_memory.worlds.models import (
    ColliderKind,
    Dimensions3D,
    ObjectCategory,
    PhysicalProperties,
    PrimitiveCollider,
    TrainingWorld,
    Vec2,
    WorldObject,
)
from muscle_memory.worlds.rules import NumericRange, WorldRules, load_world_rules

_CATEGORY_ORDER = tuple(ObjectCategory)
_ANCHOR_PAIRS = (
    (Vec2(x=0.8, y=0.8), Vec2(x=7.2, y=5.2)),
    (Vec2(x=0.8, y=5.2), Vec2(x=7.2, y=0.8)),
    (Vec2(x=7.2, y=0.8), Vec2(x=0.8, y=5.2)),
    (Vec2(x=7.2, y=5.2), Vec2(x=0.8, y=0.8)),
)


def _sample(rng: random.Random, allowed: NumericRange) -> float:
    return round(rng.uniform(allowed.minimum, allowed.maximum), 3)


def _candidate_object(
    rng: random.Random,
    *,
    index: int,
    rules: WorldRules,
) -> WorldObject:
    category = _CATEGORY_ORDER[rng.randrange(len(_CATEGORY_ORDER))]
    rule = rules.objects[category]
    kind = rule.approved_colliders[rng.randrange(len(rule.approved_colliders))]
    length = _sample(rng, rule.length_m)
    width = _sample(rng, rule.width_m)
    if kind is ColliderKind.CYLINDER:
        diameter_range = NumericRange(
            minimum=max(rule.length_m.minimum, rule.width_m.minimum),
            maximum=min(rule.length_m.maximum, rule.width_m.maximum),
        )
        length = width = _sample(rng, diameter_range)
    dimensions = Dimensions3D(
        length_m=length,
        width_m=width,
        height_m=_sample(rng, rule.height_m),
    )
    half_extent = max(length, width) / 2.0
    boundary = rules.object_boundary_clearance_m + half_extent
    position = Vec2(
        x=round(rng.uniform(boundary, rules.template.width_m - boundary), 3),
        y=round(rng.uniform(boundary, rules.template.depth_m - boundary), 3),
    )
    yaw = 0.0 if kind is ColliderKind.CYLINDER else float(90 * rng.randrange(4))
    return WorldObject(
        object_id=f"{category.value.replace('_', '-')}-{index:02d}",
        category=category,
        position=position,
        yaw_degrees=yaw,
        collider=PrimitiveCollider(kind=kind, dimensions=dimensions),
        physical=PhysicalProperties(
            mass_kg=_sample(rng, rule.mass_kg),
            sliding_friction=_sample(rng, rule.sliding_friction),
            restitution=_sample(rng, rule.restitution),
            movable=category is not ObjectCategory.TABLE,
        ),
    )


def _can_place(
    candidate: WorldObject,
    placed: list[WorldObject],
    *,
    start: Vec2,
    destination: Vec2,
    rules: WorldRules,
) -> bool:
    candidate_bounds = object_aabb(candidate)
    navigation_margin = rules.robot_radius_m + rules.minimum_clearance_m
    if candidate_bounds.inflated(navigation_margin).contains(start):
        return False
    if candidate_bounds.inflated(navigation_margin).contains(destination):
        return False
    separation = rules.object_separation_m / 2.0
    inflated = candidate_bounds.inflated(separation)
    return not any(inflated.overlaps(object_aabb(item).inflated(separation)) for item in placed)


def _generate_candidate(rng: random.Random, seed: int, rules: WorldRules) -> TrainingWorld | None:
    start, destination = _ANCHOR_PAIRS[rng.randrange(len(_ANCHOR_PAIRS))]
    object_count = rng.randint(rules.object_count.minimum, rules.object_count.maximum)
    placed: list[WorldObject] = []
    for index in range(object_count):
        for _ in range(200):
            obstacle = _candidate_object(rng, index=index, rules=rules)
            if _can_place(
                obstacle,
                placed,
                start=start,
                destination=destination,
                rules=rules,
            ):
                placed.append(obstacle)
                break
        else:
            return None
    return TrainingWorld(
        world_id=f"train-v{rules.generation_version}-{seed:016x}",
        seed=seed,
        generation_version=rules.generation_version,
        template=rules.template,
        start=start,
        destination=destination,
        objects=tuple(placed),
    )


def generate_training_world(
    seed: int,
    rules: WorldRules | None = None,
) -> ValidatedTrainingWorld:
    """Generate a deterministic training world and pass the strict gate."""
    active_rules = rules or load_world_rules()
    if not 0 <= seed <= (2**63) - 1:
        raise ValueError("seed must be between 0 and 2**63 - 1")
    rng = random.Random(seed ^ (active_rules.generation_version << 48))
    last_error: WorldValidationError | None = None
    for _ in range(100):
        candidate = _generate_candidate(rng, seed, active_rules)
        if candidate is None:
            continue
        try:
            return validate_training_world(candidate, active_rules)
        except WorldValidationError as error:
            last_error = error
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"could not generate a valid world for seed {seed}{detail}")
