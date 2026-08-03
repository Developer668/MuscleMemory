"""Load and validate the versioned world-generation contract."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, FiniteFloat, model_validator

from muscle_memory.paths import REPOSITORY_ROOT
from muscle_memory.worlds.models import (
    ColliderKind,
    FrozenModel,
    ObjectCategory,
    WorldTemplate,
)


class NumericRange(FrozenModel):
    """Inclusive numeric safety range."""

    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def ordered(self) -> "NumericRange":
        """Reject inverted ranges before they reach generation."""
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self

    def contains(self, value: float) -> bool:
        """Return whether value falls inside this inclusive range."""
        return self.minimum <= value <= self.maximum


class IntegerRange(FrozenModel):
    """Inclusive integer generation range."""

    minimum: int = Field(ge=1)
    maximum: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self) -> "IntegerRange":
        """Reject inverted ranges before they reach generation."""
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class ObjectRule(FrozenModel):
    """Approved colliders and safe physical bounds for one category."""

    approved_colliders: tuple[ColliderKind, ...] = Field(min_length=1)
    length_m: NumericRange
    width_m: NumericRange
    height_m: NumericRange
    mass_kg: NumericRange
    sliding_friction: NumericRange
    restitution: NumericRange


class WorldRules(FrozenModel):
    """Complete deterministic generation and validation contract."""

    generation_version: int = Field(ge=1)
    template: WorldTemplate
    grid_resolution_m: FiniteFloat = Field(gt=0.0)
    robot_radius_m: FiniteFloat = Field(gt=0.0)
    minimum_clearance_m: FiniteFloat = Field(gt=0.0)
    object_separation_m: FiniteFloat = Field(ge=0.0)
    object_boundary_clearance_m: FiniteFloat = Field(ge=0.0)
    minimum_start_goal_distance_m: FiniteFloat = Field(gt=0.0)
    object_count: IntegerRange
    objects: dict[ObjectCategory, ObjectRule]

    @model_validator(mode="after")
    def complete_categories(self) -> "WorldRules":
        """Require a physical contract for every and only approved category."""
        if set(self.objects) != set(ObjectCategory):
            raise ValueError("object rules must cover every approved category exactly")
        return self


DEFAULT_RULES_PATH = REPOSITORY_ROOT / "config" / "worlds" / "foundation-v1.json"


@lru_cache(maxsize=1)
def load_world_rules(path: Path = DEFAULT_RULES_PATH) -> WorldRules:
    """Load the immutable default rules from version-controlled JSON."""
    return WorldRules.model_validate(json.loads(path.read_text(encoding="utf-8")))
