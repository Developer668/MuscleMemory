"""Immutable world descriptions with no generation or evaluation behavior."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class FrozenModel(BaseModel):
    """Base for immutable, strict world records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ObjectCategory(StrEnum):
    """Object categories admitted by the bounded apartment MVP."""

    CHAIR = "chair"
    BOX = "box"
    TABLE = "table"
    STOOL = "stool"
    LAUNDRY_BASKET = "laundry_basket"


class ColliderKind(StrEnum):
    """Deterministic physics primitives approved for obstacle collision."""

    BOX = "box"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"


class Vec2(FrozenModel):
    """A point in world-floor coordinates, in metres."""

    x: FiniteFloat
    y: FiniteFloat


class Dimensions3D(FrozenModel):
    """Full collider extents, rather than half-extents, in metres."""

    length_m: FiniteFloat = Field(gt=0.0)
    width_m: FiniteFloat = Field(gt=0.0)
    height_m: FiniteFloat = Field(gt=0.0)


class PrimitiveCollider(FrozenModel):
    """Physics-only collider description.

    This schema intentionally has no asset or mesh field. Strict extra-field rejection
    prevents a detailed visual mesh from being supplied as collision geometry.
    """

    kind: ColliderKind
    dimensions: Dimensions3D
    centre_offset: Vec2 = Vec2(x=0.0, y=0.0)


class PhysicalProperties(FrozenModel):
    """Physical values subject to category-specific safety bounds."""

    mass_kg: FiniteFloat = Field(gt=0.0)
    sliding_friction: FiniteFloat = Field(ge=0.0)
    restitution: FiniteFloat = Field(ge=0.0, le=1.0)
    movable: bool


class WorldObject(FrozenModel):
    """An obstacle with a primitive physics shape and optional cosmetic asset."""

    object_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: ObjectCategory
    position: Vec2
    yaw_degrees: FiniteFloat
    collider: PrimitiveCollider
    physical: PhysicalProperties
    visual_asset_uri: str | None = None


class WorldTemplate(FrozenModel):
    """The one frozen apartment footprint supported by the MVP."""

    template_id: Literal["bounded-apartment-8x6-v1"]
    width_m: FiniteFloat = Field(gt=0.0)
    depth_m: FiniteFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def frozen_dimensions(self) -> "WorldTemplate":
        """Reject any footprint other than the frozen 8 x 6 metre template."""
        if self.width_m != 8.0 or self.depth_m != 6.0:
            raise ValueError("world template dimensions must be exactly 8 x 6 metres")
        return self


class WorldDefinition(FrozenModel):
    """Shared physical world fields without granting access to either data split."""

    world_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    seed: int = Field(ge=0, le=(2**63) - 1)
    generation_version: int = Field(ge=1)
    template: WorldTemplate
    start: Vec2
    destination: Vec2
    objects: tuple[WorldObject, ...]


class TrainingWorld(WorldDefinition):
    """A generated training candidate that has not necessarily passed its gate."""

    world_id: str = Field(min_length=1, pattern=r"^train-v[0-9]+-[0-9a-f]{16}$")
    split: Literal["training"] = "training"


class HeldOutWorld(WorldDefinition):
    """A frozen evaluation world that training interfaces must reject."""

    world_id: str = Field(min_length=1, pattern=r"^heldout-v[0-9]+-[0-9a-f]{16}$")
    split: Literal["held_out"] = "held_out"


EpisodeWorld = TrainingWorld | HeldOutWorld
