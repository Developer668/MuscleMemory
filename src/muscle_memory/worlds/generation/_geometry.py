"""Conservative floor-plane geometry for generation and validation."""

import math
from dataclasses import dataclass

from muscle_memory.worlds.models import PrimitiveCollider, Vec2, WorldObject


@dataclass(frozen=True, slots=True)
class Aabb:
    """Axis-aligned floor-plane bounds in metres."""

    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float

    def inflated(self, amount: float) -> "Aabb":
        """Expand all four sides by amount."""
        return Aabb(
            minimum_x=self.minimum_x - amount,
            maximum_x=self.maximum_x + amount,
            minimum_y=self.minimum_y - amount,
            maximum_y=self.maximum_y + amount,
        )

    def contains(self, point: Vec2) -> bool:
        """Return whether a point lies inside or on these bounds."""
        return (
            self.minimum_x <= point.x <= self.maximum_x
            and self.minimum_y <= point.y <= self.maximum_y
        )

    def overlaps(self, other: "Aabb") -> bool:
        """Return whether two closed boxes intersect or touch."""
        return not (
            self.maximum_x < other.minimum_x
            or other.maximum_x < self.minimum_x
            or self.maximum_y < other.minimum_y
            or other.maximum_y < self.minimum_y
        )


def collider_aabb(position: Vec2, yaw_degrees: float, collider: PrimitiveCollider) -> Aabb:
    """Return a conservative AABB for a rotated primitive collider."""
    centre_x = position.x + collider.centre_offset.x
    centre_y = position.y + collider.centre_offset.y
    half_length = collider.dimensions.length_m / 2.0
    half_width = collider.dimensions.width_m / 2.0
    angle = math.radians(yaw_degrees)
    extent_x = abs(math.cos(angle)) * half_length + abs(math.sin(angle)) * half_width
    extent_y = abs(math.sin(angle)) * half_length + abs(math.cos(angle)) * half_width
    return Aabb(
        minimum_x=centre_x - extent_x,
        maximum_x=centre_x + extent_x,
        minimum_y=centre_y - extent_y,
        maximum_y=centre_y + extent_y,
    )


def object_aabb(obstacle: WorldObject) -> Aabb:
    """Return the floor-plane bounds used by the conservative safety gate."""
    return collider_aabb(obstacle.position, obstacle.yaw_degrees, obstacle.collider)
