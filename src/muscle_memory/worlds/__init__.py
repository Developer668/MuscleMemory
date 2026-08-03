"""Schemas shared by world producers and consumers.

Training-only generation and pathfinding live under ``muscle_memory.worlds.generation``
so evaluation code can depend on these schemas without acquiring an expert planner.
"""

from muscle_memory.worlds.models import (
    ColliderKind,
    Dimensions3D,
    HeldOutWorld,
    ObjectCategory,
    PhysicalProperties,
    PrimitiveCollider,
    TrainingWorld,
    Vec2,
    WorldObject,
    WorldTemplate,
)

__all__ = [
    "ColliderKind",
    "Dimensions3D",
    "HeldOutWorld",
    "ObjectCategory",
    "PhysicalProperties",
    "PrimitiveCollider",
    "TrainingWorld",
    "Vec2",
    "WorldObject",
    "WorldTemplate",
]
