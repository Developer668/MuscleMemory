"""Training-only world generation and validation gate.

Evaluation packages must not import this module: it contains the A* expert planner.
"""

from muscle_memory.worlds.generation.generator import generate_training_world
from muscle_memory.worlds.generation.models import ValidatedTrainingWorld
from muscle_memory.worlds.generation.validation import (
    ValidationIssue,
    WorldValidationError,
    validate_training_world,
)

__all__ = [
    "ValidatedTrainingWorld",
    "ValidationIssue",
    "WorldValidationError",
    "generate_training_world",
    "validate_training_world",
]
