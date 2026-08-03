"""Result records available only with the training-generation package."""

from pydantic import Field

from muscle_memory.worlds.models import FrozenModel, TrainingWorld, Vec2


class ValidatedTrainingWorld(FrozenModel):
    """A training world plus its validation-gated A* teacher path."""

    world: TrainingWorld
    baseline_path: tuple[Vec2, ...] = Field(min_length=2)
