"""The only command surface exposed to a learned task policy."""

import math
from dataclasses import dataclass

MAX_FORWARD_SPEED_MPS = 1.0
MAX_TURNING_RATE_RAD_S = 1.0
STOP_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class TaskCommand:
    """A high-level command with no direct locomotion or lateral control."""

    forward_speed_mps: float
    turning_rate_rad_s: float
    stop_probability: float

    def __post_init__(self) -> None:
        values = (
            self.forward_speed_mps,
            self.turning_rate_rad_s,
            self.stop_probability,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("task command values must be finite")
        if not -MAX_FORWARD_SPEED_MPS <= self.forward_speed_mps <= MAX_FORWARD_SPEED_MPS:
            raise ValueError(
                f"forward_speed_mps must be within +/-{MAX_FORWARD_SPEED_MPS}"
            )
        if not -MAX_TURNING_RATE_RAD_S <= self.turning_rate_rad_s <= MAX_TURNING_RATE_RAD_S:
            raise ValueError(
                f"turning_rate_rad_s must be within +/-{MAX_TURNING_RATE_RAD_S}"
            )
        if not 0.0 <= self.stop_probability <= 1.0:
            raise ValueError("stop_probability must be within [0, 1]")

    def frozen_controller_command(self) -> tuple[float, float, float]:
        """Return the gait command as forward, fixed-zero lateral, and yaw."""
        if self.stop_probability >= STOP_THRESHOLD:
            return (0.0, 0.0, 0.0)
        return (self.forward_speed_mps, 0.0, self.turning_rate_rad_s)
