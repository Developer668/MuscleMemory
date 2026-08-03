"""Evaluation-safe obstacle-unaware baseline used for measured V0 comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path

from muscle_memory.policy.observation import NavigationObservation
from muscle_memory.robot.command import TaskCommand


class DirectGoalPolicy:
    """Turn toward the destination and drive directly without an obstacle strategy."""

    policy_id = "delivery-v0-direct-goal"

    def __init__(self) -> None:
        self.policy_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def command(self, observation: NavigationObservation) -> TaskCommand:
        distance = observation.destination_distance_m
        bearing = observation.destination_bearing_rad
        if distance <= 0.43:
            return TaskCommand(0.0, 0.0, 1.0)
        turning = max(-0.5, min(0.5, 1.4 * bearing))
        if abs(bearing) > 0.22:
            return TaskCommand(0.0, turning, 0.0)
        forward = 0.18 if distance < 0.8 else 0.3
        return TaskCommand(forward, turning, 0.0)
