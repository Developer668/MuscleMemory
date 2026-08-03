"""Safe-delivery success criteria and deterministic evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class SafeDeliveryCriteria:
    """Canonical thresholds for every safe-delivery requirement."""

    maximum_time_to_resident_seconds: float = 30.0
    maximum_stop_distance_metres: float = 0.5
    maximum_facing_error_degrees: float = 30.0
    maximum_falls: int = 0
    maximum_body_collisions: int = 0
    minimum_obstacle_clearance_metres: float = 0.25
    maximum_tray_tilt_degrees_exclusive: float = 12.0
    package_slip_allowed: bool = False
    maximum_human_interventions: int = 0
    stopped_required: bool = True


SAFE_DELIVERY_CRITERIA = SafeDeliveryCriteria()


@dataclass(frozen=True, slots=True)
class Vector2:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("vector coordinates must be finite")


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Measurements required to decide whether one delivery succeeded."""

    time_to_resident_seconds: float | None
    robot_stop_position: Vector2
    resident_position: Vector2
    robot_forward: Vector2
    stopped: bool
    falls: int
    body_collisions: int
    minimum_obstacle_clearance_metres: float
    maximum_tray_tilt_degrees: float
    package_slipped: bool
    human_interventions: int

    def __post_init__(self) -> None:
        if self.time_to_resident_seconds is not None and (
            not math.isfinite(self.time_to_resident_seconds)
            or self.time_to_resident_seconds < 0
        ):
            raise ValueError("time_to_resident_seconds must be finite and non-negative")
        for name in ("falls", "body_collisions", "human_interventions"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "minimum_obstacle_clearance_metres",
            "maximum_tray_tilt_degrees",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


class FailureReason(StrEnum):
    DID_NOT_REACH_RESIDENT_IN_TIME = "Resident was not reached within the time limit"
    DID_NOT_STOP = "Robot did not stop"
    STOPPED_TOO_FAR_AWAY = "Robot stopped too far from the resident"
    NOT_FACING_RESIDENT = "Robot was not facing the resident"
    FELL = "Robot fell"
    BODY_COLLISION = "Robot had a body collision"
    INSUFFICIENT_OBSTACLE_CLEARANCE = "Minimum obstacle clearance was too small"
    EXCESSIVE_TRAY_TILT = "Tray tilt reached or exceeded the limit"
    PACKAGE_SLIPPED = "The simulated package slipped"
    HUMAN_INTERVENTION = "Human intervention occurred"


@dataclass(frozen=True, slots=True)
class SuccessEvaluation:
    success: bool
    failed_reasons: tuple[FailureReason, ...]
    stop_distance_metres: float
    facing_error_degrees: float | None


def calculate_stop_distance_metres(outcome: EpisodeOutcome) -> float:
    return math.hypot(
        outcome.resident_position.x - outcome.robot_stop_position.x,
        outcome.resident_position.y - outcome.robot_stop_position.y,
    )


def calculate_facing_error_degrees(outcome: EpisodeOutcome) -> float | None:
    """Return planar angular error, or ``None`` for an undefined forward vector."""

    target_x = outcome.resident_position.x - outcome.robot_stop_position.x
    target_y = outcome.resident_position.y - outcome.robot_stop_position.y
    target_length = math.hypot(target_x, target_y)
    forward_length = math.hypot(outcome.robot_forward.x, outcome.robot_forward.y)
    if forward_length == 0:
        return None
    if target_length == 0:
        return 0.0

    cosine = (
        outcome.robot_forward.x * target_x + outcome.robot_forward.y * target_y
    ) / (forward_length * target_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def evaluate_safe_delivery(
    outcome: EpisodeOutcome,
    criteria: SafeDeliveryCriteria = SAFE_DELIVERY_CRITERIA,
) -> SuccessEvaluation:
    """Evaluate every requirement and retain every reason for failure."""

    failures: list[FailureReason] = []
    stop_distance = calculate_stop_distance_metres(outcome)
    facing_error = calculate_facing_error_degrees(outcome)

    if (
        outcome.time_to_resident_seconds is None
        or outcome.time_to_resident_seconds > criteria.maximum_time_to_resident_seconds
    ):
        failures.append(FailureReason.DID_NOT_REACH_RESIDENT_IN_TIME)
    if criteria.stopped_required and not outcome.stopped:
        failures.append(FailureReason.DID_NOT_STOP)
    if stop_distance > criteria.maximum_stop_distance_metres:
        failures.append(FailureReason.STOPPED_TOO_FAR_AWAY)
    if facing_error is None or facing_error > criteria.maximum_facing_error_degrees:
        failures.append(FailureReason.NOT_FACING_RESIDENT)
    if outcome.falls > criteria.maximum_falls:
        failures.append(FailureReason.FELL)
    if outcome.body_collisions > criteria.maximum_body_collisions:
        failures.append(FailureReason.BODY_COLLISION)
    if outcome.minimum_obstacle_clearance_metres < criteria.minimum_obstacle_clearance_metres:
        failures.append(FailureReason.INSUFFICIENT_OBSTACLE_CLEARANCE)
    if outcome.maximum_tray_tilt_degrees >= criteria.maximum_tray_tilt_degrees_exclusive:
        failures.append(FailureReason.EXCESSIVE_TRAY_TILT)
    if outcome.package_slipped and not criteria.package_slip_allowed:
        failures.append(FailureReason.PACKAGE_SLIPPED)
    if outcome.human_interventions > criteria.maximum_human_interventions:
        failures.append(FailureReason.HUMAN_INTERVENTION)

    return SuccessEvaluation(
        success=not failures,
        failed_reasons=tuple(failures),
        stop_distance_metres=stop_distance,
        facing_error_degrees=facing_error,
    )
