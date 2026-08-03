from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from muscle_memory.evaluation import (
    SAFE_DELIVERY_CRITERIA,
    EpisodeOutcome,
    FailureReason,
    Vector2,
    calculate_facing_error_degrees,
    calculate_stop_distance_metres,
    evaluate_safe_delivery,
)


@pytest.fixture
def passing_outcome() -> EpisodeOutcome:
    return EpisodeOutcome(
        time_to_resident_seconds=30.0,
        robot_stop_position=Vector2(0.0, 0.0),
        resident_position=Vector2(0.5, 0.0),
        robot_forward=Vector2(1.0, 0.0),
        stopped=True,
        falls=0,
        body_collisions=0,
        minimum_obstacle_clearance_metres=0.25,
        maximum_tray_tilt_degrees=11.999,
        package_slipped=False,
        human_interventions=0,
    )


def test_canonical_criteria_match_safe_delivery_requirements() -> None:
    assert SAFE_DELIVERY_CRITERIA.maximum_time_to_resident_seconds == 30.0
    assert SAFE_DELIVERY_CRITERIA.maximum_stop_distance_metres == 0.5
    assert SAFE_DELIVERY_CRITERIA.minimum_obstacle_clearance_metres == 0.25
    assert SAFE_DELIVERY_CRITERIA.maximum_tray_tilt_degrees_exclusive == 12.0
    assert SAFE_DELIVERY_CRITERIA.maximum_falls == 0
    assert SAFE_DELIVERY_CRITERIA.maximum_body_collisions == 0
    assert not SAFE_DELIVERY_CRITERIA.package_slip_allowed
    assert SAFE_DELIVERY_CRITERIA.maximum_human_interventions == 0
    assert SAFE_DELIVERY_CRITERIA.stopped_required


def test_all_boundary_conditions_pass(passing_outcome: EpisodeOutcome) -> None:
    result = evaluate_safe_delivery(passing_outcome)

    assert result.success
    assert result.failed_reasons == ()
    assert result.stop_distance_metres == pytest.approx(0.5)
    assert result.facing_error_degrees == pytest.approx(0.0)


FailureMutation = Callable[[EpisodeOutcome], EpisodeOutcome]

FAILURE_CASES: tuple[tuple[FailureMutation, FailureReason], ...] = (
    (
        lambda outcome: replace(outcome, time_to_resident_seconds=None),
        FailureReason.DID_NOT_REACH_RESIDENT_IN_TIME,
    ),
    (
        lambda outcome: replace(outcome, time_to_resident_seconds=30.001),
        FailureReason.DID_NOT_REACH_RESIDENT_IN_TIME,
    ),
    (lambda outcome: replace(outcome, stopped=False), FailureReason.DID_NOT_STOP),
    (
        lambda outcome: replace(outcome, resident_position=Vector2(0.501, 0.0)),
        FailureReason.STOPPED_TOO_FAR_AWAY,
    ),
    (
        lambda outcome: replace(outcome, robot_forward=Vector2(0.0, 1.0)),
        FailureReason.NOT_FACING_RESIDENT,
    ),
    (lambda outcome: replace(outcome, falls=1), FailureReason.FELL),
    (
        lambda outcome: replace(outcome, body_collisions=1),
        FailureReason.BODY_COLLISION,
    ),
    (
        lambda outcome: replace(outcome, minimum_obstacle_clearance_metres=0.249),
        FailureReason.INSUFFICIENT_OBSTACLE_CLEARANCE,
    ),
    (
        lambda outcome: replace(outcome, maximum_tray_tilt_degrees=12.0),
        FailureReason.EXCESSIVE_TRAY_TILT,
    ),
    (
        lambda outcome: replace(outcome, package_slipped=True),
        FailureReason.PACKAGE_SLIPPED,
    ),
    (
        lambda outcome: replace(outcome, human_interventions=1),
        FailureReason.HUMAN_INTERVENTION,
    ),
)


@pytest.mark.parametrize(("mutate", "reason"), FAILURE_CASES)
def test_each_requirement_has_an_explicit_failure_reason(
    passing_outcome: EpisodeOutcome,
    mutate: FailureMutation,
    reason: FailureReason,
) -> None:
    result = evaluate_safe_delivery(mutate(passing_outcome))

    assert not result.success
    assert reason in result.failed_reasons


def test_evaluation_reports_all_failures(passing_outcome: EpisodeOutcome) -> None:
    result = evaluate_safe_delivery(
        replace(
            passing_outcome,
            time_to_resident_seconds=None,
            stopped=False,
            falls=1,
            body_collisions=2,
            minimum_obstacle_clearance_metres=0.1,
            maximum_tray_tilt_degrees=13.0,
            package_slipped=True,
            human_interventions=1,
        )
    )

    assert set(result.failed_reasons) == {
        FailureReason.DID_NOT_REACH_RESIDENT_IN_TIME,
        FailureReason.DID_NOT_STOP,
        FailureReason.FELL,
        FailureReason.BODY_COLLISION,
        FailureReason.INSUFFICIENT_OBSTACLE_CLEARANCE,
        FailureReason.EXCESSIVE_TRAY_TILT,
        FailureReason.PACKAGE_SLIPPED,
        FailureReason.HUMAN_INTERVENTION,
    }


def test_distance_and_facing_calculations() -> None:
    outcome = EpisodeOutcome(
        time_to_resident_seconds=1.0,
        robot_stop_position=Vector2(1.0, 1.0),
        resident_position=Vector2(4.0, 5.0),
        robot_forward=Vector2(0.0, 2.0),
        stopped=True,
        falls=0,
        body_collisions=0,
        minimum_obstacle_clearance_metres=1.0,
        maximum_tray_tilt_degrees=0.0,
        package_slipped=False,
        human_interventions=0,
    )

    assert calculate_stop_distance_metres(outcome) == pytest.approx(5.0)
    assert calculate_facing_error_degrees(outcome) == pytest.approx(36.8698976458)


def test_zero_forward_vector_fails_facing(passing_outcome: EpisodeOutcome) -> None:
    result = evaluate_safe_delivery(replace(passing_outcome, robot_forward=Vector2(0.0, 0.0)))

    assert result.facing_error_degrees is None
    assert FailureReason.NOT_FACING_RESIDENT in result.failed_reasons
