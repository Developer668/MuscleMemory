import math
from dataclasses import fields

import pytest

from muscle_memory.robot.command import TaskCommand


def test_task_command_is_the_complete_policy_output_boundary() -> None:
    assert [field.name for field in fields(TaskCommand)] == [
        "forward_speed_mps",
        "turning_rate_rad_s",
        "stop_probability",
    ]
    assert TaskCommand(0.4, -0.2, 0.1).frozen_controller_command() == (0.4, 0.0, -0.2)


def test_stop_probability_yields_zero_velocity_command() -> None:
    assert TaskCommand(0.4, -0.2, 0.5).frozen_controller_command() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "command",
    [
        TaskCommand(0.0, 0.0, 0.0),
        TaskCommand(-1.0, 1.0, 1.0),
    ],
)
def test_valid_commands_are_finite(command: TaskCommand) -> None:
    assert all(math.isfinite(value) for value in command.frozen_controller_command())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((float("nan"), 0.0, 0.0), "finite"),
        ((1.01, 0.0, 0.0), "forward_speed"),
        ((0.0, -1.01, 0.0), "turning_rate"),
        ((0.0, 0.0, 1.01), "stop_probability"),
    ],
)
def test_invalid_task_commands_fail_closed(
    values: tuple[float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TaskCommand(*values)
