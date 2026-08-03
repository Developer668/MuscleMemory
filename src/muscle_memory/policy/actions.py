"""Fixed action vocabulary behavior-cloned from continuous expert commands."""

from enum import IntEnum

from muscle_memory.robot.command import TaskCommand


class PolicyAction(IntEnum):
    STOP = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    DRIVE_LEFT = 3
    DRIVE_STRAIGHT = 4
    DRIVE_RIGHT = 5


POLICY_ACTION_COUNT = len(PolicyAction)
POLICY_ACTION_COMMANDS = {
    PolicyAction.STOP: TaskCommand(0.0, 0.0, 1.0),
    PolicyAction.TURN_LEFT: TaskCommand(0.0, 0.5, 0.0),
    PolicyAction.TURN_RIGHT: TaskCommand(0.0, -0.5, 0.0),
    PolicyAction.DRIVE_LEFT: TaskCommand(0.3, 0.12, 0.0),
    PolicyAction.DRIVE_STRAIGHT: TaskCommand(0.3, 0.0, 0.0),
    PolicyAction.DRIVE_RIGHT: TaskCommand(0.3, -0.12, 0.0),
}
