from muscle_memory.robot.identity import (
    CONTROLLER_INFERENCE_HZ,
    CONTROLLER_SUPERVISOR_HZ,
    PHYSICS_HZ,
    TASK_POLICY_HZ,
)
from muscle_memory.simulation.runtime import (
    CONTROLLER_INTERVAL_STEPS,
    CONTROLLER_SUPERVISOR_INTERVAL_STEPS,
    PHYSICS_DT,
    TASK_POLICY_INTERVAL_STEPS,
)


def test_runtime_rates_are_explicit_and_separate() -> None:
    assert PHYSICS_HZ == 500
    assert PHYSICS_DT == 0.002
    assert CONTROLLER_SUPERVISOR_HZ == 100
    assert CONTROLLER_SUPERVISOR_INTERVAL_STEPS == 5
    assert CONTROLLER_INFERENCE_HZ == 50
    assert CONTROLLER_INTERVAL_STEPS == 10
    assert TASK_POLICY_HZ == 10
    assert TASK_POLICY_INTERVAL_STEPS == 50
    assert CONTROLLER_INFERENCE_HZ != CONTROLLER_SUPERVISOR_HZ
