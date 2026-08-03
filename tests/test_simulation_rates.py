from itertools import pairwise
from pathlib import Path

import onnx
import pytest

from muscle_memory.paths import G1_POLICY_ONNX, MM01_CONTROLLER_ONNX
from muscle_memory.robot.command import TaskCommand
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
    STOP_FORWARD_DECELERATION_MPS2,
    STOP_TURN_DECELERATION_RAD_S2,
    TASK_POLICY_INTERVAL_STEPS,
    HeadlessG1Simulation,
)


def test_runtime_rates_are_explicit_and_separate() -> None:
    assert PHYSICS_HZ == 500
    assert PHYSICS_DT == 0.002
    assert CONTROLLER_SUPERVISOR_HZ == 100
    assert CONTROLLER_SUPERVISOR_INTERVAL_STEPS == 5
    assert CONTROLLER_INFERENCE_HZ == 100
    assert CONTROLLER_INTERVAL_STEPS == 5
    assert TASK_POLICY_HZ == 10
    assert TASK_POLICY_INTERVAL_STEPS == 50
    assert CONTROLLER_INFERENCE_HZ == CONTROLLER_SUPERVISOR_HZ


def test_qualification_runtime_can_execute_policy_at_100_hz(tmp_path: Path) -> None:
    policy = onnx.load_model(G1_POLICY_ONNX)
    policy.metadata_props.add(key="controller_rate_hz", value="100")
    policy_path = tmp_path / "declared-100hz.onnx"
    onnx.save_model(policy, policy_path)
    simulation = HeadlessG1Simulation(
        controller_policy_path=policy_path,
        controller_inference_hz=100,
    )

    for _ in range(20):
        simulation.step(lambda _time: TaskCommand(0.0, 0.0, 1.0))

    assert simulation.controller.controller_hz == 100
    assert simulation.controller.inference_count == 4
    assert simulation.controller_supervisor_ticks == 4
    assert simulation.task_policy_updates == 1


def test_production_runtime_loads_the_qualified_controller_at_100_hz() -> None:
    simulation = HeadlessG1Simulation()

    for _ in range(20):
        simulation.step(lambda _time: TaskCommand(0.0, 0.0, 1.0))

    assert MM01_CONTROLLER_ONNX.is_file()
    assert simulation.controller.controller_hz == 100
    assert simulation.controller.inference_count == 4


def test_stop_request_is_rate_bounded_by_the_100_hz_supervisor() -> None:
    simulation = HeadlessG1Simulation()

    for _ in range(TASK_POLICY_INTERVAL_STEPS):
        simulation.step(lambda _time: TaskCommand(0.4, -0.5, 0.0))
    assert simulation.controller_command == TaskCommand(0.4, -0.5, 0.0)

    observed: list[TaskCommand] = []
    for _ in range(TASK_POLICY_INTERVAL_STEPS):
        simulation.step(lambda _time: TaskCommand(0.4, -0.5, 1.0))
        if simulation.step_index % CONTROLLER_SUPERVISOR_INTERVAL_STEPS == 1:
            observed.append(simulation.controller_command)

    assert observed[0] == TaskCommand(0.3925, -0.45, 0.0)
    assert observed[-1].forward_speed_mps == pytest.approx(0.325)
    assert observed[-1].turning_rate_rad_s == 0.0
    assert observed[-1].stop_probability == 0.0
    assert all(
        left.forward_speed_mps > right.forward_speed_mps
        for left, right in pairwise(observed)
    )

    for _ in range(5 * TASK_POLICY_INTERVAL_STEPS):
        simulation.step(lambda _time: TaskCommand(0.4, -0.5, 1.0))
    assert simulation.controller_command == TaskCommand(0.0, 0.0, 1.0)
    assert STOP_FORWARD_DECELERATION_MPS2 == 0.75
    assert STOP_TURN_DECELERATION_RAD_S2 == 5.0


def test_non_stop_commands_still_transfer_without_shaping() -> None:
    simulation = HeadlessG1Simulation()

    simulation.step(lambda _time: TaskCommand(-0.3, 0.25, 0.0))

    assert simulation.controller_command == TaskCommand(-0.3, 0.25, 0.0)
