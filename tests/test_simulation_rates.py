from pathlib import Path

import onnx

from muscle_memory.paths import G1_POLICY_ONNX
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
    TASK_POLICY_INTERVAL_STEPS,
    HeadlessG1Simulation,
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
