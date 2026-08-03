from muscle_memory.robot.command import TaskCommand
from muscle_memory.simulation.smoke import run_smoke


def test_real_mujoco_onnx_smoke_moves_without_falling() -> None:
    result = run_smoke(duration_seconds=1.0, command=TaskCommand(0.4, 0.0, 0.0))

    assert result.candidate_bundle_valid is True
    assert result.smoke_passed is True
    assert result.finite_state is True
    assert result.moved is True
    assert result.fell is False
    assert result.physics_steps == 500
    assert result.task_policy_updates == 10
    assert result.controller_supervisor_ticks == 100
    assert result.controller_inferences == 50
    assert result.qualified is False
    assert any("50 Hz" in blocker for blocker in result.qualification_blockers)
    assert any("stopping" in blocker for blocker in result.qualification_blockers)
