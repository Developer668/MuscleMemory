"""Fast real-physics smoke test for the unqualified candidate controller."""

from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from muscle_memory.robot.command import TaskCommand
from muscle_memory.robot.identity import (
    CONTROLLER_INFERENCE_HZ,
    CONTROLLER_SUPERVISOR_HZ,
    PHYSICS_HZ,
    TASK_POLICY_HZ,
    verify_candidate_bundle,
)
from muscle_memory.simulation.runtime import HeadlessG1Simulation

QUALIFICATION_BLOCKERS = (
    "candidate ONNX gait inference is 50 Hz, not the required 100 Hz",
    "zero-command stopping has not passed qualification",
)


class SmokeResult(BaseModel):
    """Measured smoke evidence, explicitly distinct from qualification."""

    model_config = ConfigDict(frozen=True)

    candidate_bundle_valid: bool
    smoke_passed: bool
    qualified: bool
    qualification_blockers: tuple[str, ...]
    physics_hz: int
    controller_supervisor_hz: int
    controller_inference_hz: int
    task_policy_hz: int
    simulated_seconds: float
    physics_steps: int
    task_policy_updates: int
    controller_supervisor_ticks: int
    controller_inferences: int
    finite_state: bool
    moved: bool
    fell: bool
    planar_displacement_m: float = Field(ge=0.0)
    start_position_m: tuple[float, float, float]
    end_position_m: tuple[float, float, float]
    minimum_pelvis_height_m: float
    minimum_torso_up_z: float
    render_path: str | None


def _render(simulation: HeadlessG1Simulation, render_path: Path) -> str:
    render_path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(simulation.model, height=480, width=640) as renderer:
        renderer.update_scene(simulation.data, camera="track")
        pixels = renderer.render()
    Image.fromarray(pixels).save(render_path)
    return render_path.resolve().as_posix()


def run_smoke(
    *,
    duration_seconds: float = 2.0,
    command: TaskCommand | None = None,
    render_path: Path | None = None,
) -> SmokeResult:
    """Run real MuJoCo and ONNX steps without claiming controller acceptance."""
    bundle = verify_candidate_bundle()
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive and finite")
    physics_steps = round(duration_seconds * PHYSICS_HZ)
    if not np.isclose(physics_steps / PHYSICS_HZ, duration_seconds):
        raise ValueError("duration_seconds must align to the 500 Hz physics clock")

    task_command = command or TaskCommand(0.4, 0.0, 0.0)
    simulation = HeadlessG1Simulation()
    start = simulation.data.qpos[:3].copy()
    minimum_height = float(start[2])
    minimum_up_z = float(simulation.data.sensor("upvector_torso").data[2])
    finite_state = True
    fell = False

    for _ in range(physics_steps):
        simulation.step(lambda _time: task_command)
        height = float(simulation.data.qpos[2])
        up_z = float(simulation.data.sensor("upvector_torso").data[2])
        minimum_height = min(minimum_height, height)
        minimum_up_z = min(minimum_up_z, up_z)
        finite_state = finite_state and bool(
            np.isfinite(simulation.data.qpos).all()
            and np.isfinite(simulation.data.qvel).all()
        )
        fell = fell or up_z <= 0.0 or height <= 0.35

    end = simulation.data.qpos[:3].copy()
    displacement = float(np.linalg.norm(end[:2] - start[:2]))
    moved = displacement >= 0.05
    saved_render = _render(simulation, render_path) if render_path is not None else None
    smoke_passed = bundle.valid and finite_state and moved and not fell
    return SmokeResult(
        candidate_bundle_valid=bundle.valid,
        smoke_passed=smoke_passed,
        qualified=False,
        qualification_blockers=QUALIFICATION_BLOCKERS,
        physics_hz=PHYSICS_HZ,
        controller_supervisor_hz=CONTROLLER_SUPERVISOR_HZ,
        controller_inference_hz=CONTROLLER_INFERENCE_HZ,
        task_policy_hz=TASK_POLICY_HZ,
        simulated_seconds=physics_steps / PHYSICS_HZ,
        physics_steps=physics_steps,
        task_policy_updates=simulation.task_policy_updates,
        controller_supervisor_ticks=simulation.controller_supervisor_ticks,
        controller_inferences=simulation.controller.inference_count,
        finite_state=finite_state,
        moved=moved,
        fell=fell,
        planar_displacement_m=displacement,
        start_position_m=(float(start[0]), float(start[1]), float(start[2])),
        end_position_m=(float(end[0]), float(end[1]), float(end[2])),
        minimum_pelvis_height_m=minimum_height,
        minimum_torso_up_z=minimum_up_z,
        render_path=saved_render,
    )
