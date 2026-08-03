"""Command-line entry points kept deliberately thin."""

from muscle_memory.robot.identity import verify_candidate_bundle
from muscle_memory.simulation.smoke import run_smoke


def verify_robot_main() -> None:
    """Verify every frozen candidate asset against its manifest."""
    result = verify_candidate_bundle()
    print(result.model_dump_json(indent=2))


def smoke_main() -> None:
    """Run a real headless MuJoCo episode using the frozen candidate policy."""
    result = run_smoke()
    print(result.model_dump_json(indent=2))
