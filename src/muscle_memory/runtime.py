"""Production Muscle Memory backend composition root."""

from __future__ import annotations

from collections.abc import Mapping

from muscle_memory.backend.api_backend import MuscleMemoryApiBackend
from muscle_memory.backend.approvals import CoordinatorApprovalLedger
from muscle_memory.backend.config import BackendConfig
from muscle_memory.backend.episode_journal import CoordinatorEpisodeJournal
from muscle_memory.backend.episode_runtime import OperationalEpisodeRuntime
from muscle_memory.backend.evaluation_import import admit_held_out_evaluation_from_env
from muscle_memory.backend.graph_prerequisites import CoordinatorGraphPrerequisiteResolver
from muscle_memory.backend.providers import build_provider_bundle
from muscle_memory.backend.rocketride_artifact import (
    ReviewedPipelineArtifact,
    ReviewedPipelineError,
)
from muscle_memory.backend.rocketride_callback import (
    CallbackCredential,
    FixedStepDispatcher,
)
from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.episodes import EpisodeService
from muscle_memory.robot.identity import verify_mm01_bundle


def build_api_backend(
    environ: Mapping[str, str] | None = None,
) -> MuscleMemoryApiBackend:
    """Build injected runtime services while failing closed on local corruption."""

    config = BackendConfig.from_env(environ)
    verification = verify_mm01_bundle()
    if not verification.valid or not verification.qualified:
        raise RuntimeError("the qualified MM-01 bundle did not pass verification")

    coordinator = CoordinatorStore(config.coordinator_path)
    try:
        approval_ledger = CoordinatorApprovalLedger(coordinator)
        providers = build_provider_bundle(
            config,
            approval_ledger=approval_ledger,
            coordinator=coordinator,
        )
        evaluation_admission = admit_held_out_evaluation_from_env(
            coordinator,
            config.environ,
        )
        if evaluation_admission is not None:
            admitted_policy_ids = {
                evaluation_admission.baseline_policy_id,
                evaluation_admission.candidate_policy_id,
            }
            for checkpoint in coordinator.evaluated_checkpoints():
                if checkpoint.policy_id in admitted_policy_ids:
                    providers.graph_memory.record_evaluated_policy(checkpoint)
        journal = CoordinatorEpisodeJournal(
            coordinator,
            expected_robot_checksum=verification.robot_checksum,
        )
        episode_service = EpisodeService(
            telemetry_backend=providers.laserdata,
            telemetry_store=providers.laserdata.spool,
            graph_memory=providers.graph_memory,
            journal=journal,
            graph_prerequisites=CoordinatorGraphPrerequisiteResolver(
                coordinator,
                expected_robot_checksum=verification.robot_checksum,
            ),
        )
        episode_runtime = OperationalEpisodeRuntime(
            episode_service,
            expected_robot_checksum=verification.robot_checksum,
        )
        try:
            reviewed_artifact = ReviewedPipelineArtifact.from_env(config.environ)
        except ReviewedPipelineError:
            rocketride_callback = None
        else:
            rocketride_callback = FixedStepDispatcher(
                coordinator=coordinator,
                episodes=episode_service,
                graph_memory=providers.graph_memory,
                credential=CallbackCredential(reviewed_artifact.coordinator_token),
            )
        return MuscleMemoryApiBackend(
            coordinator=coordinator,
            journal=journal,
            episode_runtime=episode_runtime,
            providers=providers,
            approval_ledger=approval_ledger,
            rocketride_callback=rocketride_callback,
        )
    except BaseException:
        coordinator.close()
        raise


def create_api_backend() -> MuscleMemoryApiBackend:
    """Zero-argument ``MM_API_BACKEND_FACTORY`` entrypoint."""

    return build_api_backend()


__all__ = ["build_api_backend", "create_api_backend"]
