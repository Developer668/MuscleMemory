"""Production Muscle Memory backend composition root."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

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
from muscle_memory.live import (
    BoundedVideoService,
    EvaluatedPolicySelection,
    LiveEpisodeConfig,
    LiveEpisodeController,
    LiveEpisodeManager,
    LiveWorldCatalog,
)
from muscle_memory.paths import REPOSITORY_ROOT
from muscle_memory.robot.identity import verify_mm01_bundle


def _repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _live_episode_config(environ: Mapping[str, str]) -> LiveEpisodeConfig:
    try:
        return LiveEpisodeConfig(
            maximum_duration_seconds=float(
                environ.get("MM_LIVE_MAX_DURATION_SECONDS", "30")
            ),
            render_width=int(environ.get("MM_LIVE_RENDER_WIDTH", "320")),
            render_height=int(environ.get("MM_LIVE_RENDER_HEIGHT", "240")),
            jpeg_quality=int(environ.get("MM_LIVE_JPEG_QUALITY", "82")),
        )
    except ValueError as exc:
        raise RuntimeError("live episode environment configuration is invalid") from exc


def build_api_backend(
    environ: Mapping[str, str] | None = None,
) -> MuscleMemoryApiBackend:
    """Build injected runtime services while failing closed on local corruption."""

    config = BackendConfig.from_env(environ)
    verification = verify_mm01_bundle()
    if not verification.valid or not verification.qualified:
        raise RuntimeError("the qualified MM-01 bundle did not pass verification")

    coordinator = CoordinatorStore(config.coordinator_path)
    providers = None
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
        live_world_catalog = LiveWorldCatalog.load()
        episode_service = EpisodeService(
            telemetry_backend=providers.laserdata,
            telemetry_store=providers.laserdata.spool,
            graph_memory=providers.graph_memory,
            journal=journal,
            graph_prerequisites=CoordinatorGraphPrerequisiteResolver(
                coordinator,
                expected_robot_checksum=verification.robot_checksum,
                catalog=live_world_catalog,
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
        backend = MuscleMemoryApiBackend(
            coordinator=coordinator,
            journal=journal,
            episode_runtime=episode_runtime,
            providers=providers,
            approval_ledger=approval_ledger,
            rocketride_callback=rocketride_callback,
            stable_policy_alias=(
                config.environ.get("MM_STABLE_POLICY_ALIAS", "stable").strip()
                or "stable"
            ),
        )
        if evaluation_admission is not None:
            evaluation_path = _repository_path(
                config.environ["MM_HELDOUT_EVALUATION_ARTIFACT"]
            )
            candidate_selection = EvaluatedPolicySelection.load(
                checkpoint_path=_repository_path(
                    config.environ["MM_HELDOUT_CANDIDATE_CHECKPOINT"]
                ),
                evaluation_path=evaluation_path,
                expected_evaluation_evidence_hash=evaluation_admission.artifact_hash,
            )
            baseline_selection = EvaluatedPolicySelection.load_baseline(
                evaluation_path=evaluation_path,
                expected_evaluation_evidence_hash=evaluation_admission.artifact_hash,
            )
            if (
                candidate_selection.policy.policy_id
                != evaluation_admission.candidate_policy_id
            ):
                raise RuntimeError(
                    "live policy identity differs from the admitted held-out candidate"
                )
            if (
                baseline_selection.policy.policy_id
                != evaluation_admission.baseline_policy_id
            ):
                raise RuntimeError(
                    "live policy identity differs from the admitted held-out baseline"
                )
            evaluated = {
                checkpoint.policy_id: checkpoint
                for checkpoint in coordinator.evaluated_checkpoints()
            }
            for selection in (baseline_selection, candidate_selection):
                admitted = evaluated[selection.policy.policy_id]
                if admitted.checkpoint_hash != selection.policy.policy_hash:
                    raise RuntimeError(
                        "live policy hash differs from the admitted evaluated policy"
                    )
                if (
                    admitted.evaluation_evidence_hash
                    != selection.evaluation_evidence_hash
                ):
                    raise RuntimeError(
                        "live policy evidence differs from the admitted evaluation"
                    )
            video = BoundedVideoService(
                maximum_frame_sets=900,
                maximum_bytes=256 << 20,
                maximum_total_bytes=64 << 20,
            )
            manager = LiveEpisodeManager(
                lifecycle=episode_runtime,
                video=video,
                maximum_concurrent_episodes=1,
            )
            live_config = _live_episode_config(config.environ)
            controller = LiveEpisodeController(
                manager=manager,
                worlds=live_world_catalog,
                policies=(baseline_selection, candidate_selection),
                stable_policy_id=coordinator.current_policy(
                    config.environ.get("MM_STABLE_POLICY_ALIAS", "stable").strip()
                    or "stable"
                ),
                config=live_config,
            )
            setattr(backend, "live_episode_controller", controller)  # noqa: B010
        return backend
    except BaseException:
        if providers is not None:
            with suppress(Exception):
                providers.laserdata.spool.close()
        coordinator.close()
        raise


def create_api_backend() -> MuscleMemoryApiBackend:
    """Zero-argument ``MM_API_BACKEND_FACTORY`` entrypoint."""

    return build_api_backend()


__all__ = ["build_api_backend", "create_api_backend"]
