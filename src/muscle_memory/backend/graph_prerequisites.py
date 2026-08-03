"""Trusted graph parents derived from validated domain artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from muscle_memory.coordinator import CoordinatorStore
from muscle_memory.episodes import EpisodeIdentity
from muscle_memory.episodes.service import EpisodeGraphPrerequisites
from muscle_memory.evaluation.runner import PolicyEpisodeResult
from muscle_memory.graph_memory import (
    ObstacleMemoryRecord,
    WorldMemoryRecord,
    WorldSplit,
    canonical_json,
)
from muscle_memory.live.catalog import LiveWorldCatalog
from muscle_memory.worlds.models import TrainingWorld


@dataclass(frozen=True, slots=True)
class DerivedTrainingWorldArtifacts:
    definition: TrainingWorld
    world: WorldMemoryRecord
    obstacles: tuple[ObstacleMemoryRecord, ...]
    baseline_path_digest: str


def derive_training_world_artifacts(
    seed: int,
    *,
    recorded_at: datetime,
    catalog: LiveWorldCatalog | None = None,
) -> DerivedTrainingWorldArtifacts:
    """Load stable graph parents without making the path teacher runtime-reachable."""

    active_catalog = catalog or LiveWorldCatalog.load()
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware")
    try:
        validated = active_catalog.world_for_seed(seed)
    except KeyError as exc:
        raise ValueError("training world seed is not in the admitted live catalog") from exc
    world = validated.world
    world_hash = hashlib.sha256(world.model_dump_json().encode("utf-8")).hexdigest()
    baseline_path_digest = validated.baseline_path_sha256
    validation_hash = hashlib.sha256(
        canonical_json(
            {
                "baseline_path_digest": baseline_path_digest,
                "generation_version": world.generation_version,
                "world_hash": world_hash,
            }
        ).encode("utf-8")
    ).hexdigest()
    world_record = WorldMemoryRecord(
        world_id=world.world_id,
        world_hash=world_hash,
        split=WorldSplit.TRAINING,
        seed=world.seed,
        generation_version=world.generation_version,
        validation_hash=validation_hash,
        validated=True,
        recorded_at=active_catalog.validated_at,
    )
    obstacles = tuple(
        ObstacleMemoryRecord(
            obstacle_id=f"{world.world_id}:{item.object_id}",
            obstacle_hash=hashlib.sha256(
                canonical_json(item.model_dump(mode="json")).encode("utf-8")
            ).hexdigest(),
            world_id=world.world_id,
            category=item.category.value,
            collider_kind=item.collider.kind.value,
            physical_properties_approved=True,
            recorded_at=active_catalog.validated_at,
        )
        for item in world.objects
    )
    return DerivedTrainingWorldArtifacts(
        definition=world,
        world=world_record,
        obstacles=obstacles,
        baseline_path_digest=baseline_path_digest,
    )


class CoordinatorGraphPrerequisiteResolver:
    """Bind an episode to a catalog world and immutable evaluated checkpoint."""

    def __init__(
        self,
        coordinator: CoordinatorStore,
        *,
        expected_robot_checksum: str,
        catalog: LiveWorldCatalog | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._expected_robot_checksum = expected_robot_checksum
        self._catalog = catalog or LiveWorldCatalog.load()

    def resolve(
        self,
        identity: EpisodeIdentity,
        result: PolicyEpisodeResult,
        *,
        recorded_at: datetime,
    ) -> EpisodeGraphPrerequisites:
        if identity.robot_checksum != self._expected_robot_checksum:
            raise ValueError("episode robot checksum does not match the qualified MM-01 bundle")
        if identity.world_split is not WorldSplit.TRAINING:
            raise ValueError("operational graph prerequisites accept training worlds only")
        derived = derive_training_world_artifacts(
            result.world_seed,
            recorded_at=recorded_at,
            catalog=self._catalog,
        )
        policies = {
            checkpoint.policy_id: checkpoint
            for checkpoint in self._coordinator.evaluated_checkpoints()
        }
        policy = policies.get(identity.policy_id)
        if policy is None or policy.checkpoint_hash != identity.policy_hash:
            raise ValueError(
                "episode policy is not an exact immutable evaluated coordinator checkpoint"
            )
        return EpisodeGraphPrerequisites(
            world=derived.world,
            obstacles=derived.obstacles,
            policy=policy,
        )


__all__ = [
    "CoordinatorGraphPrerequisiteResolver",
    "DerivedTrainingWorldArtifacts",
    "derive_training_world_artifacts",
]
