"""Immutable contracts for generated appearance assets and world admission."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from muscle_memory.worlds import Dimensions3D, ObjectCategory

SHA256_PATTERN = r"^[0-9a-f]{64}$"


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value deterministically for hashing."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


class FrozenAssetModel(BaseModel):
    """Base class for strict immutable asset records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderState(StrEnum):
    """Truthful lifecycle state for a network-backed generation provider."""

    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    HEALTHY = "healthy"
    DEGRADED = "degraded"


class ProviderSnapshot(FrozenAssetModel):
    """Current provider state without implying an end-to-end generation."""

    provider: str = Field(min_length=1)
    state: ProviderState
    detail: str = Field(min_length=1)


class ArtifactRole(StrEnum):
    """A role that prevents appearance artifacts from entering physics."""

    REFERENCE_INPUT = "reference_input"
    RENDERING_ONLY = "rendering_only"


class VisualMeshFormat(StrEnum):
    GLB = "glb"
    OBJ = "obj"


class GeneratedArtifact(FrozenAssetModel):
    """Bytes returned by a provider before they enter the verified cache."""

    data: bytes = Field(min_length=1)
    media_type: str = Field(min_length=1)
    role: ArtifactRole
    mesh_format: VisualMeshFormat | None = None

    @model_validator(mode="after")
    def enforce_role_shape(self) -> GeneratedArtifact:
        if self.role is ArtifactRole.REFERENCE_INPUT and self.mesh_format is not None:
            raise ValueError("reference images cannot declare a mesh format")
        if self.role is ArtifactRole.RENDERING_ONLY and self.mesh_format is None:
            raise ValueError("rendering artifacts must declare a mesh format")
        return self


class ArtifactDescriptor(FrozenAssetModel):
    """Integrity metadata for one content-addressed blob."""

    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    media_type: str = Field(min_length=1)
    role: ArtifactRole


class AssetRequest(FrozenAssetModel):
    """Provider-neutral request for a cosmetic obstacle asset."""

    request_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    prompt: str = Field(min_length=1, max_length=4_000)
    mesh_format: VisualMeshFormat = VisualMeshFormat.GLB

    @model_validator(mode="after")
    def reject_blank_prompt(self) -> AssetRequest:
        if not self.prompt.strip():
            raise ValueError("asset prompt cannot be blank")
        return self

    @property
    def request_sha256(self) -> str:
        payload = {
            "mesh_format": self.mesh_format.value,
            "prompt": self.prompt,
            "schema_version": 1,
        }
        return sha256_bytes(canonical_json_bytes(payload))


class AssetManifest(FrozenAssetModel):
    """Integrity manifest for a complete reference-image and visual-mesh pair."""

    schema_version: Literal[1] = 1
    bundle_id: str = Field(pattern=SHA256_PATTERN)
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_image: ArtifactDescriptor
    visual_mesh: ArtifactDescriptor
    visual_mesh_format: VisualMeshFormat
    reference_provider: str = Field(min_length=1)
    mesh_provider: str = Field(min_length=1)
    verified_fallback: bool
    live_generation: bool

    @model_validator(mode="after")
    def validate_manifest_contract(self) -> AssetManifest:
        if self.reference_image.role is not ArtifactRole.REFERENCE_INPUT:
            raise ValueError("reference descriptor has the wrong role")
        if self.visual_mesh.role is not ArtifactRole.RENDERING_ONLY:
            raise ValueError("visual mesh must be marked rendering-only")
        if self.verified_fallback and self.live_generation:
            raise ValueError("a verified fallback cannot claim live generation")
        if self.bundle_id != self.expected_bundle_id:
            raise ValueError("asset bundle content address does not match its manifest")
        return self

    @property
    def content_payload(self) -> dict[str, object]:
        return {
            "live_generation": self.live_generation,
            "mesh_provider": self.mesh_provider,
            "reference_image": self.reference_image.model_dump(mode="json"),
            "reference_provider": self.reference_provider,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "verified_fallback": self.verified_fallback,
            "visual_mesh": self.visual_mesh.model_dump(mode="json"),
            "visual_mesh_format": self.visual_mesh_format.value,
        }

    @property
    def expected_bundle_id(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.content_payload))


class ProposalOrigin(StrEnum):
    VERIFIED_CATALOG = "verified_catalog"
    AGENT = "agent"


class PhysicalField(StrEnum):
    DIMENSIONS = "dimensions"
    MASS = "mass"
    FRICTION = "friction"
    MOBILITY = "mobility"
    COLLIDER = "collider"


SAFETY_CRITICAL_FIELDS = (
    PhysicalField.DIMENSIONS,
    PhysicalField.MASS,
    PhysicalField.FRICTION,
    PhysicalField.MOBILITY,
    PhysicalField.COLLIDER,
)


class ColliderConstruction(StrEnum):
    DETERMINISTIC_PRIMITIVE = "deterministic_primitive"
    DETERMINISTIC_CONVEX = "deterministic_convex"


class AssetColliderKind(StrEnum):
    BOX = "box"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    CONVEX_HULL = "convex_hull"


class ColliderProposal(FrozenAssetModel):
    """Physics proposal produced independently from the appearance mesh."""

    kind: AssetColliderKind
    construction: ColliderConstruction
    geometry_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prior_approval_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_construction(self) -> ColliderProposal:
        if self.construction is ColliderConstruction.DETERMINISTIC_PRIMITIVE:
            if self.kind is AssetColliderKind.CONVEX_HULL:
                raise ValueError("convex hulls are not primitive colliders")
            if self.geometry_sha256 is not None:
                raise ValueError("primitive colliders cannot carry mesh bytes")
        else:
            if self.kind is not AssetColliderKind.CONVEX_HULL:
                raise ValueError("deterministic convex construction requires a convex hull")
            if self.geometry_sha256 is None:
                raise ValueError("convex colliders require independent geometry bytes")
        return self


class PhysicalProposal(FrozenAssetModel):
    """Safety-critical properties proposed for an obstacle."""

    category: ObjectCategory
    dimensions: Dimensions3D
    mass_kg: FiniteFloat = Field(gt=0.0)
    sliding_friction: FiniteFloat = Field(ge=0.0)
    restitution: FiniteFloat = Field(ge=0.0, le=1.0)
    movable: bool
    collider: ColliderProposal
    origin: ProposalOrigin
    uncertain_fields: tuple[PhysicalField, ...] = ()

    @model_validator(mode="after")
    def validate_catalog_convex_approval(self) -> PhysicalProposal:
        if len(set(self.uncertain_fields)) != len(self.uncertain_fields):
            raise ValueError("uncertain physical fields must be unique")
        if (
            self.origin is ProposalOrigin.VERIFIED_CATALOG
            and self.collider.construction is ColliderConstruction.DETERMINISTIC_CONVEX
            and self.collider.prior_approval_id is None
        ):
            raise ValueError("catalog convex colliders require a prior human approval reference")
        return self

    @property
    def requires_human_approval(self) -> bool:
        return self.origin is ProposalOrigin.AGENT or bool(self.uncertain_fields)

    @property
    def review_fields(self) -> tuple[PhysicalField, ...]:
        if self.origin is ProposalOrigin.AGENT:
            return SAFETY_CRITICAL_FIELDS
        return self.uncertain_fields

    @property
    def proposal_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))


class ColliderSource(StrEnum):
    DETERMINISTIC_PRIMITIVE = "deterministic_primitive"
    APPROVED_CONVEX = "approved_convex"


class PhysicsCollider(FrozenAssetModel):
    """Physics-only geometry; detailed rendering mesh fields are forbidden."""

    role: Literal["physics_only"] = "physics_only"
    kind: AssetColliderKind
    source: ColliderSource
    dimensions: Dimensions3D
    geometry_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    approval_reference: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def enforce_approved_geometry(self) -> PhysicsCollider:
        if self.source is ColliderSource.DETERMINISTIC_PRIMITIVE:
            if self.kind is AssetColliderKind.CONVEX_HULL:
                raise ValueError("primitive source cannot describe a convex hull")
            if self.geometry_sha256 is not None:
                raise ValueError("primitive colliders do not load mesh geometry")
        else:
            if self.kind is not AssetColliderKind.CONVEX_HULL:
                raise ValueError("approved convex source must describe a convex hull")
            if self.geometry_sha256 is None or self.approval_reference is None:
                raise ValueError("convex colliders require geometry and human approval")
        return self


class AdmittedPhysicalProperties(FrozenAssetModel):
    category: ObjectCategory
    mass_kg: FiniteFloat = Field(gt=0.0)
    sliding_friction: FiniteFloat = Field(ge=0.0)
    restitution: FiniteFloat = Field(ge=0.0, le=1.0)
    movable: bool
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    approval_reference: str | None = Field(default=None, min_length=1)


class WorldAdmissibleAsset(FrozenAssetModel):
    """The only asset-pipeline record permitted at world assembly."""

    asset_id: str = Field(pattern=SHA256_PATTERN)
    visual_mesh: ArtifactDescriptor
    visual_mesh_format: VisualMeshFormat
    physical: AdmittedPhysicalProperties
    collider: PhysicsCollider

    @model_validator(mode="after")
    def enforce_render_physics_separation(self) -> WorldAdmissibleAsset:
        if self.visual_mesh.role is not ArtifactRole.RENDERING_ONLY:
            raise ValueError("world appearance must use a rendering-only descriptor")
        if self.collider.geometry_sha256 == self.visual_mesh.sha256:
            raise ValueError("rendering mesh bytes cannot be reused as collision geometry")
        return self


class GenerationRoute(StrEnum):
    LIVE_PROVIDER = "live_provider"
    REQUEST_CACHE = "request_cache"
    VERIFIED_FALLBACK = "verified_fallback"


class AdmissionState(StrEnum):
    READY = "ready"
    BLOCKED_APPROVAL = "blocked_approval"
    REJECTED = "rejected"


class AssetPipelineResult(FrozenAssetModel):
    """Truthful generation result with an optional world-admission output."""

    route: GenerationRoute
    manifest: AssetManifest
    admission_state: AdmissionState
    approval_requirement_id: str | None = None
    world_asset: WorldAdmissibleAsset | None = None
    provider_snapshots: tuple[ProviderSnapshot, ...]
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_admission_state(self) -> AssetPipelineResult:
        if self.admission_state is AdmissionState.READY and self.world_asset is None:
            raise ValueError("ready assets require a world-admission record")
        if self.admission_state is not AdmissionState.READY and self.world_asset is not None:
            raise ValueError("blocked or rejected assets cannot enter a world")
        if self.route is GenerationRoute.VERIFIED_FALLBACK and (
            not self.manifest.verified_fallback or self.manifest.live_generation
        ):
            raise ValueError("fallback route requires a verified non-live manifest")
        return self
