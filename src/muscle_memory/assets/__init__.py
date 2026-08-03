"""Safe appearance-asset generation, caching, and world admission."""

from muscle_memory.assets.approvals import (
    AssetApprovalError,
    AssetApprovalLedger,
    AssetApprovalRequirement,
    HumanAssetDecision,
    HumanVerdict,
)
from muscle_memory.assets.builtin import seed_verified_fallback
from muscle_memory.assets.cache import (
    AssetCacheIntegrityError,
    AssetCacheMutationError,
    ContentAddressedAssetCache,
)
from muscle_memory.assets.models import (
    AdmissionState,
    ArtifactDescriptor,
    ArtifactRole,
    AssetColliderKind,
    AssetManifest,
    AssetPipelineResult,
    AssetRequest,
    ColliderConstruction,
    ColliderProposal,
    ColliderSource,
    GeneratedArtifact,
    GenerationRoute,
    PhysicalField,
    PhysicalProposal,
    ProposalOrigin,
    ProviderSnapshot,
    ProviderState,
    VisualMeshFormat,
    WorldAdmissibleAsset,
)
from muscle_memory.assets.providers import (
    ReferenceImageHttpAdapter,
    TrellisHttpAdapter,
    UrllibJsonTransport,
)
from muscle_memory.assets.service import AssetGenerationPipeline

__all__ = [
    "AdmissionState",
    "ArtifactDescriptor",
    "ArtifactRole",
    "AssetApprovalError",
    "AssetApprovalLedger",
    "AssetApprovalRequirement",
    "AssetCacheIntegrityError",
    "AssetCacheMutationError",
    "AssetColliderKind",
    "AssetGenerationPipeline",
    "AssetManifest",
    "AssetPipelineResult",
    "AssetRequest",
    "ColliderConstruction",
    "ColliderProposal",
    "ColliderSource",
    "ContentAddressedAssetCache",
    "GeneratedArtifact",
    "GenerationRoute",
    "HumanAssetDecision",
    "HumanVerdict",
    "PhysicalField",
    "PhysicalProposal",
    "ProposalOrigin",
    "ProviderSnapshot",
    "ProviderState",
    "ReferenceImageHttpAdapter",
    "TrellisHttpAdapter",
    "UrllibJsonTransport",
    "VisualMeshFormat",
    "WorldAdmissibleAsset",
    "seed_verified_fallback",
]
