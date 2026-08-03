"""Resilient asset-generation pipeline with cache and human-approval gates."""

from __future__ import annotations

from typing import Protocol

from muscle_memory.assets.approvals import (
    AssetApprovalLedger,
    AssetApprovalRequirement,
    HumanVerdict,
)
from muscle_memory.assets.builtin import seed_verified_fallback
from muscle_memory.assets.cache import ContentAddressedAssetCache
from muscle_memory.assets.models import (
    AdmissionState,
    AdmittedPhysicalProperties,
    AssetManifest,
    AssetPipelineResult,
    AssetRequest,
    ColliderConstruction,
    ColliderSource,
    GeneratedArtifact,
    GenerationRoute,
    PhysicalProposal,
    PhysicsCollider,
    ProviderSnapshot,
    ProviderState,
    VisualMeshFormat,
    WorldAdmissibleAsset,
)
from muscle_memory.assets.providers import AssetProviderError


class ReferenceImageProvider(Protocol):
    @property
    def snapshot(self) -> ProviderSnapshot: ...

    def generate(self, request: AssetRequest) -> GeneratedArtifact: ...


class VisualMeshProvider(Protocol):
    @property
    def snapshot(self) -> ProviderSnapshot: ...

    def generate(
        self,
        reference_image: GeneratedArtifact,
        *,
        mesh_format: VisualMeshFormat,
    ) -> GeneratedArtifact: ...


class AssetGenerationPipeline:
    """Generate appearance, survive provider failure, and gate world admission."""

    def __init__(
        self,
        *,
        cache: ContentAddressedAssetCache,
        approvals: AssetApprovalLedger,
        reference_provider: ReferenceImageProvider,
        mesh_provider: VisualMeshProvider,
    ) -> None:
        self._cache = cache
        self._approvals = approvals
        self._reference_provider = reference_provider
        self._mesh_provider = mesh_provider
        self._fallback = seed_verified_fallback(cache)

    @property
    def provider_snapshots(self) -> tuple[ProviderSnapshot, ...]:
        return (self._reference_provider.snapshot, self._mesh_provider.snapshot)

    @property
    def fallback_bundle_id(self) -> str:
        return self._fallback.bundle_id

    def generate(
        self,
        request: AssetRequest,
        physical_proposal: PhysicalProposal,
    ) -> AssetPipelineResult:
        """Resolve cache, try live providers, or return the verified fallback."""
        cached = self._cache.lookup_request(request.request_sha256)
        if cached is not None:
            return self._apply_approval_gate(
                manifest=cached.manifest,
                physical_proposal=physical_proposal,
                route=GenerationRoute.REQUEST_CACHE,
                detail="request resolved from a checksum-verified cache entry",
            )

        if any(
            snapshot.state is ProviderState.UNCONFIGURED for snapshot in self.provider_snapshots
        ):
            return self._fallback_result(
                physical_proposal,
                reason="one or more generation providers are unconfigured",
            )

        try:
            reference = self._reference_provider.generate(request)
            visual_mesh = self._mesh_provider.generate(
                reference,
                mesh_format=request.mesh_format,
            )
            manifest = self._cache.store_bundle(
                request_sha256=request.request_sha256,
                reference_image=reference,
                visual_mesh=visual_mesh,
                reference_provider=self._reference_provider.snapshot.provider,
                mesh_provider=self._mesh_provider.snapshot.provider,
                verified_fallback=False,
                live_generation=True,
            )
        except (AssetProviderError, OSError, TimeoutError) as exc:
            return self._fallback_result(
                physical_proposal,
                reason=f"generation provider failed ({type(exc).__name__})",
            )
        return self._apply_approval_gate(
            manifest=manifest,
            physical_proposal=physical_proposal,
            route=GenerationRoute.LIVE_PROVIDER,
            detail="providers returned validated artifacts now stored in the verified cache",
        )

    def refresh_admission(
        self,
        *,
        bundle_id: str,
        physical_proposal: PhysicalProposal,
        route: GenerationRoute,
    ) -> AssetPipelineResult:
        """Re-evaluate a blocked result after the human interface records a decision."""
        manifest = self._cache.verify_bundle(bundle_id)
        return self._apply_approval_gate(
            manifest=manifest,
            physical_proposal=physical_proposal,
            route=route,
            detail="world admission was re-evaluated against the immutable human decision ledger",
        )

    def _fallback_result(
        self,
        physical_proposal: PhysicalProposal,
        *,
        reason: str,
    ) -> AssetPipelineResult:
        fallback = self._cache.load_bundle(self._fallback.bundle_id)
        if not fallback.manifest.verified_fallback or fallback.manifest.live_generation:
            raise RuntimeError("configured fallback manifest is not a verified cached fallback")
        return self._apply_approval_gate(
            manifest=fallback.manifest,
            physical_proposal=physical_proposal,
            route=GenerationRoute.VERIFIED_FALLBACK,
            detail=f"verified cached fallback used because {reason}",
        )

    def _apply_approval_gate(
        self,
        *,
        manifest: AssetManifest,
        physical_proposal: PhysicalProposal,
        route: GenerationRoute,
        detail: str,
    ) -> AssetPipelineResult:
        requirement: AssetApprovalRequirement | None = None
        approval_reference = physical_proposal.collider.prior_approval_id
        if physical_proposal.requires_human_approval:
            requirement = AssetApprovalRequirement.create(
                bundle_id=manifest.bundle_id,
                proposal=physical_proposal,
            )
            self._approvals.submit(requirement)
            decision = self._approvals.decision_for(requirement.requirement_id)
            if decision is None:
                return AssetPipelineResult(
                    route=route,
                    manifest=manifest,
                    admission_state=AdmissionState.BLOCKED_APPROVAL,
                    approval_requirement_id=requirement.requirement_id,
                    provider_snapshots=self.provider_snapshots,
                    detail=f"{detail}; physical properties are blocked on human approval",
                )
            if decision.verdict is HumanVerdict.REJECT:
                return AssetPipelineResult(
                    route=route,
                    manifest=manifest,
                    admission_state=AdmissionState.REJECTED,
                    approval_requirement_id=requirement.requirement_id,
                    provider_snapshots=self.provider_snapshots,
                    detail=f"{detail}; the human reviewer rejected the physical proposal",
                )
            approval_reference = requirement.requirement_id

        world_asset = self._world_asset(
            manifest=manifest,
            proposal=physical_proposal,
            approval_reference=approval_reference,
        )
        return AssetPipelineResult(
            route=route,
            manifest=manifest,
            admission_state=AdmissionState.READY,
            approval_requirement_id=(requirement.requirement_id if requirement else None),
            world_asset=world_asset,
            provider_snapshots=self.provider_snapshots,
            detail=detail,
        )

    @staticmethod
    def _world_asset(
        *,
        manifest: AssetManifest,
        proposal: PhysicalProposal,
        approval_reference: str | None,
    ) -> WorldAdmissibleAsset:
        if proposal.collider.construction is ColliderConstruction.DETERMINISTIC_PRIMITIVE:
            collider_source = ColliderSource.DETERMINISTIC_PRIMITIVE
        else:
            collider_source = ColliderSource.APPROVED_CONVEX
        collider = PhysicsCollider(
            kind=proposal.collider.kind,
            source=collider_source,
            dimensions=proposal.dimensions,
            geometry_sha256=proposal.collider.geometry_sha256,
            approval_reference=approval_reference,
        )
        physical = AdmittedPhysicalProperties(
            category=proposal.category,
            mass_kg=proposal.mass_kg,
            sliding_friction=proposal.sliding_friction,
            restitution=proposal.restitution,
            movable=proposal.movable,
            proposal_sha256=proposal.proposal_sha256,
            approval_reference=approval_reference,
        )
        return WorldAdmissibleAsset(
            asset_id=manifest.bundle_id,
            visual_mesh=manifest.visual_mesh,
            visual_mesh_format=manifest.visual_mesh_format,
            physical=physical,
            collider=collider,
        )
