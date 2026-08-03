from __future__ import annotations

import base64
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from muscle_memory.assets import (
    AdmissionState,
    AssetApprovalError,
    AssetApprovalLedger,
    AssetCacheIntegrityError,
    AssetColliderKind,
    AssetGenerationPipeline,
    AssetRequest,
    ColliderConstruction,
    ColliderProposal,
    ContentAddressedAssetCache,
    GenerationRoute,
    HumanAssetDecision,
    HumanVerdict,
    PhysicalField,
    PhysicalProposal,
    ProposalOrigin,
    ProviderState,
    ReferenceImageHttpAdapter,
    TrellisHttpAdapter,
    seed_verified_fallback,
)
from muscle_memory.assets.models import PhysicsCollider
from muscle_memory.worlds import Dimensions3D, ObjectCategory


def _glb() -> bytes:
    payload = b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[]}]}'
    payload += b" " * ((-len(payload)) % 4)
    chunk = struct.pack("<I4s", len(payload), b"JSON") + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


class StaticTransport:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.timeouts: list[float] = []

    def post_json(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        del endpoint, payload, headers
        self.timeouts.append(timeout_seconds)
        return self.response


class TimeoutTransport:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def post_json(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        del endpoint, payload, headers
        self.timeouts.append(timeout_seconds)
        raise TimeoutError("fixture timeout")


def _proposal(
    *,
    origin: ProposalOrigin = ProposalOrigin.VERIFIED_CATALOG,
    uncertain_fields: tuple[PhysicalField, ...] = (),
) -> PhysicalProposal:
    return PhysicalProposal(
        category=ObjectCategory.BOX,
        dimensions=Dimensions3D(length_m=0.6, width_m=0.5, height_m=0.4),
        mass_kg=3.0,
        sliding_friction=0.6,
        restitution=0.1,
        movable=True,
        collider=ColliderProposal(
            kind=AssetColliderKind.BOX,
            construction=ColliderConstruction.DETERMINISTIC_PRIMITIVE,
        ),
        origin=origin,
        uncertain_fields=uncertain_fields,
    )


def _unconfigured_pipeline(tmp_path: Path) -> AssetGenerationPipeline:
    return AssetGenerationPipeline(
        cache=ContentAddressedAssetCache(tmp_path / "cache"),
        approvals=AssetApprovalLedger(tmp_path / "approvals"),
        reference_provider=ReferenceImageHttpAdapter(endpoint=None),
        mesh_provider=TrellisHttpAdapter(endpoint=None),
    )


def test_timeout_uses_verified_cached_fallback_without_live_claim(tmp_path: Path) -> None:
    image_transport = StaticTransport(
        {
            "data_base64": base64.b64encode(b"fixture-image").decode("ascii"),
            "media_type": "image/png",
        }
    )
    mesh_transport = TimeoutTransport()
    reference = ReferenceImageHttpAdapter(
        endpoint="https://reference.example.test/generate",
        timeout_seconds=4.0,
        transport=image_transport,
    )
    trellis = TrellisHttpAdapter(
        endpoint="https://trellis.example.test/generate",
        timeout_seconds=7.0,
        transport=mesh_transport,
    )
    pipeline = AssetGenerationPipeline(
        cache=ContentAddressedAssetCache(tmp_path / "cache"),
        approvals=AssetApprovalLedger(tmp_path / "approvals"),
        reference_provider=reference,
        mesh_provider=trellis,
    )

    result = pipeline.generate(
        AssetRequest(request_id="timeout-case", prompt="isolated household box"),
        _proposal(),
    )

    assert result.route is GenerationRoute.VERIFIED_FALLBACK
    assert result.manifest.verified_fallback
    assert not result.manifest.live_generation
    assert result.admission_state is AdmissionState.READY
    assert image_transport.timeouts == [4.0]
    assert mesh_transport.timeouts == [7.0]
    assert reference.snapshot.state is ProviderState.HEALTHY
    assert trellis.snapshot.state is ProviderState.DEGRADED


def test_provider_timeouts_have_a_hard_upper_bound() -> None:
    with pytest.raises(ValueError, match="within"):
        TrellisHttpAdapter(endpoint="https://example.test", timeout_seconds=30.1)


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://provider.example.test/generate",
        "file:///tmp/provider.json",
        "https://user:password@provider.example.test/generate",
    ),
)
def test_provider_endpoints_reject_insecure_or_credential_bearing_urls(
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match=r"HTTPS|credentials"):
        ReferenceImageHttpAdapter(endpoint=endpoint)

    assert ReferenceImageHttpAdapter(endpoint="http://127.0.0.1:9000/generate")


def test_fallback_manifest_and_blob_checksums_are_verified(tmp_path: Path) -> None:
    cache = ContentAddressedAssetCache(tmp_path / "cache")
    manifest = seed_verified_fallback(cache)
    repeated = seed_verified_fallback(cache)
    bundle = cache.load_bundle(manifest.bundle_id)

    assert repeated.bundle_id == manifest.bundle_id
    assert hashlib.sha256(bundle.reference_image).hexdigest() == manifest.reference_image.sha256
    assert hashlib.sha256(bundle.visual_mesh).hexdigest() == manifest.visual_mesh.sha256
    assert bundle.visual_mesh.startswith(b"glTF")
    json_length, json_kind = struct.unpack_from("<I4s", bundle.visual_mesh, 12)
    scene = json.loads(bundle.visual_mesh[20 : 20 + json_length].decode("ascii"))
    assert json_kind == b"JSON"
    assert scene["scenes"][0]["nodes"] == [0]
    assert scene["nodes"][0]["mesh"] == 0
    assert scene["meshes"][0]["primitives"][0]["attributes"]["POSITION"] == 0


def test_cache_tampering_fails_closed(tmp_path: Path) -> None:
    cache = ContentAddressedAssetCache(tmp_path / "cache")
    manifest = seed_verified_fallback(cache)
    visual_path = cache.blob_path(manifest.visual_mesh)
    visual_path.write_bytes(b"tampered")

    with pytest.raises(AssetCacheIntegrityError, match=r"size|checksum"):
        cache.load_bundle(manifest.bundle_id)


def test_manifest_checksum_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = ContentAddressedAssetCache(root)
    manifest = seed_verified_fallback(cache)
    checksum_path = root / "manifests" / f"{manifest.bundle_id}.json.sha256"
    checksum_path.write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(AssetCacheIntegrityError, match="checksum"):
        cache.load_bundle(manifest.bundle_id)


def test_agent_physics_is_blocked_until_human_approval(tmp_path: Path) -> None:
    pipeline = _unconfigured_pipeline(tmp_path)
    proposal = _proposal(
        origin=ProposalOrigin.AGENT,
        uncertain_fields=(
            PhysicalField.DIMENSIONS,
            PhysicalField.MASS,
            PhysicalField.FRICTION,
            PhysicalField.MOBILITY,
            PhysicalField.COLLIDER,
        ),
    )
    request = AssetRequest(request_id="approval-case", prompt="isolated laundry basket")

    blocked = pipeline.generate(request, proposal)

    assert blocked.admission_state is AdmissionState.BLOCKED_APPROVAL
    assert blocked.world_asset is None
    assert blocked.approval_requirement_id is not None
    ledger = AssetApprovalLedger(tmp_path / "approvals")
    requirement = ledger.requirement_for(blocked.approval_requirement_id)
    assert requirement is not None
    assert requirement.blocking
    assert set(requirement.review_fields) == set(PhysicalField)

    ledger.record_human_decision(
        HumanAssetDecision.create(
            requirement,
            human_subject="operator@example.test",
            verdict=HumanVerdict.APPROVE,
        )
    )
    ready = pipeline.refresh_admission(
        bundle_id=blocked.manifest.bundle_id,
        physical_proposal=proposal,
        route=blocked.route,
    )
    assert ready.admission_state is AdmissionState.READY
    assert ready.world_asset is not None
    assert ready.world_asset.physical.approval_reference == requirement.requirement_id


def test_human_asset_decision_tampering_fails_closed(tmp_path: Path) -> None:
    pipeline = _unconfigured_pipeline(tmp_path)
    proposal = _proposal(origin=ProposalOrigin.AGENT)
    blocked = pipeline.generate(
        AssetRequest(request_id="tampered-decision", prompt="isolated household box"),
        proposal,
    )
    assert blocked.approval_requirement_id is not None
    ledger = AssetApprovalLedger(tmp_path / "approvals")
    requirement = ledger.requirement_for(blocked.approval_requirement_id)
    assert requirement is not None
    decision = HumanAssetDecision.create(
        requirement,
        human_subject="operator@example.test",
        verdict=HumanVerdict.REJECT,
    )
    ledger.record_human_decision(decision)

    decision_path = tmp_path / "approvals" / "decisions" / f"{requirement.requirement_id}.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload["verdict"] = "approve"
    decision_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssetApprovalError, match="checksum"):
        ledger.decision_for(requirement.requirement_id)


def test_visual_mesh_cannot_be_supplied_as_a_physics_collider(tmp_path: Path) -> None:
    result = _unconfigured_pipeline(tmp_path).generate(
        AssetRequest(request_id="separation-case", prompt="isolated household stool"),
        _proposal(),
    )
    assert result.world_asset is not None
    assert result.world_asset.visual_mesh.role.value == "rendering_only"
    assert result.world_asset.collider.role == "physics_only"
    assert result.world_asset.collider.geometry_sha256 is None

    with pytest.raises(ValidationError, match="visual_mesh_sha256"):
        PhysicsCollider.model_validate(
            {
                **result.world_asset.collider.model_dump(mode="json"),
                "visual_mesh_sha256": result.world_asset.visual_mesh.sha256,
            }
        )
