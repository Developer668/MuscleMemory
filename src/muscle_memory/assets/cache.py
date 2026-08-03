"""Content-addressed asset cache with fail-closed integrity verification."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from muscle_memory.assets.models import (
    SHA256_PATTERN,
    ArtifactDescriptor,
    ArtifactRole,
    AssetManifest,
    GeneratedArtifact,
    canonical_json_bytes,
    sha256_bytes,
)


class AssetCacheError(RuntimeError):
    """Base error for asset cache operations."""


class AssetCacheIntegrityError(AssetCacheError):
    """Cached bytes or metadata did not match their declared checksum."""


class AssetCacheMutationError(AssetCacheError):
    """An immutable content address or request alias was changed."""


@dataclass(frozen=True, slots=True)
class CachedAssetBundle:
    """A manifest whose referenced bytes were verified during this read."""

    manifest: AssetManifest
    reference_image: bytes
    visual_mesh: bytes


def _is_sha256(value: str) -> bool:
    import re

    return re.fullmatch(SHA256_PATTERN, value) is not None


class ContentAddressedAssetCache:
    """Disk cache where blobs, manifests, and aliases are checksum verified."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._blob_root = root / "blobs" / "sha256"
        self._manifest_root = root / "manifests"
        self._alias_root = root / "request-aliases"
        for directory in (self._blob_root, self._manifest_root, self._alias_root):
            directory.mkdir(parents=True, exist_ok=True)

    def store_bundle(
        self,
        *,
        request_sha256: str,
        reference_image: GeneratedArtifact,
        visual_mesh: GeneratedArtifact,
        reference_provider: str,
        mesh_provider: str,
        verified_fallback: bool,
        live_generation: bool,
        bind_request: bool = True,
    ) -> AssetManifest:
        """Persist a complete pair only after both artifacts have been produced."""
        if not _is_sha256(request_sha256):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        if reference_image.role is not ArtifactRole.REFERENCE_INPUT:
            raise ValueError("reference_image has the wrong artifact role")
        if visual_mesh.role is not ArtifactRole.RENDERING_ONLY:
            raise ValueError("visual_mesh must be rendering-only")
        if visual_mesh.mesh_format is None:
            raise ValueError("visual_mesh requires a declared format")

        reference_descriptor = self._store_blob(reference_image)
        visual_descriptor = self._store_blob(visual_mesh)
        content_payload: dict[str, object] = {
            "live_generation": live_generation,
            "mesh_provider": mesh_provider,
            "reference_image": reference_descriptor.model_dump(mode="json"),
            "reference_provider": reference_provider,
            "request_sha256": request_sha256,
            "schema_version": 1,
            "verified_fallback": verified_fallback,
            "visual_mesh": visual_descriptor.model_dump(mode="json"),
            "visual_mesh_format": visual_mesh.mesh_format.value,
        }
        bundle_id = sha256_bytes(canonical_json_bytes(content_payload))
        manifest = AssetManifest(
            bundle_id=bundle_id,
            request_sha256=request_sha256,
            reference_image=reference_descriptor,
            visual_mesh=visual_descriptor,
            visual_mesh_format=visual_mesh.mesh_format,
            reference_provider=reference_provider,
            mesh_provider=mesh_provider,
            verified_fallback=verified_fallback,
            live_generation=live_generation,
        )
        manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
        self._write_verified_pair(
            self._manifest_path(bundle_id),
            manifest_bytes,
            mutation_label="asset manifest",
        )
        if bind_request:
            self.bind_request(request_sha256, bundle_id)
        return manifest

    def bind_request(self, request_sha256: str, bundle_id: str) -> None:
        """Create an immutable request-to-bundle cache alias."""
        if not _is_sha256(request_sha256) or not _is_sha256(bundle_id):
            raise ValueError("cache aliases require SHA-256 identifiers")
        alias = canonical_json_bytes(
            {
                "bundle_id": bundle_id,
                "request_sha256": request_sha256,
                "schema_version": 1,
            }
        )
        self._write_verified_pair(
            self._alias_path(request_sha256),
            alias,
            mutation_label="request alias",
        )

    def lookup_request(self, request_sha256: str) -> CachedAssetBundle | None:
        """Resolve and verify a cached request, or return None when absent."""
        if not _is_sha256(request_sha256):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        alias_path = self._alias_path(request_sha256)
        checksum_path = self._checksum_path(alias_path)
        if not alias_path.exists() and not checksum_path.exists():
            return None
        alias_bytes = self._read_verified_pair(alias_path, label="request alias")
        try:
            import json

            alias = json.loads(alias_bytes)
        except (UnicodeDecodeError, ValueError) as exc:
            raise AssetCacheIntegrityError("request alias is not valid JSON") from exc
        if not isinstance(alias, dict):
            raise AssetCacheIntegrityError("request alias must be a JSON object")
        if alias.get("request_sha256") != request_sha256:
            raise AssetCacheIntegrityError("request alias points to a different request")
        bundle_id = alias.get("bundle_id")
        if not isinstance(bundle_id, str) or not _is_sha256(bundle_id):
            raise AssetCacheIntegrityError("request alias contains an invalid bundle ID")
        return self.load_bundle(bundle_id)

    def load_bundle(self, bundle_id: str) -> CachedAssetBundle:
        """Read a manifest and verify every byte before returning it."""
        if not _is_sha256(bundle_id):
            raise ValueError("bundle_id must be a lowercase SHA-256 digest")
        manifest_bytes = self._read_verified_pair(
            self._manifest_path(bundle_id),
            label="asset manifest",
        )
        try:
            manifest = AssetManifest.model_validate_json(manifest_bytes)
        except ValidationError as exc:
            raise AssetCacheIntegrityError("asset manifest failed schema validation") from exc
        if manifest.bundle_id != bundle_id:
            raise AssetCacheIntegrityError("manifest is stored under the wrong content address")
        reference_image = self._read_blob(manifest.reference_image)
        visual_mesh = self._read_blob(manifest.visual_mesh)
        return CachedAssetBundle(
            manifest=manifest,
            reference_image=reference_image,
            visual_mesh=visual_mesh,
        )

    def verify_bundle(self, bundle_id: str) -> AssetManifest:
        """Verify a complete bundle and return only its safe metadata."""
        return self.load_bundle(bundle_id).manifest

    def blob_path(self, descriptor: ArtifactDescriptor) -> Path:
        """Return a blob path for renderer use after independently verifying it."""
        self._read_blob(descriptor)
        return self._blob_path(descriptor.sha256)

    def _store_blob(self, artifact: GeneratedArtifact) -> ArtifactDescriptor:
        digest = sha256_bytes(artifact.data)
        self._write_once(
            self._blob_path(digest),
            artifact.data,
            mutation_label="content-addressed blob",
        )
        return ArtifactDescriptor(
            sha256=digest,
            byte_count=len(artifact.data),
            media_type=artifact.media_type,
            role=artifact.role,
        )

    def _read_blob(self, descriptor: ArtifactDescriptor) -> bytes:
        path = self._blob_path(descriptor.sha256)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise AssetCacheIntegrityError(
                f"cached {descriptor.role.value} blob is missing or unreadable"
            ) from exc
        if len(data) != descriptor.byte_count:
            raise AssetCacheIntegrityError(
                f"cached {descriptor.role.value} blob has an unexpected size"
            )
        if sha256_bytes(data) != descriptor.sha256:
            raise AssetCacheIntegrityError(
                f"cached {descriptor.role.value} blob failed checksum validation"
            )
        return data

    def _write_verified_pair(self, path: Path, data: bytes, *, mutation_label: str) -> None:
        self._write_once(path, data, mutation_label=mutation_label)
        checksum = f"{sha256_bytes(data)}\n".encode("ascii")
        self._write_once(
            self._checksum_path(path),
            checksum,
            mutation_label=f"{mutation_label} checksum",
        )

    def _read_verified_pair(self, path: Path, *, label: str) -> bytes:
        checksum_path = self._checksum_path(path)
        try:
            data = path.read_bytes()
            checksum_text = checksum_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise AssetCacheIntegrityError(f"{label} or its checksum is missing") from exc
        if not _is_sha256(checksum_text):
            raise AssetCacheIntegrityError(f"{label} checksum file is invalid")
        if sha256_bytes(data) != checksum_text:
            raise AssetCacheIntegrityError(f"{label} failed checksum validation")
        return data

    @staticmethod
    def _write_once(path: Path, data: bytes, *, mutation_label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise AssetCacheIntegrityError(
                    f"could not verify existing {mutation_label}"
                ) from exc
            if existing != data:
                raise AssetCacheMutationError(
                    f"immutable {mutation_label} already has other bytes"
                ) from None

    def _blob_path(self, digest: str) -> Path:
        return self._blob_root / digest[:2] / digest

    def _manifest_path(self, bundle_id: str) -> Path:
        return self._manifest_root / f"{bundle_id}.json"

    def _alias_path(self, request_sha256: str) -> Path:
        return self._alias_root / f"{request_sha256}.json"

    @staticmethod
    def _checksum_path(path: Path) -> Path:
        return path.with_suffix(f"{path.suffix}.sha256")
