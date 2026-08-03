"""Deterministic appearance fallback seeded into the verified asset cache."""

from __future__ import annotations

import base64
import json
import struct

from muscle_memory.assets.cache import ContentAddressedAssetCache
from muscle_memory.assets.models import (
    ArtifactRole,
    AssetManifest,
    AssetRequest,
    GeneratedArtifact,
    VisualMeshFormat,
)

_REFERENCE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _minimal_glb() -> bytes:
    vertices = (
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    )
    indices = (
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        6,
        5,
        4,
        7,
        6,
        0,
        4,
        5,
        0,
        5,
        1,
        1,
        5,
        6,
        1,
        6,
        2,
        2,
        6,
        7,
        2,
        7,
        3,
        3,
        7,
        4,
        3,
        4,
        0,
    )
    vertex_bytes = b"".join(struct.pack("<3f", *vertex) for vertex in vertices)
    index_bytes = struct.pack(f"<{len(indices)}H", *indices)
    binary = vertex_bytes + index_bytes
    scene = json.dumps(
        {
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(vertices),
                    "max": [0.5, 0.5, 0.5],
                    "min": [-0.5, -0.5, -0.5],
                    "type": "VEC3",
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": len(indices),
                    "type": "SCALAR",
                },
            ],
            "asset": {
                "generator": "MuscleMemory verified fallback v1",
                "version": "2.0",
            },
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteLength": len(vertex_bytes),
                    "byteOffset": 0,
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteLength": len(index_bytes),
                    "byteOffset": len(vertex_bytes),
                    "target": 34963,
                },
            ],
            "buffers": [{"byteLength": len(binary)}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "nodes": [{"mesh": 0, "name": "verified-fallback-box"}],
            "scene": 0,
            "scenes": [{"nodes": [0]}],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    scene += b" " * ((-len(scene)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    json_chunk = struct.pack("<I4s", len(scene), b"JSON") + scene
    binary_chunk = struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    length = 12 + len(json_chunk) + len(binary_chunk)
    return struct.pack("<4sII", b"glTF", 2, length) + json_chunk + binary_chunk


BUILTIN_FALLBACK_REQUEST = AssetRequest(
    request_id="verified-fallback-v1",
    prompt="Verified cached generic household obstacle appearance fallback, version 1.",
    mesh_format=VisualMeshFormat.GLB,
)


def seed_verified_fallback(cache: ContentAddressedAssetCache) -> AssetManifest:
    """Idempotently seed and re-read the non-network critical-demo fallback."""
    manifest = cache.store_bundle(
        request_sha256=BUILTIN_FALLBACK_REQUEST.request_sha256,
        reference_image=GeneratedArtifact(
            data=_REFERENCE_PNG,
            media_type="image/png",
            role=ArtifactRole.REFERENCE_INPUT,
        ),
        visual_mesh=GeneratedArtifact(
            data=_minimal_glb(),
            media_type="model/gltf-binary",
            role=ArtifactRole.RENDERING_ONLY,
            mesh_format=VisualMeshFormat.GLB,
        ),
        reference_provider="built-in-reference-fallback-v1",
        mesh_provider="built-in-visual-fallback-v1",
        verified_fallback=True,
        live_generation=False,
        bind_request=False,
    )
    return cache.verify_bundle(manifest.bundle_id)
