"""Seed and verify the non-network asset fallback used by the demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from muscle_memory.assets import ContentAddressedAssetCache, seed_verified_fallback


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/assets/cache"),
        help="content-addressed cache root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cache = ContentAddressedAssetCache(args.cache_dir)
    manifest = seed_verified_fallback(cache)
    verified = cache.verify_bundle(manifest.bundle_id)
    print(
        json.dumps(
            {
                "bundle_id": verified.bundle_id,
                "cache_dir": str(args.cache_dir.resolve()),
                "live_generation": verified.live_generation,
                "verified_fallback": verified.verified_fallback,
                "visual_mesh_sha256": verified.visual_mesh.sha256,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
