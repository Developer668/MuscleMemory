"""Bind the latest physically qualified checkpoint from a completed full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops.controller.remote_job import select_qualified_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--later-evidence-root",
        type=Path,
        action="append",
        default=[],
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = select_qualified_checkpoint(
        args.run_root.resolve(),
        args.checkpoint.resolve(),
        args.evidence_root.resolve(),
        tuple(path.resolve() for path in args.later_evidence_root),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
