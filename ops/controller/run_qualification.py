"""Local and remote command-line runner for source and qualification gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ops.controller.contract import (
    QualificationEvidence,
    RunMode,
    build_artifact_manifest,
    evaluate_qualification,
    verify_qualification_binding,
    verify_source_checkout,
    write_artifact_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("verify-source")
    source.add_argument("--checkout", type=Path, required=True)
    source.add_argument("--patch", type=Path, required=True)
    source.add_argument("--patched", action="store_true")

    qualify = commands.add_parser("qualify")
    qualify.add_argument("--checkout", type=Path, required=True)
    qualify.add_argument("--patch", type=Path, required=True)
    qualify.add_argument("--run-root", type=Path, required=True)
    qualify.add_argument("--evidence", type=Path, required=True)
    qualify.add_argument("--mode", type=RunMode, choices=tuple(RunMode), required=True)
    qualify.add_argument("--seed", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = verify_source_checkout(args.checkout, args.patch, patched=args.patched)
    if args.command == "verify-source":
        print(json.dumps(asdict(source), indent=2, sort_keys=True))
        return 0

    evidence_payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("qualification evidence must be a JSON object")
    evidence = QualificationEvidence.from_mapping(evidence_payload)
    verify_qualification_binding(
        evidence,
        args.run_root,
        Path(__file__).with_name("native_qualify.py"),
    )
    result = evaluate_qualification(evidence, args.mode)
    manifest = build_artifact_manifest(
        args.run_root,
        mode=args.mode,
        seed=args.seed,
        source=source,
        qualification=result,
    )
    destination = write_artifact_manifest(args.run_root, manifest)
    print(destination.read_text(encoding="utf-8"), end="")
    return 0 if result.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
