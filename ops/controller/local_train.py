"""Run the pinned controller bootstrap through a local CPU checkout."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from ops.controller.contract import RunMode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", type=RunMode, choices=tuple(RunMode), default=RunMode.SMOKE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--confirm-full", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode is RunMode.FULL and not args.confirm_full:
        raise ValueError("full mode requires --confirm-full")
    checkout = args.checkout.resolve()
    patch = args.patch.resolve()
    python = checkout / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"pinned checkout environment is missing {python}")
    if not patch.is_file():
        raise FileNotFoundError(patch)

    os.environ["MM_CONTROLLER_CHECKOUT"] = checkout.as_posix()
    os.environ["MM_CONTROLLER_PATCH"] = patch.as_posix()
    os.environ["MM_CONTROLLER_PYTHON"] = python.as_posix()
    from ops.controller.remote_job import run_training

    run_id = args.run_id or (
        f"g1-100hz-{args.mode.value}-seed-{args.seed}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_root = args.output_root.resolve() / run_id
    print(f"run_id={run_id}", flush=True)
    manifest = run_training(
        args.mode,
        args.seed,
        run_root,
        run_id=run_id,
        execution_backend="local-macos-cpu",
    )
    print(manifest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
