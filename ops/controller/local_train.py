"""Run the pinned controller bootstrap through a local CPU checkout."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from ops.controller.contract import RunMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _positive_cpu_device_count(value: str) -> int:
    count = int(value)
    available = os.cpu_count() or 1
    if count < 1 or count > available:
        raise argparse.ArgumentTypeError(
            f"CPU device count must be between 1 and {available} on this host"
        )
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", type=RunMode, choices=tuple(RunMode), default=RunMode.SMOKE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument(
        "--cpu-devices",
        type=_positive_cpu_device_count,
        default=1,
        help="Number of virtual JAX CPU devices used for the immutable run",
    )
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
    os.environ["PYTHONPATH"] = REPOSITORY_ROOT.as_posix()
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={args.cpu_devices}"
    )
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
        execution_backend=f"local-macos-cpu-{args.cpu_devices}-device",
    )
    print(manifest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
