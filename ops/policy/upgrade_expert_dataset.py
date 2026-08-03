"""Recover continuous teacher targets for the deterministic expert corpus."""

from __future__ import annotations

import json

from muscle_memory.training.dataset import upgrade_expert_dataset_with_commands


def main() -> int:
    metadata = upgrade_expert_dataset_with_commands()
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
