"""Inspect one episode in the durable telemetry spool without provider access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muscle_memory.episodes import telemetry_digest
from muscle_memory.telemetry import DurableTelemetrySpool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", required=True, type=Path)
    parser.add_argument("--episode-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spool_path: Path = args.spool
    episode_id: str = args.episode_id
    if not spool_path.is_file():
        raise SystemExit(f"telemetry spool does not exist: {spool_path}")
    with DurableTelemetrySpool(spool_path) as spool:
        records = spool.records_for(episode_id)
    if not records:
        raise SystemExit(f"episode has no telemetry records: {episode_id}")

    report = {
        "episode_id": episode_id,
        "frame_join_key": "frame_id",
        "record_count": len(records),
        "records": [
            {
                "frame_id": record.frame_id,
                "sequence": record.sequence,
                "sim_time_seconds": record.sim_time_seconds,
            }
            for record in records
        ],
        "telemetry_digest": telemetry_digest(records),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
