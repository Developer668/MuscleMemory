"""Append and replay one real LaserData telemetry event."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid

from muscle_memory.robot.identity import verify_mm01_bundle
from muscle_memory.telemetry import (
    EpisodeTelemetryRecord,
    LaserDataConfig,
    LaserDataProviderState,
    LaserDataTelemetryBackend,
    SensorSnapshot,
    SignalUseLabel,
)


def _fixed_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


async def verify_live_provider() -> tuple[int, dict[str, object]]:
    config = LaserDataConfig.from_env()
    if not config.configured:
        return 2, {
            "provider": "LaserData",
            "state": LaserDataProviderState.UNCONFIGURED,
            "verified": False,
            "detail": "LASERDATA_CONNECTION_STRING is unset",
        }

    robot = verify_mm01_bundle()
    episode_id = f"laserdata-verification-{uuid.uuid4().hex}"
    frame_id = f"{episode_id}:00000000"
    record = EpisodeTelemetryRecord.create(
        episode_id=episode_id,
        world_id="world-laserdata-verification",
        policy_id="policy-laserdata-verification",
        sequence=0,
        sim_time_seconds=0.0,
        robot_checksum=robot.robot_checksum,
        policy_hash=_fixed_hash("laserdata-provider-verification-policy"),
        world_hash=_fixed_hash("laserdata-provider-verification-world"),
        signal_use=SignalUseLabel.LOGGED_ONLY,
        sensors=SensorSnapshot.all_unavailable(),
        payload={
            "event": "provider_verification",
            "numeric_telemetry_hz": 20,
        },
        frame_id=frame_id,
    )
    backend = LaserDataTelemetryBackend(config)
    try:
        startup = await backend.initialize()
        if startup.state is not LaserDataProviderState.HEALTHY:
            return 1, {
                "provider": startup.provider,
                "state": startup.state,
                "verified": False,
                "detail": startup.detail,
                "pending_local_records": startup.pending_local_records,
            }
        # LaserData is credited as live only after the same content-addressed event
        # is appended and read back from the provider, not merely after connecting.
        append = await backend.append(record)
        position = await backend.verify_event(append.event_id)
        health = backend.health
        verified = (
            position is not None and health.state is LaserDataProviderState.END_TO_END_VERIFIED
        )
        return (0 if verified else 1), {
            "provider": health.provider,
            "state": health.state,
            "verified": verified,
            "episode_id": episode_id,
            "event_id": append.event_id,
            "frame_id": frame_id,
            "provider_position": position,
            "delivery": append.delivery,
            "pending_local_records": health.pending_local_records,
            "robot_checksum": robot.robot_checksum,
            "detail": health.detail,
        }
    finally:
        await backend.close()


def main() -> None:
    exit_code, evidence = asyncio.run(verify_live_provider())
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
