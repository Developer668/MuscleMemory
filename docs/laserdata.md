# LaserData telemetry integration

Muscle Memory writes episode telemetry to LaserData's durable Apache Iggy log through the
official Python SDK. The adapter targets one stream and one topic, partitions by `episode_id`,
and publishes the exact canonical telemetry envelope as JSON. `frame_id` remains the only join
key between provider events and the separately transported video stream.

## Runtime dependency

The production image must install the official pre-release currently exercised by the adapter:

```text
laser-sdk==0.0.1rc16
```

The import is lazy so an unconfigured developer process can still start and report its real
state. An SDK import, fixture transport, or local spool does not count as provider health.

## Configuration

Copy the LaserData variables from `.env.example` into the deployment's secret manager. The
connection string contains credentials and is deliberately excluded from object reprs, health
details, and verification output.

```text
LASERDATA_CONNECTION_STRING=iggy://username:password@deployment-host:8090
LASERDATA_STREAM=muscle-memory
LASERDATA_TOPIC=episode-telemetry-v1
LASERDATA_PARTITIONS=4
LASERDATA_TIMEOUT_SECONDS=5
MUSCLE_MEMORY_TELEMETRY_SPOOL=artifacts/telemetry/laserdata-spool.sqlite3
```

LaserData Cloud connection strings can use managed TLS. Local or self-hosted Apache Iggy uses
the same client surface, but it is development evidence rather than LaserData Cloud proof.

## Delivery contract

Every record contains the permanent robot checksum, world and policy hashes, monotonic episode
sequence, simulation timestamp, all eight labeled sensor categories, the canonical payload
checksum, and an optional synchronized `frame_id`. Numeric telemetry is scheduled at exactly
20 Hz by `NumericTelemetryCadence`; the 500 Hz physics clock emits every 25 physics steps.

The backend always appends locally first. The SQLite spool uses triggers to reject every update
or delete to episode records, provider acknowledgements, and replay verifications. A provider
failure therefore leaves a durable outbox entry without rewriting history.

The delivery result is explicit:

- `laserdata_and_durable_cache`: the official provider accepted the append and the immutable
  local record remains available for recovery.
- `durable_local_cache_only`: the provider did not accept the append. This is degraded or
  unconfigured operation, never a successful LaserData write.

Provider health progresses through `unconfigured`, `configured`, `healthy`, and
`end_to_end_verified`; a failed probe, publish, or replay is `degraded`. `healthy` requires a
real connection, topic ensure, and capability probe. `end_to_end_verified` additionally
requires reading the exact event back from the provider and retaining its log position.

## Live verification

After injecting real deployment configuration, run:

```bash
uv run python -m ops.sponsors.verify_laserdata
```

The command verifies the frozen MM-01 bundle, appends one checksummed eight-category event,
replays that exact `event_id`, and prints the provider position. It exits `0` only for
`end_to_end_verified`, `1` for configured but unverified/degraded, and `2` for unconfigured.
The ignored SQLite spool is resilience evidence, not provider evidence.

Official references:

- [LaserData platform](https://laserdata.com/)
- [LaserData Cloud documentation](https://docs.laserdata.cloud/)
- [Official Python SDK](https://pypi.org/project/laser-sdk/)
