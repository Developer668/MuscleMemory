# Backend deployment

The repository ships one reproducible container image and a Compose stack for backend
development. The API runs with persistent coordinator, telemetry-outbox, graph-cache, and
asset-cache volumes. The local data services are pinned self-hosted instances of FalkorDB and
the LaserData Apache Iggy fork required by the stable SDK's VSR wire protocol.

This stack is deployment tooling, not cloud-readiness evidence. A healthy local FalkorDB or
Iggy container proves only the self-hosted development path. Managed LaserData, FalkorDB,
Guild.ai, RocketRide, reference-image, and TRELLIS integrations remain unconfigured until
their runtime variables are supplied and their provider-specific verification succeeds.

## Start and stop

Docker Desktop or a compatible Docker Engine with Compose is required.

```bash
./ops/deployment/start.sh
./ops/deployment/stop.sh
```

The first start creates `.env.backend.local` with mode `0600`, random local-provider
passwords, a random operator bearer token, and only the token's SHA-256 digest for the API
container. Existing files are validated and never overwritten. The plaintext operator token
is not passed into the container. Retrieve it locally when making an authenticated mutation:

```bash
sed -n 's/^MM_API_OPERATOR_TOKEN=//p' .env.backend.local
```

Normal stop preserves every named volume. Destructive local teardown is explicit:

```bash
./ops/deployment/stop.sh --purge
```

Set `MM_DEPLOY_ENV_FILE` to use another ignored environment path. The committed
[`backend-deployment.env.example`](../config/services/backend-deployment.env.example) lists
the accepted values but intentionally contains no usable secret.

## Readiness boundary

Compose waits for both local data services and the API process. `start.sh` then executes the
real LaserData verification command inside the API container. The command publishes one
canonical eight-category event at the declared 20 Hz telemetry contract and reads the exact
event back through the live consumer before the final API health smoke can pass. A TCP socket,
SDK import, or local outbox alone is not accepted as provider readiness.

Iggy carries numeric sensor data, actions, lifecycle events, and synchronized `frame_id`
metadata. Video bytes stay on their dedicated media transport; `frame_id` is their only join
key. Iggy is operational experience infrastructure and never enters the robot's control path.

The API exposes:

```text
http://127.0.0.1:8000/api/v1/health
```

Run the smoke independently when diagnosing a live stack:

```bash
python3 -m ops.deployment.smoke \
  --require-provider LaserData \
  --require-provider FalkorDB
```

Container liveness intentionally checks that the API returns typed health, even when a
provider reports `degraded`. This keeps the diagnostic surface available. The external smoke
is stricter and requires the named providers to report `healthy` or
`end_to_end_verified`.

## Runtime configuration

The API factory is the zero-argument composition root:

```text
MM_API_BACKEND_FACTORY=muscle_memory.runtime:create_api_backend
```

Local defaults connect the API to `laserdata:8090` and `falkordb:6379` on the private Compose
network. Laser uses the SDK's bare `user:password@host:port` connection form. Supplying
`LASERDATA_CONNECTION_STRING` or `MUSCLE_MEMORY_FALKORDB_URL` overrides
those addresses for the API without relabeling the local services as managed-provider proof.
The Guild.ai, RocketRide, and asset-provider variables are passed through only when present;
empty values remain truthfully unconfigured.

All host-published ports bind to `127.0.0.1` unless explicitly changed. Provider ports are
available only for local verification:

| Service | Host default | Container |
| --- | --- | --- |
| Backend API | `127.0.0.1:8000` | `8000` |
| Apache Iggy TCP | `127.0.0.1:8090` | `8090` |
| FalkorDB | `127.0.0.1:6379` | `6379` |

## Container hardening and persistence

The API image uses digest-pinned Python and uv stages, installs only locked runtime
dependencies, runs as UID/GID `10001`, drops Linux capabilities, enables
`no-new-privileges`, and mounts a read-only root filesystem with a bounded `/tmp` tmpfs. It
contains the frozen robot bundle, its validation evidence, the backend code, and sponsor
contracts, but no `.env` file or provider credential.

The data volumes are:

- `coordinator-data`: append-only coordinator SQLite history
- `telemetry-data`: durable LaserData outbox and receipts
- `graph-cache`: append-only FalkorDB recovery records
- `asset-cache`: cached rendering artifacts and approval ledger
- `falkordb-data`: self-hosted graph persistence
- `iggy-data`: self-hosted operational event-log persistence

Iggy receives the `SYS_NICE`, unconfined seccomp, and unlimited memlock settings its
documented `io_uring` runtime requires. The public LaserData Iggy server image is pinned by
immutable multi-platform digest because the stable Laser SDK wheel always uses VSR; standard
Apache Iggy images use the classic protocol and cannot satisfy the append/readback gate.
FalkorDB uses its server-only image with AOF and `appendfsync everysec`. Image upgrades must
be intentional and followed by the deployment tests and real provider readback.

Provider references:

- [Apache Iggy Docker deployment](https://iggy.apache.org/docs/server/docker/)
- [FalkorDB Docker deployment](https://docs.falkordb.com/operations/docker)
- [Laser SDK local server contract](https://github.com/laserdata/laser-sdk)
