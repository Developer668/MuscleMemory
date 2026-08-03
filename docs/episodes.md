# Episode backend

`muscle_memory.episodes.EpisodeService` is the lifecycle boundary between measured
simulation output, append-only telemetry, and post-episode graph memory.

## Lifecycle

1. Open an episode with an `EpisodeIdentity`. Robot checksum, world ID/hash/split,
   and policy ID/hash are permanent after this call.
2. Append `EpisodeTelemetryRecord` values in contiguous sequence order on the exact
   20 Hz numeric clock. The injected telemetry backend must durably store the exact
   record before returning its provider receipt.
3. Close once with a measured `PolicyEpisodeResult`. Closure hashes the ordered
   provider envelopes, changes the episode state to `closed`, and only then records
   the episode and its failure facts in graph memory.
4. Replay a closed episode in sequence order. `frame_id` is the sole video join key;
   simulation time remains ordering metadata and is never a media join key.

The close result reports telemetry and graph delivery independently. A durable local
LaserData spool or append-only FalkorDB cache is reported as a partial provider
delivery, never as sponsor-provider success. Failed episodes retain signed MuJoCo
clearance, including negative penetration distance.

## Corrections

Route and keep-out geometry is canonicalized and content-addressed when submitted.
Submission does not make it training data. An `AuthenticatedHuman` must approve the
pending correction, and the approval is recorded as a separate immutable fact before
the graph-provider write is attempted.

Only `muscle_memory.episodes.training.TrainingCorrectionFeed` exposes approved
corrections. Policy, robot control, simulation control, and evaluation modules have an
import-firewall test preventing access to the episode correction boundary. Evaluation
therefore cannot consume a user route, keep-out region, graph lesson, or path teacher.

## Wiring

Construct the service with:

- `LaserDataTelemetryBackend` as `telemetry_backend`
- that backend's `spool` as `telemetry_store`
- `ResilientGraphMemory` as `graph_memory`

Initialize the LaserData backend before opening an episode and close it during process
shutdown. Provider health and exact replay verification remain the responsibility of
the provider adapter; the episode service carries its per-event receipts unchanged.

## Replay inspection

Inspect the durable local spool without contacting a provider:

```bash
uv run python -m ops.episodes.inspect_replay \
  --spool artifacts/telemetry/laserdata-spool.sqlite3 \
  --episode-id episode-001
```

The command prints the ordered sequence, each optional `frame_id`, and the immutable
telemetry digest used by episode closure.
