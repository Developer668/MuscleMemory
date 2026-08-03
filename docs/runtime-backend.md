# Runtime backend composition

The production API factory is:

```text
MM_API_BACKEND_FACTORY=muscle_memory.runtime:create_api_backend
```

Construction verifies the qualified MM-01 bundle before opening operational state.
The factory composes the SQLite coordinator, append-only LaserData spool, FalkorDB
mirror, exact three-role Guild coordinator, fixed RocketRide executor, verified
asset cache, and FastAPI backend adapter. Invalid local identity or immutable state
fails startup. Provider connection failure remains visible as `degraded` so the
health endpoint can diagnose it.

## Durable boundaries

- The operational episode journal accepts training episodes only. Held-out
  evaluation remains behind the evaluation-only coordinator methods.
- Episode identity, provider receipts, closure, human route or keep-out
  corrections, and authenticated approvals are append-only coordinator facts.
- Guild reviews and RocketRide run snapshots are content-addressed and recovered
  after restart. A run can resume only from its exact reviewed plan.
- Guild role evidence is strict, content-addressed, and anchored to a coordinator
  provider-evidence record. The world reviewer receives only world evidence, the
  curriculum reviewer receives only training evidence, and the safety reviewer
  receives only paired aggregate evaluation evidence.
- RocketRide calls the authenticated
  `/webhook/muscle-memory-fixed-step` route. Callback results are an append-only
  fixed-pipeline prefix, so an exact retry survives restart while changed or
  out-of-order requests fail closed.
- Plan admission requires one world identity across validation and execution, one
  episode identity across run, summary, graph query, and curriculum selection,
  and one candidate identity across training, evaluation, and final action.
- Promotion and rollback decisions are accepted only when their action, policy
  identities, checkpoint hashes, evaluation hashes, and exactly recomputed gate
  metrics match the coordinator-anchored Guild evaluation artifact. The same
  binding is rechecked when the alias action is applied.
- The qualified MM-01 checksum is checked both when opening an episode and when
  replaying the journal during startup.

## LaserData consumers

LaserData is the 20 Hz operational event bus, not a passive archive. A durable
append fans out to the live dashboard and timeline, remains available for replay,
feeds deterministic safety and failure summarization, and contributes the exact
telemetry digest used by post-episode graph and training/evaluation evidence.
Video remains a separate 30 FPS transport. `frame_id` metadata is its only join
key. No event consumer is reachable from the robot controller or task-policy
command path.

Health reports producer and consumer readiness separately. Provider deployment is
labeled `self-hosted`, `cloud`, or `unconfigured`; self-hosted development evidence
is never presented as sponsor-cloud verification. `end_to_end_verified` requires a
recorded provider-side operation and evidence identifier.

## Verification

```bash
uv run python -m ops.api.verify_backend
```

The verifier prints only redacted public health fields and closes all opened
resources. An unconfigured sponsor remains `unconfigured`; a local cache or
simulation does not change that state.

The RocketRide callback origin and bearer token are supplied through
`ROCKETRIDE_MM_COORDINATOR_URL` and `ROCKETRIDE_MM_COORDINATOR_TOKEN`. The URL is
an origin because the reviewed pipe resolves the fixed webhook name; the shipped
backend owns the resulting route. A live step succeeds only when its domain fact
already exists, its plan and approval evidence match, and the callback returns a
validated content-addressed result. Cached checkpoints are identified as existing
artifacts and are never described as a newly completed training run.
