# Coordinator Store

`CoordinatorStore` is the coordinator's durable SQLite boundary. It complements, rather than
replaces, the sponsor systems: LaserData remains the live append-only telemetry log, FalkorDB
remains post-episode explicit memory, Guild.ai remains the reasoning coordinator, and
RocketRide remains the fixed-pipeline executor.

## Durable facts

- Episode metadata is content-addressed and carries the permanent robot checksum, world hash,
  and policy hash. Lifecycle changes are appended as separate events.
- Training and held-out evaluation metadata use separate APIs and separate scope storage. The
  training list/get API cannot return a held-out world-set identifier.
- Workflow plans, step events, provider evidence references, approval requirements, and human
  decisions are immutable audit facts.
- Evaluated checkpoints are immutable records. A policy action appends an alias event; it never
  rewrites checkpoint data.
- Each paired held-out artifact scopes its own forty episode results. Later candidates can be
  compared with the same immutable baseline without rebinding the baseline checkpoint or mixing
  results from separate runs.
- Promotion requires the numeric held-out gate to pass and a distinct approved human decision.
  Rollback likewise requires a numeric decision and a distinct approved human decision.

Every history table has `BEFORE UPDATE` and `BEFORE DELETE` triggers. SQLite uses WAL mode,
full synchronous commits, foreign keys, and an immediate transaction for each state transition.
After a process restart, current state is reconstructed from the latest append-only event.

## Use

```python
from pathlib import Path

from muscle_memory.coordinator import CoordinatorStore

with CoordinatorStore(Path(".state/coordinator.sqlite3")) as store:
    current_policy = store.current_policy("stable")
```

Provider credentials and video payloads do not belong in this database. Evidence rows store
only provider object identifiers and content hashes. Video remains on its dedicated transport,
joined to telemetry exclusively through `frame_id`.
