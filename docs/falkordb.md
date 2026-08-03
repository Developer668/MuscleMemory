# FalkorDB graph memory

FalkorDB stores explicit experience after an episode closes. It is not imported by the
simulation runtime, learned policy, walking controller, or evaluation runner, and a graph
query can never affect a command already being executed.

## Install and configure

The production adapter uses the official Python client:

```text
falkordb>=1.6,<2
```

Copy the variable names from `config/services/falkordb.env.example` into the deployment's
secret manager. `MUSCLE_MEMORY_FALKORDB_URL` accepts `redis://` or `rediss://`; the credential-
bearing URL is held as a Pydantic secret and is never included in health details or receipts.

```python
from muscle_memory.graph_memory import build_graph_memory

memory = build_graph_memory()
configuration = memory.configuration_health()
health = memory.health()
```

The status values have narrow meanings:

| State | Meaning |
| --- | --- |
| `unconfigured` | No provider URL is present. |
| `configured` | Credentials were loaded, but no provider query has proved connectivity. |
| `healthy` | A read-only query completed against FalkorDB. |
| `unavailable` | Configuration exists, but the provider could not complete the operation. |
| `end_to_end_verified` | Reserved for orchestration evidence that writes and reads the real graph. The adapter never infers this from a health probe. |

## Immutable graph

The adapter records content-addressed `World`, `Obstacle`, `PolicyVersion`, `Episode`,
`Failure`, `Correction`, and `Lesson` nodes. An evaluated policy version is immutable: trying
to reuse its identity for different checkpoint or evaluation evidence raises
`GraphMemoryIntegrityError`. Episodes require the permanent robot checksum and must reference
an existing validated world and evaluated policy with matching hashes.

Relationships are created only after their referenced facts exist:

```text
(World)-[:CONTAINS]->(Obstacle)
(Episode)-[:RAN_IN]->(World)
(Episode)-[:USED]->(PolicyVersion)
(Episode)-[:OBSERVED_FAILURE]->(Failure)
(Episode)-[:FAILED_NEAR]->(Obstacle)
(Correction)-[:CORRECTS]->(Failure)
(Correction)-[:PRODUCED]->(Lesson)
(Lesson)-[:TRAINED_INTO]->(PolicyVersion)
(PolicyVersion)-[:EVALUATED_AGAINST]->(PolicyVersion)
(PolicyVersion)-[:OUTPERFORMED]->(PolicyVersion)
```

An authenticated approval materializes a deterministic `Lesson` from the correction and
failure signature. RocketRide links only graph-present lesson IDs from admitted curriculum
evidence to the immutable candidate checkpoint. Every paired held-out run creates an
`EVALUATED_AGAINST` edge with its measured deltas and `promote` or `roll_back` action;
`OUTPERFORMED` is added only when the recomputed numeric gate proposes promotion.

Every value is passed through the official client's parameter dictionary. No record value is
interpolated into Cypher text.

## Curriculum traversal

`query_curriculum()` follows policy to episode to failure to approved correction to lesson,
with the associated obstacle and trained policy. It groups recurring lesson signatures and
ranks them by distinct source episodes. Both the FalkorDB query and local implementation
hard-filter source episodes to the `training` split. Held-out experience is stored for measured
evaluation history but is structurally unavailable to curriculum selection.

The result reports whether it came from `falkordb` or `local_cache`. Curriculum selection is a
proposal for the Guild.ai approval flow; reading a lesson does not authorize a curriculum
change.

## Outage behavior

Every fact is first checksummed into a durable append-only JSONL cache. If FalkorDB succeeds,
the receipt says the fact was stored remotely and mirrored locally. If FalkorDB is unavailable,
the receipt says `local_cache` and `unavailable`; it never claims sponsor delivery. After
recovery, `synchronize_local_cache()` replays all facts in dependency order. Remote writes are
idempotent, and any conflicting identity stops replay.

The cache supports a resilient demo, not provider proof. End-to-end evidence still requires a
real provider identifier plus a successful write and multi-hop read for the same episode.
