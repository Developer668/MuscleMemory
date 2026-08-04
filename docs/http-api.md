# Muscle Memory HTTP API

The HTTP surface is an injected FastAPI transport around the existing episode,
telemetry, graph-memory, orchestration, coordinator, policy-evaluation, and asset
domains. Route code never writes provider storage directly and does not receive a
policy/control object.

## Runtime dependencies

The implementation was verified with these pins. They must be added centrally to
`pyproject.toml` and `uv.lock` before the API is packaged:

```text
fastapi==0.141.1
uvicorn[standard]==0.52.1
httpx==0.28.1  # dev/test only
```

`src/muscle_memory/api/contracts.py` defines `ApiBackend`. The production composition
root implements that protocol by delegating to domain services, then exposes a zero-argument
factory through `MM_API_BACKEND_FACTORY=module:function`. The service intentionally refuses
to start without this factory:

```bash
uv run python -m ops.api.serve
```

During application construction, `bind_live_publisher` supplies the backend with the
API-owned bounded hub. The episode ingestion boundary publishes accepted telemetry only
after its durable append succeeds and publishes lifecycle status through the same hook.

Validate the generated contract without opening providers:

```bash
uv run python -m ops.api.validate_openapi
```

## Authentication

Every POST route fails closed. The API accepts bearer tokens but stores only SHA-256
digests from `MM_API_AUTH_CREDENTIALS_JSON`; authenticated subjects and scopes are derived
server-side and cannot be supplied in request JSON. The write scopes are:

- `approvals:write`
- `workflows:write`
- `corrections:write`
- `episodes:write` (live episodes and episode review notes)
- `training:write` (bounded local task-policy training)

Use [the service environment example](../config/services/http-api.env.example) as the
shape reference. Provider URLs, API keys, and bearer values never appear in response
models or OpenAPI schemas. Provider error detail is redacted before projection.

## Versioned routes

All HTTP routes use `/api/v1`:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service and sponsor-provider state |
| `GET` | `/episodes` | Operational episode list |
| `GET` | `/episodes/{episode_id}` | Episode detail |
| `GET` | `/episodes/{episode_id}/notes` | Active operator review notes |
| `POST` | `/episodes/{episode_id}/notes` | Persist an authenticated review note |
| `PATCH` | `/episodes/{episode_id}/notes/{note_id}` | Update or archive a review note |
| `GET` | `/episodes/{episode_id}/telemetry` | Paginated append-only telemetry |
| `GET` | `/episodes/{episode_id}/replay` | Ordered replay records |
| `WS` | `/episodes/{episode_id}/live` | Bounded live telemetry and status |
| `GET` | `/approvals/pending` | Blocking decisions |
| `POST` | `/approvals/{requirement_id}/decision` | Immutable human verdict |
| `POST` | `/workflows/review` | Exact three-role Guild review |
| `GET` | `/workflows/{run_id}` | RocketRide run state |
| `POST` | `/workflows/{run_id}/execute` | Execute the reviewed fixed pipeline |
| `POST` | `/workflows/{run_id}/resume` | Resume after a blocking decision |
| `POST` | `/episodes/{episode_id}/corrections` | Route or keep-out correction |
| `POST` | `/corrections/{correction_id}/decision` | Correction verdict |
| `GET` | `/policies` | Evaluation summaries |
| `GET` | `/policies/promotion-eligibility` | Numeric gate evidence only |
| `POST` | `/training/jobs` | Start one bounded task-policy training job |
| `GET` | `/training/jobs` and `/training/jobs/{job_id}` | Read durable training history |
| `GET` | `/assets` and `/assets/{asset_id}` | Asset generation and admission state |

Review notes are workspace annotations. They reference a training episode, carry the authenticated
subject, tags, and timestamps, and do not rewrite episode transitions, telemetry, policy hashes, or
robot checksums. `PATCH .../notes/{note_id}` is scoped to the episode in the URL; archiving removes
the note from the default list while retaining it for an explicit `include_archived=true` read.

The frontend's Episode Review export is a client-side JSON artifact assembled from the typed episode
detail and the currently loaded replay page. It is labeled as an export, not as a new provider
receipt.

## Durable training history

The bounded local training manager writes `job.json` atomically inside each
`task-policy-<id>/` output directory. Job manifests contain the dataset digest, policy identity,
timestamps, terminal metrics, and artifact digests. On process restart, queued or running jobs are
recovered as `failed` with `error_type=process_restart`; incomplete work is never promoted or
reported as complete. The `/training/jobs` read surface therefore remains useful across a service
restart while the actual checkpoint and evidence files stay immutable.

`GET /episodes` has no scope selector and its response enum contains only training,
development, and demo episodes. Held-out world identifiers and episode rows remain behind
the evaluation-only domain boundary. Policy endpoints may expose aggregate held-out metrics,
but never individual held-out world identities or telemetry.

## Provider truthfulness

Every provider response uses one of these states:

```text
unconfigured | configured | healthy | degraded | end_to_end_verified | simulation | cached
```

Simulation and exact-plan cache results are never reported as live health. An available
local spool does not imply that LaserData accepted an event, and a local graph cache does
not imply FalkorDB persistence.

## Live stream

Each WebSocket subscriber gets a 40-item queue and output is capped at 20 Hz. When a
consumer falls behind, the oldest stale item is discarded and `dropped_before` reports the
loss on a newer event; episode ingestion is never blocked by a dashboard client.

`frame_id` is the only video join key in HTTP telemetry, replay records, and WebSocket
messages. Status messages carry no frame join value. Video bytes can use a separate media
transport, but it must join API metadata only through that field.

## Errors and shutdown

HTTP failures use one envelope containing `code`, `message`, `request_id`, and optional
safe details. Validation errors omit the submitted value, preventing accidental credential
reflection. Lifespan startup opens the injected backend; shutdown closes live subscribers
before closing provider and persistence services.
