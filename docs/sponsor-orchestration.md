# Sponsor Orchestration

`muscle_memory.orchestration` implements the boundary between Guild.ai reasoning and
RocketRide execution. It does not implement policy training or evaluation logic. It accepts
content-addressed commands for those tools and enforces who may decide, what may execute, and
when a human must intervene.

## Fixed contract

Guild has exactly three roles, in this order:

1. `World and Physics Agent`
2. `Failure and Curriculum Agent`
3. `Safety and Evaluation Agent`

Each role reviews the same immutable `ExecutionPlan`. RocketRide receives the plan only when
all three recommendations are `proceed`; `revise` and `block` both prevent execution. Agent
review text is never passed to the executor.

Every plan has exactly these eight commands:

1. `validate_world`
2. `run_episode`
3. `summarize_telemetry`
4. `query_graph_memory`
5. `select_curriculum`
6. `train_candidate_policy`
7. `evaluate_candidate_policy`
8. `promote_or_roll_back`

The plan digest binds the run ID, command order, and canonical payload checksums. Completed
results must contain the same eight-step prefix and every result is checksummed. Evaluation
accepts only baseline and candidate policy IDs plus a held-out world-set ID. It cannot accept
an expert path or any teacher input.

## Human gates

Approval requirements are derived from commands, not from provider prose:

| Command condition | Required decision |
| --- | --- |
| `uncertain_physical_properties` is true | Approve proposed physical properties |
| `curriculum_change_requested` is true | Approve the curriculum change |
| `reward_change_requested` is true | Approve the reward-function change |
| Final action is `promote` | Approve policy promotion |
| Final action is `roll_back` | Approve policy rollback |

`FixedPipelineExecutor` checks the append-only approval ledger immediately before the gated
command. A missing decision pauses without invoking the tool, and a rejected decision blocks
the run. The caller that records a decision must supply an authenticated human subject; Guild
and RocketRide adapters receive no approval-ledger write capability.

World validation is fail-closed. Unless the first tool returns `{"world_valid": true}`, no
downstream command runs, and that failed result cannot be resumed.

## Provider states

Every adapter exposes a provider, mode, health state, timestamp, and detail. These labels are
part of the returned record rather than UI-only wording.

Health progresses from `unconfigured` to `configured` to `healthy`.
`end_to_end_verified` is reserved for retained evidence from the complete real-provider path;
neither a successful health probe nor a fake-transport test may set it. Provider failures are
reported as `degraded` or `unhealthy`.

| Mode | Meaning |
| --- | --- |
| `unconfigured` | Required sponsor configuration is absent; calls fail closed. |
| `live` | A real provider adapter is selected. Health is unverified until a successful call. |
| `simulation` | Injected deterministic behavior; no sponsor request was sent. |
| `cached` | The live provider failed and an exact-plan prior result was returned. |

Cached fallback never generalizes. A cache hit requires the current plan digest to match the
stored review or completed run exactly, and the returned object carries a `FallbackRecord`
with the failure reason and source digest. Without an exact match, the provider error is
surfaced. Simulation is selected explicitly and is never presented as live or cached.

## Guild.ai adapter

Create one published Guild API trigger for each exact role. Construct `GuildApiConfig` with the
workspace owner, workspace slug, and three `GuildRoleEndpoint` values in role order. Each
endpoint credential is the trigger's `api_key_id:api_key_secret` pair. The adapter sends an
HTTP Basic authenticated `api_trigger` session request and polls its session for a structured
review containing:

```json
{
  "plan_digest": "<same digest>",
  "role": "<same exact role>",
  "recommendation": "proceed",
  "summary": "review rationale",
  "requested_approvals": []
}
```

The base URL must use HTTPS. Credentials belong in the deployment secret store and must not be
committed. A session response with a mismatched role or plan digest, an unknown approval kind,
or unstructured output marks the adapter unhealthy and fails the review.

Guild API reference: <https://docs.guild.ai/>

## RocketRide adapter

`RocketRideSdkTransport` uses the official asynchronous Python client contract:

- construct `RocketRideClient(uri=..., auth=..., request_timeout=..., persist=False)`;
- enter the async client context;
- call `use(filepath=...)` with a prevalidated `.pipe` file;
- send one canonical command envelope to the returned task token;
- terminate the task before leaving the client context.

`RocketRideSdkConfig` requires an HTTPS or WSS URI, a non-empty API key, an existing pipeline
file, and its expected SHA-256 digest. The adapter intentionally does not synthesize a provider
pipeline: deployment must supply and review the real `.pipe` artifact. Install the official
`rocketride` package in the live worker image; it remains optional for local contract tests.

RocketRide Python reference: <https://docs.rocketride.org/develop/python.md>

## Verification boundary

The repository tests use fake HTTP and SDK transports to verify request shape, authentication,
task lifetime, role separation, approval pauses, provider state, and fallback behavior. They do
not prove sponsor connectivity. A live claim requires captured Guild session IDs for all three
roles and a completed RocketRide task token using deployment credentials and the reviewed
pipeline artifact. Until that evidence exists, the adapters are implemented and locally
verified, but sponsor execution is not end-to-end verified.
