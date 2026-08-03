# RocketRide fixed-step execution

This bundle is the reviewed RocketRide execution artifact for Muscle Memory. It runs one
already-decided domain command per task input. It has no LLM, agent, Python tool, MCP tool,
dynamic URL, or control connection, so it cannot decide which command to run or reorder the
fixed application pipeline.

```text
SDK canonical envelope -> webhook -> configured authenticated callback -> typed result
```

The complete order remains:

1. `validate_world`
2. `run_episode`
3. `summarize_telemetry`
4. `query_graph_memory`
5. `select_curriculum`
6. `train_candidate_policy`
7. `evaluate_candidate_policy`
8. `promote_or_roll_back`

`FixedStepDispatcher` requires exactly those eight handler bindings in that order. Its
`SequenceLedger` rejects skipped and reordered calls, and returns the original result only for
an exact retry. The process-local ledger is suitable for tests and a single uninterrupted demo;
a production callback must replace it with durable coordinator state.

## Why the callback exists

RocketRide's current generic HTTP and MCP client nodes are agent-invoked tools with no data
lanes. Connecting them would require an agent to choose and invoke the request. The current
official `tool_n8n` node also has a deterministic data-lane face: it posts lane input to one
configured host and path, supports bearer authentication, waits synchronously, and emits the
response downstream. `fixed-step.pipe` uses only that face. It does not require an n8n workflow;
the Muscle Memory callback implements the two routes that face calls:

- `GET /healthz`
- `POST /webhook/muscle-memory-fixed-step`

The provider name is therefore retained honestly in the `.pipe`; this is a narrow reuse of its
documented configured-webhook lane, not a claim that n8n is part of the product. The inner
pipeline matches RocketRide's official version-1 `.pipe` fields and includes the explicit source.
The file uses the SDK-supported outer `pipeline` wrapper. `use(filepath=...)` unwraps that form,
while RocketRide Cloud 3.3.0 requires it for the current `rrext_validate` path because its engine
validator expects a wrapped document. The static validator pins the wrapper, every node, lane,
security setting, UI field, and checksum.

Official sources reviewed on 2026-08-03:

- <https://docs.rocketride.org/pipeline-reference/>
- <https://docs.rocketride.org/develop/python/>
- <https://docs.rocketride.org/nodes/tool_http_request/>
- <https://github.com/rocketride-org/rocketride-server/tree/develop/nodes/src/nodes/tool_n8n>

The pinned upstream source commit and reviewed paths are in `source-review.json`.

## Callback contract

The SDK sends a canonical JSON object containing `contract_version`, `run_id`, `plan_digest`,
`step`, and `payload`. Gated commands also carry exactly one `approval_evidence` record. The
callback does not trust that record by itself: an injected verifier must find the decision in
the coordinator's append-only approval ledger and confirm its plan, step, human subject, and
approved verdict.

The canonical JSON is transported as `text/plain` so RocketRide routes it through the reviewed
`text` lane. Labeling it `application/json` routes it away from that lane and produces an empty
callback payload even though the bytes themselves are valid JSON.

The following always require verified human evidence:

- proposed uncertain physical properties
- reward changes
- curriculum changes
- policy promotion
- policy rollback

Evaluation accepts only baseline and candidate policy identifiers plus the held-out world-set
identifier. Extra runtime inputs are rejected before a handler is called.

Every callback response is a canonical result object containing the request hash, output hash,
run, plan, and step. RocketRide's `response_text` node uses its required `text` lane and the SDK
returns that lane in its response mapping. `live_verify.py` extracts and validates the wrapper.
Other consumers must do the same; the existing core SDK adapter was intentionally not changed as
part of this isolated artifact.

## Backend wiring

The callback library deliberately has no fake production handlers. The application backend
must inject all eight real functions and a ledger-backed approval verifier:

```python
from integrations.rocketride.callback import make_callback_server
from integrations.rocketride.protocol import FixedStepDispatcher, SequenceLedger

dispatcher = FixedStepDispatcher(
    handlers={
        "validate_world": validate_world,
        "run_episode": run_episode,
        "summarize_telemetry": summarize_telemetry,
        "query_graph_memory": query_graph_memory,
        "select_curriculum": select_curriculum,
        "train_candidate_policy": train_candidate_policy,
        "evaluate_candidate_policy": evaluate_candidate_policy,
        "promote_or_roll_back": promote_or_roll_back,
    },
    approval_verifier=approval_ledger.contains_matching_decision,
    sequence=SequenceLedger(),
)
server = make_callback_server(dispatcher, bearer_token=deployment_secret)
server.serve_forever()
```

Backend composition can load the checked artifact without duplicating paths or hashes:

```python
from integrations.rocketride import ReviewedPipelineArtifact

artifact = ReviewedPipelineArtifact.from_env()
# artifact.pipeline_path, artifact.pipeline_sha256, artifact.sdk_environment
```

`public_evidence` deliberately omits the callback token. Provider URI and API-key ownership
remain with the backend provider registry; this object only supplies reviewed pipeline data.

In production, terminate TLS in front of the callback, keep the bearer token in the deployment
secret store, bind the callback to a private network, and use durable sequence/idempotency state.
The health route exposes no credentials or execution data.

## Static verification

```bash
uv run python -m ops.sponsors.validate_rocketride
uv run pytest tests/test_rocketride_pipeline.py
```

Static validation proves the checked-in shape, safety constraints, and checksums. Fake-server
tests prove local HTTP/auth/protocol behavior. The inner shape is pinned to the official
`examples/n8n-roundtrip.pipe`, and the wrapper compatibility is pinned to the reviewed server and
SDK sources. Only a successful provider `validate` call proves Cloud accepts it.

## Live verification

The official SDK package is optional for normal local tests. The live verifier was checked
against `rocketride==1.3.0`; pin that version in the deployment image:

```bash
uv pip install rocketride==1.3.0
```

Set these only in the deployment secret/runtime environment. Runtime configuration reprs and
verification evidence never render either credential in plaintext:

```text
ROCKETRIDE_URI=https://cloud.rocketride.ai
ROCKETRIDE_APIKEY=<provider key>
ROCKETRIDE_MM_COORDINATOR_URL=https://<private callback host>
ROCKETRIDE_MM_COORDINATOR_TOKEN=<at least 32 random characters>
ROCKETRIDE_VERIFY_ENVELOPE_FILE=/absolute/path/to/a-reviewed-canonical-envelope.json
```

Then run:

```bash
uv run python -m ops.sponsors.verify_rocketride
```

The verifier first validates the local checksums, calls the provider's `validate`, starts the
reviewed `.pipe` with `use`, sends the supplied real command, validates the typed callback
result, and terminates the task. Successful evidence includes the task-token checksum, pipeline hash,
request hash, result hash, output hash, and provider state. Missing SDK/configuration exits
nonzero and is labeled explicitly; it never emits fabricated provider proof.
