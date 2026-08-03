# Daytona production runtime

The public CPU coordinator runs in a Daytona container sandbox directly from the locked `uv`
environment. Docker and Compose are optional local-development packaging only; they are not
used in the Daytona runtime and do not count as cloud-delivery evidence.

## Required sandbox shape

The long-running sandbox must be public, have auto-stop disabled, and mount an independent
Daytona volume at `/data`. Daytona can stop a container after its inactivity interval even
while an internal server process is running, so the production sandbox uses an auto-stop
interval of `0`.

```bash
daytona volume create muscle-memory-backend-data --size 10
daytona create \
  --name muscle-memory-backend \
  --snapshot muscle-memory-runtime \
  --auto-stop 0 \
  --auto-archive 43200 \
  --auto-delete -1 \
  --public \
  --target us \
  --volume muscle-memory-backend-data:/data
```

The snapshot supplies the Daytona base runtime and `uv`. Application dependencies are still
installed from `uv.lock` inside the sandbox. `/data` holds the coordinator database,
LaserData outbox, FalkorDB recovery cache, asset cache, approval ledger, logs, and process
metadata. It survives replacement of the sandbox.

Daytona environment variables hold only deployment configuration. Supply provider credentials
through the Daytona environment at sandbox creation; never commit a real value or pass one as
a deploy-script argument. The accepted names are listed in `.env.example` and
`config/services/http-api.env.example`. A public deployment also needs a non-empty
`MM_API_AUTH_CREDENTIALS_JSON` containing only token digests, never plaintext tokens.
The four `MM_HELDOUT_*` values are all-or-none: the canonical paired artifact,
its canonical SHA-256, the candidate checkpoint, and a timezone-aware evaluation
timestamp. A partial set stops the runner before the API starts. `MM_STABLE_POLICY_ALIAS`
defaults to `stable`.

## Deploy one revision

The deploy command requires an explicit full commit SHA. It refuses a stopped sandbox, a
non-public preview, enabled auto-stop, or a missing `/data` volume. It clones or updates the
repository, runs a frozen Python production sync, installs the locked frontend dependencies,
builds `frontend/dist`, verifies the permanent robot bundle, restarts the API, discovers the
public sandbox proxy, strips its temporary discovery query, and exercises `/api/v1/health`
through the persistent unsigned HTTPS origin. If a post-start check fails, the supervisor
stops the failed replacement process instead of leaving an unverified revision running.

```bash
./ops/deployment/daytona_deploy.sh <commit-sha>
```

The default smoke proves the Daytona process and HTTP route only. It deliberately accepts
truthfully `unconfigured` providers and prints `Runtime-only smoke passed`; that is not sponsor
evidence.

After the final provider configuration is injected, enable the strict gate:

```bash
MM_DAYTONA_REQUIRE_SPONSORS=1 \
  ./ops/deployment/daytona_deploy.sh <commit-sha>
```

The strict path performs a live LaserData append/readback, a real RocketRide task/callback
probe using `ROCKETRIDE_VERIFY_ENVELOPE_FILE`, and the public health smoke for LaserData and
FalkorDB. Guild and RocketRide intentionally begin a cold process as `configured`; generic
health cannot honestly promote them merely because credentials exist. Guild proof is therefore
the durable three-role review on the exact workflow, and RocketRide proof is the task result
returned through the authenticated public callback. Both remain workflow-scoped evidence as
described in `docs/sponsor-orchestration.md`.

The live options endpoint labels a checkpoint `stable_deployed` only when its policy ID equals
the current durable stable alias. Any other admitted checkpoint is `candidate_live_test` and is
never selected as the production default; an operator must choose it explicitly.

The runner itself is foreground-safe and can also be invoked over SSH:

```bash
daytona exec muscle-memory-backend -- \
  sh -lc 'cd /home/daytona/MuscleMemory && ./ops/deployment/daytona_run.sh'
```

Daytona references:

- [CLI](https://www.daytona.io/docs/en/tools/cli/)
- [Preview URLs](https://www.daytona.io/docs/en/preview/)
- [Persistence and lifecycle](https://www.daytona.io/docs/en/persistence/)
- [Volumes](https://www.daytona.io/docs/en/volumes/)
