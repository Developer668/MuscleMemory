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

## Deploy one revision

The deploy command requires an explicit full commit SHA. It refuses a stopped sandbox, a
non-public preview, enabled auto-stop, or a missing `/data` volume. It clones or updates the
repository, runs a frozen production sync, verifies the permanent robot bundle, restarts the
API, obtains a one-hour signed preview URL, and exercises `/api/v1/health` through Daytona's
public proxy.

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

The strict path performs a live LaserData append/readback and requires LaserData, FalkorDB,
Guild.ai, and RocketRide to report `healthy` or `end_to_end_verified`. It fails closed when any
provider is merely configured, cached, simulated, degraded, or unconfigured. Guild publishing
and RocketRide task-token verification remain separate provider-specific evidence commands in
`docs/sponsor-orchestration.md`; the API health response does not replace them.

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
