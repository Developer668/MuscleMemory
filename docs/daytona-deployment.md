# Daytona production runtime

The public CPU coordinator runs in a Daytona container sandbox directly from the locked `uv`
environment. Docker and Compose are optional local-development packaging only; they are not
used in the Daytona runtime and do not count as cloud-delivery evidence.

## Required sandbox shape

The long-running sandbox must be public, have auto-stop and auto-delete disabled, and mount an
independent Daytona volume at `/data`. Disable auto-archive when the workspace permits it;
otherwise use Daytona's maximum 30-day interval (`43200` minutes).

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
installed from `uv.lock` inside the sandbox. Live mutable state is rooted at
`/home/daytona/mm-data`: coordinator SQLite and WAL files, the LaserData outbox, FalkorDB
JSONL recovery journal, asset/approval state, logs, and PID metadata all require normal
filesystem locking and atomic replacement.

`/data` is a Mountpoint-S3 object volume, so it must never host a live database, WAL, JSONL
journal, PID, or log. It holds only uniquely named immutable recovery snapshots under
`/data/muscle-memory-snapshots`. The supervisor stops the API before export, creates local
consistent SQLite backups, writes each snapshot object once, and writes `manifest.json` last.
Recovery considers only snapshots with a complete manifest, verifies every size and SHA-256,
and restores only when the managed sandbox state is empty. A partial configuration that puts
any mutable runtime path under `/data` fails preflight before the API starts. These snapshots
survive sandbox replacement; the live `/home/daytona/mm-data` tree survives ordinary restarts
for as long as the sandbox itself is retained.

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
non-public preview, any enabled lifecycle timer, or a missing `/data` volume. It stops and
snapshots the prior process, fetches the requested revision, resets the cloud checkout to that
exact commit, removes all tracked, untracked, and ignored residue, and proves the tree is clean
before rebuilding. It then runs a frozen Python production sync, installs the locked frontend
dependencies, builds `frontend/dist`, verifies the permanent robot bundle, restarts the API,
discovers the public sandbox proxy, strips its temporary discovery query, and exercises
`/api/v1/health` through the persistent unsigned HTTPS origin. If a post-start check fails,
the supervisor stops and snapshots the failed replacement process instead of leaving an
unverified revision running.

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

## GitHub production delivery

`.github/workflows/deploy-production.yml` verifies the frozen robot bundle, Python checks,
tests, and the locked frontend build before deploying the exact pushed `main` commit. The
deployment job uses the protected GitHub `production` environment and requires only a scoped
`DAYTONA_API_KEY`; sponsor and operator credentials remain in the Daytona runtime environment.

The public application is served through `https://musclememory.space`. A Cloudflare edge proxy
forwards the same-origin HTTP, media, and WebSocket traffic to the stable public Daytona preview
origin while suppressing the Daytona preview warning. `www.musclememory.space` redirects to the
apex. The deploy script binds the API to `0.0.0.0` inside the sandbox and, when
`MM_PRODUCTION_HEALTH_URL` is set, refuses success until the custom-domain health route answers.

The live options endpoint labels a checkpoint `stable_deployed` only when its policy ID equals
the current durable stable alias. Any other admitted checkpoint is `candidate_live_test` and is
never selected as the production default; an operator must choose it explicitly.

The supervisor is the supported SSH entry point because it owns stop, snapshot, recovery,
PID identity, and startup ordering:

```bash
daytona exec muscle-memory-backend -- \
  sh -lc 'cd /home/daytona/MuscleMemory && uv run --frozen --no-sync python -m ops.deployment.daytona_process'
```

Daytona references:

- [CLI](https://www.daytona.io/docs/en/tools/cli/)
- [Preview URLs](https://www.daytona.io/docs/en/preview/)
- [Persistence and lifecycle](https://www.daytona.io/docs/en/persistence/)
- [Volumes](https://www.daytona.io/docs/en/volumes/)
