# Guild specialist agents

This directory contains exactly three independently publishable Guild agent source
projects for the fixed Muscle Memory orchestration roster:

1. `world-and-physics`
2. `failure-and-curriculum`
3. `safety-and-evaluation`

Each project is an auto-managed, one-shot TypeScript agent with strict Zod input and
output schemas. The agents receive the same immutable execution-plan digest and fixed
pipeline order. They have `noTools`: they cannot run RocketRide commands, call sponsor
services, write an approval, or invoke another agent. Their only nondeterministic action
is one schema-constrained Guild LLM call after deterministic safety checks pass.

## Contract boundary

The common input fields match `GuildApiCoordinator`:

- `contract_version`
- exact `role`
- immutable `plan_digest`
- `run_id`
- exact eight-entry `pipeline_steps`
- `requested_output`

Each agent also accepts one strict role-specific evidence object. Missing evidence is a
valid request but produces `revise`; a digest and step names alone are not enough to
support a substantive review.

- World and Physics reviews validated-world and physical-property evidence. Uncertain
  or agent-proposed properties request `uncertain_physical_properties` and still require
  the independent human gate.
- Failure and Curriculum accepts training-split evidence only. Its schema has no field
  for held-out episodes or evaluation world-set identifiers. A proposed curriculum
  change requests `curriculum_change` but cannot apply it.
- Safety and Evaluation recomputes the numeric promotion gate from paired evaluation
  metrics. A passing review requests `policy_promotion` or `policy_rollback`; it never
  changes a policy alias itself.

The canonical names, environment variable names, and non-secret project metadata are in
`environment.manifest.json`. Trigger credentials are never stored here.

## Local validation

```bash
node integrations/guild/scripts/validate.mjs
```

The validator checks the exact roster and pipeline, project isolation, import and tool
boundaries, strict schemas, environment metadata, and role fixtures. It is static
contract verification. Only Guild can compile the runtime-provided
`@guildai/agents-sdk` and `zod` packages.

## Publish and verify

The publisher follows Guild's current CLI workflow while keeping its CLI-managed
`guild.json` and Git history out of this repository. It creates an isolated temporary
project for each agent, initializes it with Guild, publishes a validated version, and
runs that exact version with the valid fixture.

```bash
export MUSCLE_MEMORY_GUILD_OWNER='<owner-name>'
export MUSCLE_MEMORY_GUILD_WORKSPACE='<workspace-name>'
node integrations/guild/scripts/publish-and-verify.mjs
```

On complete success it writes `integrations/guild/evidence/live-verification.json` with
the real agent, version, and session IDs returned by Guild. The directory is ignored so
credentials or operational artifacts cannot be committed accidentally. The script
writes no evidence on partial or failed runs.

Publishing and a CLI session do not create API triggers. Guild currently requires API
trigger keys to be created in the workspace UI. Create one API trigger for each exact
role, store each combined `api_key_id:api_key_secret` value under the environment name
in the manifest, and retain subsequent API-trigger session IDs as end-to-end evidence.

References:

- <https://docs.guild.ai/guide/coded-agents>
- <https://docs.guild.ai/guide/llms>
- <https://docs.guild.ai/cli/getting-started>
- <https://docs.guild.ai/platform/triggers>

