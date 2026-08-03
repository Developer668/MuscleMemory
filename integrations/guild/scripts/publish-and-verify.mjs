#!/usr/bin/env node

import { spawnSync } from "node:child_process"
import {
  accessSync,
  constants,
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import { tmpdir } from "node:os"
import { basename, dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const options = parseArguments(process.argv.slice(2))

if (options.help) {
  printHelp()
  process.exit(0)
}

const manifest = JSON.parse(readFileSync(join(root, "environment.manifest.json"), "utf8"))
const guildBin = findGuildBinary()
const owner = process.env.MUSCLE_MEMORY_GUILD_OWNER?.trim()
const workspace = process.env.MUSCLE_MEMORY_GUILD_WORKSPACE?.trim()
const blockers = []

runStaticValidation()

if (!guildBin) {
  blockers.push(
    "Guild CLI is not installed or discoverable; install @guildai/cli and optionally set GUILD_BIN",
  )
}
if (!owner) {
  blockers.push("MUSCLE_MEMORY_GUILD_OWNER is not configured")
}
if (!workspace) {
  blockers.push("MUSCLE_MEMORY_GUILD_WORKSPACE is not configured")
}

let cliVersion
if (guildBin) {
  const versionResult = runProcess(guildBin, ["--version"], root)
  if (versionResult.status === 0) {
    cliVersion = versionResult.stdout.trim()
  } else {
    blockers.push(`Guild CLI version probe failed: ${safeOutput(versionResult)}`)
  }

  const authResult = runProcess(
    guildBin,
    ["--mode", "json", "--non-interactive", "auth", "status"],
    root,
  )
  const authPayload = parseOptionalJsonOutput(authResult)
  if (authResult.status !== 0 || authPayload?.success !== true) {
    blockers.push(`Guild CLI is not authenticated: ${safeOutput(authResult)}`)
  }
}

if (blockers.length > 0) {
  console.error("Guild publish/verify is blocked:")
  for (const blocker of blockers) {
    console.error(`- ${blocker}`)
  }
  console.error("No provider evidence was written.")
  process.exit(2)
}

const stagingRoot = mkdtempSync(join(tmpdir(), "muscle-memory-guild-"))
const records = []

try {
  for (const project of manifest.projects) {
    records.push(
      publishAndVerifyProject({
        guildBin,
        owner,
        workspace,
        project,
        stagingRoot,
      }),
    )
  }

  const evidencePath = resolve(
    options.evidenceOut ?? join(root, "evidence", "live-verification.json"),
  )
  const evidence = {
    schema_version: 1,
    provider: "guild.ai",
    provider_mode: "live",
    provider_health: "healthy",
    end_to_end_verified: false,
    checked_at: new Date().toISOString(),
    cli_version: cliVersion,
    owner,
    workspace,
    api_trigger_verified: false,
    api_trigger_note:
      "Version-pinned CLI sessions succeeded; API triggers and their credentials require separate workspace configuration and proof.",
    agents: records,
  }
  writeJsonAtomically(evidencePath, evidence)
  console.log(`Guild publish/verify completed for ${records.length} agents.`)
  console.log(`Evidence: ${evidencePath}`)
} finally {
  if (options.keepStaging) {
    console.log(`Staging retained at ${stagingRoot}`)
  } else {
    rmSync(stagingRoot, { recursive: true, force: true })
  }
}

function publishAndVerifyProject({ guildBin, owner, workspace, project, stagingRoot }) {
  const sourceRoot = join(root, project.directory)
  const stageRoot = join(stagingRoot, project.directory)
  const initResult = runProcess(
    guildBin,
    [
      "--mode",
      "json",
      "--non-interactive",
      "agent",
      "init",
      "--name",
      project.agent_name,
      "--template",
      "AUTO_MANAGED_STATE",
      "--agent-type",
      "GUILD_TYPESCRIPT",
      "--owner",
      owner,
      "--directory",
      stageRoot,
    ],
    stagingRoot,
  )
  requireSuccess(initResult, `${project.role} initialization`)
  if (!existsSync(join(stageRoot, "guild.json"))) {
    throw new Error(`${project.role} initialization did not create CLI-managed guild.json`)
  }

  for (const filename of ["agent.ts", "package.json", "tsconfig.json", "README.md", ".gitignore"]) {
    copyFileSync(join(sourceRoot, filename), join(stageRoot, filename))
  }
  cpSync(join(sourceRoot, "fixtures"), join(stageRoot, "fixtures"), { recursive: true })

  requireSuccess(runProcess("git", ["add", "."], stageRoot), `${project.role} staging`)
  const saveResult = runProcess(
    guildBin,
    [
      "--mode",
      "json",
      "--non-interactive",
      "agent",
      "save",
      "--all",
      "--message",
      "Publish Muscle Memory specialist contract",
      "--wait",
      "--publish",
      "--no-bump",
    ],
    stageRoot,
  )
  requireSuccess(saveResult, `${project.role} publish`)
  const saved = parseJsonOutput(saveResult, `${project.role} publish`)

  const getResult = runProcess(
    guildBin,
    ["--mode", "json", "--non-interactive", "agent", "get"],
    stageRoot,
  )
  requireSuccess(getResult, `${project.role} agent lookup`)
  const agentInfo = parseJsonOutput(getResult, `${project.role} agent lookup`)

  const versionsResult = runProcess(
    guildBin,
    ["--mode", "json", "--non-interactive", "agent", "versions", "--limit", "1"],
    stageRoot,
  )
  requireSuccess(versionsResult, `${project.role} version lookup`)
  const versions = parseJsonOutput(versionsResult, `${project.role} version lookup`)

  const agentId =
    findNamedString(agentInfo, ["agent_id", "id"]) ?? findNamedString(saved, ["agent_id"])
  const versionId =
    findNamedString(saved, ["version_id"]) ??
    findNamedString(versions, ["version_id", "id"])
  if (!agentId || !versionId) {
    throw new Error(`${project.role} provider output did not contain agent and version IDs`)
  }

  const fixture = JSON.parse(
    readFileSync(join(sourceRoot, "fixtures", "valid-input.json"), "utf8"),
  )
  const chatResult = runProcess(
    guildBin,
    [
      "--mode",
      "json",
      "--non-interactive",
      "agent",
      "chat",
      "--path",
      stageRoot,
      "--workspace",
      `${owner}~${workspace}`,
      "--agent-version",
      versionId,
      "--no-splash",
    ],
    stageRoot,
    `${JSON.stringify(fixture)}\n`,
  )
  requireSuccess(chatResult, `${project.role} version-pinned verification session`)
  const session = parseJsonOutput(chatResult, `${project.role} verification session`)
  const sessionId = findNamedString(session, ["session_id", "id"])
  const review = findReview(session)
  if (!sessionId || !review) {
    throw new Error(`${project.role} verification output lacked a session ID or structured review`)
  }
  validateLiveReview(project, fixture, review)

  console.log(`${project.role}: agent=${agentId} version=${versionId} session=${sessionId}`)
  return {
    role: project.role,
    agent_name: project.agent_name,
    agent_id: agentId,
    version_id: versionId,
    verification_session_id: sessionId,
    plan_digest: fixture.plan_digest,
    recommendation: review.recommendation,
    api_trigger_verified: false,
  }
}

function validateLiveReview(project, fixture, review) {
  if (review.plan_digest !== fixture.plan_digest) {
    throw new Error(`${project.role} returned a mismatched plan digest`)
  }
  if (review.role !== project.role) {
    throw new Error(`${project.role} returned a mismatched role`)
  }
  if (!["proceed", "revise", "block"].includes(review.recommendation)) {
    throw new Error(`${project.role} returned an invalid recommendation`)
  }
  if (typeof review.summary !== "string" || review.summary.trim() === "") {
    throw new Error(`${project.role} returned an empty summary`)
  }
  if (!Array.isArray(review.requested_approvals)) {
    throw new Error(`${project.role} returned invalid requested approvals`)
  }
  const permitted = new Set(
    project.role === "World and Physics Agent"
      ? ["uncertain_physical_properties"]
      : project.role === "Failure and Curriculum Agent"
        ? ["curriculum_change"]
        : ["policy_promotion", "policy_rollback"],
  )
  for (const approval of review.requested_approvals) {
    if (!permitted.has(approval)) {
      throw new Error(`${project.role} returned out-of-scope approval '${approval}'`)
    }
  }
}

function findReview(value) {
  if (typeof value === "string") {
    try {
      return findReview(JSON.parse(value))
    } catch {
      return undefined
    }
  }
  if (Array.isArray(value)) {
    for (let index = value.length - 1; index >= 0; index -= 1) {
      const found = findReview(value[index])
      if (found) {
        return found
      }
    }
    return undefined
  }
  if (!value || typeof value !== "object") {
    return undefined
  }
  if (
    "plan_digest" in value &&
    "role" in value &&
    "recommendation" in value &&
    "summary" in value
  ) {
    return value
  }
  for (const child of Object.values(value)) {
    const found = findReview(child)
    if (found) {
      return found
    }
  }
  return undefined
}

function findNamedString(value, names) {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findNamedString(item, names)
      if (found) {
        return found
      }
    }
    return undefined
  }
  if (!value || typeof value !== "object") {
    return undefined
  }
  for (const name of names) {
    if (typeof value[name] === "string" && value[name]) {
      return value[name]
    }
  }
  for (const child of Object.values(value)) {
    const found = findNamedString(child, names)
    if (found) {
      return found
    }
  }
  return undefined
}

function parseJsonOutput(result, label) {
  const output = result.stdout.trim()
  try {
    return JSON.parse(output)
  } catch {
    const lines = output.split("\n").filter(Boolean)
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      try {
        return JSON.parse(lines[index])
      } catch {
        continue
      }
    }
  }
  throw new Error(`${label} did not return machine-readable JSON`)
}

function parseOptionalJsonOutput(result) {
  try {
    return parseJsonOutput(result, "Guild CLI probe")
  } catch {
    return undefined
  }
}

function runStaticValidation() {
  const result = runProcess(process.execPath, [join(root, "scripts", "validate.mjs")], root)
  requireSuccess(result, "static Guild contract validation")
  process.stdout.write(result.stdout)
}

function findGuildBinary() {
  const candidates = []
  if (process.env.GUILD_BIN) {
    candidates.push(resolve(process.env.GUILD_BIN))
  }
  const which = spawnSync("/usr/bin/which", ["guild"], { encoding: "utf8" })
  if (which.status === 0 && which.stdout.trim()) {
    candidates.push(which.stdout.trim())
  }
  const npmPrefix = spawnSync("npm", ["prefix", "-g"], { encoding: "utf8" })
  if (npmPrefix.status === 0 && npmPrefix.stdout.trim()) {
    candidates.push(join(npmPrefix.stdout.trim(), "bin", "guild"))
  }
  for (const candidate of candidates) {
    try {
      accessSync(candidate, constants.X_OK)
      return candidate
    } catch {
      continue
    }
  }
  return undefined
}

function runProcess(command, args, cwd, input) {
  return spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    input,
    env: process.env,
    maxBuffer: 16 * 1024 * 1024,
  })
}

function requireSuccess(result, label) {
  if (result.error) {
    throw new Error(`${label} could not start: ${result.error.message}`)
  }
  if (result.status !== 0) {
    throw new Error(`${label} failed: ${safeOutput(result)}`)
  }
}

function safeOutput(result) {
  const text = `${result.stderr ?? ""}\n${result.stdout ?? ""}`.trim()
  return text || `exit status ${result.status}`
}

function writeJsonAtomically(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.tmp`)
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  })
  renameSync(temporary, path)
}

function parseArguments(args) {
  const parsed = { evidenceOut: undefined, help: false, keepStaging: false }
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === "--help" || argument === "-h") {
      parsed.help = true
    } else if (argument === "--keep-staging") {
      parsed.keepStaging = true
    } else if (argument === "--evidence-out") {
      index += 1
      if (index >= args.length) {
        throw new Error("--evidence-out requires a path")
      }
      parsed.evidenceOut = args[index]
    } else {
      throw new Error(`unknown argument '${argument}'`)
    }
  }
  return parsed
}

function printHelp() {
  console.log(`Usage: node integrations/guild/scripts/publish-and-verify.mjs [options]

Publishes all three Guild specialists from isolated temporary projects and runs each
published version with its role fixture. Evidence is written only after every project
returns real agent, version, and session IDs.

Required environment:
  MUSCLE_MEMORY_GUILD_OWNER
  MUSCLE_MEMORY_GUILD_WORKSPACE

Options:
  --evidence-out <path>  Override the ignored evidence output path
  --keep-staging         Retain temporary Guild projects for diagnosis
  -h, --help             Show this help`)
}
