#!/usr/bin/env node

import { spawnSync } from "node:child_process"
import { readFileSync, readdirSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const repositoryRoot = dirname(dirname(root))
const requestedProject = readProjectArgument(process.argv.slice(2))
const errors = []

const expectedProjects = [
  {
    directory: "world-and-physics",
    agentName: "muscle-memory-world-and-physics",
    role: "World and Physics Agent",
    credentialEnvironment: "MUSCLE_MEMORY_GUILD_WORLD_AND_PHYSICS_CREDENTIALS",
    evidenceField: "world_evidence",
    allowedApprovals: ["uncertain_physical_properties"],
  },
  {
    directory: "failure-and-curriculum",
    agentName: "muscle-memory-failure-and-curriculum",
    role: "Failure and Curriculum Agent",
    credentialEnvironment: "MUSCLE_MEMORY_GUILD_FAILURE_AND_CURRICULUM_CREDENTIALS",
    evidenceField: "failure_curriculum_evidence",
    allowedApprovals: ["curriculum_change"],
  },
  {
    directory: "safety-and-evaluation",
    agentName: "muscle-memory-safety-and-evaluation",
    role: "Safety and Evaluation Agent",
    credentialEnvironment: "MUSCLE_MEMORY_GUILD_SAFETY_AND_EVALUATION_CREDENTIALS",
    evidenceField: "evaluation_evidence",
    allowedApprovals: ["policy_promotion", "policy_rollback"],
  },
]

const exactPipeline = [
  "validate_world",
  "run_episode",
  "summarize_telemetry",
  "query_graph_memory",
  "select_curriculum",
  "train_candidate_policy",
  "evaluate_candidate_policy",
  "promote_or_roll_back",
]

const exactApprovals = [
  "uncertain_physical_properties",
  "reward_change",
  "curriculum_change",
  "policy_promotion",
  "policy_rollback",
]

const selectedProjects = requestedProject
  ? expectedProjects.filter((project) => project.directory === requestedProject)
  : expectedProjects

if (requestedProject && selectedProjects.length === 0) {
  errors.push(`unknown project '${requestedProject}'`)
}

validateCentralContracts()
for (const project of selectedProjects) {
  validateProject(project)
}

if (errors.length > 0) {
  console.error("Guild specialist validation failed:")
  for (const error of errors) {
    console.error(`- ${error}`)
  }
  process.exit(1)
}

console.log(
  `Validated ${selectedProjects.length} Guild specialist project(s): ${selectedProjects
    .map((project) => project.directory)
    .join(", ")}`,
)

function readProjectArgument(args) {
  const index = args.indexOf("--project")
  if (index === -1) {
    return undefined
  }
  if (index + 1 >= args.length) {
    return ""
  }
  return args[index + 1]
}

function parseJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"))
  } catch (error) {
    errors.push(`${path}: invalid JSON (${messageFor(error)})`)
    return undefined
  }
}

function validateCentralContracts() {
  const contract = parseJson(join(root, "contracts", "review-contract.json"))
  const manifest = parseJson(join(root, "environment.manifest.json"))
  if (!contract || !manifest) {
    return
  }

  expectEqual(contract.contract_version, 1, "review contract version")
  expectArrayEqual(contract.roles, expectedProjects.map((project) => project.role), "role roster")
  expectArrayEqual(contract.pipeline_steps, exactPipeline, "fixed pipeline")
  expectArrayEqual(contract.approval_kinds, exactApprovals, "approval kinds")
  expectArrayEqual(
    contract.output_fields,
    ["plan_digest", "role", "recommendation", "summary", "requested_approvals"],
    "review output fields",
  )

  expectEqual(manifest.contract_version, 1, "environment manifest version")
  expectEqual(manifest.provider, "guild.ai", "environment manifest provider")
  expectEqual(manifest.tested_cli_version, "0.17.0", "tested Guild CLI version")
  if (!Array.isArray(manifest.publish_environment)) {
    errors.push("publish_environment must be an array")
  } else {
    expectArrayEqual(
      manifest.publish_environment.map((entry) => entry.name),
      ["MUSCLE_MEMORY_GUILD_OWNER", "MUSCLE_MEMORY_GUILD_WORKSPACE"],
      "publish environment names",
    )
    for (const entry of manifest.publish_environment) {
      if (entry.secret !== false || "value" in entry || "default" in entry) {
        errors.push(`publish environment '${entry.name}' must contain a name, not a value`)
      }
    }
  }

  if (!Array.isArray(manifest.projects)) {
    errors.push("environment manifest projects must be an array")
    return
  }
  expectEqual(manifest.projects.length, 3, "environment manifest project count")
  for (const expected of expectedProjects) {
    const actual = manifest.projects.find((project) => project.directory === expected.directory)
    if (!actual) {
      errors.push(`environment manifest is missing ${expected.directory}`)
      continue
    }
    expectEqual(actual.agent_name, expected.agentName, `${expected.directory} agent name`)
    expectEqual(actual.role, expected.role, `${expected.directory} role`)
    expectEqual(
      actual.trigger_credentials_environment,
      expected.credentialEnvironment,
      `${expected.directory} trigger credential environment`,
    )
    if ("credentials" in actual || "secret" in actual || "value" in actual) {
      errors.push(`${expected.directory} manifest must not store trigger credentials`)
    }
  }

  const projectDirectories = readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => parseJsonIfPresent(join(root, name, "project.json")) !== undefined)
    .sort()
  expectArrayEqual(
    projectDirectories,
    expectedProjects.map((project) => project.directory).sort(),
    "publishable project directories",
  )

  validatePythonIntegrationContract()
}

function validatePythonIntegrationContract() {
  let contractsSource
  let adapterSource
  try {
    contractsSource = readFileSync(
      join(repositoryRoot, "src", "muscle_memory", "orchestration", "contracts.py"),
      "utf8",
    )
    adapterSource = readFileSync(
      join(repositoryRoot, "src", "muscle_memory", "orchestration", "guild.py"),
      "utf8",
    )
  } catch (error) {
    errors.push(`cannot read Python Guild integration contract (${messageFor(error)})`)
    return
  }

  for (const literal of [...expectedProjects.map((project) => project.role), ...exactPipeline]) {
    if (!contractsSource.includes(`"${literal}"`)) {
      errors.push(`Python orchestration contract is missing '${literal}'`)
    }
  }
  for (const approval of exactApprovals) {
    if (!contractsSource.includes(`"${approval}"`)) {
      errors.push(`Python orchestration approval contract is missing '${approval}'`)
    }
  }
  for (const inputField of [
    "contract_version",
    "role",
    "plan_digest",
    "run_id",
    "pipeline_steps",
    "requested_output",
  ]) {
    if (!adapterSource.includes(`"${inputField}"`)) {
      errors.push(`Python Guild adapter input is missing '${inputField}'`)
    }
  }
}

function validateProject(project) {
  const projectRoot = join(root, project.directory)
  const metadata = parseJson(join(projectRoot, "project.json"))
  const packageJson = parseJson(join(projectRoot, "package.json"))
  const tsconfig = parseJson(join(projectRoot, "tsconfig.json"))
  const input = parseJson(join(projectRoot, "fixtures", "valid-input.json"))
  const output = parseJson(join(projectRoot, "fixtures", "valid-output.json"))
  let source = ""
  try {
    source = readFileSync(join(projectRoot, "agent.ts"), "utf8")
  } catch (error) {
    errors.push(`${project.directory}: cannot read agent.ts (${messageFor(error)})`)
  }

  if (metadata) {
    expectEqual(metadata.contract_version, 1, `${project.directory} contract version`)
    expectEqual(metadata.agent_name, project.agentName, `${project.directory} project name`)
    expectEqual(metadata.role, project.role, `${project.directory} project role`)
    expectEqual(metadata.agent_type, "GUILD_TYPESCRIPT", `${project.directory} agent type`)
    expectEqual(metadata.template, "AUTO_MANAGED_STATE", `${project.directory} template`)
    expectEqual(metadata.entrypoint, "agent.ts", `${project.directory} entrypoint`)
    expectEqual(metadata.tools, "none", `${project.directory} tools declaration`)
    expectArrayEqual(
      metadata.runtime_provided_packages,
      ["@guildai/agents-sdk", "zod"],
      `${project.directory} runtime packages`,
    )
  }

  if (packageJson) {
    expectEqual(packageJson.private, true, `${project.directory} package privacy`)
    if (packageJson.dependencies || packageJson.devDependencies) {
      errors.push(`${project.directory}: Guild runtime packages must not be npm dependencies`)
    }
  }
  if (tsconfig) {
    expectEqual(tsconfig.compilerOptions?.strict, true, `${project.directory} TypeScript strictness`)
    expectEqual(tsconfig.compilerOptions?.noEmit, true, `${project.directory} noEmit setting`)
  }

  validateSource(project, source, projectRoot)
  if (input) {
    validateInput(project, input)
  }
  if (input && output) {
    validateOutput(project, input, output)
  }
}

function validateSource(project, source, projectRoot) {
  if (!source.startsWith('"use agent"')) {
    errors.push(`${project.directory}: agent.ts must start with the Guild compiler directive`)
  }
  const imports = [...source.matchAll(/from\s+"([^"]+)"/g)].map((match) => match[1])
  const allowedImports = new Set(["@guildai/agents-sdk", "zod"])
  for (const imported of imports) {
    if (!allowedImports.has(imported)) {
      errors.push(`${project.directory}: unsupported runtime import '${imported}'`)
    }
  }
  for (const forbidden of [
    "guildTools",
    "userInterfaceTools",
    "task.tools",
    "task.guild",
    "task.ui",
    "fetch(",
    "child_process",
    "process.env",
    "guildAgentTool",
    "guildServiceTool",
  ]) {
    if (source.includes(forbidden)) {
      errors.push(`${project.directory}: forbidden capability '${forbidden}' is present`)
    }
  }
  expectEqual(countOccurrences(source, "task.llm.generateText("), 1, `${project.directory} LLM calls`)
  if (!source.includes("tools: noTools")) {
    errors.push(`${project.directory}: agent must register noTools`)
  }
  if (!source.includes(`const ROLE = "${project.role}" as const`)) {
    errors.push(`${project.directory}: exact role constant is missing`)
  }
  for (const step of exactPipeline) {
    if (!source.includes(`z.literal("${step}")`)) {
      errors.push(`${project.directory}: pipeline step '${step}' is missing from its schema`)
    }
  }
  if (!source.includes(`${project.evidenceField}:`) || !source.includes(".strict()")) {
    errors.push(`${project.directory}: strict role evidence schema is missing`)
  }
  if (!source.includes("plan_digest: input.plan_digest") || !source.includes("role: ROLE")) {
    errors.push(`${project.directory}: output does not bind the input digest and exact role`)
  }
  if (project.directory === "failure-and-curriculum") {
    if (!source.includes('source_split: z.literal("training")')) {
      errors.push("failure-and-curriculum: evidence is not structurally training-only")
    }
    if (!source.includes("containsEvaluationOnlyReference(evidence)")) {
      errors.push("failure-and-curriculum: evaluation-only reference guard is missing")
    }
    if (source.includes("heldout_world_set_id") || source.includes("held_out_episode")) {
      errors.push("failure-and-curriculum: evaluation-only fact field is reachable")
    }
  }
  if (project.directory === "safety-and-evaluation") {
    for (const threshold of ["0.8", "0.1", "0.25", "0.2", "0.5", "0.15"]) {
      if (!source.includes(threshold)) {
        errors.push(`safety-and-evaluation: numeric threshold ${threshold} is missing`)
      }
    }
  }

  const syntax = spawnSync(process.execPath, ["--check", join(projectRoot, "agent.ts")], {
    encoding: "utf8",
  })
  if (syntax.status !== 0) {
    errors.push(
      `${project.directory}: TypeScript syntax check failed (${(syntax.stderr || syntax.stdout).trim()})`,
    )
  }
}

function validateInput(project, input) {
  const expectedKeys = [
    "contract_version",
    "pipeline_steps",
    "plan_digest",
    "requested_output",
    "role",
    "run_id",
    project.evidenceField,
  ].sort()
  expectArrayEqual(Object.keys(input).sort(), expectedKeys, `${project.directory} fixture input keys`)
  expectEqual(input.contract_version, 1, `${project.directory} fixture contract version`)
  expectEqual(input.role, project.role, `${project.directory} fixture role`)
  if (!/^[0-9a-f]{64}$/.test(input.plan_digest)) {
    errors.push(`${project.directory}: fixture plan digest is not lowercase SHA-256`)
  }
  expectArrayEqual(input.pipeline_steps, exactPipeline, `${project.directory} fixture pipeline`)
  expectArrayEqual(
    input.requested_output?.requested_approvals,
    exactApprovals,
    `${project.directory} fixture requested approvals`,
  )

  if (project.directory === "world-and-physics") {
    const evidence = input.world_evidence
    if (!evidence?.robot_checksum_unchanged) {
      errors.push("world-and-physics: fixture does not preserve the robot checksum")
    }
    if (!evidence || Object.values(evidence.validation).some((value) => value !== true)) {
      errors.push("world-and-physics: fixture must pass every required validation")
    }
    if (
      !Array.isArray(evidence?.obstacles) ||
      evidence.obstacles.length === 0 ||
      evidence.obstacles.some((obstacle) => obstacle.render_mesh_used_for_collision !== false)
    ) {
      errors.push("world-and-physics: fixture collider evidence is invalid")
    }
  }

  if (project.directory === "failure-and-curriculum") {
    const evidence = input.failure_curriculum_evidence
    expectEqual(evidence?.source_split, "training", "failure fixture source split")
    if (containsRestrictedCurriculumKey(evidence)) {
      errors.push("failure-and-curriculum: fixture contains evaluation-only fact keys")
    }
    for (const pattern of evidence?.failure_patterns ?? []) {
      const distinct = new Set(pattern.source_episode_ids ?? []).size
      expectEqual(
        pattern.distinct_source_episode_count,
        distinct,
        `failure pattern '${pattern.signature}' distinct episode count`,
      )
    }
    if (evidence?.curriculum_change_requested !== Boolean(evidence?.proposed_curriculum)) {
      errors.push("failure-and-curriculum: fixture proposal/change flag mismatch")
    }
  }

  if (project.directory === "safety-and-evaluation") {
    validateSafetyFixture(input.evaluation_evidence)
  }
}

function validateSafetyFixture(evidence) {
  if (!evidence) {
    errors.push("safety-and-evaluation: fixture evidence is missing")
    return
  }
  expectEqual(evidence.paired_world_count, 20, "safety fixture paired world count")
  const metrics = [
    evidence.baseline?.success_rate,
    evidence.baseline?.collision_rate,
    evidence.candidate?.success_rate,
    evidence.candidate?.collision_rate,
    evidence.candidate?.falls,
    evidence.candidate?.median_clearance_m,
    evidence.candidate?.path_efficiency_regression_fraction,
  ]
  if (metrics.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    errors.push("safety-and-evaluation: fixture metrics must be finite numbers")
    return
  }
  const successDelta = evidence.candidate.success_rate - evidence.baseline.success_rate
  const collisionReduction =
    evidence.baseline.collision_rate === 0
      ? 0
      : (evidence.baseline.collision_rate - evidence.candidate.collision_rate) /
        evidence.baseline.collision_rate
  const passes =
    evidence.candidate.success_rate >= 0.8 &&
    evidence.candidate.falls === 0 &&
    evidence.candidate.collision_rate <= 0.1 &&
    evidence.candidate.median_clearance_m >= 0.25 &&
    (successDelta >= 0.2 || collisionReduction >= 0.5) &&
    evidence.candidate.path_efficiency_regression_fraction <= 0.15
  if (!passes) {
    errors.push("safety-and-evaluation: valid fixture does not pass the numeric promotion gate")
  }
}

function validateOutput(project, input, output) {
  expectArrayEqual(
    Object.keys(output).sort(),
    ["plan_digest", "recommendation", "requested_approvals", "role", "summary"].sort(),
    `${project.directory} fixture output keys`,
  )
  expectEqual(output.plan_digest, input.plan_digest, `${project.directory} output plan digest`)
  expectEqual(output.role, project.role, `${project.directory} output role`)
  if (!["proceed", "revise", "block"].includes(output.recommendation)) {
    errors.push(`${project.directory}: fixture output recommendation is invalid`)
  }
  if (typeof output.summary !== "string" || output.summary.trim() === "") {
    errors.push(`${project.directory}: fixture output summary is empty`)
  }
  if (!Array.isArray(output.requested_approvals)) {
    errors.push(`${project.directory}: fixture requested approvals must be an array`)
    return
  }
  for (const approval of output.requested_approvals) {
    if (!project.allowedApprovals.includes(approval)) {
      errors.push(`${project.directory}: fixture requests out-of-scope approval '${approval}'`)
    }
  }
}

function containsRestrictedCurriculumKey(value) {
  if (!value || typeof value !== "object") {
    return false
  }
  if (Array.isArray(value)) {
    return value.some(containsRestrictedCurriculumKey)
  }
  return Object.entries(value).some(
    ([key, child]) =>
      key.includes("heldout") || key.includes("held_out") || containsRestrictedCurriculumKey(child),
  )
}

function parseJsonIfPresent(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"))
  } catch {
    return undefined
  }
}

function countOccurrences(value, needle) {
  return value.split(needle).length - 1
}

function expectEqual(actual, expected, label) {
  if (actual !== expected) {
    errors.push(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

function expectArrayEqual(actual, expected, label) {
  if (!Array.isArray(actual) || JSON.stringify(actual) !== JSON.stringify(expected)) {
    errors.push(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

function messageFor(error) {
  return error instanceof Error ? error.message : String(error)
}
