# Architecture

This document records the boundaries that make the product's claims testable. It does not
weaken the invariants in `AGENTS.md` or the acceptance gates in `README.md`.

## Deployment shape

The application is split by responsibility rather than by demo screen:

- A browser operator interface renders the live simulation, robot views, telemetry, replay,
  approvals, and measured evaluation results.
- A Python coordinator owns authenticated jobs, immutable artifact references, health state,
  approval records, and browser streaming.
- A native MuJoCo worker owns one live physics world and the frozen gait-controller process.
- A training image owns expert-data generation and behavior cloning. A* exists only here.
- A separately built evaluation image cannot import the training teacher and receives only
  read-only held-out-world credentials.
- LaserData is the append-only live episode log. Video is carried separately and joined to
  event metadata only by `frame_id`.
- FalkorDB is updated after episode closure and is never reachable from the robot control path.
- Guild.ai coordinates three distinct specialist agents and the required human decisions.
- RocketRide runs the reviewed fixed pipeline and contains no policy or agent reasoning.

The production coordinator runs directly from the locked `uv` environment in a public Daytona
sandbox with all lifecycle timers disabled. Mutable databases and journals use the persistent
sandbox filesystem under `/home/daytona/mm-data`; the `/data` object volume contains only
create-once, hash-verified recovery snapshots. Docker and Compose remain optional
local-development packaging only. Simulation and training may use separate burst workers, but
sponsor adapters and acceptance evidence cannot depend on a provider-specific shortcut.

## Control boundary

The learned task policy has exactly three outputs: forward speed, turning rate, and stop
probability. A command adapter converts these to `[forward, 0, yaw]`; there is no lateral or
joint-level task action.

The vendored upstream MuJoCo Playground G1 artifact remains a provenance-only, unqualified
50 Hz candidate. Production loads `models/mm01/gait-controller-v1.onnx`, selected from one
completed 200-million-step 100 Hz run only after straight-line, turning, stopping, payload,
fall, collision, repeatability, and 60-second standstill gates passed. The task policy still
has no access to joint targets or the controller's 29-action output.

The runtime schedules both gait inference and a fixed command supervisor at 100 Hz over 500 Hz
physics. The supervisor transfers normal commands unchanged and applies a fixed 0.75 m/s^2
deceleration envelope only after a stop request, preventing a discontinuous command transition
without modifying controller weights, the robot, rewards, sensors, or task-policy outputs.

`config/robot/mm01-v1.json` binds the selected ONNX bytes, candidate robot assets, controller
runtime, command adapter, sensor profile, qualification program, training contract, parity
record, and native trial evidence. Startup and episode assembly recompute this manifest and the
permanent robot checksum before execution; the unqualified candidate remains reachable only by
the qualification fixture.

## Data and identity

Every world, policy, correction, episode, and evaluation result is content-addressed. Every
episode record includes the permanent MM-01 checksum, world hash, policy hash, monotonically
increasing sequence, simulation timestamp, and optional synchronized `frame_id`.

Evaluated checkpoints are immutable objects. Promotion changes only an alias after the numeric
gate passes and a human approves it. Held-out worlds are stored behind credentials unavailable
to training and curriculum processes.

## Sponsor readiness states

Health reporting distinguishes `unconfigured`, `configured`, `healthy`, and
`end_to_end_verified` per service.

The observed cloud state on 2026-08-03 is deliberately split from final workflow proof:

- LaserData has a managed deployment and the four-partition `muscle-memory / episode-events-v2`
  topic. The repository verifier completed a real append and exact provider readback. A full
  20 Hz episode remains the stronger demo artifact.
- FalkorDB has a healthy managed instance. The application adapter wrote immutable world,
  episode, failure, lesson, and policy facts and completed a matching multi-hop curriculum
  query. The graph remains outside every robot-control path.
- RocketRide Cloud accepted the reviewed fixed pipeline and completed a real task startup and
  teardown. A validated callback result through the public Daytona coordinator is still
  required before calling the workflow end to end verified.
- Guild.ai has an authenticated workspace and locally schema-validated source for the exact
  three roles. Provider publication is currently blocked by Guild's own private package
  registry returning HTTP 401 during dependency installation; the existing safety agent is not
  substituted for the required three-role evidence.
- Daytona has the public, persistent production sandbox shape. A pushed immutable revision,
  provider environment, public UI/API smoke, and cross-provider callback are required before
  the application itself is described as deployed.

## End-to-end sponsor proof

One synthetic episode must pass through the real RocketRide pipeline, append and replay ordered
LaserData records, write and query the matching FalkorDB traversal, invoke all three Guild.ai
roles, block on a human decision, and then roll back or promote exactly once. The evidence bundle
must retain service identifiers, offsets, graph IDs, agent session IDs, artifact hashes, and the
unchanged robot checksum.
