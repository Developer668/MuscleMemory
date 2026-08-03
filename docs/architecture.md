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

The first cloud target is a public CPU coordinator plus burst simulation/training workers. The
exact provider remains an operational choice; service adapters and acceptance evidence cannot
depend on a provider-specific shortcut.

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

- RocketRide account access and an active hackathon key record are present; the secret and a
  completed pipeline run still need verification.
- Guild.ai account access is present and an internal safety agent exists; three published,
  schema-validated agents and observed approval traces are still required.
- LaserData Cloud signup is pending administrator approval. Local or self-hosted Laser Stack is
  an acceptable development path, but it is not cloud proof.
- FalkorDB has no current repository configuration. Development may use its official container;
  final proof requires a persistent remotely reachable instance.

## End-to-end sponsor proof

One synthetic episode must pass through the real RocketRide pipeline, append and replay ordered
LaserData records, write and query the matching FalkorDB traversal, invoke all three Guild.ai
roles, block on a human decision, and then roll back or promote exactly once. The evidence bundle
must retain service identifiers, offsets, graph IDs, agent session IDs, artifact hashes, and the
unchanged robot checksum.
