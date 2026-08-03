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

The upstream MuJoCo Playground G1 artifact is a real executable candidate, but it is not the
final MM-01 controller. Its immutable ONNX inference cadence is 50 Hz while the product requires
a 100 Hz frozen walking subsystem, and characterization shows unacceptable motion after a zero
command. The candidate therefore sits behind a fail-closed qualification gate. It must not be
described as the final frozen robot or used for a promotion claim.

The implementation schedules a controller supervisor at 100 Hz and physics at 500 Hz. A final
controller is eligible for MM-01 only after its own 100 Hz artifact passes straight-line,
turning, stopping, payload, fall, and 60-second stability tests. Once accepted, the complete
robot/controller/sensor manifest is hashed and cannot be replaced in place.

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
