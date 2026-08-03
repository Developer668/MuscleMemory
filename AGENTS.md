# AGENTS.md

Operating guide for coding agents working in this repository. Read [README.md](README.md) first for the product concept; this file covers the rules that are easy to violate and expensive to undo.

> **Status:** the repository is currently empty — no source, build system, or test harness exists yet. Sections marked _TBD_ must be filled in by the first change that establishes them, in the same commit.

---

## Hard invariants

These are not preferences. A change that breaks one of these is wrong even if it passes tests.

### 1. MM-01 never changes

The robot's body, mass, dimensions, joint set, sensor set, and gait policy are frozen. Only training worlds, stored experience, and the learned high-level task policy may change.

- Never edit the Unitree G1 skeleton, the frozen walking controller weights, or the sensor configuration to make an episode succeed.
- Every episode record must carry the robot checksum. If a change would alter that checksum, stop and surface it to the user — do not regenerate the checksum to match.
- If a task appears to require changing the robot, the task is misspecified. Say so.

### 2. The frozen walking controller is not trainable

Standing, balance, walking, turning, foot placement, joint-torque control, and fall prevention belong to the frozen 100 Hz controller from MuJoCo Playground. The learned policy emits only **forward speed**, **turning rate**, and **stop probability** at 5–10 Hz. Do not add locomotion outputs to the task policy or route task-policy gradients into the controller.

### 3. A* is a teacher, not a runtime component

A* generates expert paths for behavior cloning only. It must be unreachable from any evaluation code path. Evaluation runs the policy on sensor input alone.

### 4. Held-out worlds stay held out

Twenty validation worlds are frozen before training. Never use them for training, curriculum generation, hyperparameter selection, or failure mining. Keep them behind an interface that makes accidental use structurally difficult, not merely discouraged.

### 5. Meshes are cosmetic; colliders are physics

TRELLIS-generated GLB meshes drive rendering only. Physics always uses the deterministic primitive or convex collider. Never feed a detailed generated mesh to MuJoCo as collision geometry.

### 6. No LiDAR

MM-01's depth is derived from stereo vision (32–64 sectors). Do not add LiDAR to the sensor profile, the UI, or the docs, and do not label anything as LiDAR-equivalent.

### 7. Positioning language

MM-01 is *inspired by* the publicly described 1X NEO sensor and control architecture. Never describe it as an official NEO digital twin, an official 1X model, or a validated reproduction — in code comments, UI copy, docs, or commit messages. 1X has not published its complete robot model, controller, or sensor API.

### 8. Worlds are validated before use

No world reaches training or evaluation without passing all validation checks: no overlapping objects, start and destination connected, all passages meet minimum robot clearance, obstacles have approved colliders, a valid baseline path exists, and physical parameters are within safe limits. Validation is a gate, not a warning.

### 9. Improvement is measured, not asserted

Different actions on identical input is not evidence of learning. Any claim that a policy improved must cite measured results on the held-out worlds under the promotion gate in the README. Do not report a policy as promotable without running the comparison and showing the numbers.

---

## Runtime agents (the product's agents, not you)

Coordinated by **Guild.ai**. Do not collapse these roles or let one take another's decisions.

| Agent | Owns |
| --- | --- |
| World and Physics Agent | Obstacle physical properties, world assembly, world validation |
| Failure and Curriculum Agent | Failure pattern mining, targeted world generation, curriculum selection |
| Safety and Evaluation Agent | Evaluation runs, promotion-gate enforcement, rollback |

**RocketRide** executes approved tools in a fixed pipeline: `validate world → run episode → summarize telemetry → query graph memory → select curriculum → train candidate policy → evaluate candidate → promote or roll back`.

Guild.ai decides who reasons. RocketRide executes. Keep reasoning out of the executor and execution out of the reasoners.

### Human approval is required for

- Uncertain or agent-proposed physical properties of a new asset
- Reward function changes
- Curriculum changes
- Policy promotion or rollback

Build these as blocking gates. An agent proposing a change must not be able to apply it.

---

## Rates

Respect these when adding loops, streams, or UI updates. Mismatched rates are a common source of silent desync.

| Layer | Rate |
| --- | --- |
| Physics | 500–1,000 Hz |
| Frozen walking controller | 100 Hz |
| Learned task policy | 5–10 Hz |
| Numeric telemetry | 20 Hz |
| Dashboard charts | 10 Hz |
| Robot POV | 30 FPS |

Video may stream directly to the interface; LaserData carries synchronized frame IDs and event metadata. Keep frame IDs the single join key between the two.

---

## Where things live

| Concern | System | Rule |
| --- | --- | --- |
| Live experience | LaserData | Append-only episode telemetry. Do not mutate past episodes. |
| Explicit experience | FalkorDB | Graph of worlds, obstacles, episodes, failures, corrections, lessons, policy versions. Never in the robot's control path. |
| Behavioral memory | Policy checkpoints | Versioned and immutable once evaluated. New behavior means a new version, never an in-place edit. |

---

## Success criteria for the demo task

A delivery run succeeds only if **all** hold. Treat these as constants in one place, not as scattered literals.

- Reaches the resident within 30 s
- Stops within 0.5 m, facing them
- No falls
- No body collisions
- Minimum obstacle clearance ≥ 0.25 m
- Tray tilt < 12°
- No simulated package slip
- No human intervention

---

## UI rules

- Every sensor signal must be labeled **Used by policy**, **Logged only**, or **Simulator ground truth**. A new signal without a label is an incomplete change.
- The sensor rail covers all eight categories; do not silently drop one because it has no data yet — show it as unavailable.
- Robot POV shows left-eye RGB, right-eye RGB, stereo composite, derived depth, and simulator-debug segmentation.

---

## Demo resilience

The demo must finish even if live asset generation or live training times out.

- Cache TRELLIS assets and verified policy checkpoints.
- Any live path (generation, training) needs a timeout and a cached fallback.
- Never make the demo's completion depend on a network call succeeding.

---

## Working conventions

- **Scope:** MVP excludes grasping. Doors, stairs, moving pets, and multiple rooms are stretch goals — do not build them into core abstractions preemptively, but do not make them impossible either.
- **Determinism:** worlds are seeded. Any change that affects world generation must keep a given seed reproducible, or must bump a generation version so old seeds are not silently reinterpreted.
- **Safety limits:** physical parameter bounds (mass, friction, dimensions) are validation constraints. Widening them to make something work requires user approval.
- **Verification:** run the project's checks and report actual output before claiming a change works. If no test harness exists yet for the area you touched, say that plainly rather than implying verification happened.

### Build, test, and run commands

Python is pinned by `.python-version`; `uv` owns the environment and lockfile.

```text
setup:  uv sync --frozen --group dev
test:   uv run pytest
lint:   uv run ruff check . && uv run mypy src
run:    uv run mm-smoke
```

---

## Quick red flags

Stop and check with the user if you find yourself about to:

- Modify the robot model, controller, or sensor config to fix a failing episode
- Touch a held-out validation world
- Use A* inside evaluation
- Pass a generated mesh to the physics engine as a collider
- Apply an agent-proposed physical property, reward change, or promotion without approval
- Report improvement without held-out numbers
- Write "digital twin", "official NEO", or "LiDAR" anywhere in this project
