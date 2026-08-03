# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read these first

- **`AGENTS.md`** — the nine hard invariants (frozen MM-01, non-trainable gait controller, A\* as
  teacher only, held-out worlds stay held out, colliders vs. cosmetic meshes, sensor constraints, positioning
  language, world validation as a gate, improvement measured not asserted). A change that breaks one
  is wrong even if tests pass. Its `Status:` line claiming the repository is empty is stale.
- **`README.md`** — product concept, the eight delivery success criteria, the promotion gate, and the
  12-point definition of a complete v1. That list is the project's real progress rubric.
- **`docs/architecture.md`** — deployment split, control boundary, content-addressing rules, and
  per-sponsor readiness states. `docs/` has a file per subsystem; consult the matching one before
  changing a subsystem.

## Commands

`uv` owns the environment and lockfile; Python is pinned by `.python-version`.

```bash
uv sync --frozen --group dev
uv run pytest
uv run ruff check . && uv run mypy src
uv run mm-verify-robot     # fails closed unless robot identity matches config/robot/mm01-v1.json
uv run mm-smoke
```

Single test or subset:

```bash
uv run pytest tests/test_task_policy.py
uv run pytest tests/test_task_policy.py::test_name
uv run pytest -k 'heldout or promotion'
```

`pyproject.toml` declares a `slow` marker, but no test currently uses it — `-m 'not slow'` is a no-op
and `uv run pytest` is already the whole suite (~290 tests, well under a minute).

`mm-smoke` and `mm-verify-robot` are the only console scripts. Everything in `ops/` runs as a module
from the repo root; `ops/` is intentionally excluded from the wheel (`packages = ["src/muscle_memory"]`):

```bash
uv run python -m ops.policy.evaluate_development
uv run python -m ops.policy.evaluate_heldout
uv run python -m ops.policy.train_sensor_fusion
uv run python -m ops.worlds.freeze_heldout
uv run python -m ops.api.serve
```

Frontend (`frontend/`, Node >= 22) is a separate npm project: `npm run dev`, `npm run build`
(`tsc -b && vite build`), `npm run lint`. It serves two pages — `LandingPage.tsx` (marketing) and
`OperatorConsole.tsx` (the real operator UI, which talks to the backend through `src/operator/api.ts`).

## Package layout

Each subpackage under `src/muscle_memory/` states its charter in its `__init__.py` docstring. The
grouping that isn't discoverable from names alone:

| Concern | Packages |
|---|---|
| Frozen substrate and control seam | `robot`, `simulation`, `policy` |
| World production and admission | `worlds`, `assets` |
| Teacher / test split | `training`, `evaluation` |
| The three memories plus durable state | `telemetry` (LaserData), `graph_memory` (FalkorDB), `episodes`, `coordinator` |
| Agent coordination | `orchestration` (Guild reasoning + RocketRide execution) |
| Runtime serving | `live`, `api`, `backend` |

Three of these enforce invariants **structurally**, which is the single most important thing to
preserve when adding imports:

- `training` is training-only and excluded from evaluation images. A\* lives here. Nothing on an
  evaluation path may import it.
- `evaluation` loads held-out access on demand rather than at import time.
- `policy` exposes evaluation-safe observation and inference interfaces — the seam that lets
  evaluation run on sensor input alone.

An import that crosses these lines defeats invariants 3 and 4 in `AGENTS.md` without failing a test.

`src/muscle_memory/dashboard/` is an empty vestigial directory; the operator UI lives in `frontend/`.

## Evidence and content-addressing

Artifacts are content-addressed, and this is enforced rather than documentary. Every evidence JSON in
`evidence/` records the sha256 of the checkpoint, dataset, world bundle, and robot it describes;
`backend/evidence_admission.py` and `backend/graph_prerequisites.py` check them.

The practical failure mode: retraining a checkpoint regenerates `training.json` but **not** the
evaluation files. An evaluation whose `candidate_policy_sha256` no longer matches the checkpoint on
disk is stale and describes a policy that no longer exists. When a checkpoint changes, re-run the
evaluation — do not copy, rename, or hand-edit an evaluation record to match. Verify with:

```bash
shasum -a 256 models/policy/delivery-v2.npz   # compare to candidate_policy_sha256 in the evidence
```

Superseded evaluations are kept as distinct rounds (`development-evaluation-round1.json`) rather than
overwritten, and rejected candidates are marked `selection_status = rejected_before_heldout`.

## World bundles

`config/worlds/` holds three bundles with different rules: `foundation-v1.json` (generation template),
`live-training-v1.json` (live session catalog), and **`heldout-v1.json` — the 20 frozen validation
worlds**. The held-out bundle must never be used for training, curriculum, hyperparameter selection,
or failure mining, and new bundles must not overlap its seed range. Any change affecting world
generation must keep a given seed reproducible or bump the generation version.

## Claiming a policy improved

Only `ops/policy/evaluate_heldout.py` produces a promotion verdict, comparing the candidate against
the `DirectGoalPolicy` baseline on the frozen worlds. Development evaluations run on
avoidance-selected seeds and are marked `development_only_not_held_out`; their numbers are flattered
relative to held-out and are not promotion evidence.

Note that command accuracy from behavior cloning is action agreement with the A\* teacher, not
delivery success — a policy can match the teacher closely and still fail the gate. Report
`promotable` with the measured numbers behind it, never a training metric as a proxy.
