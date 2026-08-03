# Live MM-01 simulation

`muscle_memory.live` is the operator-facing episode runtime. It runs the real bundled
MuJoCo scene and frozen walking controller in a bounded worker, while keeping video bytes
out of LaserData and all memory systems out of the control path.

## Admission boundary

A live run accepts only:

- a `ValidatedRuntimeWorld` loaded from the content-addressed
  `config/worlds/live-training-v1.json` catalog produced by the offline strict gate;
- an `EvaluatedPolicySelection` whose checkpoint SHA-256, policy ID, and policy hash match
  every candidate result, the aggregate decision, and the independently admitted canonical
  held-out evidence hash; and
- an episode ID that has not previously been opened.

The selection records whether the measured gate considered the policy promotable. That
field is disclosure only. Starting a training-world run does not promote the policy and
does not bypass the required human promotion gate.

The catalog fixes one timezone-aware validation timestamp. Graph handoff reuses that
provenance for the immutable world and obstacle parents on every visit to the same seed;
episode records continue to carry their own measured end times.

The API process never generates a world from a request. It selects one of the pinned catalog
seeds, so the live package does not import the path teacher or the training expert. Policy inference
receives only `NavigationObservation` and can return only forward speed, turning rate, and
stop probability. World assembly re-verifies the fixed robot checksum before execution.

## Rate-separated execution

| Surface | Rate | Transport |
| --- | ---: | --- |
| Physics | 500 Hz | In-process MuJoCo |
| Frozen walking controller | 100 Hz | Existing controller boundary |
| Learned task policy | 10 Hz | Existing sensor-only observation boundary |
| Numeric telemetry | exactly 20 Hz | EpisodeService to LaserData plus durable outbox |
| Direct video | 30 FPS | `BoundedVideoService`, never LaserData |

Telemetry sequence `n` always has simulation time `n / 20`. Every immutable tick contains
typed IMU, joint effort, contacts, tray state, tactile/slip, battery, policy action,
collision, reward/progress, safety-marker, and completion fields. The existing LaserData
transport partitions on `episode_id` and indexes `episode_id`, `world_id`, `policy_id`,
`failure_type`, `event_time`, `sequence`, and `event_id`.

Every 30 FPS camera frame is assigned to the first 20 Hz tick at or after its scheduled
simulation time. The tick includes metadata and SHA-256 values for each assigned frame,
but no encoded bytes. Every product metadata record includes its direct MJPEG URL and exact-frame
URL. `frame_id` remains the only video/telemetry join key.

## Direct video products

Each `VideoFrameSet` contains six JPEG products captured from one MuJoCo state:

- third-person view;
- left-eye RGB;
- right-eye RGB;
- stereo composite;
- stereo-derived depth visualization; and
- simulator-debug segmentation.

`BoundedVideoService` limits both retained frame sets and total bytes, drops only the
oldest frame set when full, reports the drop count, and closes waiting streams when the
episode ends. It exposes exact-frame lookup, latest-frame lookup, and per-product MJPEG
iteration. FastAPI exposes each product at
`/api/v1/episodes/{episode_id}/video/{product}.mjpeg` and exact buffered frames at
`/api/v1/episodes/{episode_id}/video/{product}/frames/{frame_index}`. Historical frame URLs
truthfully return unavailable after a frame leaves the bounded cloud process buffer.

## Operator API

`GET /api/v1/live/options` reports whether a deployment has a catalog and evaluated policy
admitted. It returns an explicit disabled state rather than fixture options when it does not.

Starting and cancelling use `POST /api/v1/live/episodes` and
`POST /api/v1/live/episodes/{episode_id}/cancel`. Both require a bearer credential with the
dedicated `episodes:write` scope. `GET /api/v1/live/episodes/{episode_id}` exposes bounded worker
state without reflecting credentials or raw internal exceptions. Production shutdown cancels and
joins the simulator worker before provider connections close.

The production composition enables these routes only when the four `MM_HELDOUT_*` admission
variables identify an exact verified artifact and checkpoint. Partial or changed admission
configuration fails closed during startup.

## Supervisor hooks

`LiveEpisodeManager` is the integration surface:

- `start_episode(...)` queues one validation-gated real simulation;
- `status(...)` returns phase, simulation time, wall lag, counts, provider state, measured
  result, and post-close provider completeness;
- `cancel(...)` requests a stop and writes the terminal completion event on the next exact
  20 Hz tick;
- `latest_video_frame(...)` and `video_frame(...)` return direct frame sets;
- `iter_mjpeg(...)` streams one named visual product; and
- `shutdown(...)` cancels workers and waits for deterministic closure.

Concurrency is fixed when the manager is constructed. Starts above the limit fail closed.
Wall-clock lag and dropped-frame counts downgrade live health instead of being hidden.

Normal completion and cancellation both produce a measured `PolicyEpisodeResult` and call
the existing `EpisodeService.close_episode`. That service durably records closure before
performing the post-episode FalkorDB handoff. If execution raises unexpectedly, status is
`failed` with the error type; the manager does not claim a completed episode.

## Verification

Run the focused real-simulator checks with:

```text
uv run pytest tests/test_live_simulation.py
uv run pytest tests/test_live_http_api.py
```

They verify the exact clocks, all frame joins and products, byte-free telemetry metadata,
bounded MJPEG behavior, evidence binding, raw-world rejection, cancellation, graph handoff,
and a short run through the real MuJoCo model and frozen controller.
