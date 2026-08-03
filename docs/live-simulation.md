# Live MM-01 simulation

`muscle_memory.live` is the operator-facing episode runtime. It runs the real bundled
MuJoCo scene and frozen walking controller in a bounded worker, while keeping video bytes
out of LaserData and all memory systems out of the control path.

## Admission boundary

A live run accepts only:

- the concrete `ValidatedTrainingWorld` produced by the strict training-world gate;
- an `EvaluatedPolicySelection` whose checkpoint SHA-256, policy ID, and policy hash match
  every candidate result and the aggregate decision in held-out evaluation evidence; and
- an episode ID that has not previously been opened.

The selection records whether the measured gate considered the policy promotable. That
field is disclosure only. Starting a training-world run does not promote the policy and
does not bypass the required human promotion gate.

The live package does not import the path teacher or the training expert. Policy inference
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
but no encoded bytes. `frame_id` remains the only video/telemetry join key.

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
iteration. These are ready to wrap with normal FastAPI JSON, JPEG, and
`multipart/x-mixed-replace` responses.

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
```

They verify the exact clocks, all frame joins and products, byte-free telemetry metadata,
bounded MJPEG behavior, evidence binding, raw-world rejection, cancellation, graph handoff,
and a short run through the real MuJoCo model and frozen controller.
