# Controller Qualification

A controller can become MM-01's permanent frozen walking controller only when all checks pass on
the exact candidate bytes and matching robot assets.

## Required checks

1. Physics runs at 500 Hz and the complete walking controller runs at 100 Hz.
2. The controller accepts only velocity/turn/stop commands and emits only joint targets.
3. A 60-second command schedule completes with finite state and no fall.
4. Forward commands meet a bounded cross-track and heading error.
5. Left and right turns meet bounded angular tracking error without a fall.
6. A stop request reaches the configured low-speed threshold without freezing physics.
7. The robot remains upright at zero command for 60 seconds inside a bounded drift radius.
8. The approved tray and medicine-pouch payload pass the same stability and stop tests.
9. Repeated runs from the same seed produce matching metrics within declared numeric tolerance.
10. The entire robot, controller, sensor profile, and qualification evidence hash to one manifest.

## Current candidate

The vendored MuJoCo Playground ONNX policy is executable and remains upright during a 60-second
mixed walk/turn smoke schedule. It is not qualified: neural inference is 50 Hz, and a zero
command does not produce an acceptable physical stop. A successful load or a non-fall run must
not be reported as controller acceptance.

## Pinned 100 Hz bootstrap

The offline bootstrap uses MuJoCo Playground v0.2.0 at commit
`124a73fa3303f75a62f8fe04d329b829ed0ebdfb` and MuJoCo Menagerie at commit
`1b86ece576591213e2b666ebf59508454200ca97`. The verified patch changes only the controller
period, the episode step count needed to preserve the same 20-second physical horizon, and the
upstream-documented phase hold at zero command. It does not alter rewards, policy networks,
robot assets, or sensors.

Cloud smoke run `g1-100hz-smoke-seed-2-20260803T070839Z` completed one fresh attempt at 100 Hz
and exported a dynamic-batch ONNX controller. Independent JAX/ONNX comparison over eight inputs
measured a maximum absolute delta of `1.6838312149047852e-06`, below the `1e-05` limit. A second
invocation with the same run ID recovered the completed checkpoint and exported it without
creating another training attempt. Its manifest remains explicitly unqualified because smoke
training can never satisfy the controller gate.

The earlier seed-1 cloud smoke controller was exercised through native 500 Hz MuJoCo physics.
It failed the physical gate with seven falls, 178.97 degrees of forward heading error, 73.31 and
55.17 degrees of turn error, 0.563 m/s stop speed, 1.199 m of standstill drift, 152.63 degrees of
payload tray tilt, and package slip. Those failures are retained as evidence; they are not
papered over by changing MM-01, its sensors, the upstream reward, or any gate threshold.

Modal interruption recovery is deliberately conservative. A completion marker and valid
checkpoint allow export to resume. An incomplete attempt remains immutable evidence and the
next attempt starts the entire configured plan fresh with the same seed. Brax's parameter-only
restore is not represented as optimizer-state continuation.
