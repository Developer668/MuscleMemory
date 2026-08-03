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
