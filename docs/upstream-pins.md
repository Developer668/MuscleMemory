# Upstream Pins

MM-01 source inputs are pinned before any permanent robot checksum is created.

| Component | Pin | License |
| --- | --- | --- |
| MuJoCo Playground | `v0.2.0` / `124a73fa3303f75a62f8fe04d329b829ed0ebdfb` | Apache-2.0 |
| MuJoCo Menagerie G1 | `1b86ece576591213e2b666ebf59508454200ca97` | BSD-3-Clause |
| MuJoCo | `3.6.0` | Apache-2.0 |
| ONNX Runtime | `1.28.0` | MIT |
| Candidate G1 policy | SHA-256 `db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b` | Apache-2.0 |

The candidate policy expects `float32[1, 103]` observations and returns 29 joint-position
actions. It is used only with Playground's matching `g1_mjx_feetonly` model variant and
Menagerie assets.

The upstream package published to PyPI omits `experimental/sim2sim`, so the repository vendors
the exact required files and their licenses. `config/robot/mm01-candidate.json` records each
bundled file checksum and upstream commit. `mm-verify-robot` fails on any mismatch.

The permanent MM-01 manifest will be a separate file. Creating it is blocked until a controller
passes every qualification in `docs/controller-qualification.md`.
