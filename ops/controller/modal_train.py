"""Modal entrypoint for pinned MM-01 gait training and qualification.

Smoke mode is intentionally small. Full mode requires an explicit CLI guard and
has not been launched as part of repository setup.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import modal

from ops.controller.contract import MENAGERIE_COMMIT, PLAYGROUND_COMMIT, RunMode

APP_NAME = "muscle-memory-g1-controller"
VOLUME_NAME = "muscle-memory-controller-artifacts"
REMOTE_CHECKOUT = "/opt/mujoco_playground"
REMOTE_CONTROLLER_OPS = "/opt/mm/ops/controller"
LOCAL_CONTROLLER_OPS = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "git", "libegl1", "libgl1", "libglfw3", "patch")
    .run_commands(
        f"git clone https://github.com/google-deepmind/mujoco_playground.git {REMOTE_CHECKOUT}",
        f"git -C {REMOTE_CHECKOUT} checkout --detach {PLAYGROUND_COMMIT}",
        (
            "git clone https://github.com/google-deepmind/mujoco_menagerie.git "
            f"{REMOTE_CHECKOUT}/mujoco_playground/external_deps/mujoco_menagerie"
        ),
        (
            f"git -C {REMOTE_CHECKOUT}/mujoco_playground/external_deps/mujoco_menagerie "
            f"checkout --detach {MENAGERIE_COMMIT}"
        ),
        "python -m pip install uv==0.8.15",
        f"cd {REMOTE_CHECKOUT} && uv sync --frozen --extra cuda",
        (
            f"uv pip install --python {REMOTE_CHECKOUT}/.venv/bin/python "
            "onnx==1.20.1 onnxruntime==1.28.0"
        ),
    )
    .add_local_dir(LOCAL_CONTROLLER_OPS, REMOTE_CONTROLLER_OPS, copy=True)
    .run_commands(
        (
            f"PYTHONPATH=/opt/mm python {REMOTE_CONTROLLER_OPS}/run_qualification.py "
            f"verify-source --checkout {REMOTE_CHECKOUT} "
            f"--patch {REMOTE_CONTROLLER_OPS}/mm01_g1_100hz.patch"
        ),
        f"git -C {REMOTE_CHECKOUT} apply {REMOTE_CONTROLLER_OPS}/mm01_g1_100hz.patch",
        (
            f"PYTHONPATH=/opt/mm python {REMOTE_CONTROLLER_OPS}/run_qualification.py "
            f"verify-source --checkout {REMOTE_CHECKOUT} "
            f"--patch {REMOTE_CONTROLLER_OPS}/mm01_g1_100hz.patch --patched"
        ),
    )
    .env(
        {
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "MUJOCO_GL": "egl",
            "PYTHONPATH": "/opt/mm",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=24 * 60 * 60,
    volumes={"/artifacts": artifacts},
)
def train(mode_value: str, seed: int, run_id: str) -> dict[str, object]:
    from ops.controller.remote_job import run_training

    mode = RunMode(mode_value)
    artifacts.reload()
    manifest = run_training(
        mode,
        seed,
        Path("/artifacts") / run_id,
        run_id=run_id,
        execution_backend="modal-l4",
    )
    artifacts.commit()
    return {"run_id": run_id, "manifest": manifest}


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/artifacts": artifacts},
)
def export(run_id: str) -> dict[str, object]:
    from ops.controller.remote_job import export_training_run

    artifacts.reload()
    run_root = Path("/artifacts") / run_id
    manifest = export_training_run(run_root)
    artifacts.commit()
    return {"run_id": run_id, "manifest": manifest}


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/artifacts": artifacts},
)
def qualify(run_id: str) -> dict[str, object]:
    from ops.controller.remote_job import qualify_training_run

    artifacts.reload()
    run_root = Path("/artifacts") / run_id
    manifest = qualify_training_run(run_root)
    artifacts.commit()
    return {"run_id": run_id, "manifest": manifest}


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    seed: int = 1,
    run_id: str = "",
    confirm_full: bool = False,
) -> None:
    if mode in {"export", "qualify"}:
        if not run_id:
            raise ValueError(f"--run-id is required for {mode}")
        print(export.remote(run_id) if mode == "export" else qualify.remote(run_id))
        return

    selected_mode = RunMode(mode)
    if selected_mode is RunMode.FULL and not confirm_full:
        raise ValueError("full mode requires --confirm-full")
    stable_run_id = run_id or (
        f"g1-100hz-{selected_mode.value}-seed-{seed}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    print(train.remote(selected_mode.value, seed, stable_run_id))
