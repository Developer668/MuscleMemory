"""Remote training and qualification implementation used inside Modal."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ops.controller.contract import (
    ENVIRONMENT_NAME,
    TRAINING_PLANS,
    ContractError,
    QualificationEvidence,
    RunMode,
    build_artifact_manifest,
    evaluate_qualification,
    verify_qualification_binding,
    verify_source_checkout,
    write_artifact_manifest,
)

CHECKOUT_ROOT = Path("/opt/mujoco_playground")
PATCH_PATH = Path("/opt/mm/ops/controller/mm01_g1_100hz.patch")
VENV_PYTHON = CHECKOUT_ROOT / ".venv" / "bin" / "python"
RUN_ID_PATTERN = re.compile(
    r"g1-100hz-(?P<mode>smoke|full)-seed-(?P<seed>[0-9]+)-"
    r"(?P<timestamp>[0-9]{8}T[0-9]{6}Z)"
)


def validate_run_id(run_id: str, *, mode: RunMode | None = None, seed: int | None = None) -> None:
    """Reject traversal and cross-plan reuse of a controller run directory."""

    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise ContractError("controller run ID does not match the immutable naming contract")
    if mode is not None and match.group("mode") != mode.value:
        raise ContractError("controller run ID mode differs from the requested mode")
    if seed is not None and int(match.group("seed")) != seed:
        raise ContractError("controller run ID seed differs from the requested seed")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _export_checkpoint(
    run_root: Path,
    seed: int,
    checkpoint_path: Path,
) -> dict[str, object]:
    command = (
        str(VENV_PYTHON),
        "-m",
        "ops.controller.export_onnx",
        "--run-root",
        str(run_root),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint_path),
    )
    subprocess.run(command, cwd=CHECKOUT_ROOT, check=True)
    payload = json.loads((run_root / "onnx-parity.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise RuntimeError("ONNX export did not persist passing parity evidence")
    return payload


def _training_command(
    mode: RunMode,
    seed: int,
    attempt_root: Path,
) -> tuple[str, ...]:
    plan = TRAINING_PLANS[mode]
    return (
        str(VENV_PYTHON),
        str(CHECKOUT_ROOT / "learning" / "train_jax_ppo.py"),
        f"--env_name={ENVIRONMENT_NAME}",
        "--impl=jax",
        f"--seed={seed}",
        f"--logdir={attempt_root}",
        f"--suffix=mm01-{mode.value}-seed-{seed}",
        "--use_wandb=false",
        "--use_tb=false",
        "--domain_randomization=false",
        "--num_videos=1",
        *plan.cli_overrides(),
    )


def _new_contract(
    run_id: str,
    mode: RunMode,
    seed: int,
    source: object,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "mode": mode.value,
        "seed": seed,
        "source": asdict(source),
        "training_plan": asdict(TRAINING_PLANS[mode]),
        "environment": {
            "name": ENVIRONMENT_NAME,
            "ctrl_dt_seconds": 0.01,
            "controller_hz": 100,
            "physics_hz": 500,
            "physical_episode_seconds": 20.0,
        },
        "preserved_upstream_settings": {
            "reward_configuration": True,
            "g1_network_configuration": True,
            "robot_and_sensors": True,
        },
        "restart_semantics": {
            "optimizer_state_restored": False,
            "policy_warm_start_used": False,
            "interrupted_attempt_action": "start a fresh complete plan with the same seed",
            "completed_attempt_requirement": "one attempt must finish the entire plan",
        },
        "status": "pending",
        "selected_attempt_id": None,
        "attempts": [],
    }


def _load_or_create_contract(
    run_root: Path,
    run_id: str,
    mode: RunMode,
    seed: int,
    source: object,
) -> dict[str, Any]:
    contract_path = run_root / "training-contract.json"
    if not contract_path.exists():
        contract = _new_contract(run_id, mode, seed, source)
        _atomic_write_json(contract_path, contract)
        return contract

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("training contract must be a JSON object")
    expected = {
        "schema_version": 2,
        "run_id": run_id,
        "mode": mode.value,
        "seed": seed,
        "source": asdict(source),
        "training_plan": asdict(TRAINING_PLANS[mode]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(f"existing training contract changed at {key}")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise ContractError("training contract attempts must be a list of objects")
    return payload


def _log_has_training_completion(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        return any(line.rstrip() == "Done training." for line in stream)


def _validate_checkpoint_config(checkpoint_path: Path) -> None:
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        raise ContractError("completed checkpoint is missing its environment config")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("checkpoint environment config must be a JSON object")
    if payload.get("ctrl_dt") != 0.01:
        raise ContractError("persisted checkpoint config does not describe a 100 Hz controller")
    if payload.get("episode_length") != 2000:
        raise ContractError("persisted checkpoint config does not preserve a 20 second horizon")


def _latest_checkpoint(run_root: Path) -> Path:
    checkpoints = tuple(
        path
        for path in run_root.rglob("checkpoints/*")
        if path.is_dir() and path.name.isdigit()
    )
    if not checkpoints:
        raise ValueError("training did not produce a numeric checkpoint")
    return max(
        checkpoints,
        key=lambda path: (int(path.name), path.stat().st_mtime_ns, path.as_posix()),
    )


def _attempt_checkpoint(run_root: Path, attempt: dict[str, Any]) -> Path | None:
    relative_root = attempt.get("root")
    if not isinstance(relative_root, str):
        return None
    attempt_root = run_root / relative_root
    try:
        checkpoint_path = _latest_checkpoint(attempt_root)
        _validate_checkpoint_config(checkpoint_path)
    except (ContractError, ValueError):
        return None
    return checkpoint_path


def _recover_attempts(run_root: Path, contract: dict[str, Any]) -> Path | None:
    """Recover an interrupted Modal retry without pretending optimizer continuation."""

    selected: Path | None = None
    changed = False
    attempts = contract["attempts"]
    assert isinstance(attempts, list)
    for raw_attempt in attempts:
        assert isinstance(raw_attempt, dict)
        status = raw_attempt.get("status")
        log_relative = raw_attempt.get("log")
        log_path = run_root / log_relative if isinstance(log_relative, str) else Path()
        checkpoint_path = _attempt_checkpoint(run_root, raw_attempt)
        completed = checkpoint_path is not None and _log_has_training_completion(log_path)
        if completed:
            if status != "training_complete":
                raw_attempt["status"] = "training_complete"
                raw_attempt["recovered_after_restart"] = True
                raw_attempt["completed_at"] = _timestamp()
                raw_attempt["checkpoint"] = checkpoint_path.relative_to(run_root).as_posix()
                changed = True
            selected = checkpoint_path
            contract["selected_attempt_id"] = raw_attempt.get("attempt_id")
            contract["status"] = "training_complete"
        elif status == "running":
            raw_attempt["status"] = "interrupted"
            raw_attempt["interrupted_at"] = _timestamp()
            raw_attempt["interruption_disposition"] = (
                "checkpoint retained as evidence only; next attempt starts the complete plan fresh"
            )
            changed = True
    if changed:
        _atomic_write_json(run_root / "training-contract.json", contract)
    return selected


def _record_attempt_failure(
    run_root: Path,
    contract: dict[str, Any],
    attempt: dict[str, Any],
    error: BaseException,
) -> None:
    attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    attempt["ended_at"] = _timestamp()
    attempt["error_type"] = type(error).__name__
    attempt["error"] = str(error)
    contract["status"] = attempt["status"]
    _atomic_write_json(run_root / "training-contract.json", contract)


def _execute_attempt(
    mode: RunMode,
    seed: int,
    run_root: Path,
    contract: dict[str, Any],
) -> Path:
    attempts = contract["attempts"]
    assert isinstance(attempts, list)
    attempt_number = len(attempts) + 1
    attempt_id = f"attempt-{attempt_number:04d}"
    attempt_root = run_root / "attempts" / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    log_path = attempt_root / "training.log"
    command = _training_command(mode, seed, attempt_root)
    if "--episode_length=2000" not in command:
        raise ContractError("training command did not preserve the 20 second episode horizon")

    attempt: dict[str, Any] = {
        "attempt_id": attempt_id,
        "root": attempt_root.relative_to(run_root).as_posix(),
        "log": log_path.relative_to(run_root).as_posix(),
        "status": "running",
        "started_at": _timestamp(),
        "command": list(command),
        "restart_kind": "fresh_complete_plan",
        "optimizer_state_restored": False,
        "policy_warm_start_used": False,
    }
    attempts.append(attempt)
    contract["status"] = "running"
    _atomic_write_json(run_root / "training-contract.json", contract)

    environment = os.environ.copy()
    environment.update(
        {
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "PYTHONHASHSEED": str(seed),
            "MUJOCO_GL": "egl",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    try:
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=CHECKOUT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            saw_training_completion = False
            stopped_after_smoke_checkpoint = False
            for line in process.stdout:
                output.write(line)
                output.flush()
                if line.rstrip() == "Done training.":
                    saw_training_completion = True
                    if mode is RunMode.SMOKE:
                        stopped_after_smoke_checkpoint = True
                        process.send_signal(signal.SIGTERM)
                        break
            process.stdout.close()
            return_code = process.wait(timeout=30)
        attempt["process_return_code"] = return_code
        attempt["stopped_after_smoke_checkpoint"] = stopped_after_smoke_checkpoint
        if not saw_training_completion:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            raise RuntimeError("training process exited without its completion marker")

        checkpoint_path = _latest_checkpoint(attempt_root)
        _validate_checkpoint_config(checkpoint_path)
        attempt["status"] = "training_complete"
        attempt["completed_at"] = _timestamp()
        attempt["checkpoint"] = checkpoint_path.relative_to(run_root).as_posix()
        contract["status"] = "training_complete"
        contract["selected_attempt_id"] = attempt_id
        _atomic_write_json(run_root / "training-contract.json", contract)
        return checkpoint_path
    except BaseException as error:
        _record_attempt_failure(run_root, contract, attempt, error)
        raise


def run_training(
    mode: RunMode,
    seed: int,
    run_root: Path,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
    """Run one complete pinned PPO attempt and persist an unqualified manifest."""

    resolved_run_id = run_id or run_root.name
    validate_run_id(resolved_run_id, mode=mode, seed=seed)
    if run_root.name != resolved_run_id:
        raise ContractError("controller run directory and run ID differ")
    source = verify_source_checkout(CHECKOUT_ROOT, PATCH_PATH, patched=True)
    run_root.mkdir(parents=True, exist_ok=True)
    contract = _load_or_create_contract(run_root, resolved_run_id, mode, seed, source)
    checkpoint_path = _recover_attempts(run_root, contract)
    if checkpoint_path is None:
        checkpoint_path = _execute_attempt(mode, seed, run_root, contract)

    _export_checkpoint(run_root, seed, checkpoint_path)
    contract["status"] = "exported_unqualified"
    contract["exported_checkpoint"] = checkpoint_path.relative_to(run_root).as_posix()
    _atomic_write_json(run_root / "training-contract.json", contract)
    manifest = build_artifact_manifest(run_root, mode=mode, seed=seed, source=source)
    write_artifact_manifest(run_root, manifest)
    return manifest


def _contract_checkpoint(run_root: Path, contract_payload: dict[str, Any]) -> Path:
    relative = contract_payload.get("exported_checkpoint")
    if isinstance(relative, str):
        checkpoint_path = run_root / relative
        _validate_checkpoint_config(checkpoint_path)
        return checkpoint_path
    return _latest_checkpoint(run_root)


def export_training_run(run_root: Path) -> dict[str, object]:
    """Attach a verified ONNX export to an existing unqualified training run."""

    validate_run_id(run_root.name)
    source = verify_source_checkout(CHECKOUT_ROOT, PATCH_PATH, patched=True)
    contract_payload = json.loads(
        (run_root / "training-contract.json").read_text(encoding="utf-8")
    )
    if not isinstance(contract_payload, dict):
        raise ContractError("training contract must be a JSON object")
    mode = RunMode(contract_payload["mode"])
    seed = int(contract_payload["seed"])
    checkpoint_path = _contract_checkpoint(run_root, contract_payload)
    _export_checkpoint(run_root, seed, checkpoint_path)
    manifest = build_artifact_manifest(run_root, mode=mode, seed=seed, source=source)
    write_artifact_manifest(run_root, manifest)
    return manifest


def qualify_training_run(run_root: Path) -> dict[str, object]:
    """Evaluate separately produced physical evidence and replace the manifest."""

    validate_run_id(run_root.name)
    source = verify_source_checkout(CHECKOUT_ROOT, PATCH_PATH, patched=True)
    contract_payload = json.loads(
        (run_root / "training-contract.json").read_text(encoding="utf-8")
    )
    evidence_payload = json.loads(
        (run_root / "qualification-evidence.json").read_text(encoding="utf-8")
    )
    if not isinstance(contract_payload, dict):
        raise ContractError("training contract must be a JSON object")
    if not isinstance(evidence_payload, dict):
        raise ContractError("qualification evidence must be a JSON object")
    mode = RunMode(contract_payload["mode"])
    seed = int(contract_payload["seed"])
    evidence = QualificationEvidence.from_mapping(evidence_payload)
    verify_qualification_binding(evidence, run_root, Path(__file__).with_name("native_qualify.py"))
    result = evaluate_qualification(evidence, mode)
    manifest = build_artifact_manifest(
        run_root,
        mode=mode,
        seed=seed,
        source=source,
        qualification=result,
    )
    write_artifact_manifest(run_root, manifest)
    return manifest
