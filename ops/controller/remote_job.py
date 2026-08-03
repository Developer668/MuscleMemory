"""Remote training and qualification implementation used inside Modal."""

from __future__ import annotations

import json
import os
import re
import shutil
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
    SourceVerification,
    build_artifact_manifest,
    evaluate_qualification,
    sha256_file,
    verify_qualification_binding,
    verify_source_checkout,
    write_artifact_manifest,
)

CHECKOUT_ROOT = Path(os.environ.get("MM_CONTROLLER_CHECKOUT", "/opt/mujoco_playground"))
PATCH_PATH = Path(
    os.environ.get("MM_CONTROLLER_PATCH", "/opt/mm/ops/controller/mm01_g1_100hz.patch")
)
VENV_PYTHON = Path(
    os.environ.get("MM_CONTROLLER_PYTHON", CHECKOUT_ROOT / ".venv" / "bin" / "python")
)
RUN_ID_PATTERN = re.compile(
    r"g1-100hz-(?P<mode>smoke|full)-seed-(?P<seed>[0-9]+)-"
    r"(?P<timestamp>[0-9]{8}T[0-9]{6}Z)"
)
LOCAL_CPU_BACKEND_PATTERN = re.compile(r"local-macos-cpu-(?P<devices>[1-9][0-9]*)-device")


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


def _selection_provenance(
    contract: dict[str, Any],
    selected_checkpoint: str,
) -> tuple[str, str]:
    """Preserve the first selection's time and previously exported checkpoint."""
    existing = contract.get("checkpoint_selection")
    same_selection = (
        isinstance(existing, dict)
        and existing.get("selected_checkpoint") == selected_checkpoint
    )
    previous = (
        existing.get("previous_exported_checkpoint")
        if same_selection
        else contract.get("exported_checkpoint")
    )
    selected_at = existing.get("selected_at") if same_selection else _timestamp()
    if not isinstance(previous, str) or not isinstance(selected_at, str):
        raise ContractError("checkpoint selection provenance is incomplete")
    return previous, selected_at


def _is_local_cpu_backend(execution_backend: str) -> bool:
    return LOCAL_CPU_BACKEND_PATTERN.fullmatch(execution_backend) is not None


def _execution_configuration(execution_backend: str) -> dict[str, object]:
    match = LOCAL_CPU_BACKEND_PATTERN.fullmatch(execution_backend)
    if match is not None:
        return {
            "jax_platform": "cpu",
            "host_platform_device_count": int(match.group("devices")),
        }
    if execution_backend == "modal-l4":
        return {"jax_platform": "gpu", "accelerator": "L4"}
    raise ContractError(f"unsupported controller execution backend: {execution_backend}")


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
    source: SourceVerification,
    execution_backend: str = "modal-l4",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "mode": mode.value,
        "seed": seed,
        "execution_backend": execution_backend,
        "execution_configuration": _execution_configuration(execution_backend),
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
    source: SourceVerification,
    execution_backend: str,
) -> dict[str, Any]:
    contract_path = run_root / "training-contract.json"
    if not contract_path.exists():
        contract = _new_contract(run_id, mode, seed, source, execution_backend)
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
        "execution_backend": execution_backend,
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
            assert checkpoint_path is not None
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
    execution_backend: str,
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
        "execution_backend": execution_backend,
    }
    attempts.append(attempt)
    contract["status"] = "running"
    _atomic_write_json(run_root / "training-contract.json", contract)

    environment = os.environ.copy()
    environment.update(
        {
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "PYTHONHASHSEED": str(seed),
            "MUJOCO_GL": "cgl" if _is_local_cpu_backend(execution_backend) else "egl",
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
    execution_backend: str = "modal-l4",
) -> dict[str, object]:
    """Run one complete pinned PPO attempt and persist an unqualified manifest."""

    resolved_run_id = run_id or run_root.name
    validate_run_id(resolved_run_id, mode=mode, seed=seed)
    if run_root.name != resolved_run_id:
        raise ContractError("controller run directory and run ID differ")
    source = verify_source_checkout(CHECKOUT_ROOT, PATCH_PATH, patched=True)
    run_root.mkdir(parents=True, exist_ok=True)
    contract = _load_or_create_contract(
        run_root,
        resolved_run_id,
        mode,
        seed,
        source,
        execution_backend,
    )
    checkpoint_path = _recover_attempts(run_root, contract)
    if checkpoint_path is None:
        checkpoint_path = _execute_attempt(
            mode,
            seed,
            run_root,
            contract,
            execution_backend,
        )

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


def _load_qualification_probe(
    evidence_root: Path,
    *,
    mode: RunMode,
) -> tuple[QualificationEvidence, dict[str, object]]:
    evidence_payload = json.loads(
        (evidence_root / "qualification-evidence.json").read_text(encoding="utf-8")
    )
    parity_payload = json.loads(
        (evidence_root / "onnx-parity.json").read_text(encoding="utf-8")
    )
    if not isinstance(evidence_payload, dict) or not isinstance(parity_payload, dict):
        raise ContractError("checkpoint probe evidence must contain JSON objects")
    if parity_payload.get("passed") is not True:
        raise ContractError("checkpoint probe does not contain passing ONNX parity")
    evidence = QualificationEvidence.from_mapping(evidence_payload)
    verify_qualification_binding(
        evidence,
        evidence_root,
        Path(__file__).with_name("native_qualify.py"),
    )
    result = evaluate_qualification(evidence, mode)
    return evidence, {
        **parity_payload,
        "qualified": result.qualified,
        "failures": list(result.failures),
    }


def select_qualified_checkpoint(
    run_root: Path,
    checkpoint_path: Path,
    evidence_root: Path,
    later_evidence_roots: tuple[Path, ...],
) -> dict[str, object]:
    """Select the latest physically qualified checkpoint from one completed full attempt."""

    validate_run_id(run_root.name)
    contract_path = run_root / "training-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ContractError("training contract must be a JSON object")
    mode = RunMode(contract["mode"])
    if mode is not RunMode.FULL:
        raise ContractError("only a completed full run can select a qualified checkpoint")
    if contract.get("status") not in {"exported_unqualified", "qualified_checkpoint_selected"}:
        raise ContractError("training run has not completed and exported successfully")

    attempts = contract.get("attempts")
    selected_attempt_id = contract.get("selected_attempt_id")
    if not isinstance(attempts, list) or not isinstance(selected_attempt_id, str):
        raise ContractError("training contract has no selected completed attempt")
    attempt = next(
        (
            item
            for item in attempts
            if isinstance(item, dict) and item.get("attempt_id") == selected_attempt_id
        ),
        None,
    )
    if attempt is None or attempt.get("status") != "training_complete":
        raise ContractError("selected training attempt is not complete")
    attempt_relative = attempt.get("root")
    if not isinstance(attempt_relative, str):
        raise ContractError("selected training attempt has no artifact root")
    attempt_root = (run_root / attempt_relative).resolve(strict=True)
    checkpoint = checkpoint_path.resolve(strict=True)
    if not checkpoint.is_relative_to(attempt_root) or not checkpoint.name.isdigit():
        raise ContractError("selected checkpoint is outside the completed attempt")
    _validate_checkpoint_config(checkpoint)

    seed = int(contract["seed"])
    selected_step = int(checkpoint.name)
    evidence, selected_probe = _load_qualification_probe(evidence_root, mode=mode)
    if selected_probe["qualified"] is not True:
        raise ContractError("selected checkpoint did not pass full physical qualification")
    parity_checkpoint = Path(str(selected_probe.get("checkpoint", ""))).name
    if parity_checkpoint != checkpoint.name:
        raise ContractError("selected qualification probe names a different checkpoint")

    later_checkpoints = {
        path.name
        for path in checkpoint.parent.iterdir()
        if path.is_dir() and path.name.isdigit() and int(path.name) > selected_step
    }
    later_outcomes: list[dict[str, object]] = []
    provided_later: set[str] = set()
    for later_root in later_evidence_roots:
        later_evidence, later_probe = _load_qualification_probe(later_root, mode=mode)
        later_name = Path(str(later_probe.get("checkpoint", ""))).name
        if not later_name.isdigit() or int(later_name) <= selected_step:
            raise ContractError("later qualification evidence does not name a later checkpoint")
        if later_probe["qualified"] is True:
            raise ContractError("a later checkpoint passed and must be selected instead")
        provided_later.add(later_name)
        later_outcomes.append(
            {
                "checkpoint": later_name,
                "controller_onnx_sha256": later_evidence.controller_onnx_sha256,
                "failures": later_probe["failures"],
            }
        )
    if provided_later != later_checkpoints:
        raise ContractError(
            "qualification evidence must cover every checkpoint later than the selection"
        )

    source = verify_source_checkout(CHECKOUT_ROOT, PATCH_PATH, patched=True)
    _export_checkpoint(run_root, seed, checkpoint)
    if sha256_file(run_root / "controller.onnx") != evidence.controller_onnx_sha256:
        raise ContractError("fresh selected-checkpoint export differs from qualification evidence")
    shutil.copyfile(
        evidence_root / "qualification-evidence.json",
        run_root / "qualification-evidence.json",
    )
    shutil.copyfile(
        evidence_root / "qualification-trials.json",
        run_root / "qualification-trials.json",
    )
    verify_qualification_binding(
        evidence,
        run_root,
        Path(__file__).with_name("native_qualify.py"),
    )

    selected_relative = checkpoint.relative_to(run_root.resolve()).as_posix()
    previous_checkpoint, selected_at = _selection_provenance(contract, selected_relative)
    contract["exported_checkpoint"] = selected_relative
    contract["status"] = "qualified_checkpoint_selected"
    contract["checkpoint_selection"] = {
        "strategy": "latest_checkpoint_passing_full_native_qualification",
        "selected_at": selected_at,
        "selected_checkpoint": selected_relative,
        "previous_exported_checkpoint": previous_checkpoint,
        "controller_onnx_sha256": evidence.controller_onnx_sha256,
        "qualification_trials_sha256": evidence.qualification_trials_sha256,
        "later_rejected_checkpoints": sorted(
            later_outcomes,
            key=lambda item: int(str(item["checkpoint"])),
        ),
    }
    _atomic_write_json(contract_path, contract)
    qualification = evaluate_qualification(evidence, mode)
    manifest = build_artifact_manifest(
        run_root,
        mode=mode,
        seed=seed,
        source=source,
        qualification=qualification,
    )
    write_artifact_manifest(run_root, manifest)
    return manifest
