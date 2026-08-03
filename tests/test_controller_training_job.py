from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ops.controller.contract import (  # noqa: E402
    CONTROLLER_HZ,
    CTRL_DT_SECONDS,
    EPISODE_LENGTH,
    PATCH_SHA256,
    PATCHED_JOYSTICK_SHA256,
    PHYSICAL_EPISODE_SECONDS,
    PHYSICS_HZ,
    PLAYGROUND_COMMIT,
    TRAINING_PLANS,
    UNPATCHED_SOURCE_SHA256,
    ContractError,
    QualificationEvidence,
    RunMode,
    SourceVerification,
    build_artifact_manifest,
    evaluate_qualification,
    sha256_file,
    verify_qualification_binding,
    verify_source_checkout,
)
from ops.controller.export_onnx import (  # noqa: E402
    ACTION_SIZE,
    OBSERVATION_SIZE,
    POLICY_LAYER_SIZES,
    _resolve_checkpoint,
    build_policy_model,
    latest_checkpoint,
    numpy_policy,
    verify_onnx_parity,
)
from ops.controller.remote_job import (  # noqa: E402
    _new_contract,
    _recover_attempts,
    _training_command,
    validate_run_id,
)

PATCH_PATH = REPOSITORY_ROOT / "ops" / "controller" / "mm01_g1_100hz.patch"
MODAL_ENTRYPOINT = REPOSITORY_ROOT / "ops" / "controller" / "modal_train.py"
LOCAL_ENTRYPOINT = REPOSITORY_ROOT / "ops" / "controller" / "local_train.py"
REMOTE_JOB = REPOSITORY_ROOT / "ops" / "controller" / "remote_job.py"
NATIVE_QUALIFIER = REPOSITORY_ROOT / "ops" / "controller" / "native_qualify.py"
VENDORED_JOYSTICK = (
    REPOSITORY_ROOT
    / "third_party"
    / "mujoco_playground"
    / "mujoco_playground"
    / "_src"
    / "locomotion"
    / "g1"
    / "joystick.py"
)


@pytest.fixture
def passing_evidence() -> QualificationEvidence:
    return QualificationEvidence(
        controller_hz=CONTROLLER_HZ,
        ctrl_dt_seconds=CTRL_DT_SECONDS,
        physics_hz=PHYSICS_HZ,
        task_command_outputs=3,
        gait_action_outputs=29,
        finite_state=True,
        fall_count=0,
        body_collision_count=0,
        forward_progress_metres=2.0,
        forward_cross_track_error_metres=0.1,
        forward_heading_error_degrees=5.0,
        left_turn_error_degrees=5.0,
        right_turn_error_degrees=5.0,
        stop_speed_metres_per_second=0.01,
        standstill_duration_seconds=60.0,
        standstill_drift_metres=0.1,
        payload_stop_speed_metres_per_second=0.01,
        maximum_payload_tray_tilt_degrees=5.0,
        payload_package_slipped=False,
        deterministic_repeat_max_metric_delta=1e-6,
        robot_checksum="frozen-mm01-checksum",
        controller_onnx_sha256="controller-checksum",
        qualification_program_sha256="qualifier-checksum",
        qualification_trials_sha256="trials-checksum",
    )


def test_patch_is_exact_and_applies_to_pinned_joystick(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    target = checkout / "mujoco_playground" / "_src" / "locomotion" / "g1" / "joystick.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(VENDORED_JOYSTICK, target)

    assert sha256_file(PATCH_PATH) == PATCH_SHA256
    assert sha256_file(target) == UNPATCHED_SOURCE_SHA256[target.relative_to(checkout).as_posix()]
    subprocess.run(("git", "apply", "--check", str(PATCH_PATH)), cwd=checkout, check=True)
    subprocess.run(("git", "apply", str(PATCH_PATH)), cwd=checkout, check=True)

    assert sha256_file(target) == PATCHED_JOYSTICK_SHA256


def test_modal_copy_path_matches_ops_import_namespace() -> None:
    entrypoint = MODAL_ENTRYPOINT.read_text(encoding="utf-8")
    remote_job = REMOTE_JOB.read_text(encoding="utf-8")

    assert 'REMOTE_CONTROLLER_OPS = "/opt/mm/ops/controller"' in entrypoint
    assert '"/opt/mm/ops/controller/mm01_g1_100hz.patch"' in remote_job
    assert entrypoint.count('gpu="L4"') == 3
    assert entrypoint.count("artifacts.reload()") == 3
    assert "train.remote(selected_mode.value, seed, stable_run_id)" in entrypoint
    assert "ops.controller.export_onnx import" not in remote_job

    import_audit = subprocess.run(
        (sys.executable, "-S", "-c", "import ops.controller.remote_job"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_audit.returncode == 0, import_audit.stderr


def test_local_entrypoint_uses_explicit_backend_and_runtime_paths(tmp_path: Path) -> None:
    source = LOCAL_ENTRYPOINT.read_text(encoding="utf-8")
    assert 'execution_backend="local-macos-cpu"' in source
    assert 'os.environ["MM_CONTROLLER_CHECKOUT"]' in source
    remote_source = REMOTE_JOB.read_text(encoding="utf-8")
    assert '"cgl" if execution_backend == "local-macos-cpu" else "egl"' in remote_source
    contract = _new_contract(
        "g1-100hz-full-seed-1-20260803T070102Z",
        RunMode.FULL,
        1,
        SourceVerification(
            playground_commit=PLAYGROUND_COMMIT,
            playground_tag_commit=PLAYGROUND_COMMIT,
            menagerie_commit="1b86ece576591213e2b666ebf59508454200ca97",
            patch_sha256=PATCH_SHA256,
            joystick_sha256=PATCHED_JOYSTICK_SHA256,
            patched=True,
        ),
        "local-macos-cpu",
    )
    assert contract["execution_backend"] == "local-macos-cpu"

    environment = os.environ.copy()
    environment.update(
        {
            "MM_CONTROLLER_CHECKOUT": str(tmp_path / "checkout"),
            "MM_CONTROLLER_PATCH": str(tmp_path / "controller.patch"),
            "MM_CONTROLLER_PYTHON": str(tmp_path / "python"),
        }
    )
    audit = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            "from ops.controller.remote_job import CHECKOUT_ROOT, PATCH_PATH, VENV_PYTHON; "
            "print(CHECKOUT_ROOT, PATCH_PATH, VENV_PYTHON, sep='\\n')",
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert audit.returncode == 0, audit.stderr
    assert audit.stdout.splitlines() == [
        str(tmp_path / "checkout"),
        str(tmp_path / "controller.patch"),
        str(tmp_path / "python"),
    ]


def test_patch_changes_only_rate_and_documented_phase_hold() -> None:
    patch = PATCH_PATH.read_text(encoding="utf-8")
    changed_lines = tuple(
        line
        for line in patch.splitlines()
        if line.startswith(("+", "-")) and line[0:3] not in {"+++", "---"}
    )

    assert "-      ctrl_dt=0.02," in changed_lines
    assert "+      ctrl_dt=0.01," in changed_lines
    assert "-      episode_length=1000," in changed_lines
    assert "+      episode_length=2000," in changed_lines
    assert "+    state.info[\"phase\"] = jp.where(" in changed_lines
    assert all("reward" not in line.lower() for line in changed_lines)
    assert all("sensor" not in line.lower() for line in changed_lines)


def test_smoke_stops_after_checkpoint_before_upstream_rendering() -> None:
    remote_job = REMOTE_JOB.read_text(encoding="utf-8")

    assert 'line.rstrip() == "Done training."' in remote_job
    assert "mode is RunMode.SMOKE" in remote_job
    assert "process.send_signal(signal.SIGTERM)" in remote_job
    assert '"-m",\n        "ops.controller.export_onnx"' in remote_job


def test_native_qualification_cannot_import_world_generator_or_path_teacher() -> None:
    source = NATIVE_QUALIFIER.read_text(encoding="utf-8")
    assert "muscle_memory.worlds.generation" not in source
    audit = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import ops.controller.native_qualify; "
            "assert not any(name.startswith('muscle_memory.worlds.generation') "
            "for name in sys.modules)",
        ),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert audit.returncode == 0, audit.stderr


def test_both_modes_preserve_twenty_second_physical_horizon() -> None:
    assert EPISODE_LENGTH * CTRL_DT_SECONDS == PHYSICAL_EPISODE_SECONDS
    for plan in TRAINING_PLANS.values():
        assert plan.episode_length == 2000
        assert "--episode_length=2000" in plan.cli_overrides()


def test_full_mode_keeps_upstream_g1_ppo_configuration() -> None:
    plan = TRAINING_PLANS[RunMode.FULL]

    assert plan.num_timesteps == 200_000_000
    assert plan.num_envs == 8192
    assert plan.num_evals == 20
    assert plan.cli_overrides() == ("--episode_length=2000",)


def test_full_attempt_is_fresh_and_cannot_claim_parameter_only_resume(tmp_path: Path) -> None:
    command = _training_command(RunMode.FULL, 11, tmp_path / "attempt-0001")

    assert not any("load_checkpoint" in argument for argument in command)
    assert "--episode_length=2000" in command
    assert not any("num_timesteps" in argument for argument in command)


def test_run_id_rejects_traversal_and_cross_plan_reuse() -> None:
    valid = "g1-100hz-full-seed-7-20260803T070102Z"
    validate_run_id(valid, mode=RunMode.FULL, seed=7)

    with pytest.raises(ContractError, match="naming contract"):
        validate_run_id("../full-controller")
    with pytest.raises(ContractError, match="mode differs"):
        validate_run_id(valid, mode=RunMode.SMOKE, seed=7)
    with pytest.raises(ContractError, match="seed differs"):
        validate_run_id(valid, mode=RunMode.FULL, seed=8)


def test_interrupted_attempt_is_retained_and_never_selected(tmp_path: Path) -> None:
    source = SourceVerification(
        playground_commit=PLAYGROUND_COMMIT,
        playground_tag_commit=PLAYGROUND_COMMIT,
        menagerie_commit="1b86ece576591213e2b666ebf59508454200ca97",
        patch_sha256=PATCH_SHA256,
        joystick_sha256=PATCHED_JOYSTICK_SHA256,
        patched=True,
    )
    run_id = "g1-100hz-full-seed-1-20260803T070102Z"
    contract = _new_contract(run_id, RunMode.FULL, 1, source)
    attempt_root = tmp_path / "attempts" / "attempt-0001"
    attempt_root.mkdir(parents=True)
    log_path = attempt_root / "training.log"
    log_path.write_text("training began\n", encoding="utf-8")
    contract["attempts"] = [
        {
            "attempt_id": "attempt-0001",
            "root": "attempts/attempt-0001",
            "log": "attempts/attempt-0001/training.log",
            "status": "running",
        }
    ]

    recovered = _recover_attempts(tmp_path, contract)

    assert recovered is None
    attempt = contract["attempts"][0]
    assert attempt["status"] == "interrupted"
    assert "next attempt starts the complete plan fresh" in attempt["interruption_disposition"]
    assert contract["selected_attempt_id"] is None


def test_completed_attempt_is_recovered_for_export_without_retraining(tmp_path: Path) -> None:
    source = SourceVerification(
        playground_commit=PLAYGROUND_COMMIT,
        playground_tag_commit=PLAYGROUND_COMMIT,
        menagerie_commit="1b86ece576591213e2b666ebf59508454200ca97",
        patch_sha256=PATCH_SHA256,
        joystick_sha256=PATCHED_JOYSTICK_SHA256,
        patched=True,
    )
    run_id = "g1-100hz-full-seed-1-20260803T070102Z"
    contract = _new_contract(run_id, RunMode.FULL, 1, source)
    attempt_root = tmp_path / "attempts" / "attempt-0001"
    checkpoint = attempt_root / "experiment" / "checkpoints" / "200000000"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "config.json").write_text(
        json.dumps({"ctrl_dt": 0.01, "episode_length": 2000}),
        encoding="utf-8",
    )
    log_path = attempt_root / "training.log"
    log_path.write_text("Done training.\nrendering interrupted\n", encoding="utf-8")
    contract["attempts"] = [
        {
            "attempt_id": "attempt-0001",
            "root": "attempts/attempt-0001",
            "log": "attempts/attempt-0001/training.log",
            "status": "running",
        }
    ]

    recovered = _recover_attempts(tmp_path, contract)

    assert recovered == checkpoint
    assert contract["attempts"][0]["status"] == "training_complete"
    assert contract["attempts"][0]["recovered_after_restart"] is True
    assert contract["selected_attempt_id"] == "attempt-0001"


def test_latest_checkpoint_searches_nested_attempts_by_step(tmp_path: Path) -> None:
    older = tmp_path / "attempts" / "attempt-0001" / "run" / "checkpoints" / "100"
    newer = tmp_path / "attempts" / "attempt-0002" / "run" / "checkpoints" / "200"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    assert latest_checkpoint(tmp_path) == newer


def test_checkpoint_resolution_preserves_a_symlinked_volume_mount(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    checkpoint = backing / "attempt" / "run" / "checkpoints" / "200"
    checkpoint.mkdir(parents=True)
    mount = tmp_path / "artifacts"
    mount.symlink_to(backing, target_is_directory=True)
    logical_checkpoint = mount / checkpoint.relative_to(backing)

    resolved = _resolve_checkpoint(mount, logical_checkpoint)

    assert resolved == logical_checkpoint
    assert resolved.relative_to(mount) == Path("attempt/run/checkpoints/200")


def test_mode_overrides_cannot_change_rewards_or_networks() -> None:
    forbidden = ("reward", "learning_rate", "hidden_layer", "policy_obs", "value_obs")
    for plan in TRAINING_PLANS.values():
        assert not any(term in argument for term in forbidden for argument in plan.cli_overrides())


def test_source_verification_fails_closed_on_wrong_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("ops.controller.contract.sha256_file", lambda _: PATCH_SHA256)
    monkeypatch.setattr("ops.controller.contract._git_output", lambda *_: "wrong-commit")

    with pytest.raises(ContractError, match="Playground commit mismatch"):
        verify_source_checkout(tmp_path, PATCH_PATH, patched=False)


def test_full_evidence_can_qualify(passing_evidence: QualificationEvidence) -> None:
    result = evaluate_qualification(passing_evidence, RunMode.FULL)

    assert result.qualified
    assert result.failures == ()


def test_smoke_evidence_never_qualifies(passing_evidence: QualificationEvidence) -> None:
    result = evaluate_qualification(passing_evidence, RunMode.SMOKE)

    assert not result.qualified
    assert "smoke training cannot qualify a controller" in result.failures


def test_qualification_reports_every_failed_gate(
    passing_evidence: QualificationEvidence,
) -> None:
    failing = replace(
        passing_evidence,
        controller_hz=50,
        physics_hz=250,
        task_command_outputs=4,
        gait_action_outputs=30,
        finite_state=False,
        fall_count=1,
        body_collision_count=1,
        forward_progress_metres=0.5,
        forward_cross_track_error_metres=0.5,
        forward_heading_error_degrees=30.0,
        left_turn_error_degrees=30.0,
        right_turn_error_degrees=30.0,
        stop_speed_metres_per_second=0.2,
        standstill_duration_seconds=59.0,
        standstill_drift_metres=0.5,
        payload_stop_speed_metres_per_second=0.2,
        maximum_payload_tray_tilt_degrees=12.0,
        payload_package_slipped=True,
        deterministic_repeat_max_metric_delta=0.1,
        robot_checksum="",
        controller_onnx_sha256="",
        qualification_program_sha256="",
        qualification_trials_sha256="",
    )

    result = evaluate_qualification(failing, RunMode.FULL)

    assert not result.qualified
    assert len(result.failures) == 23


def test_evidence_rejects_missing_or_extra_fields(
    passing_evidence: QualificationEvidence,
) -> None:
    payload = asdict(passing_evidence)
    payload.pop("robot_checksum")
    payload["unexpected"] = True

    with pytest.raises(ContractError, match=r"missing=.*robot_checksum.*extra=.*unexpected"):
        QualificationEvidence.from_mapping(payload)


def test_qualification_binding_rejects_evidence_for_other_artifacts(
    passing_evidence: QualificationEvidence,
    tmp_path: Path,
) -> None:
    controller = tmp_path / "controller.onnx"
    qualifier = tmp_path / "native_qualify.py"
    trials = tmp_path / "qualification-trials.json"
    controller.write_bytes(b"controller")
    qualifier.write_bytes(b"qualifier")
    trials.write_bytes(b"trials")
    bound = replace(
        passing_evidence,
        controller_onnx_sha256=sha256_file(controller),
        qualification_program_sha256=sha256_file(qualifier),
        qualification_trials_sha256=sha256_file(trials),
    )
    verify_qualification_binding(bound, tmp_path, qualifier)

    controller.write_bytes(b"different controller")
    with pytest.raises(ContractError, match="controller artifact checksum"):
        verify_qualification_binding(bound, tmp_path, qualifier)


def test_artifact_manifest_hashes_files_and_records_episode_length(
    passing_evidence: QualificationEvidence,
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoint.bin").write_bytes(b"checkpoint")
    source = SourceVerification(
        playground_commit=PLAYGROUND_COMMIT,
        playground_tag_commit=PLAYGROUND_COMMIT,
        menagerie_commit="1b86ece576591213e2b666ebf59508454200ca97",
        patch_sha256=PATCH_SHA256,
        joystick_sha256=PATCHED_JOYSTICK_SHA256,
        patched=True,
    )
    result = evaluate_qualification(passing_evidence, RunMode.FULL)

    manifest = build_artifact_manifest(
        tmp_path,
        mode=RunMode.FULL,
        seed=7,
        source=source,
        qualification=result,
    )

    assert manifest["episode_length"] == 2000
    assert manifest["physical_episode_seconds"] == 20.0
    assert manifest["qualification"] == {"qualified": True, "failures": ()}
    assert manifest["files"] == [
        {
            "path": "checkpoint.bin",
            "size_bytes": 10,
            "sha256": "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
        }
    ]
    json.dumps(manifest)


def test_exported_onnx_matches_independent_policy_math(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    mean = rng.normal(size=OBSERVATION_SIZE).astype(np.float32)
    std = rng.uniform(0.5, 2.0, size=OBSERVATION_SIZE).astype(np.float32)
    layers = []
    input_size = OBSERVATION_SIZE
    for output_size in POLICY_LAYER_SIZES:
        layers.append(
            (
                rng.normal(0.0, 0.03, size=(input_size, output_size)).astype(np.float32),
                rng.normal(0.0, 0.03, size=output_size).astype(np.float32),
            )
        )
        input_size = output_size
    frozen_layers = tuple(layers)
    observations = rng.normal(size=(8, OBSERVATION_SIZE)).astype(np.float32)
    expected = numpy_policy(observations, mean, std, frozen_layers)
    model_path = tmp_path / "controller.onnx"
    onnx.save_model(build_policy_model(mean, std, frozen_layers), model_path)

    maximum_delta = verify_onnx_parity(model_path, observations, expected)
    session = ort.InferenceSession(model_path.as_posix(), providers=("CPUExecutionProvider",))

    assert maximum_delta <= 1e-5
    assert session.get_inputs()[0].name == "obs"
    assert session.get_inputs()[0].shape == [None, OBSERVATION_SIZE]
    assert session.get_outputs()[0].name == "continuous_actions"
    assert session.get_outputs()[0].shape == [None, ACTION_SIZE]
