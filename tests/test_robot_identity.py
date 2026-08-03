import json
from pathlib import Path

import pytest

import muscle_memory.robot.identity as identity
from muscle_memory.paths import MM01_MANIFEST, ROBOT_MANIFEST
from muscle_memory.robot.identity import (
    MM01_COMMAND_SOURCE,
    CandidateBundleError,
    robot_bundle_checksum,
    verify_candidate_bundle,
    verify_mm01_bundle,
)


def test_vendored_candidate_bundle_matches_manifest() -> None:
    result = verify_candidate_bundle()

    assert result.valid is True
    assert result.qualified is False
    assert result.file_count > 30
    assert any("50 Hz" in blocker for blocker in result.blockers)
    assert any("stopping" in blocker for blocker in result.blockers)


def test_manifest_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(ROBOT_MANIFEST.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    manifest = tmp_path / "tampered.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateBundleError, match="aggregate hash mismatch"):
        verify_candidate_bundle(manifest_path=manifest)


def test_qualification_claim_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(ROBOT_MANIFEST.read_text(encoding="utf-8"))
    payload["qualification"]["qualified"] = True
    manifest = tmp_path / "false-qualification.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateBundleError, match="cannot claim qualification"):
        verify_candidate_bundle(manifest_path=manifest)


def test_qualified_mm01_bundle_matches_physical_evidence() -> None:
    result = verify_mm01_bundle()

    assert result.valid is True
    assert result.qualified is True
    assert result.robot_id == "MM-01"
    assert result.controller_id == "gait-controller-v1"
    assert result.file_count == 15
    assert result.selected_checkpoint.endswith("000181043200")
    assert result.blockers == ()


def test_qualified_manifest_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(MM01_MANIFEST.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    manifest = tmp_path / "tampered-qualified.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateBundleError, match="aggregate hash mismatch"):
        verify_mm01_bundle(manifest_path=manifest)


def test_robot_checksum_covers_task_command_source(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[Path] = []
    original = identity._file_sha256

    def capture(path: Path) -> str:
        observed.append(path.resolve())
        return original(path)

    monkeypatch.setattr(identity, "_file_sha256", capture)
    robot_bundle_checksum(identity.MM01_CONTROLLER_ONNX)

    assert MM01_COMMAND_SOURCE.resolve() in observed
