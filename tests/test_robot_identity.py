import json
from pathlib import Path

import pytest

from muscle_memory.paths import ROBOT_MANIFEST
from muscle_memory.robot.identity import CandidateBundleError, verify_candidate_bundle


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
