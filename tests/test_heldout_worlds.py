"""Frozen-split integrity and teacher-isolation checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from muscle_memory.evaluation.heldout import (
    HELDOUT_WORLD_COUNT,
    HeldOutBundleError,
    load_heldout_worlds,
)
from muscle_memory.paths import HELDOUT_WORLDS_BUNDLE
from muscle_memory.simulation.world_scene import assemble_episode_scene
from muscle_memory.worlds.generation import validate_training_world


def test_frozen_heldout_split_has_twenty_physically_qualified_worlds() -> None:
    worlds = load_heldout_worlds()

    assert len(worlds) == HELDOUT_WORLD_COUNT
    assert len({item.world.world_id for item in worlds}) == HELDOUT_WORLD_COUNT
    assert len({item.world.seed for item in worlds}) == HELDOUT_WORLD_COUNT
    assert all(item.world.split == "held_out" for item in worlds)
    assert all(item.certificate.physical_qualification.success for item in worlds)
    assert min(
        item.certificate.physical_qualification.minimum_obstacle_clearance_metres
        for item in worlds
    ) >= 0.25
    assert max(
        item.certificate.physical_qualification.maximum_tray_tilt_degrees
        for item in worlds
    ) < 12.0


def test_evaluation_loader_cannot_import_teacher_or_training_package() -> None:
    audit = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; from muscle_memory.evaluation.heldout import "
            "load_heldout_worlds; worlds=load_heldout_worlds(); assert len(worlds)==20; "
            "assert not any(name.startswith('muscle_memory.worlds.generation') "
            "for name in sys.modules); assert not any(name.startswith('muscle_memory.training') "
            "for name in sys.modules)",
        ),
        capture_output=True,
        check=False,
        text=True,
    )

    assert audit.returncode == 0, audit.stderr


def test_heldout_world_cannot_enter_training_validation() -> None:
    heldout = load_heldout_worlds()[0]

    with pytest.raises(TypeError, match="only TrainingWorld"):
        validate_training_world(heldout.world)  # type: ignore[arg-type]


def test_heldout_envelope_assembles_without_exposing_teacher_path() -> None:
    heldout = load_heldout_worlds()[0]
    scene = assemble_episode_scene(heldout)

    assert scene.world == heldout.world
    assert not hasattr(heldout, "baseline_path")
    assert not hasattr(heldout, "expert_path")


def test_heldout_bundle_tampering_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(HELDOUT_WORLDS_BUNDLE.read_text(encoding="utf-8"))
    payload["records"][0]["world"]["destination"]["x"] += 0.1
    tampered = tmp_path / "heldout-tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HeldOutBundleError, match="aggregate hash mismatch"):
        load_heldout_worlds(tampered)
