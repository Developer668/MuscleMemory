"""Stable repository and bundled-asset paths."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
THIRD_PARTY_ROOT = REPOSITORY_ROOT / "third_party" / "mujoco_playground"
PLAYGROUND_ROOT = THIRD_PARTY_ROOT / "mujoco_playground"
MENAGERIE_ROOT = THIRD_PARTY_ROOT / "mujoco_menagerie"
G1_XML_ROOT = PLAYGROUND_ROOT / "_src" / "locomotion" / "g1" / "xmls"
G1_SCENE_XML = G1_XML_ROOT / "scene_mjx_feetonly_flat_terrain.xml"
G1_POLICY_ONNX = PLAYGROUND_ROOT / "experimental" / "sim2sim" / "onnx" / "g1_policy.onnx"
ROBOT_MANIFEST = REPOSITORY_ROOT / "config" / "robot" / "mm01-candidate.json"
