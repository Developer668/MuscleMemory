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
MM01_CONTROLLER_ONNX = REPOSITORY_ROOT / "models" / "mm01" / "gait-controller-v1.onnx"
MM01_MANIFEST = REPOSITORY_ROOT / "config" / "robot" / "mm01-v1.json"
MM01_CONTROLLER_EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "controller" / "gait-v1"
MM01_ONNX_PARITY_EVIDENCE = MM01_CONTROLLER_EVIDENCE_ROOT / "onnx-parity.json"
MM01_QUALIFICATION_EVIDENCE = (
    MM01_CONTROLLER_EVIDENCE_ROOT / "qualification-evidence.json"
)
MM01_QUALIFICATION_TRIALS = MM01_CONTROLLER_EVIDENCE_ROOT / "qualification-trials.json"
MM01_TRAINING_CONTRACT = MM01_CONTROLLER_EVIDENCE_ROOT / "training-contract.json"
HELDOUT_WORLDS_BUNDLE = REPOSITORY_ROOT / "config" / "worlds" / "heldout-v1.json"
EXPERT_DATASET_V1 = REPOSITORY_ROOT / "artifacts" / "policy" / "expert-v1.npz"
EXPERT_DATASET_V1_METADATA = (
    REPOSITORY_ROOT / "artifacts" / "policy" / "expert-v1.metadata.json"
)
POLICY_V1_CHECKPOINT = REPOSITORY_ROOT / "models" / "policy" / "delivery-v1.npz"
POLICY_V1_TRAINING_EVIDENCE = (
    REPOSITORY_ROOT / "evidence" / "policy" / "delivery-v1" / "training.json"
)
