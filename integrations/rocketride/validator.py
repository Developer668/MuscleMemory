"""Static validation for the reviewed RocketRide pipeline bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from integrations.rocketride.protocol import FIXED_STEPS, ContractError, canonical_json

BUNDLE_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = BUNDLE_ROOT / "fixed-step.pipe"
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"
EXPECTED_ARTIFACTS = frozenset(
    {
        "README.md",
        "__init__.py",
        "callback.py",
        "envelope.schema.json",
        "fixed-step.pipe",
        "live_verify.py",
        "protocol.py",
        "result.schema.json",
        "runtime.py",
        "source-review.json",
        "validator.py",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pipeline(path: Path = PIPELINE_PATH) -> dict[str, object]:
    document = _read_json(path)
    if set(document) != {"pipeline"} or not isinstance(document["pipeline"], dict):
        raise ContractError("pipeline document must contain one RocketRide pipeline wrapper")
    pipe = cast(dict[str, Any], document["pipeline"])
    required_top = {"components", "project_id", "source", "viewport", "version"}
    if set(pipe) != required_top:
        raise ContractError("pipeline top-level fields do not match RocketRide's reviewed shape")
    if pipe["version"] != 1:
        raise ContractError("pipeline must use RocketRide document version 1")
    if pipe["project_id"] != "00000000-0000-0000-0000-000000000000":
        raise ContractError("pipeline must retain the portable reviewed project id")
    if pipe["source"] != "fixed_step_source":
        raise ContractError("pipeline source changed from the reviewed artifact")
    if pipe["viewport"] != {"x": 0, "y": 0, "zoom": 1}:
        raise ContractError("pipeline viewport changed from the reviewed artifact")
    components = pipe["components"]
    if not isinstance(components, list) or len(components) != 3:
        raise ContractError("pipeline must contain exactly three deterministic components")
    if not all(isinstance(item, dict) for item in components):
        raise ContractError("pipeline components must be JSON objects")
    typed_components = cast(list[dict[str, Any]], components)
    ids = tuple(component.get("id") for component in typed_components)
    providers = tuple(component.get("provider") for component in typed_components)
    if ids != ("fixed_step_source", "coordinator_callback", "typed_result"):
        raise ContractError("pipeline component ids or order changed")
    if providers != ("webhook", "tool_n8n", "response_text"):
        raise ContractError("pipeline may use only the reviewed deterministic providers")
    if len(set(ids)) != len(ids):
        raise ContractError("pipeline component identity is invalid")
    if any("control" in component for component in typed_components):
        raise ContractError("agent/control connections are forbidden in the execution pipe")

    source, callback, result = typed_components
    expected_ui = (
        {
            "formDataValid": True,
            "measured": {"height": 66, "width": 150},
            "nodeType": "default",
            "position": {"x": 20, "y": 200},
        },
        {
            "formDataValid": True,
            "measured": {"height": 66, "width": 150},
            "nodeType": "default",
            "position": {"x": 240, "y": 200},
        },
        {
            "formDataValid": True,
            "measured": {"height": 66, "width": 150},
            "nodeType": "default",
            "position": {"x": 460, "y": 200},
        },
    )
    if tuple(component.get("ui") for component in typed_components) != expected_ui:
        raise ContractError("pipeline UI metadata changed from RocketRide's reviewed shape")
    if "input" in source:
        raise ContractError("source component cannot consume a lane")
    if callback.get("input") != [{"from": "fixed_step_source", "lane": "text"}]:
        raise ContractError("coordinator callback lane changed")
    if result.get("input") != [{"from": "coordinator_callback", "lane": "text"}]:
        raise ContractError("typed result lane changed")

    expected_callback_config = {
        "apiKey": "",
        "asyncTimeout": 120,
        "baseUrl": "${ROCKETRIDE_MM_COORDINATOR_URL}",
        "mode": "sync",
        "parameters": {},
        "payloadMode": "simple",
        "readOnly": True,
        "syncTimeout": 120,
        "type": "tool_n8n",
        "verifyTls": True,
        "webhookAuth": "bearer",
        "webhookToken": "${ROCKETRIDE_MM_COORDINATOR_TOKEN}",
        "workflow": "muscle-memory-fixed-step",
    }
    if callback.get("config") != expected_callback_config:
        raise ContractError("callback security or delivery configuration changed")
    if result.get("config") != {"laneName": "text"}:
        raise ContractError("typed-result response configuration changed")
    source_config = source.get("config")
    if source_config != {
        "hideForm": True,
        "mode": "Source",
        "parameters": {},
        "type": "webhook",
    }:
        raise ContractError("source configuration changed")
    return {
        "component_ids": list(ids),
        "pipeline_sha256": sha256_file(path),
        "providers": list(providers),
    }


def validate_schema_documents(root: Path = BUNDLE_ROOT) -> None:
    envelope = _read_json(root / "envelope.schema.json")
    result = _read_json(root / "result.schema.json")
    for name, schema in (("envelope", envelope), ("result", result)):
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise ContractError(f"{name} schema must reject non-object or extra fields")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ContractError(f"{name} schema properties are missing")
        step = properties.get("step")
        if not isinstance(step, dict) or tuple(step.get("enum", ())) != FIXED_STEPS:
            raise ContractError(f"{name} schema step enum does not match fixed pipeline")


def validate_manifest(root: Path = BUNDLE_ROOT) -> dict[str, str]:
    manifest = _read_json(root / "manifest.json")
    if set(manifest) != {"algorithm", "artifacts", "bundle_version"}:
        raise ContractError("checksum manifest fields changed")
    if manifest["algorithm"] != "sha256" or manifest["bundle_version"] != 1:
        raise ContractError("unsupported checksum manifest")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ContractError("checksum manifest has no artifacts")
    if set(artifacts) != EXPECTED_ARTIFACTS:
        raise ContractError("checksum manifest does not cover the exact reviewed artifact set")
    checked: dict[str, str] = {}
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ContractError("checksum manifest entries must be string pairs")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ContractError("checksum manifest path escapes the bundle")
        artifact = root / relative
        if not artifact.is_file():
            raise ContractError(f"manifest artifact is missing: {relative}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise ContractError(f"manifest checksum mismatch: {relative}")
        checked[relative] = actual
    return checked


def validate_bundle(root: Path = BUNDLE_ROOT) -> dict[str, object]:
    pipeline = validate_pipeline(root / "fixed-step.pipe")
    validate_schema_documents(root)
    artifacts = validate_manifest(root)
    review = _read_json(root / "source-review.json")
    source_commit = review.get("official_source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ContractError("source review does not pin an official commit")
    return {
        "artifacts": artifacts,
        "bundle_sha256": hashlib.sha256(canonical_json(artifacts).encode()).hexdigest(),
        "pipeline": pipeline,
        "valid": True,
    }


def main() -> None:
    print(json.dumps(validate_bundle(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
