"""Run one real RocketRide task through the authenticated callback pipeline."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from integrations.rocketride.protocol import (
    ContractError,
    StepEnvelope,
    canonical_json,
    sha256_text,
    validate_result,
)
from integrations.rocketride.validator import BUNDLE_ROOT, validate_bundle


@dataclass(frozen=True, slots=True)
class LiveVerificationConfig:
    uri: str
    api_key: str
    coordinator_url: str
    coordinator_token: str
    envelope_path: Path
    pipeline_path: Path = BUNDLE_ROOT / "fixed-step.pipe"
    request_timeout_ms: float = 120_000.0

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> tuple[LiveVerificationConfig | None, tuple[str, ...]]:
        values = os.environ if environ is None else environ
        required = {
            "ROCKETRIDE_URI": values.get("ROCKETRIDE_URI", "").strip(),
            "ROCKETRIDE_APIKEY": values.get("ROCKETRIDE_APIKEY", "").strip(),
            "ROCKETRIDE_MM_COORDINATOR_URL": values.get(
                "ROCKETRIDE_MM_COORDINATOR_URL", ""
            ).strip(),
            "ROCKETRIDE_MM_COORDINATOR_TOKEN": values.get(
                "ROCKETRIDE_MM_COORDINATOR_TOKEN", ""
            ).strip(),
            "ROCKETRIDE_VERIFY_ENVELOPE_FILE": values.get(
                "ROCKETRIDE_VERIFY_ENVELOPE_FILE", ""
            ).strip(),
        }
        missing = tuple(name for name, value in required.items() if not value)
        if missing:
            return None, missing
        timeout_value = values.get("ROCKETRIDE_REQUEST_TIMEOUT_MS", "120000")
        try:
            timeout = float(timeout_value)
        except ValueError as exc:
            raise ContractError("ROCKETRIDE_REQUEST_TIMEOUT_MS must be numeric") from exc
        config = cls(
            uri=required["ROCKETRIDE_URI"],
            api_key=required["ROCKETRIDE_APIKEY"],
            coordinator_url=required["ROCKETRIDE_MM_COORDINATOR_URL"],
            coordinator_token=required["ROCKETRIDE_MM_COORDINATOR_TOKEN"],
            envelope_path=Path(required["ROCKETRIDE_VERIFY_ENVELOPE_FILE"]),
            request_timeout_ms=timeout,
        )
        config.validate()
        return config, ()

    def validate(self) -> None:
        if not self.uri.startswith(("https://", "wss://")):
            raise ContractError("live RocketRide verification requires HTTPS or WSS")
        if not self.coordinator_url.startswith("https://"):
            raise ContractError("live coordinator callback requires HTTPS")
        if len(self.coordinator_token) < 32:
            raise ContractError("coordinator callback token must contain at least 32 characters")
        if self.request_timeout_ms <= 0:
            raise ContractError("RocketRide request timeout must be positive")
        if not self.envelope_path.is_file():
            raise ContractError("verification envelope file does not exist")
        if not self.pipeline_path.is_file():
            raise ContractError("reviewed RocketRide pipeline file does not exist")


def _client_factory() -> Callable[..., Any]:
    try:
        module = importlib.import_module("rocketride")
    except ImportError as exc:
        raise RuntimeError(
            "official RocketRide SDK is missing; install the `rocketride` package"
        ) from exc
    return cast(Callable[..., Any], module.RocketRideClient)


def _validation_errors(value: object) -> list[object]:
    if not isinstance(value, Mapping):
        return []
    errors: list[object] = []
    if value.get("valid") is False or value.get("success") is False:
        errors.append(dict(value))
    direct = value.get("errors")
    if isinstance(direct, list):
        errors.extend(direct)
    nested = value.get("result")
    if isinstance(nested, Mapping):
        errors.extend(_validation_errors(nested))
    return errors


def _extract_result(response: object) -> str:
    if isinstance(response, Mapping):
        required = {
            "contract_version",
            "output",
            "output_sha256",
            "plan_digest",
            "request_sha256",
            "run_id",
            "status",
            "step",
        }
        if set(response) == required:
            return canonical_json(dict(response))
        for key in ("result", "text", "table", "json"):
            if key not in response:
                continue
            candidate = response[key]
            if isinstance(candidate, list) and len(candidate) == 1:
                candidate = candidate[0]
            if isinstance(candidate, Mapping):
                return canonical_json(dict(candidate))
            if isinstance(candidate, str):
                try:
                    decoded = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    return canonical_json(decoded)
    raise ContractError("RocketRide response does not contain one typed callback result")


async def verify_live_provider(
    config: LiveVerificationConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, object]]:
    if config is None:
        try:
            config, missing = LiveVerificationConfig.from_env(environ)
        except ContractError as exc:
            return 2, {
                "provider": "rocketride.ai",
                "state": "unconfigured",
                "verified": False,
                "detail": str(exc),
            }
        if config is None:
            return 2, {
                "provider": "rocketride.ai",
                "state": "unconfigured",
                "verified": False,
                "missing": list(missing),
                "detail": "required live verification configuration is absent",
            }
    try:
        config.validate()
        bundle = validate_bundle(config.pipeline_path.parent)
        encoded_envelope = config.envelope_path.read_text(encoding="utf-8")
        envelope = StepEnvelope.parse(encoded_envelope)
        factory = client_factory or _client_factory()
    except RuntimeError as exc:
        return 3, {
            "provider": "rocketride.ai",
            "state": "dependency_missing",
            "verified": False,
            "detail": str(exc),
        }
    except (ContractError, OSError) as exc:
        return 2, {
            "provider": "rocketride.ai",
            "state": "unconfigured",
            "verified": False,
            "detail": str(exc),
        }

    token = ""
    validation: object = None
    try:
        client = factory(
            uri=config.uri,
            auth=config.api_key,
            env={
                "ROCKETRIDE_MM_COORDINATOR_URL": config.coordinator_url,
                "ROCKETRIDE_MM_COORDINATOR_TOKEN": config.coordinator_token,
            },
            request_timeout=config.request_timeout_ms,
            persist=False,
            module="muscle-memory-live-verification",
        )
        async with client:
            pipeline = json.loads(config.pipeline_path.read_text(encoding="utf-8"))
            validation = await client.validate(pipeline)
            errors = _validation_errors(validation)
            if errors:
                raise ContractError(f"RocketRide rejected pipeline: {errors}")
            task = await client.use(filepath=str(config.pipeline_path))
            token_value = task.get("token") if isinstance(task, Mapping) else None
            if not isinstance(token_value, str) or not token_value:
                raise ContractError("RocketRide use() returned no task token")
            token = token_value
            try:
                response = await client.send(
                    token,
                    encoded_envelope,
                    objinfo={"name": f"{envelope.step}.json"},
                    mimetype="application/json",
                )
            finally:
                await client.terminate(token)
        result_json = _extract_result(response)
        result = validate_result(result_json, envelope)
        request_sha256 = sha256_text(encoded_envelope)
        if result["request_sha256"] != request_sha256:
            raise ContractError("callback result request checksum mismatch")
        return 0, {
            "provider": "rocketride.ai",
            "mode": "live",
            "state": "end_to_end_verified",
            "verified": True,
            "run_id": envelope.run_id,
            "step": envelope.step,
            "task_token": token,
            "pipeline_sha256": cast(dict[str, Any], bundle["pipeline"])[
                "pipeline_sha256"
            ],
            "request_sha256": request_sha256,
            "result_sha256": sha256_text(result_json),
            "output_sha256": result["output_sha256"],
            "detail": "real RocketRide task returned a validated coordinator result",
        }
    except Exception as exc:
        return 1, {
            "provider": "rocketride.ai",
            "mode": "live",
            "state": "unhealthy",
            "verified": False,
            "task_token": token or None,
            "pipeline_sha256": cast(dict[str, Any], bundle["pipeline"])[
                "pipeline_sha256"
            ],
            "detail": str(exc),
            "pipeline_validation_received": validation is not None,
        }


def main() -> None:
    import asyncio

    exit_code, evidence = asyncio.run(verify_live_provider())
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
