"""Authenticated HTTP surface used by the deterministic RocketRide lane."""

from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from integrations.rocketride.protocol import (
    ApprovalRejectedError,
    ContractError,
    FixedStepDispatcher,
    HandlerExecutionError,
    SequenceError,
    canonical_json,
)

CALLBACK_PATH = "/webhook/muscle-memory-fixed-step"
MAX_BODY_BYTES = 1_100_000


def make_callback_server(
    dispatcher: FixedStepDispatcher,
    *,
    bearer_token: str,
    host: str = "127.0.0.1",
    port: int = 8788,
) -> ThreadingHTTPServer:
    """Build a callback server; callers retain lifecycle ownership."""

    if len(bearer_token) < 32:
        raise ValueError("callback bearer token must contain at least 32 characters")

    class CallbackHandler(BaseHTTPRequestHandler):
        server_version = "MuscleMemoryRocketRideCallback/1"

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(
                HTTPStatus.OK,
                {
                    "service": "muscle-memory-rocketride-callback",
                    "state": "healthy",
                },
            )

        def do_POST(self) -> None:
            if self.path != CALLBACK_PATH:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            expected = f"Bearer {bearer_token}"
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, expected):
                self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if self.headers.get_content_type() != "application/json":
                self._write_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "content_type_must_be_application_json"},
                )
                return
            raw_length = self.headers.get("Content-Length", "")
            try:
                content_length = int(raw_length)
            except ValueError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                return
            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                self._write_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_too_large"})
                return
            body = self.rfile.read(content_length)
            try:
                wrapper = json.loads(body)
                if not isinstance(wrapper, dict) or set(wrapper) != {"data"}:
                    raise ContractError("callback body must contain only the data field")
                encoded_envelope = wrapper["data"]
                if not isinstance(encoded_envelope, str):
                    raise ContractError("callback data field must be a string")
                result_json = dispatcher.dispatch(encoded_envelope)
            except ApprovalRejectedError as exc:
                self._write_error(HTTPStatus.FORBIDDEN, "approval_rejected", exc)
                return
            except SequenceError as exc:
                self._write_error(HTTPStatus.CONFLICT, "sequence_violation", exc)
                return
            except ContractError as exc:
                self._write_error(HTTPStatus.UNPROCESSABLE_ENTITY, "contract_violation", exc)
                return
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._write_error(HTTPStatus.BAD_REQUEST, "invalid_json", exc)
                return
            except HandlerExecutionError as exc:
                self._write_error(HTTPStatus.BAD_GATEWAY, "handler_failed", exc)
                return
            encoded = result_json.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _write_error(
            self,
            status: HTTPStatus,
            code: str,
            exc: BaseException,
        ) -> None:
            self._write_json(status, {"error": code, "detail": str(exc)})

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = canonical_json(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return ThreadingHTTPServer((host, port), CallbackHandler)
