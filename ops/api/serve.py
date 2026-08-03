"""Load an injected backend and serve the production FastAPI application."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from typing import cast

from fastapi import FastAPI

from muscle_memory.api import ApiBackend, authenticator_from_env, create_app
from muscle_memory.training.jobs import TaskPolicyTrainingManager

BACKEND_FACTORY_ENV = "MM_API_BACKEND_FACTORY"


def load_backend(environ: Mapping[str, str] | None = None) -> ApiBackend:
    values = os.environ if environ is None else environ
    factory_path = values.get(BACKEND_FACTORY_ENV, "").strip()
    if not factory_path or ":" not in factory_path:
        raise RuntimeError(
            f"{BACKEND_FACTORY_ENV} must name a configured module:factory"
        )
    module_name, attribute_name = factory_path.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise RuntimeError(f"{BACKEND_FACTORY_ENV} must name a configured module:factory")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise RuntimeError(f"{BACKEND_FACTORY_ENV} does not resolve to a callable")
    backend = cast(Callable[[], ApiBackend], factory)()
    return backend


def create_application() -> FastAPI:
    """Uvicorn factory; secrets stay in provider/backend objects, never app state JSON."""

    return create_app(
        backend=load_backend(),
        authenticator=authenticator_from_env(),
        training_jobs=TaskPolicyTrainingManager.from_env(),
    )


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve the HTTP API") from exc
    host = os.environ.get("MM_API_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("MM_API_PORT", "8000"))
    except ValueError as exc:
        raise RuntimeError("MM_API_PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise RuntimeError("MM_API_PORT must be between 1 and 65535")
    uvicorn.run(
        "ops.api.serve:create_application",
        factory=True,
        host=host,
        port=port,
        log_level=os.environ.get("MM_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
