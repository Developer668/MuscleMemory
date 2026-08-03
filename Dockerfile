# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY ops ./ops
COPY config ./config
COPY artifacts/policy/expert-v1.npz ./artifacts/policy/expert-v1.npz
COPY evidence ./evidence
COPY integrations ./integrations
COPY models ./models
COPY third_party ./third_party
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV HOME=/home/muscle-memory \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MM_API_HOST=0.0.0.0 \
    MM_API_PORT=8000

RUN groupadd --gid "${APP_GID}" muscle-memory \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
       --create-home --home-dir /home/muscle-memory --shell /usr/sbin/nologin \
       muscle-memory \
    && mkdir -p \
       /var/lib/muscle-memory/assets \
       /var/lib/muscle-memory/coordinator \
       /var/lib/muscle-memory/graph \
       /var/lib/muscle-memory/telemetry \
       /var/lib/muscle-memory/training \
    && chown -R "${APP_UID}:${APP_GID}" \
       /home/muscle-memory /var/lib/muscle-memory

WORKDIR /app
COPY --from=builder --chown=${APP_UID}:${APP_GID} /app /app

USER ${APP_UID}:${APP_GID}
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-m", "ops.deployment.healthcheck"]

ENTRYPOINT ["python", "-m", "ops.api.serve"]
