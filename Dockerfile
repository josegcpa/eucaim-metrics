FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY uv.lock pyproject.toml README.md /app/

RUN uv sync --frozen --no-install-project --no-dev --no-cache

COPY eucaim_eval /app/eucaim_eval
RUN uv sync --frozen --no-dev --no-cache
RUN uv clean

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT ["uv", "run"]