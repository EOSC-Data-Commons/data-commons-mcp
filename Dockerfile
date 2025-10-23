FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim
# https://docs.astral.sh/uv/guides/integration/docker

WORKDIR /app
COPY . /app/

RUN uv sync --frozen --extra agent

ENV SERVER_PORT=8000
ENV SERVER_HOST='0.0.0.0'
ENV PYTHONUNBUFFERED='1'
EXPOSE 8000
ENTRYPOINT ["uv", "run", "uvicorn", "src.data_commons_mcp.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "6", "--log-config", "logging.yml"]
