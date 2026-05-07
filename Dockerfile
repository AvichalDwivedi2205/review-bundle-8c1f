FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md openenv.yaml ./
COPY inference.py ./
COPY configs ./configs
COPY server ./server
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "latentgoalops.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
