# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# curl backs the compose healthcheck; build-essential covers packages that
# still ship source-only wheels on arm64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are installed before the source is copied, so editing a Python
# file doesn't invalidate the (slow) dependency layer. hatchling needs the
# package directory to exist at install time, hence the stub — the editable
# install points at /srv, so the real source copied below takes over.
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install -e ".[dev]"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
