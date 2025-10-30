# syntax=docker/dockerfile:1
FROM python:3.11.9-slim AS builder
ENV VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=1
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY pyproject.toml README.md .
COPY app ./app
COPY the_assistant ./the_assistant
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .

FROM python:3.11.9-slim AS runtime
ENV VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=1 \
    PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
WORKDIR /app
COPY app/config ./app/config
COPY artifacts ./artifacts
RUN adduser --disabled-password --gecos "" genie
USER genie
ENV GENIE_LAMP_HOME=/app \
    GENIE_LAMP_CONFIG=/app/config/default.yaml \
    GENIE_LAMP_HOST=0.0.0.0 \
    GENIE_LAMP_PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl --fail http://127.0.0.1:7860/health || exit 1
CMD ["python", "-m", "genie_lamp", "serve", "--host", "0.0.0.0", "--port", "7860"]
