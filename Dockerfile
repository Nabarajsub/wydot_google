# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG APP_FILE=flash_cloud_2.5rpo_login.py
ARG REQS_FILE=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Per-app deps
COPY ${REQS_FILE} /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    pip install python-dotenv streamlit

# Source
COPY . /app

# Non-root
RUN useradd -m appuser
USER appuser

ENV PORT=8080
ENV APP_FILE=${APP_FILE}
ENV DOTENV_PATH=/etc/secrets/.env
ENV SECRETS_TOML_PATH=/etc/secrets/streamlit/secrets.toml

CMD ["python", "container_entrypoint.py"]
