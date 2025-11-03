# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG APP_FILE=flash_cloud_2.5rpo.py
ARG REQS_FILE=requirements_val.txt

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

# Unprivileged
RUN useradd -m appuser
USER appuser

ENV PORT=8080
ENV APP_FILE=${APP_FILE}
# Shared .env will be mounted here
ENV DOTENV_PATH=/etc/secrets/.env
# Streamlit will look here automatically for secrets.toml
ENV STREAMLIT_SECRETS_DIR=/app/.streamlit

CMD ["python", "-c", "\
import os, subprocess, pathlib; \
from dotenv import load_dotenv; \
p = os.getenv('DOTENV_PATH','/etc/secrets/.env'); \
if pathlib.Path(p).exists(): load_dotenv(p, override=False); \
app = os.getenv('APP_FILE','validator.py'); \
port = os.getenv('PORT','8080'); \
# ensure secrets dir exists (mount will provide file) \
pathlib.Path(os.getenv('STREAMLIT_SECRETS_DIR','/app/.streamlit')).mkdir(parents=True, exist_ok=True); \
subprocess.run(['python','-m','streamlit','run',app,'--server.port',port,'--server.address','0.0.0.0']) \
"]
