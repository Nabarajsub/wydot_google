FROM python:3.11-slim

ARG APP_FILE=
ARG REQS_FILE=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ${REQS_FILE} /tmp/requirements.txt
ARG CACHEBUST=1
RUN pip install -r /tmp/requirements.txt && \
    python -c "import chainlit; print(f'chainlit installed at: {chainlit.__file__}')" && \
    which chainlit && \
    echo "=== CHAINLIT VERIFIED ==="

# --- BAKE MODEL INTO IMAGE ---
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
COPY ingestion_service/download_model.py /app/download_model.py
RUN python3 /app/download_model.py && \
    chmod -R 777 /app/model_cache

COPY . /app

# Create temp directories for file uploads with correct permissions
# Note: These may be reset by Cloud Run, so we also create them at runtime
RUN mkdir -p /tmp/.files /tmp/.chainlit && \
    chmod -R 777 /tmp/.files /tmp/.chainlit

RUN useradd -m appuser
USER appuser

ENV PORT=8080
ENV APP_FILE=chatapp.py
ENV DOTENV_PATH=/etc/secrets/.env
ENV SECRETS_TOML_PATH=/etc/secrets/streamlit/secrets.toml
ENV CHAINLIT_CONFIG_PATH=/app/.chainlit/config.toml
ENV CHAINLIT_STORAGE_PATH=/tmp/.chainlit
ENV CHAINLIT_FILES_DIRECTORY=/tmp/.files

# Run chainlit DIRECTLY
CMD ["chainlit", "run", "chatapp.py", "--port", "8080", "--host", "0.0.0.0" "--headless"]
