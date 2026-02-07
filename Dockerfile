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
    pip install python-dotenv streamlit chainlit && \
    echo "=== VERIFYING CHAINLIT INSTALLATION ===" && \
    python -c "import chainlit; print(f'chainlit installed at: {chainlit.__file__}')" && \
    which chainlit && \
    echo "=== CHAINLIT VERIFIED ==="

COPY . /app

# Create temp directories for file uploads with correct permissions
RUN mkdir -p /tmp/.files && chmod 777 /tmp/.files

RUN useradd -m appuser
USER appuser

ENV PORT=8080
ENV APP_FILE=chatapp.py
ENV DOTENV_PATH=/etc/secrets/.env
ENV SECRETS_TOML_PATH=/etc/secrets/streamlit/secrets.toml

# Run chainlit DIRECTLY - no entrypoint script
CMD ["chainlit", "run", "chatapp.py", "--port", "8080", "--host", "0.0.0.0", "--headless"]
