FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
# --- BAKE MODEL INTO IMAGE ---
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
# Copy download script
COPY download_model.py .
# Run download script to populate cache
RUN python download_model.py && \
    chmod -R 777 /app/model_cache

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app
