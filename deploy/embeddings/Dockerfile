FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .

# Pre-download the model during build to bake it into the image
# This prevents downloading it every time a container starts (faster cold starts)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port (default 8080 for Vertex AI)
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
