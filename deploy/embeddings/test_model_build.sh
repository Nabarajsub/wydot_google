#!/bin/bash
echo "Building Embeddings container..."
docker build -t test-embeddings:latest ./deploy/embeddings

echo "Running container..."
# Run in background
docker run -d -p 8080:8080 --name test-embeddings test-embeddings:latest

echo "Waiting for service to start..."
sleep 10

echo "Testing health..."
curl -v http://localhost:8080/health

echo "Testing prediction..."
curl -v -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": ["Hello world", "This is a test"]}'

echo "Cleaning up..."
docker stop test-embeddings
docker rm test-embeddings
