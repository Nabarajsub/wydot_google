#!/usr/bin/env bash
# Setup Cloud Scheduler jobs to keep the Cloud Run service and Neo4j Aura warm.
# This prevents cold starts for users and stops Neo4j Aura Free from pausing.
#
# Usage: bash cloudrun/setup-keepalive.sh <PROJECT_ID> <SERVICE_URL>
# Example: bash cloudrun/setup-keepalive.sh my-project https://wydot-unified-pg-532721775338.us-central1.run.app

set -euo pipefail

PROJECT_ID="${1:?Usage: $0 <PROJECT_ID> <SERVICE_URL>}"
SERVICE_URL="${2:?Usage: $0 <PROJECT_ID> <SERVICE_URL>}"

echo "Setting up keep-alive jobs for: ${SERVICE_URL}"

# 1. Ping the app every 4 minutes to keep the Cloud Run instance warm
gcloud scheduler jobs create http wydot-keepalive-app \
  --project="${PROJECT_ID}" \
  --location=us-central1 \
  --schedule="*/4 * * * *" \
  --uri="${SERVICE_URL}/api/health" \
  --http-method=GET \
  --attempt-deadline=30s \
  --description="Keep Cloud Run instance warm (every 4 min)" \
  2>/dev/null || \
gcloud scheduler jobs update http wydot-keepalive-app \
  --project="${PROJECT_ID}" \
  --location=us-central1 \
  --schedule="*/4 * * * *" \
  --uri="${SERVICE_URL}/api/health" \
  --http-method=GET \
  --attempt-deadline=30s

echo "✅ App keep-alive job created (every 4 minutes)"

# 2. Ping Neo4j every 10 minutes to prevent Aura Free from pausing (pauses after 72h idle)
gcloud scheduler jobs create http wydot-keepalive-neo4j \
  --project="${PROJECT_ID}" \
  --location=us-central1 \
  --schedule="*/10 * * * *" \
  --uri="${SERVICE_URL}/api/health/neo4j" \
  --http-method=GET \
  --attempt-deadline=30s \
  --description="Keep Neo4j Aura warm (every 10 min)" \
  2>/dev/null || \
gcloud scheduler jobs update http wydot-keepalive-neo4j \
  --project="${PROJECT_ID}" \
  --location=us-central1 \
  --schedule="*/10 * * * *" \
  --uri="${SERVICE_URL}/api/health/neo4j" \
  --http-method=GET \
  --attempt-deadline=30s

echo "✅ Neo4j keep-alive job created (every 10 minutes)"
echo ""
echo "Done! To verify: gcloud scheduler jobs list --project=${PROJECT_ID} --location=us-central1"
