#!/bin/bash
# =========================================================
# WYDOT GCP LLMOps/RAGOps Setup Script
# =========================================================
# This script sets up the required GCP infrastructure for:
# 1. Automated document ingestion (Cloud Storage + Eventarc)
# 2. RAG evaluation system (Vertex AI, BigQuery)
# 3. Monitoring and observability
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - Billing enabled on your GCP project
# - Owner or Editor role on the project
# =========================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}WYDOT GCP Infrastructure Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check for required environment variables

# 1. Check argument
if [ -n "$1" ]; then
    GCP_PROJECT_ID="$1"
fi

# 2. Check PROJECT_ID fallback
if [ -z "$GCP_PROJECT_ID" ] && [ -n "$PROJECT_ID" ]; then
    GCP_PROJECT_ID="$PROJECT_ID"
fi

# 3. Prompt if missing
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${YELLOW}Enter your GCP Project ID:${NC}"
    read GCP_PROJECT_ID
fi

# Sanitize Project ID (remove trailing stuff if user copy-pasted wrong)
GCP_PROJECT_ID=$(echo "$GCP_PROJECT_ID" | tr -d '[:space:]')

# Validate Project ID format
if [[ ! "$GCP_PROJECT_ID" =~ ^[a-z0-9-]+$ ]]; then
    echo -e "${RED}ERROR: Invalid Project ID format: '$GCP_PROJECT_ID'${NC}"
    echo "Project IDs must only contain lowercase letters, numbers, and hyphens."
    echo "Please unset your invalid environment variable:"
    echo "  unset GCP_PROJECT_ID"
    echo "  unset PROJECT_ID"
    echo "Then run the script again with the ID as an argument:"
    echo "  ./scripts/setup_gcp.sh YOUR_PROJECT_ID"
    exit 1
fi

if [ -z "$GCP_REGION" ]; then
    GCP_REGION="us-central1"
fi

echo -e "\n${GREEN}Configuration:${NC}"
echo "  Project ID: $GCP_PROJECT_ID"
echo "  Region: $GCP_REGION"
echo ""

# Set project
gcloud config set project $GCP_PROJECT_ID

# =========================================================
# 1. Enable required APIs
# =========================================================
echo -e "\n${GREEN}Step 1: Enabling required APIs...${NC}"

APIS=(
    "run.googleapis.com"              # Cloud Run
    "eventarc.googleapis.com"         # Eventarc
    "storage.googleapis.com"          # Cloud Storage
    "secretmanager.googleapis.com"    # Secret Manager
    "aiplatform.googleapis.com"       # Vertex AI
    "bigquery.googleapis.com"         # BigQuery
    "cloudscheduler.googleapis.com"   # Cloud Scheduler
    "cloudbuild.googleapis.com"       # Cloud Build
    "artifactregistry.googleapis.com" # Artifact Registry
    "logging.googleapis.com"          # Cloud Logging
    "monitoring.googleapis.com"       # Cloud Monitoring
)

for api in "${APIS[@]}"; do
    echo "  Enabling $api..."
    gcloud services enable $api --quiet
done

echo -e "${GREEN}✓ APIs enabled${NC}"

# =========================================================
# 2. Create service account
# =========================================================
echo -e "\n${GREEN}Step 2: Creating service account...${NC}"

SA_NAME="cloud-run-sa"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe $SA_EMAIL 2>/dev/null; then
    gcloud iam service-accounts create $SA_NAME \
        --display-name="Cloud Run Service Account" \
        --description="Service account for WYDOT Cloud Run services"
    echo "  Created service account: $SA_EMAIL"
else
    echo "  Service account already exists: $SA_EMAIL"
fi

# Grant required roles
ROLES=(
    "roles/run.invoker"
    "roles/storage.objectAdmin"
    "roles/secretmanager.secretAccessor"
    "roles/aiplatform.user"
    "roles/bigquery.dataEditor"
    "roles/bigquery.jobUser"
    "roles/eventarc.eventReceiver"
    "roles/logging.logWriter"
)

echo "  Granting roles..."
for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role" \
        --quiet 2>/dev/null || true
done

echo -e "${GREEN}✓ Service account configured${NC}"

# =========================================================
# 3. Create Artifact Registry repository
# =========================================================
echo -e "\n${GREEN}Step 3: Creating Artifact Registry repository...${NC}"

REPO_NAME="apps"
if ! gcloud artifacts repositories describe $REPO_NAME --location=us 2>/dev/null; then
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=us \
        --description="WYDOT application containers"
    echo "  Created repository: us-docker.pkg.dev/$GCP_PROJECT_ID/$REPO_NAME"
else
    echo "  Repository already exists"
fi

echo -e "${GREEN}✓ Artifact Registry configured${NC}"

# =========================================================
# 4. Create Cloud Storage buckets
# =========================================================
echo -e "\n${GREEN}Step 4: Creating Cloud Storage buckets...${NC}"

# Documents bucket (for ingestion)
DOCS_BUCKET="wydot-documents-${GCP_PROJECT_ID}"
if ! gsutil ls gs://$DOCS_BUCKET 2>/dev/null; then
    gsutil mb -l $GCP_REGION gs://$DOCS_BUCKET
    gsutil uniformbucketlevelaccess set on gs://$DOCS_BUCKET
    echo "" | gsutil cp - gs://$DOCS_BUCKET/incoming/.keep
    echo "" | gsutil cp - gs://$DOCS_BUCKET/processed/.keep
    echo "" | gsutil cp - gs://$DOCS_BUCKET/failed/.keep
    echo "  Created bucket: gs://$DOCS_BUCKET"
else
    echo "  Bucket already exists: gs://$DOCS_BUCKET"
fi

# Evaluation bucket
EVAL_BUCKET="wydot-evaluations-${GCP_PROJECT_ID}"
if ! gsutil ls gs://$EVAL_BUCKET 2>/dev/null; then
    gsutil mb -l $GCP_REGION gs://$EVAL_BUCKET
    echo "  Created bucket: gs://$EVAL_BUCKET"
else
    echo "  Bucket already exists: gs://$EVAL_BUCKET"
fi

echo -e "${GREEN}✓ Storage buckets configured${NC}"

# =========================================================
# 5. Create secrets (placeholder values)
# =========================================================
echo -e "\n${GREEN}Step 5: Creating Secret Manager secrets...${NC}"

SECRETS=(
    "neo4j-uri"
    "neo4j-username"
    "neo4j-password"
    "gcp-project-id"
)

for secret in "${SECRETS[@]}"; do
    if ! gcloud secrets describe $secret 2>/dev/null; then
        gcloud secrets create $secret \
            --replication-policy="automatic"
        echo "  Created secret: $secret"
        
        if [ "$secret" == "gcp-project-id" ]; then
            echo -n "$GCP_PROJECT_ID" | gcloud secrets versions add $secret --data-file=-
            echo "  Added value for $secret"
        else
            echo -e "${YELLOW}  ⚠️  Please add value for $secret:${NC}"
            echo "     echo -n 'YOUR_VALUE' | gcloud secrets versions add $secret --data-file=-"
        fi
    else
        echo "  Secret already exists: $secret"
    fi
done

echo -e "${GREEN}✓ Secrets configured${NC}"

# =========================================================
# 6. Create BigQuery dataset
# =========================================================
echo -e "\n${GREEN}Step 6: Creating BigQuery dataset...${NC}"

BQ_DATASET="wydot_eval"
if ! bq show $GCP_PROJECT_ID:$BQ_DATASET 2>/dev/null; then
    bq mk --location=$GCP_REGION --dataset $GCP_PROJECT_ID:$BQ_DATASET
    echo "  Created dataset: $BQ_DATASET"
else
    echo "  Dataset already exists: $BQ_DATASET"
fi

echo -e "${GREEN}✓ BigQuery dataset configured${NC}"

# =========================================================
# 7. Grant Eventarc permissions
# =========================================================
echo -e "\n${GREEN}Step 7: Configuring Eventarc permissions...${NC}"

# Get the Cloud Storage service agent
GCS_SA="service-$(gcloud projects describe $GCP_PROJECT_ID --format='value(projectNumber)')@gs-project-accounts.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:$GCS_SA" \
    --role="roles/pubsub.publisher" \
    --quiet 2>/dev/null || true

echo -e "${GREEN}✓ Eventarc permissions configured${NC}"

# =========================================================
# Summary
# =========================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Resources created:"
echo "  📦 Artifact Registry: us-docker.pkg.dev/$GCP_PROJECT_ID/apps"
echo "  🪣 Documents bucket: gs://$DOCS_BUCKET"
echo "  🪣 Evaluations bucket: gs://$EVAL_BUCKET"
echo "  🔐 Service account: $SA_EMAIL"
echo "  📊 BigQuery dataset: $BQ_DATASET"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Add your Neo4j credentials to Secret Manager:"
echo "     echo -n 'YOUR_URI' | gcloud secrets versions add neo4j-uri --data-file=-"
echo "     echo -n 'YOUR_USER' | gcloud secrets versions add neo4j-username --data-file=-"
echo "     echo -n 'YOUR_PASS' | gcloud secrets versions add neo4j-password --data-file=-"
echo ""
echo "  2. Set up GitHub Actions secrets:"
echo "     - GCP_PROJECT: $GCP_PROJECT_ID"
echo "     - GCP_REGION: $GCP_REGION"
echo "     - WIF_PROVIDER: (your Workload Identity Federation provider)"
echo ""
echo "  3. Push to main branch to trigger deployments"
echo ""
echo "  4. Upload documents to trigger ingestion:"
echo "     gsutil cp your-document.pdf gs://$DOCS_BUCKET/incoming/"
echo ""
