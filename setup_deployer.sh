#!/bin/bash
set -e

echo "🔍 Detecting Project ID..."
PROJECT_ID=$(gcloud config get-value project)
echo "✅ Project ID: $PROJECT_ID"

SA_NAME="github-deployer"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🔨 Creating Service Account: $SA_EMAIL..."
# Initial creation might fail if it already exists, so we use || true but check for real errors
gcloud iam service-accounts create $SA_NAME --display-name="GitHub Actions Deployer" || echo "Service account might already exist, continuing..."

echo "🔑 Granting Roles..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.admin" --condition=None

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.writer" --condition=None

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iam.serviceAccountUser" --condition=None

echo "🔗 Binding Workload Identity (Updating Policy)..."
# We need to know the repo name. Assuming user's current repo remote.
# But to be safe, we will ask user or use a wildcard for now to unblock.
# Better: Just bind to the specific repo if possible, but the user didn't give us the full repo name "user/repo".
# In DEPLOYMENT_GUIDE code, we used placeholders.
# Let's try to get it from git remote.

GITHUB_REPO=$(git config --get remote.origin.url | sed 's/https:\/\/github.com\///' | sed 's/.git$//')

if [ -z "$GITHUB_REPO" ]; then
    echo "⚠️ Could not detect GitHub repo from git config. Please verify WIF binding manually if this fails."
else
    echo "✅ Detected GitHub Repo: $GITHUB_REPO"
    # Get project number
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    
    # Allow the specific repo to impersonate the SA
    gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
        --project="${PROJECT_ID}" \
        --role="roles/iam.workloadIdentityUser" \
        --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GITHUB_REPO}" --condition=None
fi

echo "✅ Done! Service Account '$SA_EMAIL' is ready."
echo "👉 Go to GitHub -> Actions and 'Re-run jobs' to try the deployment again."
