PROJECT_ID="wydotchatbot"

echo "======================================================="
echo "VERIFYING GCP PROJECT SETTINGS FOR: $PROJECT_ID"
echo "======================================================="

echo "\n[1/5] Checking Required APIs..."
gcloud services list --project=$PROJECT_ID --format="value(config.name)" | grep -E "(run|artifactregistry|sqladmin|secretmanager|iamcredentials)\.googleapis\.com" || echo "⚠️  Some required APIs might be disabled. Check your API library."

echo "\n[2/5] Checking Service Accounts..."
gcloud iam service-accounts list --project=$PROJECT_ID --format="value(email)" | grep -E "cloud-run-sa|github-deployer" || echo "⚠️  Could not find both cloud-run-sa and github-deployer service accounts."

echo "\n[3/5] Checking Artifact Registry 'apps' repo in us-central1..."
gcloud artifacts repositories list --project=$PROJECT_ID --location=us-central1 --format="value(name)" | grep "apps" || echo "⚠️  Artifact Registry repository 'apps' is missing or not in us-central1."

echo "\n[4/5] Checking Cloud SQL Instance 'wydot-db-instance'..."
gcloud sql instances list --project=$PROJECT_ID --format="value(name)" | grep "wydot-db-instance" || echo "⚠️  Cloud SQL Instance 'wydot-db-instance' missing."

echo "\nChecking Cloud SQL Database 'chat_history'..."
gcloud sql databases list -i wydot-db-instance --project=$PROJECT_ID --format="value(name)" | grep "chat_history" || echo "⚠️  Database 'chat_history' missing."

echo "\n[5/5] Checking Secret Manager for 'shared-dotenv'..."
gcloud secrets list --project=$PROJECT_ID --format="value(name)" | grep "shared-dotenv" || echo "⚠️  Secret 'shared-dotenv' missing."

echo "\n======================================================="
echo "Done! If you see any ⚠️ warnings above, double check those steps in the console."
echo "======================================================="
