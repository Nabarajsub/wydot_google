# WYDOT Assistant: Development Workflow

This guide explains how to test new features locally using Docker and how to safely push changes to Google Cloud.

## 🛠️ Step 1: Local Development & Testing
Always verify features locally before pushing. This ensures stability and saves deployment costs.

### 1. Start the Stack
Run the full environment including PostgreSQL and Embeddings services:
```bash
docker-compose up --build
```

### 2. Verify Core Functions
- **Chat**: [http://localhost:8080](http://localhost:8080)
- **Monitoring/Ingestion**: [http://localhost:8082](http://localhost:8082)

### 3. Database Inspection
Use the local `db` service to check persistence:
```bash
docker-compose exec db psql -U postgres -d wydot_db
```

---

## 🚀 Step 2: Push to Google Cloud
Once local testing is successful, use the automated CI/CD pipeline.

### 1. Commit and Push
The project is configured with GitHub Actions. Pushing to the `main` branch (if configured) will trigger a rebuild.
```bash
git add .
git commit -m "feat: your feature description"
git push origin main
```

### 2. Manual Build (Backup)
If you need to force a build via Google Cloud Build:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

---

## 🧹 Step 3: Cleanup Tips
- To reset the local database entirely: `docker-compose down -v`
- To clean up dangling Docker images: `docker image prune`

## ⚠️ Important Notes
- **Secrets**: Production secrets (Vertex AI, Neo4j) are managed in **GCP Secret Manager**. Do not commit `.env` files with production secrets.
- **Port 8080**: Both Chatbot and Admin services use port 8080 internally for Cloud Run compatibility. Local Docker Compose maps them to 8080 and 8082 respectively.
