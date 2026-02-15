# WYDOT Assistant Database Architecture

This document maps the various data stores used by the WYDOT Assistant in both Local and Cloud environments.

## 1. Primary Relational Store: PostgreSQL
All persistent application state is now unified in **PostgreSQL**.

| Component | Table(s) | Purpose |
| :--- | :--- | :--- |
| **Authentication** | `users` | Stores verified users, password hashes (PBKDF2), and verification codes. |
| **Chat History** | `messages`, `threads` | Stores all chat conversations and historical thread titles. |
| **Monitoring** | `request_metrics` | Recorded latency (retrieval/generation) and model usage per request. |
| **Evaluation** | `online_evals`, `offline_runs` | Stores RAG evaluation scores (relevancy, utilization, etc.) and batch run results. |
| **Ingestion** | `ingestion_metadata` | Tracks status, chunk counts, and timestamps of ingested documents. |

**Local Connection**: `postgresql://postgres:password@db:5432/wydot_db` (via Docker Compose)  
**Production Connection**: [Cloud SQL for PostgreSQL](https://console.cloud.google.com/sql/instances) (Connected via Cloud SQL Auth Proxy or Unix Socket).

---

## 2. Vector & Knowledge Store: Neo4j
Uses **Neo4j Aura** for Knowledge Graph and Vector Search.

- **Labels**: `Chunk`, `Document`, `Entity` (for KG).
- **Index**: `wydot_vector_index` (Vector search for RAG).
- **Hosting**: [Neo4j Aura Console](https://console.neo4j.io/).

---

## 3. Temporary / Ephemeral
- **Model Cache**: AI models are baked into the Docker images but use `/tmp/model_cache` for writable execution layers in Cloud Run.
- **Verification Codes**: Logged to `/tmp/verification_codes.txt` if an SMTP server is not configured.
