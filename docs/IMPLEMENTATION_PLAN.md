# WYDOT Chatbot & Ingestion — Enterprise Implementation Plan

**Goal:** Add database-backed login, Cloud SQL migration, Redis for conversation memory, host ingestion UI with Eventarc triggers, run chatbot on Cloud Run, and implement online/offline RAG evaluation and monitoring — at enterprise scale (~10,000 WYDOT employees).

**Scope:** Plan only. No code; implementation to follow.

---

## 1. Current State Summary

| Component | Current State |
|-----------|----------------|
| **chatapp.py** | Chainlit app with SQLite auth store (disabled), in-memory session history, Neo4j RAG, multi-model (Mistral/Gemini/OpenRouter). Chat DB path: `/tmp/wydot_chat_history.sqlite3` on Cloud Run (ephemeral). |
| **Ingestion** | Two code paths: (1) `app.py` — Eventarc-only, POST `/`/`/ingest` for GCS `incoming/`; (2) `local_ingest.py` — full UI (upload.html), `/upload`, `/transcribe`, `/library`, `/delete`, `/kg/*`, JSON tracker + `ingested_data/`. Deployed service uses `app.py` only; UI not hosted. |
| **Evaluation** | `evaluation/rag_evaluator.py`: Vertex AI EvalTask, BigQuery, batch eval. `run_evaluation_job.py`: Cloud Run Job + Scheduler. No per-request (online) metrics in chatapp. |
| **Cloud Run** | Chatbot: `flash.yaml` (scaled-chatbot). Ingestion: `ingestion_deploy.yaml` (wydot-ingestion). Evaluation: Cloud Run Job + weekly Scheduler. |

---

## 2. Database & Login (chatapp.py → Cloud SQL)

### 2.1 Feasibility

**Feasible.** You already have:

- `ChatHistoryStore` with `users` (email, password_hash, display_name) and `messages` (user_id, session_id, role, content).
- PBKDF2 password hashing (salt + 200k iterations).
- Auth callback stubbed but commented out; resume/DB write paths assume `user.metadata["db_id"]`.

### 2.2 Approach

1. **Keep SQLite for local/dev**  
   Use `CHAT_DB_PATH` (e.g. local file or `/tmp`); no Cloud SQL dependency in dev.

2. **Introduce Cloud SQL (PostgreSQL or MySQL) for production**  
   - Prefer **PostgreSQL** for JSON support, full SQL, and GCP first-class support.  
   - Add abstraction: e.g. `ChatHistoryStore` implementing an interface, with backends: `SQLiteChatHistoryStore` and `CloudSQLChatHistoryStore` (or single class that switches on `DATABASE_URL`).  
   - Use connection pooling (e.g. **SQLAlchemy** with `pg8000` or `cloud-sql-python-connector`) so Cloud Run instances don’t exhaust connections.  
   - Schema: mirror current tables (`users`, `messages`); add indexes on `(user_id, session_id)`, `session_id`, and `ts` for history and resume.

3. **Enable login in production**  
   - Uncomment and wire `@cl.password_auth_callback` to `ChatHistoryStore.authenticate` / `create_user`.  
   - Optional: add “Sign up” vs “Login” in Chainlit config so first-time users can register (or restrict to pre-provisioned users only).  
   - Enforce HTTPS and secure cookies in production (Chainlit/Cloud Run default).

4. **Secrets**  
   Store Cloud SQL credentials in Secret Manager; inject `DATABASE_URL` or `CLOUD_SQL_CONNECTION_NAME` + user/pass into Cloud Run.

5. **Migration path**  
   - Script to create Cloud SQL schema (from current SQLite DDL).  
   - One-time migration of existing SQLite data (if any) into Cloud SQL; then point production to Cloud SQL only.

### 2.3 Enterprise Considerations (~10k users)

- **Cloud SQL** instance size: start with db-f1-micro or db-g1-small; scale based on concurrent sessions and message volume.  
- **Pool size:** e.g. 5–10 connections per Cloud Run instance; max instances × pool size &lt; Cloud SQL max connections.  
- **Backups:** Enable automated backups and point-in-time recovery.  
- **IAM:** Use dedicated DB user with least privilege; avoid storing credentials in repo.

---

## 3. Conversation Memory with Redis (Reduce Latency)

### 3.1 Feasibility

**Feasible and recommended.** Today:

- Authenticated: history read via `CHAT_DB.get_recent()` and written with `add_message()` (SQLite/DB).
- Unauthenticated: history in `cl.user_session.set("memory", ...)` (in-process, lost on new instance).

Both hit DB or in-memory only; for 10k users, DB for every message turn adds latency and load.

### 3.2 Approach

1. **Use Redis as the “hot” conversation store**  
   - Key pattern: e.g. `wydot:chat:{user_id}:{session_id}` or `wydot:thread:{thread_id}`.  
   - Value: list of recent messages (e.g. last N messages as JSON or msgpack).  
   - TTL: e.g. 7–30 days so old sessions expire; optional “promote to DB” for long-term storage.

2. **Read path**  
   - On each request: try Redis first for that user/session.  
   - On cache miss: load from Cloud SQL (if authenticated) and backfill Redis.  
   - Use this list for `build_prompt_with_history()` and for resuming threads.

3. **Write path**  
   - After each assistant reply: append to Redis list (and optionally trim to last N).  
   - Optionally: async or batched write to Cloud SQL for persistence (e.g. background task or periodic flush).  
   - If you want a single source of truth in DB: write-through (Redis + DB); else Redis as cache with DB as backup.

4. **Hosting**  
   - **Google Cloud Memorystore for Redis** (recommended): VPC, managed, HA option.  
   - Ensure Cloud Run can reach Redis (VPC connector or Serverless VPC Access if Memorystore is in VPC).

5. **Compatibility with current code**  
   - Introduce a small `ConversationMemory` interface (e.g. `get_recent(user_id, session_id, limit)`, `append(user_id, session_id, role, content)`).  
   - Implementations: `RedisConversationMemory` (primary), optional `SQLConversationMemory` or current `ChatHistoryStore` for fallback/migration.  
   - In chatapp: use `ConversationMemory` for the “recent messages” used in generation; keep `ChatHistoryStore` for auth and optional long-term persistence.

### 3.3 Enterprise Considerations

- **Memorystore** sizing: start with 1 GB; scale with active sessions and message size.  
- **Eviction:** volatile-lru or allkeys-lru; avoid noeviction if memory is limited.  
- **Security:** Redis in VPC; no public IP; Cloud Run via VPC connector.  
- **Latency:** Single-digit ms from Cloud Run to Redis in same region.

---

## 4. Ingestion: Host Upload UI + Eventarc for New Documents

### 4.1 Feasibility

**Feasible.** You have:

- **Eventarc:** GCS `incoming/` → Cloud Run `app.py` `/ingest` (working in deploy workflow).  
- **Full UI logic** in `local_ingest.py` + `upload.html`: upload, transcribe, library, KG explorer, delete.

Gap: deployed service runs only `app.py`; it does not serve the HTML or the `/upload`, `/library`, `/kg/*` routes that `upload.html` expects.

### 4.2 Approach

**Option A (recommended): Single ingestion service with UI + Eventarc**

1. **Merge capabilities into one Flask app** (e.g. keep `app.py` as the main entrypoint):  
   - **Keep:** POST `/`, POST `/ingest` — Eventarc payload (GCS event); same logic as today (download from GCS, process, Neo4j, move to processed/failed).  
   - **Add:**  
     - `GET /` → serve `upload.html` (and static assets).  
     - POST `/upload` — accept multipart file; either (i) process in-place (like local_ingest) and ingest to Neo4j, or (ii) upload file to `gs://bucket/incoming/` and return immediately (Eventarc will trigger `/ingest`).  
     - `/transcribe`, `/library`, `/delete`, `/kg/stats`, `/kg/documents`, `/kg/chunks`, `/kg/search`, `/kg/update` — port from `local_ingest.py`.

2. **Persistence for “library” and tracker**  
   - Current: JSON file + `ingested_data/` directory (not suitable for stateless Cloud Run).  
   - **Replace with:**  
     - **Library/tracker:** Cloud SQL table(s) or Firestore collection (e.g. `ingestion_metadata`: filename, type, chunks count, date, GCS path).  
     - **File storage:** For UI uploads, upload to GCS `incoming/` (or a dedicated prefix); processed files live in `processed/`; metadata in DB/Firestore.  
   - **Library API:** Read from DB/Firestore; “delete” = delete from Neo4j + update/delete metadata.

3. **Transcribe**  
   - Keep Gemini transcription in the same service; ensure `GEMINI_API_KEY` (or Vertex) is available in Cloud Run.

4. **Eventarc**  
   - No change: GCS finalize → POST to same service `/ingest`.  
   - After processing in `/ingest`, write metadata to Cloud SQL/Firestore so “Ingested Library” stays in sync for both UI uploads and GCS-triggered ingestions.

5. **Interface “same as here”**  
   - Keep `upload.html` and its existing JS (calls to `/upload`, `/transcribe`, `/library`, `/delete`, `/kg/*`). Only backend and storage change.

**Option B: Two services**

- **Service 1 (UI):** Serves upload.html and `/upload` (e.g. uploads to GCS `incoming/` only); `/library` and `/kg/*` read from shared DB + Neo4j.  
- **Service 2 (Worker):** Current Eventarc target; only `/ingest`, no UI.  
- More moving parts; use Option A unless you need strict separation.

### 4.3 Enterprise Considerations

- **Auth for ingestion UI:** Restrict to internal users (e.g. IAP or OAuth with WYDOT domain).  
- **Quotas:** Large file uploads → increase Cloud Run request timeout and max request size; consider resumable uploads for very large files.  
- **Rate limiting:** Optional rate limit on `/upload` to avoid abuse.

---

## 5. Host Chatbot on Google Cloud Run

### 5.1 Current State

Already in place: `cloudrun/flash.yaml`, Dockerfile for chatapp, deploy via image `us-docker.pkg.dev/${PROJECT_ID}/apps/flash-cloud:latest`, Chainlit on port 8080.

### 5.2 What to Add/Change

1. **Config for Cloud SQL and Redis**  
   - Add env or Secret Manager for `DATABASE_URL` (or Cloud SQL connection name + credentials) and Redis host/port/auth.  
   - In Cloud Run service YAML: add env vars and/or volume mounts for secrets.

2. **VPC**  
   - If using Memorystore (and optionally private Cloud SQL), attach VPC connector to the Cloud Run service so it can reach Redis/DB.

3. **Scaling for ~10k users**  
   - Set `maxScale` to a value that supports peak concurrency (e.g. 50–200 depending on traffic).  
   - Keep `minScale: 0` for cost or set `minScale: 1` for lower cold-start latency.  
   - `containerConcurrency: 80` is reasonable; tune with load tests.

4. **Health**  
   - Startup probe already targets `/`; ensure Chainlit responds quickly.  
   - Liveness: Chainlit’s root is sufficient if it returns 200 when the app is up.

No need to change “host on Cloud Run” in principle — only add DB, Redis, and scaling/security.

---

## 6. Online RAG Evaluation & Monitoring (Inline in chatapp)

### 6.1 Feasibility

**Feasible.** You already have evaluation metrics and BigQuery in `rag_evaluator.py`; the gap is capturing per-request data in the chat path and publishing it.

### 6.2 Approach

1. **Instrument the message handler**  
   In `@cl.on_message` (and equivalent for audio path):  
   - Record timestamps: start (before retrieval), after retrieval, after generation, end.  
   - Compute: **retrieval_latency_ms**, **generation_latency_ms**, **total_latency_ms**.  
   - Optional: **num_sources**, **model_used**, **index_used**, **thread_id**, **user_id** (if authenticated).

2. **Lightweight “inline” metrics (no LLM eval in hot path)**  
   - **Citation count:** e.g. count of `[Source N]` or `[N]` in the response.  
   - **Retrieval count:** number of chunks returned.  
   - Optionally: store the question and response (or hashes) for later offline eval; avoid running Vertex Eval in the request path (too slow).

3. **Where to send**  
   - **BigQuery:** One row per query (e.g. table `wydot_chat_requests`): request_id, timestamp, user_id, thread_id, model, index, retrieval_latency_ms, generation_latency_ms, total_latency_ms, num_sources, citation_count, etc.  
   - **Cloud Monitoring:** Push custom metrics (e.g. `rag/latency`, `rag/citation_count`) for dashboards and alerts.  
   - **Logging:** Structured logs (e.g. JSON with the same fields) for debugging and Logs Explorer.

4. **Implementation**  
   - Non-blocking: e.g. fire-and-forget insert to BigQuery (or async task) so latency is not added to the user response.  
   - Use a small “telemetry” module called from the message handler; keep chatapp.py readable.

5. **Optional: sampling**  
   - For high volume, log or insert a sample (e.g. 10%) to control cost and volume.

### 6.3 Enterprise Considerations

- **PII:** Avoid logging full question/answer if they contain PII; log hashes or omit.  
- **BigQuery cost:** Partition table by date; set retention; use streaming insert only if needed, else batch.  
- **Alerts:** Define SLOs (e.g. p95 latency &lt; 5s) and alert in Cloud Monitoring.

---

## 7. Offline RAG Evaluation (Curated Dataset + Pipeline)

### 7.1 Feasibility

**Feasible.** You already have:

- `rag_evaluator.py`: WYDOTRagSystem, EvalResult, BigQuery storage, batch evaluation, report generation.  
- `run_evaluation_job.py`: Cloud Run Job + Scheduler.  
- Data sources: validator DB, JSONL, synthetic generation.

Missing: a **curated golden dataset** and a clear pipeline to run offline eval on it.

### 7.2 Approach

1. **Curated dataset format**  
   - Use the existing `EvalExample` (id, question, reference_answer, context, metadata).  
   - Store as **JSONL** (one JSON object per line):  
     - `id`, `question`, `reference_answer` (optional), `context` (optional), `metadata` (e.g. category, difficulty, spec year).  
   - Location: versioned in repo (e.g. `evaluation/datasets/wydot_golden.jsonl`) and/or in GCS (e.g. `gs://wydot-evaluations-*/datasets/wydot_golden.jsonl`) for the Cloud Run Job.

2. **Curation process**  
   - Export from validator (human-validated Q&A) using existing `load_golden_dataset_from_validator`.  
   - Add domain expert–curated Q&A pairs (WYDOT specs, common questions).  
   - Optional: use Gemini to generate candidates, then human review and add to JSONL.  
   - Version the dataset (and tag in BigQuery which dataset version was used for each run).

3. **Offline pipeline**  
   - **Input:** Dataset path (GCS or local).  
   - **Run:** `RAGEvaluator.evaluate_batch(examples)` (existing).  
   - **Output:** BigQuery rows (existing) + report JSON in GCS (existing in run_evaluation_job).  
   - **Trigger:** Cloud Run Job (existing); schedule weekly or on-demand; optionally trigger on dataset update (e.g. Cloud Build or manual).

4. **Metrics to track over time**  
   - Same as in rag_evaluator: groundedness, answer_relevance, context_relevance, coherence, citation_accuracy, section_reference_accuracy, latency (p50/p95).  
   - Store in BigQuery with `dataset_version` and `run_timestamp`; dashboards (Looker Studio or Cloud Monitoring) to track trends.

5. **Integration with online metrics**  
   - Online: latency and simple counters in production.  
   - Offline: full LLM-based metrics on curated set.  
   - Compare: e.g. offline “citation_accuracy” vs online “citation_count” distribution to validate that production behavior aligns with golden set.

### 7.3 Enterprise Considerations

- **Dataset ownership:** Clear process for who adds/approves golden examples.  
- **Reproducibility:** Pin model versions and index name in eval config; record in BigQuery.  
- **Cost:** Vertex Eval and LLM calls have cost; run offline eval on a schedule or on-demand, not on every commit.

---

## 8. Implementation Order and Dependencies

Suggested order:

1. **Cloud SQL + login (chatapp)**  
   - Add DB abstraction and Cloud SQL backend; enable auth callback; deploy with secrets.  
   - No dependency on Redis.

2. **Redis conversation memory**  
   - Add Memorystore; VPC connector; ConversationMemory interface; wire chatapp to Redis (with DB fallback if desired).  
   - Depends on: Cloud Run and optionally Cloud SQL (if you persist from Redis to DB).

3. **Online evaluation (inline metrics)**  
   - Add telemetry in chatapp; BigQuery + optional Cloud Monitoring.  
   - Can be done in parallel with 1–2.

4. **Ingestion UI + Eventarc**  
   - Merge local_ingest routes into app.py; add Cloud SQL or Firestore for library/tracker; GCS for upload path; keep Eventarc on `/ingest`.  
   - Deploy single ingestion service; test UI and GCS trigger.

5. **Offline evaluation pipeline**  
   - Define and version golden dataset (JSONL); wire Cloud Run Job to load it; add dataset_version to BigQuery; set up dashboard.  
   - Depends on: curated data (you can start with validator export + small manual set).

6. **Cloud Run tuning for scale**  
   - After 1–4 are stable: load test; tune min/max instances, concurrency, and resource limits; set SLOs and alerts.

---

## 9. Feasibility Summary

| Item | Feasible | Notes |
|------|----------|--------|
| Database in chatapp + login | Yes | Existing schema and auth logic; add Cloud SQL backend and enable callback. |
| Migrate DB to Cloud SQL | Yes | Standard migration; use pooling and secrets. |
| Redis for conversation memory | Yes | Memorystore + VPC; interface in front of Redis + optional DB. |
| Host ingestion UI, same interface | Yes | Merge local_ingest into app.py; persistent metadata in DB/Firestore; GCS for files. |
| Eventarc trigger to index new docs | Yes | Already in place; ensure UI upload path and Eventarc both write to same metadata store. |
| Chatbot on Cloud Run | Yes | Already there; add DB/Redis config and scaling. |
| Online RAG evaluation (latency + metrics) | Yes | Instrument handler; send to BigQuery/Monitoring; non-blocking. |
| Offline RAG evaluation (curated dataset) | Yes | Define JSONL dataset; use existing evaluator and Job; add versioning and dashboards. |
| Enterprise scale (~10k users) | Yes | Cloud SQL + Memorystore + scaling and IAM; no fundamental blockers. |

---

## 10. Best Practices (Enterprise, ~10k Users)

- **Auth:** Enforce login for production; consider SSO (e.g. Google Workspace) later.  
- **Secrets:** All credentials in Secret Manager; no .env in images.  
- **Networking:** Redis and Cloud SQL in VPC; Cloud Run via VPC connector.  
- **Observability:** Structured logs, custom metrics, BigQuery for request-level and eval metrics; alerts on latency and errors.  
- **Security:** IAP or equivalent for ingestion UI; least-privilege IAM for Cloud Run and jobs.  
- **Cost:** Right-size Cloud SQL and Redis; use minScale: 0 where acceptable; partition and retain BigQuery data.  
- **Disaster recovery:** Cloud SQL backups; optional Redis persistence; versioned eval datasets and configs.

This plan is ready to be used as a checklist for implementation; each section can be broken into smaller tasks and assigned as needed.
