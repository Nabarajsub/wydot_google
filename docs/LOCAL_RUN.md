# Running WYDOT Components Locally (Before Cloud)

## 1. Chatbot (chatapp.py)

**Database & login (local SQLite)**  
- Auth is **enabled**. Use the login prompt (email/password). First-time users are created on sign-up.  
- Data is stored in `utils/chat_history.sqlite3` (or `CHAT_DB_PATH` if set).

**Conversation memory (Redis optional)**  
- If `REDIS_URL` or `REDIS_SERVER` is set, recent conversation is read/written via Redis.  
- If not set, history comes from the SQLite store (and in-memory for anonymous users).

**Online telemetry**  
- Each RAG request records latency and citation count into `telemetry.sqlite3` (or `TELEMETRY_DB_PATH`).  
- No BigQuery or cloud needed for local runs.

**Run:**
```bash
# From repo root (nbc)
chainlit run chatapp.py --port 8080
```
Ensure `.env` has `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and optionally `MISTRAL_API_KEY`, `GEMINI_API_KEY`.  
Optional: `REDIS_SERVER=localhost` or `REDIS_URL=redis://localhost:6379` for Redis-backed memory.

---

## 2. Ingestion Service (unified UI + Eventarc-style)

**Local:** SQLite tracker (`ingestion_service/ingestion_tracker.sqlite3`). No GCS required.  
**Cloud:** Set `GCS_BUCKET`; POST `/ingest` will handle Eventarc GCS events.

**Run:**
```bash
cd ingestion_service
python -m venv .venv  # optional
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```
Open http://localhost:5001 — same upload UI (drag-and-drop, ingest, library, KG explorer).  
Neo4j and (for audio/video) `GEMINI_API_KEY` must be set.

---

## 3. Offline RAG Evaluation

**Dataset:** Add or edit `evaluation/datasets/wydot_golden_sample.jsonl` (one JSON object per line with `question`, `reference_answer`, optional `id`).

**Run locally:**
```bash
cd evaluation
pip install -r requirements.txt
python run_local.py --dataset datasets/wydot_golden_sample.jsonl
```
Report is written to `evaluation/reports/local_report_<timestamp>.json`.  
Requires Neo4j. For full LLM-based metrics (Vertex AI), set `GCP_PROJECT_ID` in `.env`.

---

## 4. Cloud migration (summary)

- **Chat:** Set `DATABASE_URL` (PostgreSQL) for Cloud SQL; use Memorystore Redis and `REDIS_URL`.  
- **Ingestion:** Deploy same `app.py` to Cloud Run; set `GCS_BUCKET` and wire Eventarc to `/ingest`.  
- **Telemetry:** Point telemetry module to BigQuery (or keep SQLite for dev).  
- **Eval:** Use existing Cloud Run Job + Scheduler; point dataset to GCS or baked-in path.
