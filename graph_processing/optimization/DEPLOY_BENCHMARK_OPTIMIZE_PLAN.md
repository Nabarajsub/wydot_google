# Deploy, Benchmark & Optimize WYDOT Unified RAG Pipeline

## Context

The unified service (chatbot + admin) is built but not yet deployed. Before scaling to 1,267 documents, you need to understand where time and money are spent, how the pipeline compares to industry RAG standards, and what concrete changes will get response times under 2 seconds. This plan covers deployment → baseline measurement → industry comparison → optimization in a prioritized, phased approach.

---

## Current Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Chainlit (FastAPI)                                      │
│                                                         │
│  1. Settings: model, index, hyde, multihop, reranking   │
│  2. Multimodal check → bypass RAG if media attached     │
│                                                         │
│  ┌─────────────────── RETRIEVAL ──────────────────┐     │
│  │                                                │     │
│  │  [Optional] HyDE: Gemini generates snippet     │     │
│  │       ↓ (+500-2000ms)                          │     │
│  │                                                │     │
│  │  Embedding: Gemini API (768d)                  │     │
│  │       ↓ (100-500ms)                            │     │
│  │                                                │     │
│  │  Vector Search: Neo4j index (FETCH_K=25)       │     │
│  │       ↓ (50-200ms)                             │     │
│  │                                                │     │
│  │  Metadata Enrichment: Cypher batch query        │     │
│  │    • Document→Section→Chunk traversal          │     │
│  │    • SUPERSEDES newer version injection        │     │
│  │    • Entity graph facts collection             │     │
│  │    • [Fallback] CONTAINS text scan             │     │
│  │       ↓ (50-300ms, fallback: +200-500ms)       │     │
│  │                                                │     │
│  │  [Optional] FlashRank reranking                │     │
│  │    • Rerank FETCH_K → RETRIEVAL_K=10           │     │
│  │    • Neighbor context expansion                │     │
│  │       ↓ (+50-200ms + Neo4j queries)            │     │
│  │                                                │     │
│  │  [Optional] Multihop: decompose → N parallel   │     │
│  │    searches → merge results                    │     │
│  │       ↓ (+1000-4000ms)                         │     │
│  │                                                │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌─────────────────── GENERATION ─────────────────┐     │
│  │                                                │     │
│  │  Build prompt: system + history + sources      │     │
│  │  LLM call: Gemini 2.5 Flash (streaming)       │     │
│  │    OR Mistral | OpenRouter (13 models)         │     │
│  │       ↓ (1000-5000ms)                          │     │
│  │                                                │     │
│  │  Citation enhancement: [SOURCE_X] → [X]       │     │
│  │  Source elements for side panel                │     │
│  │       ↓ (<10ms)                                │     │
│  │                                                │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  Telemetry: record latency + model + sources (async)    │
│  Chat History: Redis + SQLite/PostgreSQL                │
│  Online Eval: background quality scoring                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Latency Breakdown (Typical Query, No Optional Features)

| Stage | Latency | Notes |
|-------|---------|-------|
| Query embedding | 100-500ms | Gemini API network call |
| Vector search | 50-200ms | Neo4j vector index |
| Metadata enrichment | 50-300ms | Single batched Cypher |
| LLM generation | 1000-5000ms | Streaming, model-dependent |
| Citation enhancement | <10ms | Local string parsing |
| Telemetry record | <5ms | Background thread |
| **Total baseline** | **1.2-6s** | Dominated by LLM latency |

### With Optional Features Enabled

| Feature | Additional Latency | When Used |
|---------|-------------------|-----------|
| HyDE (query expansion) | +500-2000ms | Extra LLM call for hypothetical snippet |
| Multihop (N=2-4 sub-queries) | +1000-4000ms | Parallel searches, each with embedding+retrieval |
| Reranking (FlashRank) | +50-200ms + Neo4j | FlashRank inference + neighbor context fetch |
| Thinking mode (sync) | Same total, blocks UI | No streaming, retrieval blocks earlier |

---

## Phase 1: Deploy & Collect Baseline (Week 1-2)

### 1.1 Push the unified service
- Commit all new files: `admin_routes.py`, `Dockerfile.unified`, `cloudrun/unified-pg.yaml`, `.github/workflows/deploy-unified.yml`, `requirements.unified.txt`, and the `chatapp_gemini.py` admin mount changes
- Push to `main` → GitHub Actions builds + deploys `wydot-unified-pg`
- Verify: hit `/` (Chainlit UI), `/admin/login` (admin panel), `/api/health/neo4j` (keepalive)

### 1.2 Collect baseline metrics
- Existing `utils/telemetry.py` already records `retrieval_latency_ms`, `generation_latency_ms`, `total_latency_ms`
- Run the 5 golden dataset queries (in `utils/evaluation.py`) + 20 real user queries
- After 1-2 weeks, pull P50/P95/P99 via `get_latency_stats()`

### 1.3 Add granular stage timing
Currently telemetry lumps embedding + vector search + enrichment into one `retrieval_latency_ms`. Break it into:
- `embedding_latency_ms` — Gemini API embed call
- `search_latency_ms` — Neo4j vector index query
- `enrichment_latency_ms` — Cypher metadata enrichment
- `reranking_latency_ms` — FlashRank (if enabled)
- Add `input_tokens`, `output_tokens` from Gemini response `usage_metadata` for cost tracking

**Files:** `chatapp_gemini.py` (lines 2153-2364, `main()` handler), `utils/telemetry.py`

---

## Phase 2: Industry Comparison

### 2.1 Where WYDOT sits in the RAG taxonomy

| Architecture | Description | WYDOT Coverage | Gap? |
|---|---|---|---|
| **Naive RAG** | Embed → retrieve → generate | ✅ Baseline path | — |
| **Advanced RAG** | HyDE, reranking, query expansion | ✅ Optional toggles | — |
| **GraphRAG** | Graph-structured knowledge, entity relationships, Cypher queries | ✅ Rich graph with SUPERSEDES, Entity facts, graph-augmented context | — |
| **Modular RAG** | Pluggable components (model selection, index selection) | ✅ Gemini/Mistral/OpenRouter, multiple indexes | — |
| **Agentic RAG** | Intent routing, query decomposition, multi-step reasoning | ⚠️ Partial (multihop + intent routing exist) | No self-reflection loop |
| **RAPTOR** | Hierarchical summarization trees | ❌ | Consider for annual report comparison |
| **Corrective RAG (CRAG)** | Retrieval quality check before generation | ❌ | Low priority |
| **Self-RAG** | Adaptive retrieval with self-reflection | ❌ | Research-grade |

**Verdict:** WYDOT is an **Advanced GraphRAG** system — more sophisticated than 90% of production RAG deployments. The main gaps (RAPTOR, CRAG, Self-RAG) are research-grade and not required for a government document chatbot.

### 2.2 Strengths vs. industry

- **Graph-aware retrieval** with SUPERSEDES versioning is genuinely advanced and rare
- **Multi-model support** via OpenRouter gives flexibility most systems lack
- **Existing telemetry** is well-structured — most RAG systems ship without it
- **Streaming** implemented for all 3 model backends

### 2.3 Industry latency targets for chat

| Metric | Industry Standard | WYDOT Current (estimated) | Status |
|---|---|---|---|
| P50 total | < 2s | ~2-3s | ⚠️ Close |
| P95 total | < 5s | ~5-8s | ❌ Over |
| Time to first token | < 500ms | ~1.2-2s | ❌ Over |
| Cold start | < 3s | 5-15s | ❌ Over |

### 2.4 Key problems found in code analysis

1. **No Neo4j connection pooling** — 5 separate `GraphDatabase.driver()` calls create new TCP connections per request (lines 1114, 1193, 1404, 1639, 1739 in `chatapp_gemini.py`)
2. **CONTAINS fallback** is an O(N) full-text scan when chunk IDs are missing (line 1262) — can add 200-500ms
3. **Cold starts** are 5-15s due to 2-3GB Docker image + no min-instances
4. **PyTorch baked in image** but unused for the query path (only needed for admin ingestion's MiniLM model)
5. **No embedding cache** — every query pays the Gemini API roundtrip, even if identical
6. **No response cache** — FAQ-like repeated questions rerun the entire pipeline

---

## Phase 3: Latency Optimization (Ranked by Impact/Effort)

### P0 — Do First (~3 hours work, ~400-800ms savings per request)

#### 3.1 Neo4j Connection Pooling
Create a single global driver singleton, reuse across all requests:
```python
_NEO4J_DRIVER = None

def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        from neo4j import GraphDatabase
        _NEO4J_DRIVER = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
    return _NEO4J_DRIVER
```
Replace all 5 `GraphDatabase.driver()` calls in `chatapp_gemini.py`.

**Savings:** 100-300ms per request (TCP connection establishment + TLS handshake).
**File:** `chatapp_gemini.py` lines 1114, 1193, 1404, 1639, 1739

#### 3.2 Eliminate CONTAINS Fallback
At line 1262, when chunk IDs aren't found in the batch enrichment query, the code falls back to `c.text CONTAINS $text_snippet` — a full graph scan. Fix options:
- **Option A (recommended):** Skip the fallback — use whatever metadata the vector search already returned
- **Option B:** Add a full-text index on `Chunk.text` if the fallback is truly needed

**Savings:** 200-500ms when triggered (which is often on large graphs).
**File:** `chatapp_gemini.py` line 1262

#### 3.3 Add `min-instances: 1` to Cloud Run
Add annotation in `unified-pg.yaml`:
```yaml
run.googleapis.com/minScale: "1"
```
**Savings:** Eliminates 5-15s cold starts entirely.
**Cost:** ~$30-50/month (one warm instance always running).
**File:** `cloudrun/unified-pg.yaml`

### P1 — Do Second (~6 hours, ~300-1500ms additional savings)

#### 3.4 Reduce FETCH_K from 25 → 15
Set `FETCH_K=15` environment variable. Still enough headroom for reranking to select the best 10 from 15 candidates (vs. 25).

**Savings:** 50-150ms retrieval + 100-300ms enrichment (fewer chunks to process).

#### 3.5 Embedding Cache (LRU)
Add an in-memory LRU cache on normalized query strings:
```python
from functools import lru_cache

@lru_cache(maxsize=500)
def cached_embed_query(query_normalized: str):
    return embeddings_model.embed_query(query_normalized)
```
**Savings:** 100-500ms for repeated/similar queries (complete Gemini API call skip).
**File:** `chatapp_gemini.py` near `get_embeddings_model()`

#### 3.6 Prompt Compression
- Limit sources passed to LLM to top 5-7 (after reranking) instead of all RETRIEVAL_K
- Truncate each chunk preview to 500 tokens max
- Summarize conversation history beyond the last 3 turns

**Savings:** 200-1000ms LLM generation (fewer input tokens = faster processing).
**File:** `chatapp_gemini.py` `build_prompt_with_history()` (line 1887)

#### 3.7 Right-size Cloud Run to 2 vCPU / 2 GiB
The chatbot is I/O-bound (waiting for Gemini API + Neo4j responses), not CPU-bound. Test with 2 vCPU, 2 GiB.

**Savings:** 40-60% compute cost (~$200-400/month).
**File:** `cloudrun/unified-pg.yaml`

### P2 — Ongoing (~1-2 days)

#### 3.8 Telemetry API + Admin Dashboard
- Mount telemetry endpoints at `/api/telemetry/stats`, `/api/telemetry/timeseries`
- Build `/admin/dashboard` HTML page with Chart.js pulling from these endpoints
- Set up alerts: P95 > 5s, error rate > 5%
- Functions already exist in `utils/telemetry.py`: `get_latency_stats()`, `get_timeseries()`, `get_model_comparison()`

#### 3.9 Redis/In-Memory Response Cache
For FAQ-like queries, cache the full response:
- Key: normalized query + model + index
- TTL: 24 hours (specs don't change daily)
- Cache hit = ~5ms response
- Use Cloud Memorystore (Redis) or simpler in-memory dict for initial implementation

### P3 — Future (~1+ weeks)

#### 3.10 Slim Docker Image
Remove PyTorch pre-download from `Dockerfile.unified`. PyTorch is only needed for admin ingestion's MiniLM model, which is infrequent. The chatbot query path uses Gemini API embeddings (no local model).
- Image drops from ~2-3GB to ~500MB
- Cold start improvement even without min-instances
**File:** `Dockerfile.unified`

#### 3.11 Token Cost Tracking
Record `prompt_token_count` and `candidates_token_count` from Gemini response `usage_metadata` in the telemetry table. Gives real cost-per-query visibility.
**File:** `chatapp_gemini.py`, `utils/telemetry.py`

---

## Phase 4: Cost Optimization Summary

| Change | Monthly Savings | Effort | Risk |
|---|---|---|---|
| Cloud Run 4→2 vCPU, 4→2 GiB | ~$200-400 | 10 min | Low — monitor CPU after |
| Embedding cache (fewer API calls) | ~$5-20 | 2 hours | None |
| Prompt compression (fewer tokens) | ~$10-30 | 3 hours | Low — test answer quality |
| Slim Docker (faster cold starts) | Indirect (fewer billed seconds) | 4 hours | Low |
| min-instances: 1 | +$30-50 (cost, not savings) | 5 min | None — eliminates cold starts |

### Current Cost Drivers (Highest to Lowest)

| Service | What | Monthly Est. |
|---------|------|-------------|
| **Cloud Run** | 4 vCPU + 4 GiB, 900s timeout | ~$400-800 (if always on) |
| **Cloud SQL** | PostgreSQL instance | ~$400-600 |
| **Gemini API** | Embedding + chat tokens | Variable (depends on query volume) |
| **Neo4j Aura** | Free tier (or Pro if scaled) | $0-500 |
| **Container Registry** | 2-3 GB image storage | ~$0.20-0.30 |
| **GCS** | Document storage (Eventarc) | Minimal |

---

## Phase 5: Monitoring & Alerting

1. **Cloud Scheduler job**: Hit `/api/health/neo4j` every 5 minutes to prevent Aura Free tier pausing
2. **P95 alert**: Cloud Monitoring custom metric when P95 total latency > 5s
3. **Admin dashboard**: `/admin/dashboard` with:
   - Latency distribution chart (P50/P95/P99 over time)
   - Model comparison bar chart
   - Cost-per-query trend
   - Error rate
4. **Weekly report**: Query telemetry DB for weekly P50/P95/P99 trends

---

## Priority Matrix

| # | Change | Latency Savings | Effort | Priority |
|---|---|---|---|---|
| 1 | Neo4j connection pooling | 100-300ms | 1 hour | **P0** |
| 2 | Eliminate CONTAINS fallback | 200-500ms | 2 hours | **P0** |
| 3 | min-instances: 1 | 5-15s cold start | 5 min | **P0** |
| 4 | Reduce FETCH_K 25→15 | 50-150ms | 5 min | **P1** |
| 5 | Embedding LRU cache | 100-500ms | 2 hours | **P1** |
| 6 | Prompt compression | 200-1000ms | 3 hours | **P1** |
| 7 | Right-size Cloud Run 2/2 | — (cost only) | 10 min | **P1** |
| 8 | Telemetry API + dashboard | — (visibility) | 6 hours | **P2** |
| 9 | Redis response cache | Full pipeline skip | 1 day | **P2** |
| 10 | Slim Docker image | Cold start | 4 hours | **P3** |
| 11 | Token cost tracking | — (visibility) | 2 hours | **P3** |

**Recommended execution order:** P0 items first (3 changes, ~3 hours of work, 400-800ms savings on every request + cold start elimination). Then P1 items (~6 hours, another 300-1500ms savings). P2/P3 as ongoing improvements.

---

## Existing Caching Mechanisms (Already in Code)

| Cache | What | Scope |
|-------|------|-------|
| `_EMBEDDINGS_CACHE` | Gemini embedding model object | Module-level, reused across requests |
| `_VECTOR_STORE_CACHE` | Neo4j Vector retriever objects | Module-level, keyed by (index, use_gemini) |
| `_RANKER_INSTANCE` | FlashRank model | Module-level singleton |
| `_DOC_TITLES_CACHE` | Document sources from Neo4j | 1-hour TTL |
| Conversation memory | Recent chat history | Redis-backed (authenticated users) |

**Missing:** Query embedding cache, response cache, Neo4j connection pool.

---

## Critical Files for Implementation

| File | What Changes |
|---|---|
| `chatapp_gemini.py` | Connection pooling (5 locations), embedding cache, CONTAINS removal, stage-level telemetry, prompt compression |
| `cloudrun/unified-pg.yaml` | min-instances annotation, CPU/memory right-sizing |
| `utils/telemetry.py` | Granular stage timing columns, token cost columns, API endpoints |
| `Dockerfile.unified` | Remove PyTorch pre-download (P3) |
| `admin_routes.py` | Dashboard HTML page (P2) |

---

## Verification Plan

1. **Deploy**: Push to main, confirm GitHub Actions completes, hit `/`, `/admin/login`, `/api/health/neo4j`
2. **Baseline**: Run 25 test queries, pull P50/P95/P99 from telemetry
3. **After P0 changes**: Run same 25 queries, compare P50/P95/P99 — expect 400-800ms improvement
4. **After P1 changes**: Run same 25 queries — expect total P50 < 2s
5. **Cost**: Compare Cloud Run billing week-over-week after right-sizing
