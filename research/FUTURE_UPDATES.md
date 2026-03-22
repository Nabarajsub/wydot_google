# Future Updates: Integration Roadmap
## Unifying ChartQNA, GraphRAG, and Agentic Systems into One Chatbot
### Nabaraj Subedi — March 2026

---

## 1. Executive Summary

This document analyzes three independently developed WYDOT subsystems and proposes a phased roadmap to integrate them into a single, unified **Agentic Multimodal GraphRAG Chatbot**. The three subsystems are:

| System | Directory | Core Capability |
|--------|-----------|----------------|
| **ChartQNA / Visual Compliance** | `copalirag/` | Visual retrieval of standard plan drawings + VLM-based compliance checking |
| **Multimodal Graph RAG** | `neo4j/` | Text + Image + Video ingestion into Neo4j with dual vector indexes |
| **Agentic Orchestration** | `wydot_agents/` | Multi-agent pipeline: Coordinator → Retrieval → Compliance → Action Executor |

**Goal:** Merge these into a single chatbot where the **Agent Coordinator** decides at query time whether to use text retrieval, visual retrieval, graph traversal, or compliance checking — and routes to the right specialist automatically.

---

## 2. System-by-System Analysis

### 2.1 ChartQNA / Visual Compliance (`copalirag/`)

**What it does:**
- **Indexer** (`indexer.py`): Uses **ColPali v1.2** (a vision-language retrieval model) to encode standard plan images (JPG/PNG) into multi-vector embeddings. Stores them in a `.pt` file.
- **Retrieval** (`app.py`): Given a text query like "mailbox support post spacing", ColPali uses **MaxSim late interaction** scoring to find the most relevant standard plan drawing.
- **Reasoning** (`app.py`): Sends the retrieved image + user question to **Qwen 2.5-VL-7B** for visual question answering.
- **Compliance Mode** (`compliance_sec_gemini.py`): Takes a user-uploaded design image + retrieved standard plan → sends BOTH images to **Gemini** → generates a structured compliance audit report with PASS/FAIL verdicts, forensic reasoning, and a downloadable PDF.

**Key Techniques:**
| Technique | Implementation |
|-----------|---------------|
| Vision Retrieval | ColPali v1.2 (MaxSim late interaction scoring) |
| Visual QA | Qwen 2.5-VL-7B-Instruct |
| Compliance Reasoning | Gemini with structured JSON output |
| Index Format | PyTorch `.pt` file with image paths + ColPali embeddings |
| Top-K Retrieval | 5 most relevant standard plans |
| Audit Report | FPDF-generated PDF with segment-level PASS/FAIL verdicts |

**Strengths:**
- Can "see" engineering drawings — not just text
- Structured compliance output (segment name, check type, verdict, reasoning)
- Publication-ready audit reports with PDF export

**Limitations:**
- Runs on HPC only (requires GPU for ColPali + Qwen)
- Standalone Streamlit app — not integrated with main chatbot
- Index is a flat `.pt` file — no graph relationships between drawings
- No connection to the text-based knowledge graph

---

### 2.2 Multimodal Graph RAG (`neo4j/`)

**What it does:**
- **Ingestion** (`ingest_multimodal.py`): Processes PDFs AND videos:
  - PDFs → Extracts page images → **BLIP** captioning → **CLIP** visual embedding + **SentenceTransformer** text embedding → stores as `VisualChunk` and `TextChunk` nodes in Neo4j
  - Videos (MP4) → Extracts keyframes every 5 seconds using **OpenCV** → **BLIP** captioning → **CLIP** embedding + **Whisper** audio transcription → stores as `VideoFrame` nodes
- **Search** (`app_mrag.py`): Dual-index search:
  - `wydot_text_index`: Semantic search over text chunks
  - `wydot_visual_index`: CLIP-based visual search over page images and video frames
- **Answer Generation**: Combines text evidence + visual evidence descriptions → sends to **Gemini** for synthesis

**Key Techniques:**
| Technique | Implementation |
|-----------|---------------|
| Text Embedding | SentenceTransformer (all-MiniLM-L6-v2) |
| Visual Embedding | CLIP ViT-base-patch32 |
| Image Captioning | BLIP |
| Audio Transcription | OpenAI Whisper |
| Video Keyframes | OpenCV (every 5 seconds) |
| Graph Nodes | TextChunk, VisualChunk, VideoFrame |
| Dual Search | 2 separate Neo4j vector indexes |
| LLM | Gemini 3 Flash Preview |

**Strengths:**
- True multimodal: text + images + video + audio in one graph
- Neo4j graph structure connects visual content to source documents
- Video frame retrieval with timestamp-linked playback

**Limitations:**
- CLIP embeddings are weaker than ColPali for document/drawing retrieval
- No entity extraction or relationship mapping (flat chunks)
- No compliance checking logic
- Separate Streamlit app — not in main Chainlit chatbot

---

### 2.3 Agentic Orchestration (`wydot_agents/`)

**What it does:**
- **Coordinator** (`coordinator.py`): The "brain" — classifies user query and determines which specialists to call. Currently uses keyword matching (bridge ID, project references).
- **Retrieval Specialist** (`retrieval_specialist.py`): The "librarian" — combines HyDE expansion + Neo4j vector search + full-text search fallback. Uses the main chatbot's Neo4j indexes.
- **Compliance Analyst** (`compliance_analyst.py`): The "judge" — takes retrieved context + user query → Gemini evaluates compliance → produces confidence score (0–1) + approval flag.
- **Action Executor** (`action_executor.py`): The "hand" — generates final answer, drafts permits, sends critical notifications, and records all decisions in a **Neo4j Decision Ledger**.
- **External Tools** (`external_tools.py`): Mock tools for permit drafting and critical notification sending.
- **Proactive Alerts** (`proactive_alert.py`): Simulates event-driven agent behavior (e.g., detecting a new policy upload that impacts ongoing projects).
- **Human-in-the-Loop** (`agent_app.py`): When compliance score is below threshold or action is high-risk, presents ✅ Approve / ❌ Reject buttons to the user before execution.

**Agent Pipeline:**
```
User Query → Coordinator → Retrieval Specialist → Compliance Analyst → Action Executor
                    ↓                                      ↓
              [Project Data Specialist]           [Human-in-the-Loop if high risk]
                                                           ↓
                                                  [Neo4j Decision Ledger]
```

**Key Techniques:**
| Technique | Implementation |
|-----------|---------------|
| Orchestration | Manual state machine (not LangGraph, but same pattern) |
| Query Routing | Keyword-based coordinator |
| Retrieval | HyDE + Vector search + Full-text fallback |
| Compliance | LLM-generated confidence score + approval gate |
| Tool Use | Permit drafting, critical notifications |
| Auditability | Neo4j Decision Ledger (immutable record) |
| HITL | Chainlit Action buttons (Approve/Reject) |
| Self-Correction | Retries with broadened query if context is empty |

**Strengths:**
- Full agent pipeline with compliance scoring
- Human-in-the-loop for high-risk decisions
- Auditable decision ledger in Neo4j
- Self-correction loop for failed retrievals

**Limitations:**
- No visual/multimodal retrieval — text only
- Coordinator uses keyword matching (not LLM-based routing)
- No connection to ColPali/ChartQNA visual tools
- No graph traversal (entity navigation, SUPERSEDES chains)

---

## 3. Integration Plan: The Unified Architecture

### 3.1 The Target Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     UNIFIED WYDOT CHATBOT                       │
│                    (Single Chainlit App)                         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   SMART COORDINATOR (LLM-Based)                  │
│  Classifies query → Routes to specialist team                    │
│  Categories: TEXT | VISUAL | COMPARISON | COMPLIANCE | PROJECT   │
└────┬──────────┬───────────┬───────────┬──────────────┬──────────┘
     │          │           │           │              │
     ▼          ▼           ▼           ▼              ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐
│ TEXT     │ │ VISUAL   │ │ GRAPH    │ │ COMPLIANCE│ │ PROJECT  │
│ RETRIEVAL│ │ RETRIEVAL│ │ NAVIGATOR│ │ ANALYST   │ │ DATA     │
│          │ │          │ │          │ │           │ │ SPECIALIST│
│ Neo4j    │ │ ColPali  │ │ Cypher   │ │ Gemini    │ │ Bridge   │
│ Vector + │ │ MaxSim + │ │ entity   │ │ structured│ │ database │
│ BM25 +   │ │ Gemini   │ │ traversal│ │ output +  │ │ lookups  │
│ FlashRank│ │ VLM      │ │ + version│ │ approval  │ │          │
│          │ │          │ │ chains   │ │ gate      │ │          │
└─────────┘ └──────────┘ └──────────┘ └───────────┘ └──────────┘
     │          │           │           │              │
     └──────────┴───────────┴───────────┴──────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ACTION EXECUTOR                                │
│  Final answer synthesis + permit drafting + decision ledger       │
│  + Human-in-the-Loop approval buttons                            │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 What Changes from Current Systems

| Current State | Integrated State |
|--------------|-----------------|
| 3 separate apps (Streamlit × 2, Chainlit × 1) | 1 unified Chainlit app |
| ColPali `.pt` index on HPC | ColPali embeddings in Neo4j `visual_plan_index` |
| CLIP visual search in `neo4j/` | Replaced by ColPali (stronger for documents) |
| Keyword-based coordinator | LLM-based intent classifier using Gemini |
| No visual compliance in main chatbot | Visual Compliance Agent integrated as a specialist |
| Separate compliance logic (copalirag vs wydot_agents) | Single Compliance Analyst with text AND visual modes |

---

## 4. Phased Roadmap

### Phase 1: Smart Coordinator (Week 1–2)
**Goal:** Replace keyword-based coordinator with LLM-based intent classification.

**What to do:**
- Upgrade `coordinator.py` to use Gemini with structured output (Pydantic):
  ```python
  class QueryIntent(BaseModel):
      category: Literal["TEXT", "VISUAL", "COMPARISON", "COMPLIANCE", "PROJECT", "GENERAL"]
      requires_graph: bool
      requires_visual: bool
      sub_queries: List[str]  # For multi-hop decomposition
  ```
- The coordinator calls Gemini once to classify the query, then routes to the right specialist(s).
- **Reasoning:** Current keyword matching misses queries like "Show me the approach slab drawing" (should route to VISUAL but has no keyword trigger).

---

### Phase 2: Integrate Visual Retrieval (Week 3–4)
**Goal:** Bring ColPali visual search into the main chatbot.

**What to do:**
- Move ColPali `.pt` index into Neo4j as `StandardPlan` nodes with ColPali embeddings stored as properties.
- Create a new `visual_retrieval_specialist.py` that:
  1. Uses ColPali to find relevant standard plan drawings
  2. Sends retrieved images to Gemini Vision for QA/compliance
- Wire this into the Coordinator's routing logic.
- **Reasoning for ColPali over CLIP:** ColPali uses late interaction scoring (MaxSim) which is significantly better than CLIP's single-vector matching for document retrieval. ColPali was trained specifically on document images, while CLIP was trained on general photos.

**Key technical decision:** We use **Gemini** (API-based) instead of Qwen 2.5-VL (GPU-dependent) for the VLM reasoning step. This removes the HPC dependency and makes it deployable on Cloud Run.

---

### Phase 3: Unified Compliance Engine (Week 5–6)
**Goal:** Merge the text-based compliance logic (wydot_agents) with the visual compliance logic (copalirag) into a single Compliance Analyst.

**What to do:**
- Upgrade `compliance_analyst.py` to handle two modes:
  - **Text Compliance:** Current behavior — score retrieved text context against query.
  - **Visual Compliance:** New — when visual evidence is present, send images to Gemini with the structured audit prompt from `compliance_sec_gemini.py`.
- Unified compliance output format:
  ```json
  {
    "segments": [
      {
        "segment_name": "Rebar Spacing",
        "check_type": "Dimensional",
        "verdict": "PASS",
        "standard_requirement": "#4 bars at 12\" OC",
        "design_observation": "#4 bars at 12\" OC",
        "forensic_reasoning": "Matches WYDOT standard plan detail"
      }
    ],
    "overall_score": 0.92,
    "requires_human_approval": false
  }
  ```

---

### Phase 4: Graph Navigation Agent (Week 7–8)
**Goal:** Add a Graph Navigator specialist that uses Cypher to traverse the knowledge graph.

**What to do:**
- Create `graph_navigator.py` that executes targeted Cypher queries:
  - Entity lookup: "What specs reference AASHTO M85?" → `MATCH (e:Entity {name: 'AASHTO M85'})<-[:MENTIONS]-(c:Chunk) RETURN c.text`
  - Version comparison: "What changed in Section 401?" → traverse `[:SUPERSEDES]` edges
  - Material tracing: "What test methods apply to Portland Cement?" → `MATCH (m:Material {name: 'Portland Cement'})-[:REQUIRES]->(t:TestMethod) RETURN t.name`
- Wire into Coordinator: when `requires_graph == true`, route to Graph Navigator before or alongside Text Retrieval.
- **Reasoning:** Vector search finds semantically similar text. Graph navigation finds logically connected entities. Combining both gives the best of both worlds.

---

### Phase 5: Autonomous WYDOT Agents (Week 9–12)
**Goal:** Move from reactive (user asks → system answers) to proactive (system detects events → system acts).

**Independent Agent Roadmap:**

| Agent | Function | Trigger |
|-------|----------|---------|
| **Specification Monitor** | Detects when new spec versions are uploaded → automatically diffs against previous version → creates change summary | New file in GCS bucket |
| **Compliance Watchdog** | Periodically scans active projects against latest specifications → flags non-compliant items | Scheduled cron (weekly) |
| **Permit Processor** | Receives permit applications → retrieves load rating specs → runs compliance check → drafts permit or flags for human review | New permit application API call |
| **Training Report Generator** | Analyzes chat history to identify most-asked questions → generates training guides for new WYDOT engineers | Scheduled monthly |
| **Bridge Health Monitor** | Integrates with inspection data → matches against spec requirements → generates maintenance priority lists | After inspection data upload |

**Architecture for Autonomous Agents:**
```
┌───────────────────────────────────────────────────────┐
│              EVENT BUS (Pub/Sub or Cloud Tasks)       │
└───────┬─────────┬──────────┬──────────┬──────────────┘
        │         │          │          │
        ▼         ▼          ▼          ▼
   ┌─────────┐ ┌──────┐ ┌────────┐ ┌────────────┐
   │ Spec    │ │Permit│ │Compliance│ │ Training  │
   │ Monitor │ │Proc. │ │Watchdog │ │ Report Gen│
   └─────────┘ └──────┘ └────────┘ └────────────┘
        │         │          │          │
        └─────────┴──────────┴──────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   SHARED KNOWLEDGE GRAPH    │
        │   (Neo4j + Decision Ledger) │
        └─────────────────────────────┘
```

---

## 5. Reasoning Behind the Shift to Knowledge Graphs

### 5.1 Why Vectors Alone Aren't Enough

| Limitation | Example | Why Graph Solves It |
|-----------|---------|-------------------|
| No relationships | "What materials does Section 803 require?" | Graph: `Section 803 → [:MENTIONS] → Material nodes` |
| No version tracking | "What changed in cement specs?" | Graph: `2021 Specs → [:SUPERSEDES] → 2010 Specs` |
| No provenance | "Why did the agent approve this?" | Graph: `Decision node → [:BASED_ON] → Chunk nodes` |
| No cross-modal links | "Show me the drawing for this spec" | Graph: `Section → [:HAS_VISUAL] → StandardPlan` |
| No compliance trail | "Prove this meets AASHTO M85" | Graph: `Material → [:REQUIRES] → Standard → [:VERIFIED_BY] → TestMethod` |

### 5.2 Why ColPali Specifically

| Feature | CLIP (used in neo4j/) | ColPali (used in copalirag/) |
|---------|------|---------|
| Training data | General photos (LAION) | Document images (DocVQA, InfoVQA) |
| Embedding type | Single 512-dim vector | Multi-vector (patch-level, ~1000+ tokens) |
| Scoring method | Cosine similarity | MaxSim late interaction |
| Document understanding | Poor (treats drawings as photos) | Strong (understands layout, tables, text in images) |
| Use case fit | General image search | Engineering drawing retrieval ✓ |

### 5.3 Why Agents Specifically

| Reactive System (Current) | Agentic System (Future) |
|--------------------------|------------------------|
| User asks → System answers | System detects → System acts |
| Fixed retrieval pipeline | Dynamic tool selection |
| No self-correction | Retry with broadened query |
| No approval workflow | Human-in-the-loop for high-risk |
| No audit trail | Decision ledger in Neo4j |
| Single LLM call | Multi-step reasoning chain |

---

## 6. Slide Content for Presentation

### SLIDE: Future Direction — Unified Agentic Architecture

**Title:** The Unified Agentic Architecture (Version 5)

**Content:**
- **Goal:** Merge ChartQNA visual compliance, GraphRAG text retrieval, and multi-agent orchestration into ONE chatbot
- **Smart Coordinator:** LLM-based intent classification replaces keyword routing
- **Visual Retrieval Agent:** ColPali vision search finds relevant standard plan drawings from within the chatbot
- **Graph Navigator Agent:** Cypher traversal for entity relationships, version chains, and compliance trails
- **Unified Compliance Engine:** Handles BOTH text-based AND visual compliance checking
- **Decision Ledger:** Every agent decision is recorded in Neo4j for full auditability

---

### SLIDE: Future Direction — Autonomous WYDOT Agents

**Title:** Beyond Chat: Autonomous WYDOT Agents (Version 6)

**Content:**
- **Specification Monitor Agent:** Automatically detects new spec uploads → diffs against previous versions → alerts affected projects
- **Compliance Watchdog Agent:** Weekly scans of active projects against latest specifications → flags non-compliant items
- **Permit Processing Agent:** Receives applications → runs compliance check → drafts permit OR flags for human review
- **Training Report Agent:** Analyzes chat history → identifies knowledge gaps → generates training guides
- **Bridge Health Agent:** Integrates inspection data → matches against spec requirements → generates maintenance priority lists
- **All agents share:** The same Neo4j Knowledge Graph + Decision Ledger for full traceability

---

### SLIDE: Integration Timeline

**Title:** Phased Integration Roadmap

| Phase | Timeline | Deliverable |
|-------|----------|------------|
| **Phase 1** | Week 1–2 | Smart LLM-based Coordinator (replace keyword routing) |
| **Phase 2** | Week 3–4 | Visual Retrieval Agent (ColPali in chatbot) |
| **Phase 3** | Week 5–6 | Unified Compliance Engine (text + visual) |
| **Phase 4** | Week 7–8 | Graph Navigator Agent (Cypher traversal) |
| **Phase 5** | Week 9–12 | Autonomous event-driven agents |

**Key Architectural Principle:** Each phase is independently deployable. Phase 1 improves the chatbot immediately. Phase 5 is the long-term research goal.

---

### SLIDE: Why This Matters for WYDOT

**Title:** Impact: From Q&A Tool to Autonomous Operations Platform

| Current State | Future State |
|--------------|-------------|
| Engineers manually search 1,300+ PDFs | Chatbot retrieves text AND visual specs instantly |
| Compliance checks are manual | Visual compliance agent compares designs against standards automatically |
| Specification changes go unnoticed | Spec Monitor agent detects and alerts in real-time |
| Permit approvals take days | Permit agent pre-screens and drafts in minutes |
| No institutional memory | Decision ledger captures all AI-assisted decisions |
| Knowledge leaves with retiring engineers | Knowledge graph preserves and connects expertise permanently |

---

## 7. Summary: The Big Picture

```
Version 1-2: "Search" → Basic text retrieval
Version 3-4: "Understand" → Knowledge Graph with entities + relationships
Version 5:   "See + Understand" → Multimodal (text + visual + graph)
Version 6:   "Act Autonomously" → Event-driven agents with compliance + HITL
```

**The trajectory:** From a search tool → to an understanding engine → to an autonomous operations platform.

---

## 8. Deployment Analysis: How to Deploy on Google Cloud

### 8.1 What You Already Have (Current Architecture)

Your chatbot is **already deployed on Google Cloud Run**. Here's the full stack:

```
┌────────────────────────────────────────────────────────────┐
│                    GITHUB REPOSITORY                       │
│  Push to main → GitHub Actions CI/CD triggers              │
└───────────────────────┬────────────────────────────────────┘
                        │ docker build + push
                        ▼
┌────────────────────────────────────────────────────────────┐
│              ARTIFACT REGISTRY (GAR)                       │
│  us-docker.pkg.dev/{PROJECT}/apps/wydot-chatbot:{sha}      │
└───────────────────────┬────────────────────────────────────┘
                        │ gcloud run services replace
                        ▼
┌────────────────────────────────────────────────────────────┐
│                  GOOGLE CLOUD RUN                          │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Chatbot     │  │ Admin/Ingest │  │ Evaluation        │ │
│  │ Service     │  │ Service      │  │ Service           │ │
│  │             │  │              │  │                   │ │
│  │ chatapp.py  │  │ ingestion/   │  │ evaluation/       │ │
│  │ 4 CPU/4 GiB │  │ app.py       │  │                   │ │
│  │ Port 8080   │  │              │  │                   │ │
│  └──────┬──────┘  └──────────────┘  └───────────────────┘ │
│         │                                                  │
│         ├── Secret Manager (.env)                          │
│         ├── Cloud SQL Proxy (PostgreSQL)                   │
│         └── HuggingFace model baked into image             │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│                 EXTERNAL SERVICES                          │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Neo4j Aura  │  │ Gemini API   │  │ Mistral API       │ │
│  │ (Graph DB)  │  │ (LLM +       │  │ (LLM)             │ │
│  │             │  │  Embeddings) │  │                   │ │
│  └─────────────┘  └──────────────┘  └───────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

**Current Deployment Details:**

| Component | Current Setup |
|-----------|--------------|
| **Container** | `python:3.11-slim`, CPU-only PyTorch |
| **Resources** | 4 vCPU, 4 GiB RAM |
| **Startup** | CPU boost enabled, 15s initial delay, 30 failure threshold |
| **Concurrency** | 80 requests per instance |
| **Timeout** | 900 seconds (15 min) |
| **Database** | Cloud SQL PostgreSQL via Unix socket proxy |
| **Secrets** | Google Secret Manager (`shared-dotenv`) |
| **Model** | `all-MiniLM-L6-v2` baked into Docker image |
| **CI/CD** | GitHub Actions → Artifact Registry → Cloud Run |
| **Auth** | Workload Identity Federation (keyless) |

---

### 8.2 All Google Cloud Deployment Options Compared

| Option | Cost Model | GPU Support | Scale to Zero | Best For |
|--------|-----------|-------------|---------------|----------|
| **Cloud Run** (current) | Pay-per-request | ✅ L4 GPUs (per-second billing) | ✅ Yes | API-based chatbots, low/variable traffic |
| **GKE Autopilot** | Pay-per-pod | ✅ L4, A100, H100 GPUs | ❌ Min 1 pod | High-traffic, complex orchestration |
| **GKE Standard** | Pay-per-VM | ✅ Full GPU catalog | ❌ Pay for VMs | Full infrastructure control |
| **App Engine Standard** | Pay-per-instance-hour | ❌ No GPU | ✅ Yes | Simple web apps (not ideal for AI) |
| **App Engine Flexible** | Pay-per-VM | ❌ No GPU | ❌ Min 1 instance | Legacy apps |
| **Vertex AI** | Pay-per-prediction | ✅ Managed | ✅ Yes (endpoints) | ML model serving only |
| **Compute Engine** | Pay-per-VM-hour | ✅ All GPUs | ❌ Always on | Full control, persistent workloads |

---

### 8.3 Recommendation: Stay on Cloud Run (With Upgrades)

**Cloud Run is the right choice for your chatbot.** Here's why:

| Requirement | Cloud Run Delivers |
|------------|-------------------|
| WebSocket support (Chainlit needs it) | ✅ Yes (HTTP/2 + WebSocket) |
| Scale to zero (save money when no users) | ✅ Yes |
| CI/CD integration | ✅ Already set up with GitHub Actions |
| Secret management | ✅ Already using Secret Manager |
| Database access | ✅ Already using Cloud SQL proxy |
| GPU for future ColPali | ✅ L4 GPUs now GA, billed per-second |
| No infrastructure to manage | ✅ Fully serverless |

**When to switch to GKE Autopilot:**
- Only if you need multiple persistent GPU instances (e.g., always-on ColPali + Whisper + BLIP)
- Or if you need more than 1 GPU per instance
- Or if you need TPU access for model fine-tuning

---

### 8.4 Deployment Upgrades for the Unified Chatbot

#### Upgrade 1: Switch to Gemini Embeddings (Remove HuggingFace)

Your current Dockerfile bakes `all-MiniLM-L6-v2` into the image. Since V4 uses Gemini embeddings (API-based), you can:
```dockerfile
# REMOVE this line from Dockerfile.chatbot:
# RUN mkdir -p /app/model_cache && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# The image becomes ~500 MB smaller
# Gemini embeddings are called via API, no local model needed
```

**Impact:** Smaller Docker image → faster deployments → faster cold starts.

#### Upgrade 2: Add GPU Support for ColPali (When Ready)

When you integrate visual retrieval (Phase 2 of the roadmap), add GPU:
```yaml
# cloudrun/chatbot-gpu.yaml
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      containers:
      - image: ${IMAGE}
        resources:
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"    # ← Add L4 GPU
```

**Cost estimate with GPU:**
- L4 GPU on Cloud Run: ~$0.000225/second = ~$0.81/hour
- But with scale-to-zero: Only pay when users are actively using visual search
- vs. GKE with always-on GPU: ~$0.81/hour × 24 × 30 = ~$583/month

#### Upgrade 3: Multi-Service Architecture for Agents

When you implement autonomous agents (Phase 5), split into separate Cloud Run services:

```
┌─────────────────────────────────────────────────────────────┐
│                  CLOUD RUN SERVICES                         │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Chatbot  │ │ Visual   │ │ Agent    │ │ Spec Monitor  │  │
│  │ Service  │ │ Service  │ │ Workers  │ │ (Scheduled)   │  │
│  │ (CPU)    │ │ (GPU)    │ │ (CPU)    │ │               │  │
│  │ Port 8080│ │ Port 8081│ │ Port 8082│ │ Cloud Tasks   │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│       │            │            │              │            │
│       └────────────┴────────────┴──────────────┘            │
│                         │                                   │
│              Shared: Neo4j Aura + Cloud SQL                 │
└─────────────────────────────────────────────────────────────┘
```

**Why separate services:**
- Chatbot (CPU-only, scale-to-zero) → cheap, fast cold starts
- Visual Service (GPU, L4) → expensive, only runs when visual queries are submitted
- Agent Workers (CPU) → triggered by Cloud Tasks for async operations
- Spec Monitor → Cloud Scheduler cron job, runs weekly

---

### 8.5 Detailed Cost Estimation (Monthly)

#### Scenario A: Current Setup (Text-Only Chatbot)

This is what you are running right now.

| Service | Configuration | Monthly Cost |
|---------|--------------|-------------|
| **Cloud Run — Chatbot** | 4 vCPU / 4 GiB. Scale-to-zero. ~50–200 queries/day. | **$5–20** |
| **Cloud Run — Admin/Ingestion** | Rarely used (only when uploading new docs). | **$1–3** |
| **Cloud SQL PostgreSQL** | `db-f1-micro` instance, 24/7. $0.018/hour. | **~$13** |
| **Neo4j Aura Free** | 200K nodes / 400K relationships. | **$0** |
| **Gemini API** | LLM + Embeddings. Free tier covers ~1,500 req/day. | **$0** |
| **Secret Manager** | 1 secret, 1 version. | **$0.06** |
| **Artifact Registry** | Docker image storage (~2 GB). | **$0.20** |
| **GitHub Actions CI/CD** | ~50 builds/month (2,000 min free). | **$0** |
| **TOTAL (Scenario A)** | | **~$20–37/month** |

> **Why so cheap?** Cloud Run's scale-to-zero means you only pay for the seconds the chatbot is processing queries. For ~100 queries/day, that's roughly 5–20 minutes of actual compute per day — not 24 hours.

---

#### Scenario B: Scaled to Full WYDOT Corpus (1,300+ Documents)

When you ingest all WYDOT documents, the knowledge graph exceeds the Neo4j free tier.

| Service | Change from Scenario A | Monthly Cost |
|---------|----------------------|-------------|
| **Neo4j Aura Professional** | Upgrade from free tier. 1 GB instance (minimum). $65/GB/month. | **$65** |
| **Cloud SQL PostgreSQL** | Same `db-f1-micro`. | **~$13** |
| **Cloud Run — Chatbot** | Same config, potentially more queries. | **$10–25** |
| **Cloud Run — Admin** | Same. | **$1–3** |
| **Gemini API** | May exceed free tier with higher volume. ~$0.075/1M input tokens. | **$0–5** |
| **Everything else** | Same. | **$0.26** |
| **TOTAL (Scenario B)** | | **~$90–112/month** |

> **The big jump:** Neo4j Aura Professional at $65/month is the single largest cost. This is the price of a production knowledge graph with unlimited nodes, daily backups, and SLA.

---

#### Scenario C: Full Agentic + Visual (Future Architecture)

Adding GPU-powered ColPali visual service and autonomous agents.

| Service | Change from Scenario B | Monthly Cost |
|---------|----------------------|-------------|
| **Cloud Run + L4 GPU** | Visual retrieval service. Scale-to-zero. ~30 min/day active. L4 = ~$0.81/hour. | **$10–25** |
| **Cloud Tasks / Scheduler** | Triggers for autonomous agents. | **$1–2** |
| **Cloud Run — Agent Workers** | CPU-only async workers for permits, compliance. | **$3–8** |
| **Everything from Scenario B** | Same. | **$90–112** |
| **TOTAL (Scenario C)** | | **~$105–150/month** |

> **GPU savings:** With scale-to-zero, the L4 GPU costs $10–25/month (only active when visual queries come in). On GKE with an always-on GPU, the same L4 would cost ~$583/month.

---

#### Cost Comparison Summary

| Scenario | Monthly Cost | What You Get |
|----------|-------------|-------------|
| **A: Current** | **$20–37** | Text chatbot + free knowledge graph |
| **B: Full Corpus** | **$90–112** | Text chatbot + production knowledge graph (1,300+ docs) |
| **C: Full Agentic** | **$105–150** | Text + visual + agents + GPU + production graph |

---

### 8.6 Deployment Slide Content

#### SLIDE: Current Cloud Deployment Architecture

**Title:** Production Deployment: Google Cloud Run

**Content:**
- **Fully Serverless:** No servers to manage — Google handles scaling, patching, and load balancing
- **Scale-to-Zero:** When no one is using the chatbot, cost drops to near-zero
- **CI/CD Pipeline:** Push to GitHub → Automatic build → Deploy to Cloud Run (zero-downtime)
- **Security:** Workload Identity Federation (keyless auth), Secret Manager for credentials
- **Database:** Cloud SQL PostgreSQL for chat history, Neo4j Aura for knowledge graph
- **Current Cost:** ~$15–25/month for the full stack

---

#### SLIDE: Future Multi-Service Architecture

**Title:** Scaling to Multi-Service: GPU + Agents

**Content:**
- **Phase 1 (Now):** Single Cloud Run service (CPU-only) for text chatbot → ~$15/month
- **Phase 2 (Visual):** Add GPU-enabled Cloud Run service for ColPali visual search → billed per-second, only when visual queries are submitted
- **Phase 3 (Agents):** Separate agent worker services triggered by Cloud Tasks → async processing for permit drafting, compliance audits
- **Phase 4 (Autonomous):** Cloud Scheduler triggers weekly Spec Monitor and Compliance Watchdog agents
- **Key Principle:** Each service scales independently. The chatbot stays cheap. GPU only runs when needed.

