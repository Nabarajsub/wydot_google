# WYDOT Intelligent Document Assistant: Future Directions & Integration Roadmap
## Unifying Multimodal RAG, Knowledge Graphs, Compliance Checking, and Agentic Automation
### Nabaraj Subedi — March 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Analysis of Existing Systems](#2-analysis-of-existing-systems)
   - 2.1 ColPali Multimodal RAG (copalirag/)
   - 2.2 Neo4j Multimodal Knowledge Graph (neo4j/)
   - 2.3 WYDOT Agent System (wydot_agents/)
3. [Integration Architecture: The Unified Chatbot](#3-integration-architecture-the-unified-chatbot)
4. [Phase 1: Multimodal Knowledge Graph](#4-phase-1-multimodal-knowledge-graph)
5. [Phase 2: Agentic Retrieval & Compliance](#5-phase-2-agentic-retrieval--compliance)
6. [Phase 3: Autonomous WYDOT Agents](#6-phase-3-autonomous-wydot-agents)
7. [Technical Deep Dives](#7-technical-deep-dives)
8. [Research Contributions & Novelty](#8-research-contributions--novelty)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [References](#10-references)

---

## 1. Executive Summary

This document synthesizes three independent research explorations conducted during the WYDOT project and proposes a roadmap to unify them into a single, production-ready chatbot. The three systems are:

| System | Location | Core Innovation | Status |
|--------|----------|----------------|--------|
| **ColPali Multimodal RAG** | `copalirag/` | Visual retrieval of engineering drawings using late-interaction embeddings + VLM-based QA & compliance checking | Evaluated on HPC (78-question benchmark) |
| **Neo4j Multimodal Graph** | `neo4j/ingest_multimodal.py` + `app_mrag.py` | Dual-index graph storing both text chunks and visual chunks (images, video frames) with CLIP+BLIP+Whisper | Prototype complete |
| **WYDOT Agent System** | `wydot_agents/` | Multi-agent architecture with Coordinator, Retrieval Specialist, Compliance Analyst, Action Executor, and Proactive Alerting | Prototype with HITL (Human-in-the-Loop) |

**The Vision:** A single chatbot where a WYDOT engineer can:
1. Ask questions about specifications (text RAG)
2. Ask questions about standard plan drawings (visual RAG)
3. Upload a design for automated compliance checking
4. Get proactive alerts when policy changes affect their projects
5. Execute permitted actions (draft permits, notifications) with human approval

---

## 2. Analysis of Existing Systems

### 2.1 ColPali Multimodal RAG (`copalirag/`)

#### 2.1.1 Architecture Overview

The ColPali system implements a **Retrieve-then-Read** pipeline for visual document understanding:

```
User Query (text)
    |
    v
[ColPali Retriever] -----> Late Interaction (MaxSim) scoring
    |                       against pre-indexed page embeddings
    v
Top-K relevant page images
    |
    v
[VLM Generator] ---------> Qwen2.5-VL-7B or Gemini Flash/Pro
    |                       processes images + query
    v
Natural language answer / Compliance JSON report
```

#### 2.1.2 Key Technical Components

**1. ColPali Retriever (vidore/colpali-v1.2)**
- Based on PaliGemma vision encoder with a learned projection to 128-dim multi-vector space
- Each document page produces ~1,030 visual token embeddings (128-dim each)
- Query text is tokenized and projected to the same 128-dim space
- **MaxSim (Late Interaction)** scoring:
  ```
  Score = SUM over query tokens [MAX over visual tokens (dot_product(q_i, v_j))]
  ```
- This is fundamentally different from CLIP (single-vector cosine similarity) because it preserves spatial/token-level matching

**2. Multiple Indexing Strategies Explored**

| Indexer | File | Technique | Metadata |
|---------|------|-----------|----------|
| Basic ColPali | `indexer.py` | Raw embedding + path | None |
| Gemini-Enhanced | `index_gemini.py` | ColPali embedding + Gemini Flash metadata extraction | plan_id, revision_date, title, category, keywords |
| Binary Quantized | `colpali hpc/create_bq_index.py` | 1-bit quantization of embeddings (128 bits = 16 bytes/vector) | Same as Gemini-enhanced |
| Token-Pruned | `colpali hpc/create_pruned_index.py` | Remove low-information visual tokens to reduce index size | Same as Gemini-enhanced |
| VisionRAG Pyramid | `Visionrag/create_visionrag_index.py` | 3-level text extraction (Global/Structural/Factual) via Qwen-VL + MiniLM embeddings | Per-level text embeddings |

**3. Generation Variants**

| App | Generator | Use Case |
|-----|-----------|----------|
| `app.py` | Qwen2.5-VL-7B | Standard Plan QA (ColPali retrieve + Qwen answer) |
| `geminiapp.py` | Gemini Flash/Pro | Standard Plan QA (ColPali retrieve + Gemini answer) |
| `qwenapp.py` / `qwenapp2.py` | Qwen2.5-VL-7B | Publication-ready layout for research paper screenshots |
| `gemini_compliance.py` | Gemini Flash | Structured JSON compliance audit (PASS/FAIL/UNCLEAR) |
| `qwen_complaince.py` | Qwen2.5-VL-7B | On-premise compliance with forensic reasoning + PDF export |
| `compliance_sec_gemini.py` | Gemini Pro | Forensic audit with HALLUCINATION_SUSPECTED detection |

**4. Compliance Checking Pipeline (Key Innovation)**

```
User uploads Proposed Design (image)
    |
    v
[ColPali Retriever] -----> Retrieves top-K matching WYDOT Standard Plans
    |
    v
[VLM Compliance Agent] --> Compares Design vs Standards
    |                      Outputs structured JSON:
    |                      {
    |                        "segment_name": "Section A-A",
    |                        "check_type": "DIMENSION",
    |                        "standard_requirement": "12 ft lane width",
    |                        "design_observation": "10 ft lane width",
    |                        "verdict": "FAIL",
    |                        "forensic_reasoning": "..."
    |                      }
    v
Formal Audit Report (Streamlit table + downloadable PDF/CSV)
```

**5. Evaluation Framework**

| Component | File | Metrics |
|-----------|------|---------|
| RAG Benchmark | `evalrag.py` | Retrieval Hit Rate, Visual Similarity, Judge Score (Gemini-as-judge) |
| Compliance Eval | `evaldatacompliance.py` | Dimensional Accuracy, Visual Interpretation, Logical Reasoning (Gemini-as-judge) |
| CLIP Baseline | `benchmark/benchmark_clip.py` | Recall@5, Latency |
| LayoutLM Baseline | `benchmark/benchmark_layoutlm.py` | Recall@5, Latency |
| OCR+BGE Baseline | `benchmark/benchmark_ocr_bge.py` | Recall@5, Latency |
| VisionRAG | `Visionrag/benchmark_visionrag.py` | Recall@5 with RRF fusion, Latency |
| HPC Efficiency | `colpali hpc/benchmark_efficiency.py` | Storage reduction, Latency impact |

**6. Explainability (Key Research Contribution)**

`explanability/run_explainability.py` implements a novel visual explainability pipeline:
1. **ColPali Attention Heatmap:** Extract patch-level attention scores from MaxSim interaction matrix
2. **OCR-Guided Region Extraction:** Use Tesseract to find text regions, weight by heatmap intensity
3. **Focused Generation:** Send only the high-attention crops to Qwen-VL for targeted answer generation
4. **Visual Report Card:** Generate publication-quality images combining focus strip + Q/A/GT text

This is a **novel contribution** because it provides spatial explainability for vision-language retrieval, showing exactly which parts of an engineering drawing the model focused on.

---

### 2.2 Neo4j Multimodal Knowledge Graph (`neo4j/`)

#### 2.2.1 Architecture Overview

The multimodal graph extends the existing text-only Neo4j knowledge graph (V3/V4) with visual and audio modalities:

```
[Document]
    |
    |---[:HAS_CHUNK]--->  [TextChunk]     (MiniLM embedding, 384-dim)
    |                      - text, source, page, year
    |
    |---[:HAS_VISUAL]-->  [VisualChunk]   (CLIP embedding, 512-dim)
                           - local_path, page/timestamp, description (BLIP caption)
                           - type: "PDF Page" | "Video Frame"
```

#### 2.2.2 Multimodal Ingestion Pipeline (`ingest_multimodal.py`)

**Model Manager** loads 4 AI models on-demand:

| Model | Purpose | Output |
|-------|---------|--------|
| **SentenceTransformer (MiniLM-L6-v2)** | Text embedding | 384-dim vector |
| **CLIP (ViT-B/32)** | Image embedding | 512-dim vector |
| **BLIP (Salesforce)** | Image captioning | Natural language description |
| **Whisper (base)** | Audio transcription | Timestamped text segments |

**PDF Processing:**
1. Extract text per page → MiniLM embedding → TextChunk node
2. Render page as image (PyMuPDF 1.5x) → CLIP embedding + BLIP caption → VisualChunk node
3. Both linked to parent Document via [:HAS_CHUNK] and [:HAS_VISUAL]

**Video Processing:**
1. Extract audio → Whisper transcription → TextChunk nodes (with start/end timestamps)
2. Extract frames every 5 seconds → CLIP embedding + BLIP caption → VisualChunk nodes (with timestamp)
3. Both linked to parent Document

#### 2.2.3 Multimodal Retrieval (`app_mrag.py`)

**Dual Search Pipeline:**
```
User Query
    |
    |---> [MiniLM encode] ---> Neo4j text vector index ---> TextChunk results
    |
    |---> [CLIP text encode] ---> Neo4j visual vector index ---> VisualChunk results
    |
    v
Combined context (text snippets + visual descriptions)
    |
    v
[Gemini Flash LLM] ---> Final answer
    |
    v
UI displays: text evidence + image thumbnails + video playback at timestamp
```

**Key Innovation:** Both text and visual modalities live in the **same graph**, allowing future cross-modal graph traversal (e.g., "Show me the video frame where they discuss the bridge spec from page 45").

---

### 2.3 WYDOT Agent System (`wydot_agents/`)

#### 2.3.1 Architecture Overview

A multi-agent system built with a **Coordinator-Specialist** pattern (similar to LangGraph's supervisor architecture):

```
User Query
    |
    v
[Coordinator] ---------> Routes to specialists based on query analysis
    |
    |---> [Retrieval Specialist] --> HyDE + Neo4j Vector + Full-Text Search
    |---> [Project Data Specialist] --> Bridge/Project database lookup
    |
    v
[Compliance Analyst] ---> Risk assessment + confidence scoring
    |
    |--- if high-risk ---> [Human-in-the-Loop] --> Approve/Reject buttons
    |
    v
[Action Executor] ------> Tool execution + Decision Ledger recording
    |
    |---> draft_permit()
    |---> send_critical_notification()
    |
    v
Final Answer (with audit trail in Neo4j)
```

#### 2.3.2 Agent Breakdown

**1. AgentState (base.py)**
```python
class AgentState(TypedDict):
    query: str
    task_list: List[str]      # Which specialists to invoke
    context: List[str]        # Retrieved documents/data
    results: Dict[str, any]   # Compliance analysis results
    next_step: str            # Next agent in pipeline
    final_answer: str         # Final response to user
```

**2. Coordinator (coordinator.py) - "The Brain"**
- Analyzes query keywords to decide routing
- Currently rule-based: detects bridge IDs, project references
- Always includes retrieval_specialist; adds project_data_specialist for project queries

**3. Retrieval Specialist (retrieval_specialist.py) - "The Librarian"**
- **HyDE (Hypothetical Document Embedding):** Generates a fake technical document snippet via Gemini, then concatenates with original query for better vector retrieval
- **Neo4j Vector Search** with custom Cypher: Chunk -> Section -> Document traversal
- **Full-Text Search Fallback:** If vector results < 3, supplements with Neo4j full-text index
- Uses Gemini embedding-001 for query embedding

**4. Project Data Specialist (project_data_specialist.py) - "The Engineer"**
- Queries structured project database (mock: Snake River Bridge #74, Alkali Creek Overpass)
- Returns: bridge ratings, load limits, HMA compliance status, inspection dates, recent notes
- Enriches context with project-specific structured data alongside retrieved documents

**5. Compliance Analyst (compliance_analyst.py) - "The Judge"**
- LLM-based assessment: compares user query against retrieved context
- Outputs: Assessment summary, confidence_score (0.0-1.0), requires_human_approval flag
- Risk detection: flags overweight loads, structural modifications, emergency requests

**6. Action Executor (action_executor.py) - "The Hand"**
- **Tool Execution:** Calls external tools based on compliance score:
  - `draft_permit()` if score >= 0.90 and permit-related query
  - `send_critical_notification()` if emergency/maintenance detected
- **Decision Ledger:** Records every decision in Neo4j as a `Decision` node linked to `WYDOT_DECISION_LEDGER`
- Generates final professional response incorporating tool outputs

**7. Proactive Alert System (proactive_alert.py)**
- Autonomous event detection (simulated): scans for policy updates
- Auto-notifies affected projects when relevant standards change
- Example: New Section 401 HMA spec detected -> notifies Bridge ID-104 team about Summer 2026 overlay impact

**8. Chainlit App (agent_app.py)**
- Step-by-step execution visibility with nested Chainlit Steps
- **Self-Correction Loop:** Retries with broadened query if context is empty
- **Human-in-the-Loop (HITL):** Approve/Reject action buttons for high-risk decisions
- Full pipeline: Coordinator -> Retrieval -> [Project Data] -> Compliance -> [HITL] -> Action

---

## 3. Integration Architecture: The Unified Chatbot

### 3.1 The Problem

Currently, these three systems are isolated:
- **copalirag** runs on HPC with GPU, uses Streamlit, operates on standard plan images only
- **neo4j multimodal** runs on a different Neo4j instance, uses CLIP+MiniLM (not ColPali)
- **wydot_agents** connects to the V4 Gemini Neo4j graph, has no visual capabilities

### 3.2 The Unified Architecture

```
                          ┌────────────────────────┐
                          │     UNIFIED CHATBOT     │
                          │   (Chainlit / Cloud Run) │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │      COORDINATOR        │
                          │   (LLM-Based Router)    │
                          │                         │
                          │  Analyzes query intent:  │
                          │  - Text spec question?   │
                          │  - Visual plan question?  │
                          │  - Compliance check?      │
                          │  - Project lookup?        │
                          │  - Action request?        │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
  ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
  │  TEXT RETRIEVAL    │   │  VISUAL RETRIEVAL  │   │  PROJECT DATA     │
  │  SPECIALIST        │   │  SPECIALIST        │   │  SPECIALIST       │
  │                    │   │                    │   │                   │
  │ HyDE + Neo4j KG   │   │ ColPali MaxSim     │   │ Bridge DB lookup  │
  │ Vector + Fulltext  │   │ + Gemini metadata  │   │ Inspection data   │
  │ Graph traversal    │   │ index              │   │ Load limits       │
  │ Entity resolution  │   │                    │   │                   │
  └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │  COMPLIANCE ANALYST     │
                          │                         │
                          │ Text compliance:        │
                          │   Spec comparison       │
                          │ Visual compliance:      │
                          │   Design vs Standard    │
                          │ Risk scoring (0.0-1.0)  │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │  HUMAN-IN-THE-LOOP      │
                          │  (if high-risk)         │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │  ACTION EXECUTOR        │
                          │                         │
                          │ - Draft permits          │
                          │ - Send notifications     │
                          │ - Record to ledger       │
                          │ - Generate PDF reports   │
                          └────────────────────────┘
```

### 3.3 The Unified Knowledge Graph Schema

```
[Document]──[:SUPERSEDES]──>[Document]
    │
    ├──[:HAS_SECTION]──>[Section]
    │                       │
    │                       └──[:HAS_CHUNK]──>[Chunk]──[:NEXT_CHUNK]──>[Chunk]
    │                                           │
    │                                           └──[:MENTIONS]──>[Entity]
    │                                                              (Material, Standard,
    │                                                               TestMethod, etc.)
    │
    ├──[:HAS_VISUAL]──>[VisualChunk]
    │                     │
    │                     ├── local_path (page screenshot)
    │                     ├── clip_embedding (512-dim, for visual search)
    │                     ├── colpali_embedding (multi-vector, for late-interaction search)
    │                     ├── description (BLIP/Gemini caption)
    │                     └── type: "PDF Page" | "Standard Plan" | "Video Frame"
    │
    ├──[:HAS_AUDIO]──>[AudioChunk]
    │                     │
    │                     ├── text (Whisper transcript)
    │                     ├── text_embedding (MiniLM/Gemini)
    │                     ├── start_time, end_time
    │                     └── source_video
    │
    └──[:HAS_STANDARD_PLAN]──>[StandardPlan]
                                  │
                                  ├── plan_id: "799-1A"
                                  ├── category: "Traffic"
                                  ├── colpali_embedding (multi-vector)
                                  ├── gemini_metadata (title, revision, keywords)
                                  └──[:REFERENCES]──>[Entity]
```

**Key Addition:** StandardPlan nodes bridge the gap between the text-based knowledge graph and the visual standard plan index. A single graph query can now traverse:

```cypher
-- "Show me the standard plan that references Portland Cement in Section 801"
MATCH (e:Entity {name: "Portland Cement"})
  <-[:MENTIONS]-(c:Chunk)
  -[:HAS_CHUNK]-(s:Section {name: "SECTION 801"})
  -[:HAS_SECTION]-(d:Document)
  -[:HAS_STANDARD_PLAN]->(sp:StandardPlan)
RETURN sp.plan_id, sp.local_path, sp.category
```

---

## 4. Phase 1: Multimodal Knowledge Graph

### 4.1 Objective

Merge the ColPali visual index with the Neo4j knowledge graph so that **text chunks, visual pages, and standard plan images all live in the same graph** with cross-modal relationships.

### 4.2 Implementation Steps

**Step 1: Unify Embeddings**

| Modality | Current | Unified |
|----------|---------|---------|
| Text chunks | Gemini embedding-001 (768-dim) | Gemini embedding-001 (768-dim) - keep |
| Visual pages (PDF screenshots) | CLIP ViT-B/32 (512-dim) | Gemini multimodal embedding OR keep CLIP for graph, add ColPali for retrieval |
| Standard plans | ColPali multi-vector (128-dim x ~1030 tokens) | Store ColPali embeddings externally (too large for Neo4j properties); use Neo4j node IDs as index keys |
| Audio segments | MiniLM (384-dim) | Gemini embedding-001 (768-dim) - align with text |

**Step 2: Ingest Standard Plans into Neo4j**

```python
# New ingestion: Standard Plans → Neo4j nodes + ColPali external index
def ingest_standard_plan(pdf_path, graph, colpali_model, colpali_proc):
    pages = convert_from_path(pdf_path, dpi=200)

    # 1. Gemini metadata extraction (from index_gemini.py pattern)
    metadata = get_gemini_metadata(pages[0], os.path.basename(pdf_path))

    for i, page in enumerate(pages):
        # 2. Create StandardPlan node in Neo4j
        graph.query("""
            MERGE (d:Document {name: $doc_name})
            CREATE (sp:StandardPlan {
                plan_id: $plan_id, page: $page_num,
                category: $category, title: $title,
                local_path: $img_path,
                description: $caption
            })
            SET sp.clip_embedding = $clip_emb
            MERGE (d)-[:HAS_STANDARD_PLAN]->(sp)
        """, params)

        # 3. Store ColPali embedding in external index (too large for Neo4j)
        colpali_emb = compute_colpali_embedding(page, colpali_model, colpali_proc)
        external_index.add(node_id=sp_id, embedding=colpali_emb)
```

**Step 3: Dual-Index Retrieval**

At query time, run both text and visual retrieval in parallel:

```python
async def unified_search(query, k_text=10, k_visual=5, k_plans=3):
    # 1. Text retrieval (existing V4 pipeline)
    text_results = await neo4j_vector_search(query, index="wydot_gemini_index", k=k_text)

    # 2. Visual retrieval via ColPali (external index)
    plan_results = colpali_maxsim_search(query, external_index, k=k_plans)

    # 3. Graph enrichment: traverse from chunks/plans to entities/documents
    enriched_results = await graph_enrich(text_results + plan_results)

    # 4. Rerank with FlashRank (text) or MaxSim score (visual)
    return rerank_and_merge(enriched_results)
```

### 4.3 Research Contribution

**Novelty:** Combining late-interaction visual retrieval (ColPali) with structured knowledge graph traversal. No existing work combines MaxSim-based visual search with entity-aware graph RAG.

---

## 5. Phase 2: Agentic Retrieval & Compliance

### 5.1 Objective

Replace the current single-pipeline chatbot with a **multi-agent architecture** where the LLM decides at query time which retrieval strategy to use.

### 5.2 Agent Integration into Current Chatbot

**Upgrading `chatapp_gemini.py` (V4) to agentic:**

```
V4 Chatbot (Current):
    Query → Vector Search → Rerank → LLM Generate → Answer

V5 Chatbot (Agentic):
    Query → Coordinator → [Text Agent | Visual Agent | Project Agent | Compliance Agent]
              → Merge Context → Compliance Check → [HITL] → Action → Answer
```

### 5.3 New Agent Roster

| Agent | Source | Role | Integration |
|-------|--------|------|-------------|
| **Coordinator** | `wydot_agents/coordinator.py` | Route queries to appropriate specialists | Upgrade to LLM-based intent classification (not keyword rules) |
| **Text Retrieval Specialist** | `wydot_agents/retrieval_specialist.py` | HyDE + Neo4j vector + full-text search | Already integrated with V4 graph |
| **Visual Retrieval Specialist** | NEW (from `copalirag/`) | ColPali MaxSim search for standard plan images | New agent wrapping ColPali retrieval |
| **Compliance Analyst** | `wydot_agents/compliance_analyst.py` + `copalirag/gemini_compliance.py` | Both text-based and visual compliance checking | Merge text compliance (agent) with visual compliance (ColPali) |
| **Project Data Specialist** | `wydot_agents/project_data_specialist.py` | Bridge/project structured data lookup | Connect to real WYDOT project database |
| **Action Executor** | `wydot_agents/action_executor.py` | Draft permits, send notifications, record decisions | Integrate with external WYDOT systems |
| **Explainability Agent** | NEW (from `copalirag/explanability/`) | Generate attention heatmaps for visual answers | Provide spatial grounding for visual QA |

### 5.4 LLM-Based Coordinator (Upgrade from Rule-Based)

```python
COORDINATOR_PROMPT = """
You are the WYDOT Chatbot Coordinator. Analyze the user's query and decide
which specialist agents to invoke. Return a JSON plan.

Available Agents:
- text_retrieval: For questions about specifications, standards, technical requirements
- visual_retrieval: For questions about standard plan drawings, dimensions, diagrams
- compliance_check: For verifying designs against standards (requires uploaded image)
- project_data: For questions about specific bridges, projects, inspection data
- action_executor: For permit drafting, notifications, or operational requests

User Query: {query}
Has Attached Image: {has_image}

Return JSON:
{
    "agents": ["text_retrieval", "visual_retrieval"],
    "reasoning": "User asks about lane taper dimensions which is both in specs and drawings",
    "compliance_needed": false,
    "risk_level": "LOW"
}
"""
```

### 5.5 Unified Compliance Pipeline

The most powerful integration: merging **text-based compliance** (checking against specification text) with **visual compliance** (checking against standard plan drawings):

```
User uploads Proposed Design
    |
    v
[Coordinator] --> Detects compliance intent
    |
    |---> [Text Retrieval] --> Find relevant spec sections
    |                          (e.g., Section 401 HMA requirements)
    |
    |---> [Visual Retrieval] --> Find relevant standard plan images
    |                            (e.g., Standard Plan 799-1A lane layout)
    |
    v
[Unified Compliance Analyst]
    |
    |---> Text compliance: Check dimensions against spec values
    |---> Visual compliance: VLM compares design image against standard plan image
    |
    v
Combined Compliance Report:
{
    "text_checks": [
        {"requirement": "Section 401.3.2 - HMA thickness", "verdict": "PASS"}
    ],
    "visual_checks": [
        {"segment": "Section A-A", "check_type": "DIMENSION",
         "standard_requirement": "12 ft lane", "design_value": "10 ft", "verdict": "FAIL"}
    ],
    "overall_verdict": "FAIL",
    "confidence": 0.72,
    "requires_human_review": true
}
```

---

## 6. Phase 3: Autonomous WYDOT Agents

### 6.1 Objective

Move beyond reactive chatbot interactions to **proactive, autonomous agents** that monitor WYDOT operations and take independent action (with appropriate human oversight).

### 6.2 Independent Agent Designs

#### Agent 1: Policy Change Monitor

```
[Policy Change Monitor Agent]
    |
    |--- Watches for: New PDF uploads, standard plan revisions, policy updates
    |
    v
[Document Ingestion Pipeline]
    |
    |--- Auto-ingests new documents into the knowledge graph
    |--- Detects SUPERSEDES relationships with existing documents
    |
    v
[Impact Analysis]
    |
    |--- Queries graph: "Which projects reference affected standards?"
    |--- Example: New Section 401 spec -> Find all projects with HMA work
    |
    v
[Proactive Notification]
    |
    |--- Sends alerts to affected project managers
    |--- Creates impact summary with before/after comparison
    |--- Records alert in Decision Ledger
```

**WYDOT Use Case:** When a new edition of the Standard Specifications is published, this agent automatically:
1. Ingests the new PDF
2. Creates SUPERSEDES links to the old edition
3. Identifies all active projects that reference changed sections
4. Notifies project managers with specific change summaries

#### Agent 2: Automated Compliance Pre-Check

```
[Compliance Pre-Check Agent]
    |
    |--- Watches for: New design submissions in WYDOT portal
    |
    v
[Multi-Modal Compliance Pipeline]
    |
    |--- Step 1: Extract design type from submission (bridge, road, drainage)
    |--- Step 2: Text retrieval for relevant specification requirements
    |--- Step 3: Visual retrieval for relevant standard plan drawings
    |--- Step 4: VLM comparison (design vs standard)
    |
    v
[Auto-Generated Compliance Report]
    |
    |--- Creates draft compliance report (PDF) with PASS/FAIL/UNCLEAR per segment
    |--- Flags items needing human engineer review
    |--- Routes to appropriate WYDOT reviewer based on design category
    |
    v
[Human Review Queue]
    |
    |--- Engineer reviews auto-generated report
    |--- Approves, rejects, or requests modifications
    |--- Decision recorded in ledger
```

**WYDOT Use Case:** Instead of engineers manually checking every design against standards, this agent performs an initial automated compliance scan, reducing review time by flagging obvious violations and passing clear designs through faster.

#### Agent 3: Permit Workflow Automation

```
[Permit Workflow Agent]
    |
    |--- Triggered by: Permit application submission
    |
    v
[Application Analysis]
    |
    |--- Extract permit type (overweight, oversized, special use)
    |--- Lookup route/bridge data from project database
    |--- Check load limits against bridge ratings
    |
    v
[Compliance Verification]
    |
    |--- Verify permit requirements against current specs
    |--- Check bridge condition ratings (must be > threshold)
    |--- Verify HMA compliance status on route
    |
    v
[Decision & Routing]
    |
    |--- If score > 0.95 and load within limits: Auto-approve (with HITL confirmation)
    |--- If score 0.70-0.95: Flag for engineer review
    |--- If score < 0.70: Auto-reject with explanation
    |
    v
[Permit Generation]
    |
    |--- Generate permit document with conditions
    |--- Record in Decision Ledger
    |--- Send to applicant and file in WYDOT system
```

#### Agent 4: Bridge Health Dashboard Agent

```
[Bridge Health Agent]
    |
    |--- Scheduled: Runs daily/weekly
    |
    v
[Data Collection]
    |
    |--- Query project database for all bridges
    |--- Check inspection dates (flag overdue inspections)
    |--- Cross-reference with recent standard changes
    |
    v
[Risk Assessment]
    |
    |--- Score each bridge: condition rating + age + traffic volume + recent changes
    |--- Identify bridges needing attention
    |
    v
[Dashboard Update]
    |
    |--- Update visual dashboard with current status
    |--- Generate weekly summary report
    |--- Alert if any bridge drops below threshold
```

#### Agent 5: Training & Onboarding Assistant

```
[Training Agent]
    |
    |--- Triggered by: New employee assignment
    |
    v
[Curriculum Generation]
    |
    |--- Based on role, generate personalized learning path
    |--- Pull relevant sections from Standard Specifications
    |--- Include visual standard plans for their project type
    |
    v
[Interactive Q&A Training]
    |
    |--- Present questions from the 8,000+ validated Q&A dataset
    |--- Adaptive difficulty based on performance
    |--- Generate new questions from relevant spec sections
    |
    v
[Competency Assessment]
    |
    |--- Track correct/incorrect answers
    |--- Identify knowledge gaps
    |--- Recommend additional reading with direct links to spec sections
```

---

## 7. Technical Deep Dives

### 7.1 ColPali MaxSim vs CLIP: Why Late Interaction Matters for Engineering Drawings

**CLIP (Single-Vector):**
```
Query: "What is the lane width in the acceleration taper?"
CLIP encodes this to ONE 512-dim vector.
Each standard plan image is ONE 512-dim vector.
Comparison: Single cosine similarity.
Problem: CLIP sees the "overall vibe" of the page but misses specific dimensions.
```

**ColPali (Late Interaction):**
```
Query: "What is the lane width in the acceleration taper?"
ColPali encodes this to ~15 token vectors (128-dim each).
Each standard plan page is ~1030 patch vectors (128-dim each).
Comparison: Every query token finds its best matching visual patch.
- "lane" matches the lane diagram region
- "width" matches the dimension annotation
- "taper" matches the taper geometry
Result: Fine-grained spatial matching that CLIP fundamentally cannot do.
```

**Benchmark Evidence (from project evaluation):**
- ColPali achieves significantly higher Recall@5 on WYDOT standard plans compared to CLIP and OCR+BGE baselines
- The improvement is most dramatic on "Dimensional Accuracy" questions where spatial grounding matters

### 7.2 HyDE for Technical Document Retrieval

The WYDOT agent system uses **Hypothetical Document Embedding (HyDE)** to improve retrieval quality:

```
User Query: "What are the HMA overlay requirements for bridges?"

HyDE generates:
"Section 401.3.2 - Hot Mix Asphalt Overlay Requirements for Bridge Decks:
The minimum overlay thickness shall be 2 inches. The HMA mix design shall
conform to AASHTO M323 Superpave specifications with PG 64-28 binder grade..."

Combined Search Query: Original + HyDE snippet
Result: Better vector similarity with actual specification text
```

**Why this matters:** Technical queries often use different vocabulary than the spec documents. HyDE bridges this gap by generating text that "sounds like" the document, improving embedding alignment.

### 7.3 Binary Quantization for ColPali at Scale

The `colpali hpc/create_bq_index.py` explores 1-bit quantization:

```
Original: 128-dim float16 per visual token = 256 bytes/token
Quantized: 128 bits packed into 16 uint8 bytes = 16 bytes/token
Compression: 16x reduction in storage

For 1,300 documents x ~10 pages x 1030 tokens:
- Original: ~6.4 GB
- Quantized: ~400 MB
```

**Trade-off:** Binary quantization uses Hamming distance instead of dot product, sacrificing some recall accuracy for massive storage and speed gains. The HPC benchmarks measure this trade-off quantitatively.

### 7.4 Pyramid Indexing (VisionRAG) with RRF Fusion

The `Visionrag/` system implements a novel 3-level indexing approach:

```
Each standard plan page is analyzed by Qwen-VL to extract:

Level 1 (GLOBAL):   "This is an earthwork grading plan showing cut and fill sections"
Level 2 (STRUCTURAL): "FILL SECTION, CUT SECTION, SIDEHILL CUT, Note A, Note B..."
Level 3 (FACTUAL):   "6 inches [150mm], 95% compaction, 1V:4H slope ratio..."

Each level gets its own MiniLM embedding vector.

At query time:
- Query is encoded once
- Searched against all 3 indices
- Results fused using Reciprocal Rank Fusion (RRF):
  Score(doc) = SUM over levels [1 / (k + rank_in_level)]
```

**Research Value:** This can be compared against ColPali's native visual understanding to determine whether explicit text extraction (VisionRAG) or implicit visual understanding (ColPali) is more effective for engineering documents.

### 7.5 Decision Ledger: Auditable AI Actions

The agent system records every decision in a Neo4j graph:

```cypher
(l:Ledger {name: 'WYDOT_DECISION_LEDGER'})
    -[:HAS_RECORD]->
(d:Decision {
    timestamp: "2026-03-01T14:30:00",
    query: "Can we approve overweight permit for Bridge ID-104?",
    compliance_notes: "Load within limits, bridge rated Fair",
    final_answer: "Approved with conditions..."
})
```

**Why this matters for WYDOT:** Government agencies require auditable decision trails. Every AI-assisted decision is recorded with full context, enabling review and accountability.

---

## 8. Research Contributions & Novelty

### 8.1 Novel Contributions from Integration

| Contribution | Description | Novelty |
|-------------|-------------|---------|
| **Visual GraphRAG** | First system combining ColPali late-interaction retrieval with Neo4j knowledge graph traversal for engineering documents | No existing work combines MaxSim visual search with entity-aware graph RAG |
| **Multi-Modal Compliance Checking** | Automated compliance pipeline that checks designs against both text specifications AND visual standard plans simultaneously | Extends existing text-only compliance checking with visual grounding |
| **Agentic RAG with HITL** | Multi-agent system where the LLM autonomously decides retrieval strategy, performs compliance assessment, and executes actions with human approval gates | Combines autonomous agents with mandatory human oversight for safety-critical infrastructure decisions |
| **Visual Explainability for Engineering RAG** | ColPali attention heatmaps showing which parts of an engineering drawing the model focused on, with OCR-guided region extraction | Novel application of vision-language attention visualization to engineering document QA |
| **Pyramid Indexing vs Native Visual Understanding** | Systematic comparison of explicit text extraction (VisionRAG) vs implicit visual understanding (ColPali) for technical document retrieval | New benchmark methodology for comparing retrieval approaches on engineering drawings |
| **Binary Quantization for ColPali** | 1-bit quantization achieving 16x storage compression with measured recall trade-offs on engineering documents | First application of BQ to late-interaction visual retrieval on technical documents |
| **Autonomous Policy Change Impact Analysis** | Agent that automatically detects standard updates, identifies affected projects via graph traversal, and sends targeted notifications | Novel combination of document ingestion automation with impact analysis |

### 8.2 Potential Paper Topics

1. **"Visual GraphRAG: Integrating Late-Interaction Visual Retrieval with Knowledge Graph Traversal for Engineering Document QA"**
   - Conference: AAAI, ACL, or EMNLP
   - Core contribution: The unified graph schema + dual retrieval pipeline

2. **"Automated Visual Compliance Checking for Transportation Infrastructure Using Multi-Modal RAG"**
   - Conference: Transportation Research Board (TRB) Annual Meeting
   - Core contribution: ColPali retrieval + VLM compliance checking pipeline with real WYDOT standards

3. **"Agentic RAG with Human-in-the-Loop for Safety-Critical Infrastructure Decisions"**
   - Conference: AAMAS (Autonomous Agents) or CSCW
   - Core contribution: Multi-agent architecture with mandatory human oversight for government decision-making

4. **"Explainable Vision-Language Retrieval for Engineering Drawings: Attention Heatmaps Meet OCR"**
   - Conference: ECCV or CVPR Workshop
   - Core contribution: Spatial explainability pipeline (heatmap + OCR crops + focused generation)

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

| Task | Priority | Complexity | Dependencies |
|------|----------|-----------|--------------|
| Create unified Neo4j schema with StandardPlan and VisualChunk nodes | HIGH | Medium | Neo4j V4 graph running |
| Ingest standard plan images into Neo4j with Gemini metadata | HIGH | Low | `index_gemini.py` pattern |
| Set up ColPali external index with Neo4j node ID mapping | HIGH | Medium | ColPali model + GPU |
| Implement dual-search function (text + visual in parallel) | HIGH | Medium | Unified schema |
| Merge embedding models: align text and audio to Gemini embedding-001 | MEDIUM | Low | Gemini API |

### Phase 2: Agentic Architecture (Weeks 4-6)

| Task | Priority | Complexity | Dependencies |
|------|----------|-----------|--------------|
| Port `wydot_agents/` into existing Chainlit chatapp | HIGH | Medium | Phase 1 complete |
| Upgrade Coordinator from rule-based to LLM-based routing | HIGH | Low | Gemini API |
| Add Visual Retrieval Specialist (ColPali agent) | HIGH | Medium | Phase 1 external index |
| Merge text and visual compliance pipelines | HIGH | High | Both retrieval specialists |
| Implement HITL flow in Chainlit (approve/reject buttons) | MEDIUM | Low | Chainlit Actions |
| Add Decision Ledger recording to Neo4j | MEDIUM | Low | Neo4j connection |

### Phase 3: Autonomous Agents (Weeks 7-10)

| Task | Priority | Complexity | Dependencies |
|------|----------|-----------|--------------|
| Implement Policy Change Monitor (file watcher + auto-ingest) | MEDIUM | Medium | Phase 1 ingestion |
| Implement Compliance Pre-Check Agent | MEDIUM | High | Phase 2 compliance pipeline |
| Connect Project Data Specialist to real WYDOT database | LOW | Medium | WYDOT database access |
| Build Bridge Health Dashboard | LOW | Medium | Project database |
| Implement Training/Onboarding Assistant | LOW | Medium | Q&A dataset |

### Phase 4: Evaluation & Paper Writing (Weeks 11-14)

| Task | Priority | Complexity | Dependencies |
|------|----------|-----------|--------------|
| Run unified system evaluation on 78-question benchmark | HIGH | Medium | Full system running |
| Compare unified vs individual system performance | HIGH | Medium | Benchmark results |
| Run compliance checking evaluation on ground truth set | HIGH | Medium | Compliance pipeline |
| Write Visual GraphRAG paper | HIGH | High | Evaluation results |
| Prepare demo video for WYDOT stakeholders | MEDIUM | Low | Working system |

---

## 10. References

### Internal Project References

| File | Description |
|------|-------------|
| `copalirag/app.py` | Original ColPali + Qwen multimodal app |
| `copalirag/gemini_compliance.py` | Gemini-based visual compliance checking |
| `copalirag/explanability/run_explainability.py` | Visual explainability pipeline |
| `copalirag/Visionrag/create_visionrag_index.py` | Pyramid indexing approach |
| `copalirag/colpali hpc/create_bq_index.py` | Binary quantization research |
| `copalirag/evalrag.py` | RAG evaluation framework |
| `neo4j/ingest_multimodal.py` | Multimodal graph ingestion (PDF + Video) |
| `neo4j/app_mrag.py` | Multimodal RAG chatapp |
| `wydot_agents/coordinator.py` | Agent orchestration |
| `wydot_agents/compliance_analyst.py` | Text compliance checking |
| `wydot_agents/action_executor.py` | Tool execution + decision ledger |
| `wydot_agents/proactive_alert.py` | Autonomous event detection |
| `research/SLIDE_CONTENT.md` | V1-V4 evolution slide content |

### Academic References

1. Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models" (2024) - Foundation for visual retrieval
2. Microsoft Research, "GraphRAG: From Local to Global Text Understanding" (2024) - Knowledge graph RAG patterns
3. Gao et al., "Hypothetical Document Embeddings (HyDE)" (2023) - Query expansion technique used in retrieval specialist
4. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP" (2020) - Core RAG framework
5. Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique" (2024) - Self-reflective retrieval patterns
6. Neo4j + LangChain Integration Patterns - Graph-based retrieval implementations

---

*This document was created as part of the WYDOT Intelligent Document Assistant research project.*
*University of Wyoming - Department of Civil, Architectural & Construction Engineering*
*Principal Investigator: Ahmed Abdelaty, PhD*
