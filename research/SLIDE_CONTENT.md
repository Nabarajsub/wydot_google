# WYDOT Chatbot Evolution: Slide Content
## From Vector RAG to Knowledge Graph RAG
### Nabaraj Subedi — February 2026

---

## SLIDE 1: Title Slide

**Title:** Evolution of WYDOT Intelligent Document Assistant: From Vector Search to Knowledge Graph RAG

**Subtitle:** Version 3 & Version 4 Architecture — Reasoning Behind the Shift to Knowledge Graphs

**Presented By:** Nabaraj Subedi
MS Student and Graduate Research Assistant
Department of Civil, Architectural & Construction Engineering
University of Wyoming

**Principle Investigator:** Ahmed Abdelaty, PhD

---

## SLIDE 2: Version Timeline & Recap

**Title:** System Evolution Timeline

| Version | Date | Architecture | Key Innovation |
|---------|------|-------------|----------------|
| V1 | Fall 2025 | Multiple Vector Stores | Basic semantic search over 2 spec PDFs |
| V2 | Jan 2026 | Router + Supervisor Agent | LLM routing for intent classification + metadata filtering |
| **V3** | **Jan-Feb 2026** | **Neo4j Knowledge Graph (HuggingFace)** | **Graph-based retrieval + FlashRank reranking** |
| **V4** | **Feb 2026** | **Neo4j Knowledge Graph (Gemini)** | **Full entity extraction + semantic chunking + cross-document version chains** |

**Key Shift:** V1-V2 were "search-and-retrieve" systems. V3-V4 are "understand-and-reason" systems powered by a structured Knowledge Graph.

---

## SLIDE 3: Recap — What Was Wrong with V1 & V2

**Title:** Limitations That Drove the Architecture Shift

**V1 Issues (Vector Store Chatbot):**
- Multiple separate vector stores — no unified document representation
- High latency on multi-document queries
- Not scalable to WYDOT's 1,300+ document corpus
- No entity awareness — "Portland Cement" in 2010 and 2021 were just floating text chunks

**V2 Issues (Router + Supervisor Architecture):**
- Router pattern added complexity but the underlying retrieval was still flat vector search
- Metadata filtering by year helped but didn't capture relationships BETWEEN documents
- Comparison engine worked but was expensive — required 2 separate vector searches + LLM synthesis
- No graph structure — couldn't answer "What materials reference AASHTO M85?" (relationship query)

**Root Problem:** Both versions treated documents as bags of text chunks. They had no understanding of document structure, entity relationships, or cross-document connections.

---

## SLIDE 4: Why Knowledge Graphs? The Reasoning

**Title:** The Case for Knowledge Graph RAG (GraphRAG)

**The Core Insight:**
WYDOT documents are NOT flat text. They have rich internal structure:
- **Hierarchical:** Division 100 > Section 101 > Subsection 101.1
- **Relational:** Portland Cement (Material) → REFERENCED_BY → Section 801 → REQUIRES → AASHTO M85 (Standard)
- **Temporal:** 2021 Specs SUPERSEDES 2010 Specs

**Why a Graph Captures This Better Than Vectors:**

| Capability | Vector RAG (V1/V2) | Knowledge Graph RAG (V3/V4) |
|-----------|-------------------|---------------------------|
| "Find Portland Cement requirements" | Approximate text similarity | Exact entity lookup + traversal |
| "What changed 2010 vs 2021?" | 2 searches + LLM fusion | Follow SUPERSEDES edge directly |
| "What specs reference AASHTO M85?" | Cannot answer (no relationships) | Graph traversal via REFERENCES edge |
| "Show all materials in Section 803" | Keyword search (unreliable) | Direct graph query: Section → Chunk → MENTIONS → Material |
| Scaling to 1,300+ docs | New vector store per batch | Single graph with all documents connected |

**Academic Foundation:** Microsoft Research's GraphRAG (2024), Neo4j + LangChain Knowledge Graph patterns

---

## SLIDE 5: Version 3 Architecture Overview

**Title:** Version 3 — First Knowledge Graph Implementation

**Ingestion Pipeline (ingestneo4j.py):**
- **PDF Parser:** PyPDFLoader (LangChain built-in)
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (local, HuggingFace)
- **Chunking:** RecursiveCharacterTextSplitter (fixed 1000 chars, 100 overlap)
- **Metadata:** Extracted from filename + PDF internal properties
  - Year detected via regex from filename
  - Document type inferred (Specification, Report, Memo)
  - Section headers detected via regex (SECTION/DIVISION/CHAPTER)
- **Graph Storage:** Neo4j Aura with vector index (wydot_local_index)
- **Graph Schema:** Chunk nodes with text + embedding + metadata properties
- **Batch Processing:** 50 files per batch with garbage collection

**Chatapp (chatapp.py):**
- **Retrieval:** Neo4j vector similarity search (FETCH_K=25 candidates)
- **Reranking:** FlashRank cross-encoder (top K=10 after reranking)
- **Context Expansion:** Graph-aware neighbor retrieval (same source + same section)
- **Multi-hop:** Query decomposition into 2-3 sub-queries with parallel search
- **LLM Options:** Mistral Large, Gemini 2.5 Flash, 14 OpenRouter models (GPT-5.2, Claude Opus 4.6, etc.)
- **Features:** Authentication, chat history (SQLite), audio input, multimodal file analysis

---

## SLIDE 6: Version 3 — Graph Schema

**Title:** V3 Knowledge Graph Structure

```
[Document]
    │
    ├── source: "2021 Standard Specifications.pdf"
    ├── year: 2021
    ├── doc_type: "Specification"
    │
    └──[HAS_SECTION]──► [Section]
                            │
                            ├── name: "SECTION 101"
                            │
                            └──[HAS_CHUNK]──► [Chunk]
                                                ├── text: "..."
                                                ├── embedding: [768-dim vector]
                                                ├── page: 5
                                                └── section: "SECTION 101"
```

**What V3 Got Right:**
- Unified all documents in a single Neo4j instance
- Vector search + reranking significantly improved retrieval quality vs V1/V2
- Graph-aware context expansion (pulling adjacent chunks from same section)
- Multi-hop reasoning for complex comparison queries

**What V3 Was Missing:**
- No entity extraction — couldn't query "What materials are required?"
- No cross-document version linking
- Fixed-size chunking lost semantic boundaries
- Basic metadata — no LLM-assisted document classification
- HuggingFace embeddings (384-dim) less powerful than Gemini embeddings (768-dim)

---

## SLIDE 7: Version 4 Architecture Overview

**Title:** Version 4 — Advanced Knowledge Graph with Entity Extraction

**Ingestion Pipeline (ingestneo4j_updated.py) — 4 Phases:**

**Phase 1: Google GenAI Embeddings**
- Switched from HuggingFace (all-MiniLM-L6-v2, 384-dim) to Gemini embedding-001 (768-dim)
- Rate-limited batch processing (50 texts/batch with backoff)
- Task-specific embeddings: `task_type="retrieval_document"` for indexing

**Phase 2: Intelligent Document Parsing**
- PyMuPDF (fitz) replaces PyPDFLoader — extracts tables as Markdown
- LLM-generated metadata via Gemini 2.5 Flash with structured output (Pydantic):
  - `display_title`: Clean human-readable name
  - `document_series`: Core type (e.g., "Standard Specifications")
  - `year`: Publication year
  - `primary_category`: One of 6 categories (Regulatory, Meeting, Form, Report, Program, Public Outreach)

**Phase 3: Hybrid Chunking**
- Layer 1: Structural regex splitting by DIVISION/SECTION/CHAPTER boundaries
- Layer 2: SemanticChunker within each structural section (breakpoint by embedding similarity)
- Result: Chunks respect both document structure AND semantic coherence

**Phase 4: Entity Extraction & Graph Construction**
- LLM-based entity extraction using Gemini 2.5 Flash with Pydantic structured output
- **Inline Entity Resolution:** Each extraction call receives list of existing entities to prevent duplicates
- Parallel processing (5 threads) for extraction speed
- Cross-document version linking via `[:SUPERSEDES]` relationships

---

## SLIDE 8: Version 4 — Full Knowledge Graph Schema

**Title:** V4 Knowledge Graph Schema (Entity-Rich)

```
[Document]──[:SUPERSEDES]──►[Document]
    │                          (2021 Specs → 2010 Specs)
    │
    └──[:HAS_SECTION]──► [Section]
                            │
                            └──[:HAS_CHUNK]──► [Chunk]──[:NEXT_CHUNK]──► [Chunk]
                                                │
                                                └──[:MENTIONS]──► [Entity]
                                                                    │
                                                        Types: Material
                                                               Specification
                                                               Standard
                                                               TestMethod
                                                               Form
                                                               Committee_Or_Group
                                                               Project_Or_Corridor
                                                               Concept
                                                               Entity

[Entity]──[:REQUIRES]──► [Entity]
[Entity]──[:REFERENCES]──► [Entity]
```

**Node Types:** Document, Section, Chunk, + 9 Entity types
**Relationship Types:** HAS_SECTION, HAS_CHUNK, NEXT_CHUNK, MENTIONS, SUPERSEDES, REQUIRES, REFERENCES
**Capacity Monitoring:** Auto-checks Neo4j Aura Free limits (200K nodes / 400K relationships)

---

## SLIDE 9: Head-to-Head — V3 vs V4 Ingestion

**Title:** Ingestion Pipeline Comparison

| Component | V3 (ingestneo4j.py) | V4 (ingestneo4j_updated.py) |
|-----------|--------------------|-----------------------------|
| **PDF Parser** | PyPDFLoader (text only) | PyMuPDF + Markdown tables |
| **Embedding Model** | all-MiniLM-L6-v2 (384-dim, local) | gemini-embedding-001 (768-dim, API) |
| **Chunking** | RecursiveCharacterTextSplitter (fixed 1000 chars) | Hybrid: Structural Regex → SemanticChunker |
| **Metadata** | Filename regex + PDF properties | Gemini 2.5 Flash structured generation |
| **Document Classification** | Rule-based (filename keywords) | LLM-classified into 6 categories |
| **Entity Extraction** | None | Gemini 2.5 Flash with Inline Entity Resolution |
| **Entity Types** | None | 9 types (Material, Standard, TestMethod, etc.) |
| **Graph Relationships** | Chunk only (flat) | MENTIONS, REQUIRES, REFERENCES, SUPERSEDES |
| **Cross-Document Links** | None | SUPERSEDES (automatic version chaining) |
| **Chunk Sequencing** | None | NEXT_CHUNK linking (reading order preserved) |
| **Rate Limiting** | None | Exponential backoff + batch API calls |
| **Capacity Safety** | None | Auto-check Neo4j Aura limits before each file |

---

## SLIDE 10: Head-to-Head — V3 vs V4 Chatapp

**Title:** Chatapp Architecture Comparison

| Component | V3 (chatapp.py) | V4 (chatapp_gemini.py) |
|-----------|-----------------|----------------------|
| **Embedding for Query** | all-MiniLM-L6-v2 (local) | gemini-embedding-001 (API) |
| **Neo4j Index** | wydot_vector_index (768-dim) | wydot_gemini_index (768-dim) |
| **Database** | Default neo4j database | Dedicated Aura instance (1c9edfe6) |
| **Search Pipeline** | Vector search → FlashRank reranking | Vector search → Metadata enrichment → FlashRank reranking |
| **Metadata Enrichment** | From chunk properties only | Graph traversal: Chunk → Section → Document (fills missing source/title/year) |
| **Context Expansion** | Same (neighbor chunks from same section) | Same (neighbor chunks from same section) |
| **Multi-hop Reasoning** | Same (query decomposition) | Same (query decomposition) |
| **Streaming** | Same (Mistral/Gemini/OpenRouter) | Same (Mistral/Gemini/OpenRouter) |
| **Cloud Run Support** | Same | Same + local mode detection for Cloud SQL |
| **Default Model** | Mistral Large (initial_index=0) | Gemini 2.5 Flash (initial_index=1) |

**Key V4 Chatapp Innovation: Metadata Enrichment**
```cypher
MATCH (c:Chunk {id: $cid})-[:PART_OF|HAS_CHUNK*1..2]-(s)
      -[:PART_OF|HAS_SECTION*1..2]-(d:Document)
RETURN d.source, d.display_title, d.year, d.document_series
```
When a chunk lacks source/title metadata, V4 traverses the graph upward to the parent Document node — this is impossible in flat vector stores.

---

## SLIDE 11: The Shift in Retrieval Quality

**Title:** How Knowledge Graph Improves Retrieval

**Scenario: "What are the Portland Cement requirements in 2021?"**

**V1/V2 (Vector Only):**
1. Embed query → find similar chunks → return mixed results from 2010 AND 2021
2. No way to enforce temporal filtering at retrieval level
3. LLM must figure out which chunks are from which year

**V3 (Basic Graph):**
1. Vector search returns chunks with metadata
2. FlashRank reranks for relevance
3. Graph-aware expansion adds neighboring chunks
4. Better but still no entity-level connections

**V4 (Entity-Rich Graph):**
1. Vector search returns chunks with metadata
2. **Graph enrichment:** Each chunk knows its parent Document (year, series, category)
3. FlashRank reranks with full metadata context
4. Could potentially traverse: Portland Cement (Material) → MENTIONED_IN → Section 801 Chunk → HAS_SECTION → Document(2021)
5. **SUPERSEDES** relationship explicitly models that 2021 replaces 2010

---

## SLIDE 12: Technical Deep Dive — Hybrid Chunking

**Title:** V4's Hybrid Chunking Strategy

**Problem with V3's Fixed-Size Chunking:**
- RecursiveCharacterTextSplitter at 1000 chars splits mid-sentence, mid-paragraph, mid-section
- A single Section 101 definition might be split across 3 chunks losing context
- Table rows can be split between chunks

**V4's Two-Layer Solution:**

**Layer 1: Structural Splitting (Regex)**
```python
re.split(r'\n(?=DIVISION\s+\d+|SECTION\s+\d+|CHAPTER\s+\d+)', text)
```
- Splits document at DIVISION/SECTION/CHAPTER boundaries
- Each macro-section is a complete, self-contained unit
- Preserves the document's own organizational structure

**Layer 2: Semantic Chunking (within each section)**
```python
SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
```
- Within each section, splits at natural semantic boundaries
- Uses embedding similarity to detect topic shifts
- Result: Chunks that are both structurally AND semantically coherent

**Impact:** A query about "Section 101 definitions" retrieves a complete, meaningful section — not a random 1000-char fragment.

---

## SLIDE 13: Technical Deep Dive — Entity Extraction with Inline ER

**Title:** V4's Entity Extraction & Inline Entity Resolution

**What is Inline Entity Resolution?**
When processing chunk #500, the LLM already knows about entities from chunks #1-499.

```python
prompt = f"""
Extract entities and relationships from the text.

CRITICAL INLINE ENTITY RESOLUTION INSTRUCTIONS:
Before creating a new entity, check this list of existing canonical entities:
{existing_entities}

If the text mentions "Portland Cement" and "Type I Portland Cement"
is in the existing list, ALWAYS use the exact string from the
existing list to prevent duplicates.
"""
```

**Why This Matters:**
- Without ER: "Portland Cement", "portland cement", "Type I/II Portland Cement" → 3 separate entities
- With Inline ER: All resolve to the canonical "Portland Cement" → single entity node with multiple chunk connections

**Entity Types Extracted:**
| Entity Type | Example |
|-------------|---------|
| Material | Portland Cement, Aggregate, Asphalt Binder |
| Specification | AASHTO M85, ASTM C150 |
| Standard | AASHTO LRFD Bridge Design Specifications |
| TestMethod | AASHTO T27 (Sieve Analysis) |
| Form | Form 1170, Certified Test Reports |
| Committee_Or_Group | WYDOT Materials Program, FHWA |
| Project_Or_Corridor | I-80 Reconstruction, US 287 |
| Concept | Superpave, Quality Level Analysis |

---

## SLIDE 14: Technical Deep Dive — Cross-Document Version Chains

**Title:** V4's SUPERSEDES Relationship — Automatic Version Linking

**How It Works:**
```cypher
MATCH (d:Document)
WHERE d.year > 0 AND d.document_series IS NOT NULL
WITH d.document_series AS series, d
ORDER BY d.year DESC
WITH series, collect(d) AS docs
UNWIND range(0, size(docs)-2) AS i
WITH docs[i] AS newer, docs[i+1] AS older
MERGE (newer)-[:SUPERSEDES]->(older)
```

**Result:**
```
[2021 Standard Specs] ──[:SUPERSEDES]──► [2010 Standard Specs]
```

**Why This Is Powerful:**
- Comparison queries can now follow the SUPERSEDES edge directly
- No more guessing which documents are related — the graph encodes it explicitly
- When the user asks "What changed?", the system knows exactly which two documents to compare
- Scales automatically: Adding a 2025 edition would create 2025 → 2021 → 2010 chain

---

## SLIDE 15: Embedding Model Comparison

**Title:** Why We Switched from HuggingFace to Gemini Embeddings

| Aspect | V3: all-MiniLM-L6-v2 | V4: gemini-embedding-001 |
|--------|----------------------|--------------------------|
| **Dimensions** | 384 | 768 |
| **Model Size** | 22M parameters (local) | Google-hosted API |
| **Training Data** | General-purpose NLI/STS | Massive multilingual + code + technical |
| **Domain Performance** | Good for general text | Better for technical/engineering jargon |
| **Speed** | Fast (local CPU) | ~0.5s per batch (API latency) |
| **Cost** | Free (local) | Free tier (Gemini API) |
| **Task-Specific** | No | Yes — separate modes for retrieval_document vs retrieval_query |

**Reasoning for the Shift:**
- WYDOT documents contain highly technical language ("superpave", "AASHTO M85", "drilled shaft foundations")
- MiniLM-L6-v2 was trained on general-purpose sentence similarity
- Gemini embeddings have been trained on technical documentation and perform better on domain-specific retrieval
- Task-specific embeddings (document vs query) improve retrieval accuracy by encoding asymmetric search intent

---

## SLIDE 16: Performance & Scalability

**Title:** Scalability Improvements V3 → V4

| Metric | V3 | V4 |
|--------|----|----|
| **PDF Parsing** | Text only (misses tables) | Text + Markdown tables |
| **Chunk Quality** | Fixed 1000-char (split mid-sentence) | Semantic boundaries within structural sections |
| **Metadata Accuracy** | Rule-based (filename regex) | LLM-classified (6 categories, clean titles) |
| **Entity Awareness** | None | Full entity extraction + inline resolution |
| **Cross-Doc Links** | None | SUPERSEDES version chains |
| **Neo4j Safety** | None | Capacity monitoring (190K node / 380K rel warning) |
| **Rate Limiting** | None | Exponential backoff (2^n) for Gemini 429 errors |
| **Ingestion Speed** | ~5 min for 2 PDFs | ~15 min for 2 PDFs (entity extraction adds time) |
| **Graph Richness** | Chunks only | Documents + Sections + Chunks + Entities + Relationships |

**Scaling Projection for 1,300+ Documents:**
- V3: Would create ~1.3M chunk nodes (flat, no connections)
- V4: Would create ~1.3M chunks + ~50K entities + ~200K relationships (rich, connected graph)
- V4's entity resolution prevents entity explosion (canonical names reused)

---

## SLIDE 17: Retrieval Pipeline Comparison

**Title:** Search Architecture Side-by-Side

**V3 Pipeline:**
```
User Query
    │
    ▼
[1] Embed query (MiniLM-L6-v2, 384-dim)
    │
    ▼
[2] Neo4j vector similarity (FETCH_K=25)
    │
    ▼
[3] FlashRank cross-encoder reranking (top 10)
    │
    ▼
[4] Get neighbor chunks (same source + section)
    │
    ▼
[5] LLM generation (Mistral/Gemini/OpenRouter)
```

**V4 Pipeline:**
```
User Query
    │
    ▼
[1] Embed query (Gemini embedding-001, 768-dim)
    │
    ▼
[2] Neo4j vector similarity (FETCH_K=25)
    │
    ▼
[3] Graph metadata enrichment
    │   (Chunk → Section → Document traversal)
    │   (fills source, title, year, series)
    │
    ▼
[4] FlashRank cross-encoder reranking (top 10)
    │
    ▼
[5] Get neighbor chunks (same source + section)
    │
    ▼
[6] LLM generation (Gemini 2.5 Flash default)
```

**Key Addition:** Step [3] — Graph metadata enrichment is unique to V4. It uses Cypher traversal to fill missing chunk metadata from parent nodes, enabling better reranking decisions.

---

## SLIDE 18: Multi-Model Support

**Title:** LLM Model Availability Across Versions

**Both V3 and V4 support 16 models via 3 providers:**

| Provider | Models |
|----------|--------|
| **Local/Direct** | Mistral Large, Gemini 2.5 Flash |
| **OpenRouter** | GPT-5.2 Pro, GPT-5.2 Chat, Claude Opus 4.6, Claude Sonnet 4.6, Gemini 3 Flash, MiniMax M2.5, Step 3.5 Flash, Llama 3.1 405B, Llama 3.3 70B, DeepSeek V3, DeepSeek V3.2 Speciale, DeepSeek Coder V2, Qwen 2.5 72B, Mistral Nemo |

**Streaming:** All models support real-time token streaming in the chat interface
**Default Change:** V3 defaults to Mistral Large; V4 defaults to Gemini 2.5 Flash

**Reasoning for Default Shift:**
- Gemini 2.5 Flash is faster and free-tier compatible
- Better alignment with Gemini embeddings (same model family)
- Multimodal capability built-in (image/audio/PDF analysis)

---

## SLIDE 19: What Stayed the Same

**Title:** Shared Features Across V3 & V4

Both versions share these production-ready features:

- **Authentication System:** SQLite-backed login with email verification + guest mode
- **Chat History:** Full conversation persistence with session/thread management
- **Chat Resume:** Users can resume past conversations from the sidebar
- **FlashRank Reranking:** Cross-encoder reranking for retrieval precision
- **Multi-hop Reasoning:** Query decomposition into sub-queries for complex questions
- **Thinking Mode:** Sync generation for detailed, slower responses
- **Audio Input:** Gemini-powered speech-to-text for voice queries
- **Multimodal Analysis:** Image/PDF/audio file analysis via Gemini
- **OpenRouter Integration:** Access to 14 frontier models (GPT-5.2, Claude 4.6, etc.)
- **Cloud Run Deployment:** Production-ready with environment detection and /tmp handling
- **Citation System:** [SOURCE_X] markers transformed to clickable side-panel references
- **Telemetry:** Latency tracking, source counting, online evaluation logging

---

## SLIDE 20: Summary — The Evolution

**Title:** From Flat Vectors to Structured Knowledge

```
V1 (Fall 2025)          V2 (Jan 2026)           V3 (Jan-Feb 2026)        V4 (Feb 2026)
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ Multiple     │    │ Router +        │    │ Neo4j Graph      │    │ Full Knowledge Graph │
│ Vector       │───►│ Supervisor      │───►│ (Basic)          │───►│ (Entity-Rich)        │
│ Stores       │    │ Architecture    │    │ + FlashRank      │    │ + Semantic Chunking  │
│              │    │ + Metadata      │    │ + Multi-hop      │    │ + Inline ER          │
│              │    │   Filtering     │    │ + 16 LLM Models  │    │ + SUPERSEDES         │
└─────────────┘    └─────────────────┘    └──────────────────┘    └──────────────────────┘
     ↓                    ↓                       ↓                        ↓
 High latency      Complex routing        Graph-aware           Entity-aware
 Not scalable      Still flat vectors     retrieval             reasoning
 No structure      Metadata only          Basic chunking        Semantic chunks
```

**Each version solved the previous version's biggest limitation.**

---

## SLIDE 21: Future Directions

**Title:** Future Directions & Research Opportunities

**Immediate Next Steps:**
1. **Hybrid Retrieval:** Combine V4's entity-rich Knowledge Graph with PageIndex's hierarchical tree search for best-of-both-worlds retrieval
2. **Full Corpus Ingestion:** Scale V4 pipeline to all 1,300+ WYDOT documents with automated entity resolution
3. **GraphRAG Evaluation:** Implement RAGAS framework to quantitatively compare V3 vs V4 retrieval quality

**Research Directions:**
4. **Agentic RAG:** Let the LLM decide at query time whether to use vector search, graph traversal, or tree navigation — true autonomous retrieval
5. **Temporal Knowledge Graph:** Encode not just SUPERSEDES but also effective dates, amendment histories, and regulatory change tracking
6. **Fine-Tuned Domain Embeddings:** Train custom embeddings on WYDOT corpus for even better technical term understanding
7. **Q&A Fine-Tuning:** Use the 8,000+ validated Q&A pairs to fine-tune a domain-specific WYDOT expert model
8. **Self-RAG / CRAG:** Implement self-reflective RAG where the system evaluates its own retrieval quality and retries with different strategies

---

## SLIDE 22: The Best of Both Worlds: Hybrid Retrieval

**Title:** Do We Still Use Vector Search? Yes. (Hybrid RAG)

**The Misconception:**
Moving to a Knowledge Graph means replacing Vector Search.

**The Reality:**
Our V3/V4 architecture is a **Hybrid RAG** system. It uses Vector Search as the *starting point* for Graph Traversal.

**How it works step-by-step:**
1. **Vector Search (Semantic Match):** The user's query is converted to a vector. Neo4j's built-in vector index finds the top 25 `[Chunk]` nodes that match the meaning of the question (exactly like V1/V2).
2. **Graph Traversal (Contextual Match):** Starting from those 25 vector-matched chunks, the system traverses the graph connections (`[:MENTIONS]`, `[:HAS_SECTION]`) to pull in related Entities and Document Metadata.

**Why this is a breakthrough:**
We get the **"fuzzy semantic understanding"** of Vector Search combined with the **"strict logical relationships"** of a Knowledge Graph. We didn't replace V1/V2; we augmented them.

---

## SLIDE 23: Thank You

**Title:** Thank You

Questions?

**GitHub:** [Repository Link]
**Demo:** [Chatbot URL]

---

# APPENDIX — Quick Reference Comparison Table (for handout)

| Feature | V1 | V2 | V3 | V4 |
|---------|----|----|----|----|
| Vector Search | Yes | Yes | Yes | Yes |
| Metadata Filtering | No | Yes (year) | Yes (properties) | Yes (graph traversal) |
| Knowledge Graph | No | No | Basic | Full (entities + relationships) |
| Entity Extraction | No | No | No | Yes (9 types + inline ER) |
| Cross-Doc Links | No | No | No | Yes (SUPERSEDES) |
| Chunking | Fixed | Fixed | Fixed (1000 chars) | Hybrid (Structural + Semantic) |
| Embeddings | HuggingFace | HuggingFace | HuggingFace (384-dim) | Gemini (768-dim) |
| PDF Tables | No | No | No | Yes (Markdown) |
| Reranking | No | No | FlashRank | FlashRank |
| Multi-hop | No | Supervisor | Query Decomposition | Query Decomposition |
| LLM Models | 1 | 2 | 16 | 16 |
| Context Expansion | No | No | Graph neighbors | Graph neighbors + metadata enrichment |
| Production Features | Basic | Cloud Run | Full (auth, history, audio, multimodal) | Full (auth, history, audio, multimodal) |
