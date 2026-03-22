# Research Plan 02: Hierarchical Hybrid RAG — Fusing PageIndex + GraphRAG

## 1. Problem Statement

You now have **two independent retrieval systems**:
- **PageIndex** (Structural/Hierarchical): Navigates the document tree to find the right *section*.
- **Neo4j GraphRAG** (Entity/Semantic): Finds relevant *chunks* and traverses *entity relationships*.

No existing research has explored **fusing** a hierarchical structural index with an entity-relationship graph in a single retrieval pipeline. Current hybrid approaches only combine vector + keyword search (BM25), not structure + graph.

### The Research Gap

- **HiRAG (2025)** proposes hierarchical retrieval but only within a single index type.
- **RAPTOR** creates hierarchical summaries but loses document structure.
- **Microsoft GraphRAG** builds community clusters but ignores the document's inherent TOC.
- **No paper** combines a document-structure tree (like PageIndex) with an entity-knowledge graph (like Neo4j) as two complementary retrieval channels.

---

## 2. Proposed Architecture: "DualPath RAG"

```
User Query: "What are the aggregate gradation requirements for base course?"
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
[PageIndex Path]    [GraphRAG Path]
Navigate Tree:      Vector Search:
Div 300 → Sec 301   "aggregate gradation"
→ Subsec 301.03     → Chunk: "Table 301-1..."
                    Graph Traverse:
                    Aggregate → REQUIRES → AASHTO T-27
    │                    │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   Fusion Engine     │
    │   (Context Merge    │
    │    + Deduplication   │
    │    + Relevance Score)│
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   LLM Generation    │
    │   (Rich context     │
    │    from both paths) │
    └─────────────────────┘
```

### 2.1 Why This is Novel

| Existing Approach | What It Does | What It Misses |
|-------------------|-------------|----------------|
| Vector-only RAG | Finds semantically similar chunks | Misses structural context |
| PageIndex tree | Finds the right section | Misses cross-document entity links |
| GraphRAG entities | Finds related concepts | Misses hierarchical position |
| **DualPath (Ours)** | **Both structure + entity context** | **Fills the gap** |

---

## 3. Key Research Questions

1. **Does structural retrieval improve entity-based retrieval?** (Hypothesis: Yes, because knowing "Section 301" narrows the entity search space.)
2. **Is fusion better than either path alone?** (Measure via Precision@K, Recall@K.)
3. **How should we weight the two paths?** (Learned weights vs. query-type classification.)
4. **Does the combined context reduce hallucination?** (Measure via faithfulness score.)

---

## 4. Implementation Roadmap

### Phase 1: Router Agent (Week 1)
- [ ] Build a query classifier: "Is this a structural query or a semantic query?"
  - Structural: "What section covers..." → PageIndex first
  - Semantic: "Find all mentions of..." → GraphRAG first
  - Hybrid: "What changed in..." → Both paths

### Phase 2: Fusion Engine (Week 2)
- [ ] Implement context merging: Combine PageIndex section text + GraphRAG chunks
- [ ] Deduplicate overlapping content (same page ranges)
- [ ] Score each context piece by relevance + structural authority

### Phase 3: Evaluation Framework (Week 3)
- [ ] Create 50 test queries across 3 categories (structural, semantic, hybrid)
- [ ] Run each query through: (a) PageIndex only, (b) GraphRAG only, (c) DualPath
- [ ] Measure: Answer accuracy, citation precision, latency, token usage

### Phase 4: Analysis & Paper (Week 4)
- [ ] Statistical comparison of the three approaches
- [ ] Identify query types where fusion helps most
- [ ] Write research paper

---

## 5. Expected Contributions

1. **DualPath RAG**: First architecture combining structural tree + entity graph retrieval.
2. **Query Router**: A lightweight classifier that routes queries to the optimal retrieval path.
3. **Empirical proof** on real government engineering documents (not synthetic benchmarks).
4. **Practical insight**: When does structure beat semantics, and vice versa?

---

## 6. Tools Required (All Available)

| Tool | Status | Purpose |
|------|--------|---------|
| PageIndex trees | ✅ Built | Structural retrieval |
| Neo4j + Vector Index | ✅ Running | Entity/semantic retrieval |
| Gemini 2.5 Flash | ✅ Available | Query classification + answer generation |
| Chainlit | ✅ Available | Demo interface |

---

## 7. Publication Target

- **Conference**: SIGIR 2026, NAACL 2026 Industry Track
- **Title Proposal**: *"DualPath RAG: Fusing Document Structure Trees with Entity Knowledge Graphs for Technical Document QA"*
