# Research Plan 01: Temporal/Versioned GraphRAG for Evolving Construction Specifications

## 1. Problem Statement

WYDOT maintains **multiple editions** of the same specification manual (e.g., 2010 vs. 2021 Standard Specifications). When a user asks *"What changed in the aggregate requirements between 2010 and 2021?"*, current RAG systems retrieve chunks from both versions independently but **cannot reason about the evolution** of content across versions. They treat each document as a static, isolated corpus.

### The Research Gap

Recent papers (VersionRAG, T-GRAG, TG-RAG — all published in 2025) have introduced the concept of **temporally-aware knowledge graphs** for RAG, but:

1. **No one has applied this to government construction specifications** — All existing work focuses on news articles, Wikipedia, or financial reports.
2. **No benchmark exists** for evaluating version-aware retrieval in regulatory/engineering documents.
3. **The hierarchical structure** (Division > Section > Subsection) of specs creates a unique challenge: changes can happen at any level, and downstream sections may be affected by upstream changes.

---

## 2. Background & Related Work

| Paper | Year | Key Idea | Limitation for WYDOT |
|-------|------|----------|---------------------|
| **VersionRAG** (arXiv:2025) | 2025 | Hierarchical graph modeling document evolution | Tested on Wikipedia; no structured specs |
| **T-GRAG** (arXiv:2025) | 2025 | Temporal Knowledge Graph Generator + Query Decomposition | Focuses on news; doesn't handle section-level hierarchy |
| **TG-RAG** (arXiv:2025) | 2025 | Bi-level temporal graph with timestamped relations | No support for regulatory "supersedes" semantics |
| **Your `ingestneo4j_updated.py`** | 2024 | `[:SUPERSEDES]` links between document versions | Only links at document level, not section level |

---

## 3. Proposed Approach

### 3.1 Core Innovation: Section-Level Temporal Diff Graph

Build a **Section-Level Temporal Knowledge Graph** that captures not just *what* changed, but *how* it changed:

```
[2010:Section_801] --[:MODIFIED_IN]--> [2021:Section_801]
    ├── change_type: "Updated"
    ├── diff_summary: "Added Type IL cement requirement"
    └── affected_entities: ["Portland Cement", "Type IL"]

[2010:Section_414] --[:UNCHANGED_IN]--> [2021:Section_414]

[2010:Section_217] --[:REMOVED_IN]--> [2021:None]
    └── reason: "Merged into Section 216"
```

### 3.2 Pipeline Architecture

```
Phase 1: Structure Extraction (Already Done via PageIndex)
    ├── 2010_tree.json (140 nodes)
    └── 2021_tree.json (154 nodes)

Phase 2: Section Alignment
    ├── Exact Match: Section 801 (2010) ↔ Section 801 (2021)
    ├── Fuzzy Match: "Section 217 - Removal of Structures" ↔ "Section 216 - Removal and Disposal"
    └── Orphan Detection: Sections in 2010 not in 2021 (or vice versa)

Phase 3: Diff Generation (LLM-Powered)
    ├── For each aligned pair, extract page text from both PDFs
    ├── Send to Gemini: "What are the key differences between these two versions of Section X?"
    └── Store structured diff: {additions: [], deletions: [], modifications: []}

Phase 4: Temporal Graph Construction
    ├── Create [:MODIFIED_IN], [:UNCHANGED_IN], [:ADDED_IN], [:REMOVED_IN] edges
    ├── Attach diff metadata to edges
    └── Link affected entities to temporal edges

Phase 5: Temporal Query Engine
    ├── "What changed?" → Traverse [:MODIFIED_IN] edges
    ├── "When was X introduced?" → Find earliest [:ADDED_IN] edge
    └── "Is Section 801 still current?" → Check if latest version has [:UNCHANGED_IN]
```

### 3.3 Example Queries This Enables

| Query | Current System Answer | Temporal System Answer |
|-------|----------------------|----------------------|
| "What changed in aggregate specs?" | Returns chunks from both years, no comparison. | "Section 803 was modified: Added gradation table for RAP, removed reference to AASHTO M-147." |
| "When was Type IL cement added?" | "Found in 2021 spec." | "Type IL cement was **added** in the 2021 edition (Section 801.02). It was not present in the 2010 edition." |
| "Are the 2010 earthwork specs still valid?" | "Here are chunks from 2010..." | "Division 200 (Earthwork) was **significantly modified** in 2021. 12 of 21 sections were updated." |

---

## 4. Implementation Roadmap

### Phase 1: Data Preparation (Week 1)
- [ ] Use existing PageIndex trees (`output/*.json`) to extract section lists
- [ ] Build section alignment map (2010 ↔ 2021) using title similarity
- [ ] Extract page text for each aligned section pair

### Phase 2: Diff Engine (Week 2)
- [ ] Design Gemini prompt for structured diff extraction
- [ ] Run diff extraction for all ~140 aligned section pairs
- [ ] Store diffs as JSON: `temporal_diffs/section_XXX_diff.json`

### Phase 3: Graph Extension (Week 3)
- [ ] Extend Neo4j schema with temporal edge types
- [ ] Write Cypher to create temporal relationships
- [ ] Build temporal query functions

### Phase 4: Evaluation & Paper (Week 4)
- [ ] Create test query set (20 temporal queries)
- [ ] Compare: Standard RAG vs. Temporal RAG
- [ ] Measure: Answer accuracy, citation precision, temporal reasoning correctness

---

## 5. Expected Contributions

1. **First application** of Temporal GraphRAG to government construction specifications.
2. **Novel section-alignment algorithm** for hierarchically structured regulatory documents.
3. **WYDOT-TemporalBench**: A benchmark dataset of temporal queries over construction specs.
4. **Practical proof** that temporal edges improve answer quality for "What changed?" queries by an estimated 40–60%.

---

## 6. Estimated Costs

| Resource | Cost |
|----------|------|
| Gemini API (diff generation, ~140 pairs) | ~$5–10 |
| Neo4j Aura (existing instance) | $0 (Free tier) |
| Total new infrastructure | $0 |

---

## 7. Publication Target

- **Conference**: ACL 2026 Industry Track, EMNLP 2026, or AAAI 2026 AI for Government Workshop
- **Title Proposal**: *"VersionSpec: Temporal GraphRAG for Evolving Government Construction Specifications"*
