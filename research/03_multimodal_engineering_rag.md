# Research Plan 03: Multimodal RAG for Engineering Specifications

## 1. Problem Statement

WYDOT Standard Specifications contain **tables, figures, and diagrams** that carry critical engineering information (gradation curves, material property tables, cross-section diagrams). Current text-only RAG systems **completely ignore** this visual content, leading to incomplete or incorrect answers when the information exists only in a table or figure.

### The Research Gap

- **M-RAG (2025)**: Introduces multimodal RAG for architecture/engineering but focuses on building blueprints, not regulatory specification tables.
- **MMORE (2025)**: Handles 15+ file types but treats tables as flat text, losing row-column structure.
- **UniDoc-Bench (2025)**: Includes "Construction" as a domain but doesn't have WYDOT-style specification tables as test cases.
- **No research** has specifically addressed **regulatory specification tables** (e.g., gradation tables, material test method cross-references) in a multimodal RAG context.

---

## 2. Types of Visual Content in WYDOT Specs

| Content Type | Example | Current Handling | Impact |
|-------------|---------|-----------------|--------|
| **Gradation Tables** | Table 301-1 (Sieve sizes vs. % passing) | Extracted as messy text, structure lost | ❌ Critical |
| **Material Cross-Ref Tables** | "Test Method → AASHTO Standard" mappings | Partially captured | ⚠️ High |
| **Structural Diagrams** | Bridge reinforcement details | Completely ignored | ❌ Critical |
| **Flow Charts** | Construction inspection procedures | Completely ignored | ⚠️ Medium |

---

## 3. Proposed Approach

### 3.1 Table-Aware Ingestion Pipeline

```
PDF Page
    │
    ├── Text Extraction (PyMuPDF) → Existing pipeline
    │
    ├── Table Detection (PyMuPDF find_tables())
    │   ├── Convert to Markdown table
    │   ├── Generate LLM description: "This table shows gradation requirements..."
    │   └── Store as: TableChunk {markdown, description, page, section}
    │
    └── Figure Detection (Vision Model)
        ├── Extract figure bounding box
        ├── Send to Gemini Vision: "Describe this engineering diagram"
        └── Store as: FigureChunk {image_path, description, page, section}
```

### 3.2 Multimodal Embedding Strategy

| Content Type | Embedding Method | Storage |
|-------------|-----------------|---------|
| Text chunks | `gemini-embedding-001` (current) | Neo4j Vector Index |
| Table descriptions | `gemini-embedding-001` on LLM description | Neo4j Vector Index |
| Figure descriptions | `gemini-embedding-001` on vision description | Neo4j Vector Index |
| Raw table markdown | Stored as property on TableChunk node | Neo4j text property |

### 3.3 Retrieval Enhancement

When a query matches a TableChunk or FigureChunk, the system:
1. Returns the **LLM description** as context for answer generation.
2. Returns the **raw markdown table** for precise data extraction.
3. For figures, returns the **image** alongside the text answer.

---

## 4. Key Research Questions

1. **How much information is lost** by text-only RAG in engineering specs? (Quantify the gap.)
2. **Does table-structured context improve answer accuracy** for quantitative queries? (e.g., "What sieve size is required for Type A aggregate?")
3. **Can vision-described figures serve as useful context** for the LLM, or does the description introduce noise?
4. **What is the optimal embedding strategy** for tables: embed the markdown, the description, or both?

---

## 5. Implementation Roadmap

### Phase 1: Table Extraction Audit (Week 1)
- [ ] Run PyMuPDF `find_tables()` on both PDFs
- [ ] Count and categorize all tables (gradation, cross-reference, schedule, etc.)
- [ ] Assess quality of markdown conversion

### Phase 2: Table-Aware Ingestion (Week 2)
- [ ] Extend `ingestneo4j_updated.py` with TableChunk node type
- [ ] Generate LLM descriptions for each table
- [ ] Create embeddings for table descriptions
- [ ] Link TableChunk to parent Section and Document

### Phase 3: Figure Processing (Week 3)
- [ ] Use Gemini Vision to describe figures in both specs
- [ ] Create FigureChunk nodes with descriptions
- [ ] Test retrieval with figure-dependent queries

### Phase 4: Evaluation (Week 4)
- [ ] Create 30 table/figure-dependent test queries
- [ ] Compare: Text-only RAG vs. Multimodal RAG
- [ ] Measure: Answer completeness, numerical accuracy, citation quality

---

## 6. Expected Contributions

1. **First multimodal RAG study** on government construction specification documents.
2. **Table-structure preservation** methodology for regulatory tables in knowledge graphs.
3. **Quantified information loss** metric: How much do text-only systems miss in table-heavy specs?
4. **Practical pipeline extension** for your existing `ingestneo4j_updated.py`.

---

## 7. Publication Target

- **Conference**: ACL 2026 Industry Track, AAAI 2026 AI for Social Good
- **Title Proposal**: *"Beyond Text: Multimodal RAG for Table-Rich Government Engineering Specifications"*
