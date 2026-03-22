# PageIndex Slide Content
## Vectorless Hierarchical RAG for WYDOT Specifications
### Insert after Slide 48 (V4 Scalability) and before Slide 49 (Future Directions)

---

## SLIDE A1: PageIndex — A Different Paradigm

**Title:** PageIndex: Vectorless Reasoning-Based RAG

**The Question:** Can we retrieve and reason over WYDOT specifications *without* any vector embeddings, knowledge graphs, or API-based indexing?

**The Answer:** Yes. PageIndex uses the document's own **hierarchical structure** (Divisions, Sections, Subsections) as a navigation tree, and an LLM **reasons over the tree** at query time to find relevant pages.

**Why This Matters:**
- **Zero API cost for indexing** — Structure extracted locally using PyMuPDF bookmarks + regex TOC parsing
- **Zero embedding storage** — No vector index to maintain
- **Interpretable retrieval** — The LLM explains *why* it selected each section (not just a similarity score)
- **Scales to any structured PDF** — Works on any document with a Table of Contents or hierarchical headings

**Paradigm Comparison:**

| Approach | V1-V4 (Vector RAG / GraphRAG) | PageIndex (Reasoning RAG) |
|----------|-------------------------------|---------------------------|
| Indexing | Embed every chunk (API calls) | Parse document structure (local, free) |
| Storage | Vector database (Neo4j/FAISS) | JSON tree file (~130 KB per document) |
| Retrieval | Similarity search (approximate) | LLM tree navigation (reasoning) |
| Explainability | Opaque similarity scores | LLM reasoning trace ("I selected Section 801 because...") |
| Cost per query | Embedding API + vector search | LLM inference only |

---

## SLIDE A2: PageIndex Architecture

**Title:** PageIndex Architecture — How It Works

**Three-Stage Pipeline:**

```
Stage 1: STRUCTURE EXTRACTION (One-time, offline, zero-API)
    |
    PDF File (2021 Standard Specifications, 890 pages)
    |
    v
[PyMuPDF Bookmark Parser]
    |--- Reads the PDF's internal Table of Contents / Bookmarks
    |--- Falls back to Regex TOC parsing for scanned PDFs
    |
    v
Hierarchical JSON Tree:
{
    "title": "2021 Standard Specifications",
    "children": [
        {
            "title": "Division 100 - General Provisions",
            "page_range": [1, 98],
            "children": [
                {"title": "Section 101 - Definitions", "page_range": [1, 15]},
                {"title": "Section 102 - Bidding Requirements", "page_range": [15, 30]},
                ...
            ]
        },
        {
            "title": "Division 800 - Materials",
            "page_range": [698, 890],
            "children": [
                {"title": "Section 801 - Portland Cement", "page_range": [698, 705]},
                ...
            ]
        }
    ]
}
```

```
Stage 2: TREE SEARCH (At query time, LLM-powered)
    |
    User Query: "What are the requirements for Portland Cement?"
    |
    v
[LLM sees the tree overview]
    "Here are the top-level divisions:
     Division 100 - General Provisions (pages 1-98)
     Division 200 - Earthwork (pages 99-180)
     ...
     Division 800 - Materials (pages 698-890)"
    |
    v
[LLM reasons]: "Portland Cement is a material.
    Division 800 - Materials is the most relevant."
    |
    v
[LLM drills into Division 800 children]
    "Section 801 - Portland Cement (pages 698-705)
     Section 802 - Aggregates (pages 705-720)
     ..."
    |
    v
[LLM selects]: Section 801, pages 698-705
    |
    v
Returns: Relevant page ranges + reasoning trace
```

```
Stage 3: ANSWER GENERATION
    |
    Selected pages extracted from PDF
    |
    v
[Gemini 2.5 Flash] generates answer with page citations
    |
    v
Final answer: "According to Section 801 (p.698-705),
    Portland Cement shall conform to AASHTO M85..."
```

---

## SLIDE A3: Zero-API Structure Extraction

**Title:** direct_builder.py — How We Extract Structure Without Any API

**The Challenge:** Both the 2010 and 2021 WYDOT Standard Specifications are 800+ page PDFs with complex hierarchical structure. Most approaches would use an LLM to parse the TOC — but that costs money and adds latency.

**Our Solution: Three-Layer Extraction**

**Layer 1: PyMuPDF Bookmark Parsing**
- Most well-formatted PDFs embed a bookmark tree (clickable TOC in PDF viewers)
- PyMuPDF's `doc.get_toc()` extracts this in milliseconds
- Returns: `[(level, title, page_number), ...]`
- Works perfectly for the 2021 PDF

**Layer 2: Regex TOC Text Parsing**
- For PDFs without bookmarks (like the 2010 edition), we parse the actual TOC pages
- Custom regex patterns detect WYDOT-specific structure:
  - `DIVISION \d+` patterns for top-level divisions
  - `SECTION \d+` patterns for sections within divisions
  - Page number extraction from TOC dot-leaders
- Uses a `_division_names` lookup for clean naming (e.g., "Division 100" maps to "General Provisions")

**Layer 3: Page Offset Correction**
- PDF page numbers often don't match printed page numbers (frontmatter shifts everything)
- We detect the offset by searching for "SECTION 101" text in early pages
- Apply correction so "page 45 in TOC" maps to the correct PDF page

**Result:**
| Document | Nodes | Divisions | Processing Time | API Calls |
|----------|-------|-----------|----------------|-----------|
| 2021 Standard Specifications | 154 | 10 | ~2 seconds | 0 |
| 2010 Standard Specifications | 140 | 8 | ~2 seconds | 0 |

---

## SLIDE A4: Tree Search — LLM Reasons Over Structure

**Title:** tree_search.py — Navigating 890 Pages With Reasoning

**How Tree Search Works:**

Instead of "find the 10 most similar chunks" (vector search), PageIndex asks the LLM:
*"Given this document's structure, which sections are most relevant to the user's question?"*

**Step 1: Generate Tree Overview**
```
The LLM receives a compact overview:
"Document: 2021 Standard Specifications (890 pages)
 [1] Division 100 - General Provisions (p.1-98, 15 subsections)
 [2] Division 200 - Earthwork (p.99-180, 12 subsections)
 ...
 [8] Division 800 - Materials (p.698-890, 18 subsections)"
```

**Step 2: LLM Selects Relevant Branches**
```
Query: "What are the Portland Cement requirements?"
LLM Response: "Division 800 - Materials is most relevant because
    Portland Cement is a construction material."
```

**Step 3: Drill Into Selected Branch**
```
"Division 800 children:
 [8.1] Section 801 - Portland Cement (p.698-705)
 [8.2] Section 802 - Aggregates (p.705-720)
 [8.3] Section 803 - Asphalt Materials (p.720-740)
 ..."
LLM Response: "Section 801 directly covers Portland Cement."
```

**Step 4: Extract Pages & Generate Answer**
```
Pages 698-705 are extracted and sent to Gemini 2.5 Flash.
"According to Section 801.2 (p.699), Portland Cement shall
 conform to AASHTO Designation M85..."
```

**Key Advantage Over Vector Search:**
- Vector search might return chunks from Section 401 (which *mentions* cement) alongside Section 801 (which *specifies* cement requirements) — ranked by similarity
- PageIndex navigates directly to Section 801 because the LLM *understands* that "requirements for Portland Cement" means the Materials division, not incidental mentions elsewhere

---

## SLIDE A5: Cross-Document PageIndex Search

**Title:** Searching Across Both 2010 and 2021 Specifications

**The Chatbot searches both document trees in parallel:**

```
Query: "Compare Portland Cement requirements between 2010 and 2021"
    |
    v
[Tree Search: 2021 PDF]          [Tree Search: 2010 PDF]
    |                                 |
    v                                 v
Section 801 (p.698-705)         Section 801 (p.710-718)
    |                                 |
    v                                 v
[Gemini 2.5 Flash receives both sets of pages]
    |
    v
"The 2021 specifications (Section 801.2, p.699) require Portland
 Cement to conform to AASHTO M85, while the 2010 specifications
 (Section 801.2, p.711) referenced the same AASHTO M85 standard
 but with different testing frequency requirements..."
```

**Results on Test Queries:**

| Query | Documents Found | Sections Retrieved |
|-------|-----------------|--------------------|
| "Portland Cement requirements" | Both 2010 & 2021 | Section 801 in each |
| "HMA overlay procedures" | Both 2010 & 2021 | Section 401 in each |
| "Bridge deck concrete specifications" | Both 2010 & 2021 | Section 506 in each |
| "Aggregate gradation requirements" | Both 2010 & 2021 | Section 802 in each |
| "What are the definitions in Section 101?" | Both 2010 & 2021 | Section 101 in each |

**10 results returned per query** across both documents, with matching sections correctly identified in each edition.

---

## SLIDE A6: PageIndex Output Format

**Title:** The JSON Tree Structure

**Each processed document produces a compact JSON tree (~130 KB):**

```json
{
    "root": {
        "node_id": "root",
        "title": "2021 Standard Specifications for Road and Bridge Construction",
        "start_page": 1,
        "end_page": 890,
        "children": [
            {
                "node_id": "div_100",
                "title": "Division 100 - General Provisions",
                "start_page": 1,
                "end_page": 98,
                "children": [
                    {
                        "node_id": "sec_101",
                        "title": "Section 101 - Definitions and Terms",
                        "start_page": 1,
                        "end_page": 15,
                        "children": []
                    },
                    ...
                ]
            },
            ...
        ]
    },
    "metadata": {
        "total_nodes": 154,
        "total_pages": 890,
        "extraction_method": "pymupdf_bookmarks"
    }
}
```

**Comparison with V4 Knowledge Graph:**

| Metric | V4 (Neo4j GraphRAG) | PageIndex |
|--------|---------------------|-----------|
| Storage | Neo4j Aura (cloud) | Local JSON file (130 KB) |
| Indexing Cost | Gemini API for embeddings + entity extraction | Zero (local PyMuPDF parsing) |
| Indexing Time | ~15 minutes per document | ~2 seconds per document |
| Query Cost | Embedding API + vector search + LLM | LLM reasoning only |
| Cross-Doc Links | SUPERSEDES relationships | Parallel tree search |
| Entity Awareness | Full (9 types) | None (structure-only) |
| Best For | Entity queries ("What references AASHTO M85?") | Section navigation ("What does Section 401 say?") |

---

## SLIDE A7: PageIndex vs Vector RAG — When to Use Which

**Title:** Complementary Approaches — PageIndex + GraphRAG

**PageIndex Excels At:**
- "What does Section 506 say about drilled shaft foundations?"
  - *Navigates directly to Section 506 — no embedding needed*
- "Compare Division 800 Materials between 2010 and 2021"
  - *Finds the exact same division in both documents*
- "Summarize the General Provisions"
  - *Knows Division 100 = General Provisions from the tree structure*
- Any query where the user references **specific sections, divisions, or page ranges**

**V4 GraphRAG Excels At:**
- "What materials reference AASHTO M85?"
  - *Entity traversal: AASHTO M85 → REFERENCES → Material nodes*
- "Find all documents about HMA compliance"
  - *Metadata search across 1,300+ documents*
- "What changed in Portland Cement specs?"
  - *SUPERSEDES edge links 2021 → 2010 documents*
- Any query involving **entity relationships, cross-document connections, or keyword search**

**The Integration Opportunity:**
The Unified Chatbot (V5) could use **both** approaches:

```
User Query
    |
    v
[Coordinator Agent]
    |
    |--- Structural query? -----> PageIndex (tree navigation)
    |--- Entity query? ----------> GraphRAG (Neo4j traversal)
    |--- Visual query? ----------> ColPali (visual retrieval)
    |--- Mixed? -----------------> Run all in parallel, merge results
```

This is the **Agentic RAG** concept: the LLM decides at query time which retrieval strategy is most appropriate, rather than always using vector search.

---

## SLIDE A8: How PageIndex Fits in the Version Timeline

**Title:** Updated System Evolution Timeline

| Version | Date | Architecture | Key Innovation |
|---------|------|-------------|----------------|
| V1 | Fall 2025 | Multiple Vector Stores | Basic semantic search over 2 spec PDFs |
| V2 | Jan 2026 | Router + Supervisor Agent | LLM routing for intent classification + metadata filtering |
| V3 | Jan-Feb 2026 | Neo4j Knowledge Graph (HuggingFace) | Graph-based retrieval + FlashRank reranking |
| V4 | Feb 2026 | Neo4j Knowledge Graph (Gemini) | Full entity extraction + semantic chunking + version chains |
| **PageIndex** | **Feb-Mar 2026** | **Hierarchical Tree + LLM Reasoning** | **Zero-cost structure extraction + reasoning-based navigation** |
| V5 (Future) | Planned | Unified Agentic Chatbot | All approaches (Vector + Graph + Visual + Tree) under one coordinator |

**Key Insight:** PageIndex is not a "replacement" for V4 — it's a **complementary paradigm**. V4 excels at entity-level queries across the full corpus. PageIndex excels at structural navigation within individual documents. Together, they cover the full spectrum of WYDOT engineer queries.

---

## SLIDE A9: PageIndex Chatapp Demo

**Title:** PageIndex Chatapp — Live on Chainlit

**Implementation:** `pageindex/chatapp_pageindex.py`

**Features:**
- Loads pre-built JSON trees for both 2010 and 2021 Standard Specifications
- Searches across both documents in parallel
- Returns section-level results with page citations
- Generates answers using Gemini 2.5 Flash with extracted page content
- **No Neo4j required** — runs entirely locally with just the PDF files and JSON trees

**Architecture:**
```
[Chainlit UI]
    |
    v
[Load JSON Trees] -----> 2010_structure.json (140 nodes)
                   -----> 2021_structure.json (154 nodes)
    |
    v
[User Query] --> [tree_search.py] --> LLM navigates both trees
    |
    v
[Selected Sections] --> Extract pages from PDF --> [Gemini 2.5 Flash]
    |
    v
[Formatted Answer with Section Citations]
```

**Key Files:**
- `pageindex/direct_builder.py` — Structure extraction (PyMuPDF + regex)
- `pageindex/tree_search.py` — LLM-powered tree navigation
- `pageindex/chatapp_pageindex.py` — Chainlit chatbot interface
- `pageindex/output/*.json` — Pre-built structure trees

---

## SLIDE A10: Research Significance of PageIndex

**Title:** Why PageIndex Matters for Research

**1. Novel Retrieval Paradigm:**
- Most RAG research focuses on **embedding-based retrieval** (dense, sparse, or hybrid)
- PageIndex introduces **structure-based reasoning retrieval** — the LLM navigates the document's own organizational hierarchy
- This is closer to how a human engineer finds information: "I need cement specs → go to the Materials division → find Section 801"

**2. Zero-Cost Indexing:**
- No embedding API calls (saves money at scale)
- No vector database infrastructure (saves complexity)
- Structure extracted in 2 seconds per document (vs. 15+ minutes for V4 entity extraction)
- Especially relevant for WYDOT's 1,300+ document corpus where full entity extraction is expensive

**3. Complementary to Vector RAG:**
- PageIndex finds sections by structural reasoning
- Vector RAG finds chunks by semantic similarity
- The best system uses BOTH: structure for navigation, vectors for content matching
- This "best of both worlds" is the foundation for the Agentic RAG (V5) concept

**4. Transferable to Other Domains:**
- Any domain with structured documents (legal codes, building codes, medical guidelines, government regulations) could use the same approach
- WYDOT Standard Specifications are a representative example of hierarchically organized technical documents

**5. Evaluation Framework:**
- Direct comparison possible: Same queries answered by V4 (GraphRAG) vs PageIndex (Tree RAG) vs both combined
- Metrics: Answer accuracy, retrieval precision (did it find the right section?), cost per query, latency
