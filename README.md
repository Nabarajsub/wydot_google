# MASDR-RAG: Multi-Agent Scoped Domain Retrieval for RAG

**Solving vector search dilution in large-scale heterogeneous document collections through LLM-based query routing and domain-scoped retrieval agents.**

[![Paper](https://img.shields.io/badge/Paper-CoLM%202026-blue)](graph_processing/paper/Template_2026/colm2026_conference.tex)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j%20Aura-green)](https://neo4j.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://python.org)

---

## Overview

This project addresses a fundamental scaling problem in Retrieval-Augmented Generation (RAG): **when a document corpus grows from dozens to thousands of heterogeneous documents, embedding-based retrieval degrades significantly** because semantically similar but contextually irrelevant chunks compete for top-k positions.

We formalize this as the **Vector Search Dilution** phenomenon and propose **MASDR-RAG**, a multi-agent architecture that restores retrieval precision through LLM-based query routing and domain-scoped search agents backed by a Neo4j knowledge graph.

### Key Findings

- **Vector Search Dilution**: Precision@10 drops from 0.92 (scoped) to 0.38 (global) when scaling from 54 to 1,128 documents
- **Precision-Faithfulness Paradox**: Naive multi-agent orchestration improves precision (0.77 -> 0.86) but *degrades* faithfulness (0.61 -> 0.35) due to context fragmentation
- **Hybrid-Routed solution**: Combining regex determinism with LLM classification in a single-agent scoped search preserves both precision and faithfulness

---

## Architecture

```
                    User Query
                        |
                  +-----v------+
                  |   Hybrid   |
                  |   Router   |
                  +-----+------+
                   /    |    \
          regex   /     |     \   LLM
         match   /      |      \ classification
                /       |       \
     +---------+  +-----+-----+  +-----------+
     | Specs   |  | Construction| | Planning  |
     | Agent   |  | Agent      | | Agent     |  ... (9 domain agents)
     +---------+  +-----------+  +-----------+
          |              |              |
     +----v----+   +-----v-----+  +----v----+
     | Scoped  |   | Scoped    |  | Scoped  |
     | Vector  |   | Vector    |  | Vector  |
     | Search  |   | Search    |  | Search  |
     +---------+   +-----------+  +---------+
          \              |              /
           +-------------+-----------+
                         |
                  +------v------+
                  |   Neo4j     |
                  | Knowledge   |
                  |   Graph     |
                  +-------------+
                  152K nodes
                  339K relationships
                  89K text chunks
```

### Components

| Component | Description | Tech |
|-----------|-------------|------|
| **Knowledge Graph** | Document -> Section -> Chunk hierarchy with SUPERSEDES relationships for version tracking | Neo4j Aura + HNSW vector index + BM25 fulltext |
| **Hybrid Router** | Two-stage routing: regex patterns for deterministic matching, LLM fallback for ambiguous queries | Python + Gemini API |
| **Domain Agents** | 9 specialized agents with scoped search boundaries defined by graph metadata filters | Google GenAI tool-calling |
| **Orchestrator** | Routes queries, manages agent selection, handles cross-domain and version comparison queries | Python async |
| **Evaluation Framework** | 200-query test suite across 12 categories with automated metric computation | Custom Python harness |

---

## Mathematical Formulation

### Vector Search Dilution Factor

We formally define the dilution factor as:

```
delta = 1 - (P_global / P_scoped)
```

Where:
- `P_scoped` = Precision@k when searching within the correct domain subset
- `P_global` = Precision@k when searching the entire corpus
- `delta in [0, 1]`: higher values indicate more severe dilution

For the WYDOT corpus, measured dilution: **delta = 0.59** (P_global = 0.38, P_scoped = 0.92).

### Search Space Reduction

Each domain agent reduces the search space by 65-98%:

| Agent | Chunks | Reduction |
|-------|--------|-----------|
| Specs Agent | 3,140 | 96.5% |
| Design Agent | 1,366 | 98.5% |
| Materials Agent | 2,184 | 97.5% |
| Construction Agent | 6,641 | 92.5% |
| Planning Agent | 13,607 | 84.7% |
| Safety Agent | 30,922 | 65.2% |
| Bridge Agent | 8,076 | 90.9% |
| Admin Agent | 2,439 | 97.3% |

---

## Evaluation Results (200 Queries)

Five system configurations evaluated with statistical significance testing (permutation tests, bootstrap confidence intervals):

| Configuration | Precision@10 | Recall@10 | Faithfulness | Routing Acc |
|---------------|-------------|-----------|--------------|-------------|
| Monolithic (baseline) | 0.38 | 0.42 | 0.61 | N/A |
| Mono + RRF | 0.52 | 0.55 | 0.58 | N/A |
| LLM + Scoped Search | 0.77 | 0.71 | 0.59 | 94.5% |
| MASDR-RAG (full) | 0.86 | 0.78 | 0.35 | 97.0% |
| **Hybrid-Routed** | **0.84** | **0.76** | **0.58** | **100%** |

The **Hybrid-Routed** configuration achieves the best precision-faithfulness tradeoff.

---

## AI Safety: Precision-Faithfulness Paradox

A critical finding with implications for safe AI deployment:

> **Improving retrieval precision does NOT automatically improve answer faithfulness.** Multi-agent orchestration introduces context fragmentation where agents retrieve precise but incomplete context, leading to hallucinated gap-filling by the LLM.

This paradox highlights that **RAG system safety requires holistic evaluation** -- optimizing a single metric (precision) can degrade the metric that actually matters for trustworthy AI (faithfulness). Our hybrid-routed solution demonstrates that architectural design choices have direct implications for AI safety in deployed systems.

---

## Project Structure

```
wydot_cloud/
|-- graph_processing/
|   |-- agentic_solution/        # Multi-agent RAG architecture
|   |   |-- orchestrator.py      # Query routing + agent orchestration
|   |   |-- agents.py            # Domain-scoped retrieval agents
|   |   |-- tools.py             # Neo4j search tools (vector + fulltext + graph)
|   |   +-- config.py            # Agent-to-document-series mappings
|   |-- evaluation/              # Evaluation framework
|   |   |-- run_evaluation.py    # 5-configuration evaluation harness
|   |   |-- test_suite_200.json  # 200-query benchmark (12 categories)
|   |   |-- generate_test_suite.py  # Automated query generation from KG
|   |   |-- fill_paper.py        # Auto-populate paper with results
|   |   +-- neo4j_stats.py       # Knowledge graph statistics
|   |-- autonomous/              # Autonomous agent framework
|   |   |-- planner.py           # Task decomposition
|   |   |-- execution_engine.py  # Agent execution loop
|   |   +-- agents/              # Specialized sub-agents
|   |-- chatapp_full.py          # Production Chainlit chat interface
|   +-- paper/                   # Research paper (CoLM 2026 submission)
|-- ingestion_service/           # Document ingestion pipeline
|   |-- app.py                   # Flask API for document upload
|   +-- local_ingest.py          # Local PDF/DOCX processing
|-- ingestneo4j.py               # Neo4j graph construction
|-- chatapp_gemini.py            # Gemini-powered chat application
|-- deploy/                      # GCP deployment configs
+-- research/                    # Extended research experiments
```

---

## Technical Stack

- **Graph Database**: Neo4j Aura (HNSW vector index + BM25 fulltext index)
- **Embeddings**: text-embedding-004 (768-dim)
- **LLM**: Gemini 2.5 Flash (routing + generation + tool-calling)
- **Backend**: Python 3.11, Chainlit, Flask
- **Evaluation**: Custom harness with LLM-as-judge metrics
- **Deployment**: Google Cloud Platform (Cloud Run)

---

## Getting Started

### Prerequisites

```bash
python 3.11+
pip install chainlit neo4j google-generativeai
```

### Configuration

Set environment variables or edit `graph_processing/agentic_solution/config.py`:

```bash
export NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export GEMINI_API_KEY="your-api-key"
```

### Run the Chat Application

```bash
cd graph_processing
chainlit run chatapp_full.py -w
```

### Run the Evaluation Suite

```bash
cd graph_processing/evaluation
python run_evaluation.py
```

---

## Knowledge Graph Schema

```cypher
// Node types
(:Document {title, series, year, file_type})
(:Section {heading, level, doc_title})
(:Chunk {text, embedding, chunk_index})
(:Entity {name, type, category})

// Relationships
(:Document)-[:HAS_SECTION]->(:Section)
(:Section)-[:HAS_CHUNK]->(:Chunk)
(:Document)-[:SUPERSEDES]->(:Document)  // Version tracking
(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:HAS_FACT]->(fact_text)
```

**Scale**: 1,128 documents | 88,907 chunks | 152,231 nodes | 338,569 relationships | 9 document categories

---

## Research Paper

This work has been submitted to **CoLM 2026**. The paper formalizes the vector search dilution phenomenon, introduces the MASDR-RAG architecture, and presents evaluation results across 200 queries with statistical significance testing.

Key contributions:
1. **Formal definition** of vector search dilution with measurable dilution factor
2. **Discovery** of the precision-faithfulness paradox in multi-agent RAG
3. **Hybrid-routed architecture** that resolves the paradox
4. **Production-scale evaluation** on 1,128 real government documents

---

## Citation

```bibtex
@inproceedings{subedi2026masdr,
  title={MASDR-RAG: Diagnosing and Resolving Vector Search Dilution
         Through Multi-Agent Scoped Domain Retrieval},
  author={Subedi, Nabaraj},
  booktitle={Conference on Language Modeling (CoLM)},
  year={2026}
}
```

---

## License

This project is developed for the Wyoming Department of Transportation (WYDOT). Contact the author for licensing inquiries.

## Author

**Nabaraj Subedi** -- University of Wyoming
