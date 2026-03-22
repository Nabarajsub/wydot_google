# Literature Review: Scaling RAG Systems for Large Document Collections

## Research Area 1: Vector Search Dilution at Scale

### Problem Definition
When a RAG system scales from tens to thousands of documents, the vector index grows proportionally. Embedding similarity becomes less discriminative as semantically adjacent but contextually irrelevant chunks compete for top-k positions. This is the core problem observed in the WYDOT system scaling from 54 to 1,128 documents (1.6K to 89K chunks).

### Key Papers

1. **"Observations on Building RAG Systems in Practice"**
   - arXiv: 2404.00657
   - Documents practical failure modes in production RAG, including retrieval degradation at scale
   - Relevance: Directly validates our observed dilution phenomenon

2. **HNSW Index Degradation Studies**
   - Hierarchical Navigable Small World graphs (used by Neo4j vector indexes) show recall degradation as corpus size increases
   - The approximate nearest neighbor tradeoff becomes more pronounced with heterogeneous document collections
   - Relevance: Explains why our Neo4j vector index returns crash report chunks for specs queries

---

## Research Area 2: Query Routing in RAG Systems

### Key Papers

3. **RAGRouter: Adaptive RAG Query Routing**
   - arXiv: 2505.23052
   - Trains a lightweight classifier to route queries to optimal retrieval strategies
   - Relevance: Our LLM-based routing is a zero-shot alternative; RAGRouter uses trained classifiers

4. **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**
   - arXiv: 2403.14403
   - Routes queries based on complexity: simple (no retrieval), moderate (single retrieval), complex (multi-step)
   - Relevance: Orthogonal to our domain-based routing; could be combined

5. **LTRR: LLM-Based Text Rewriting for Retrieval**
   - arXiv: 2506.13743
   - Query rewriting before retrieval to improve recall
   - Relevance: Could augment our routing with query reformulation

6. **RouteRAG: Router-based Retrieval-Augmented Generation**
   - arXiv: 2512.09487
   - Dynamic routing between retrieval strategies based on query characteristics
   - Relevance: Similar motivation but different approach (strategy routing vs. domain routing)

---

## Research Area 3: Multi-Agent RAG Architectures

### Key Papers

7. **"Agentic RAG: A Survey on Agentic Retrieval-Augmented Generation"**
   - arXiv: 2501.09136
   - Comprehensive survey of agent-based RAG architectures
   - Taxonomy: single-agent, multi-agent, hierarchical
   - Relevance: Our orchestrator + domain agents is a hierarchical multi-agent architecture

8. **MA-RAG: Multi-Agent RAG**
   - arXiv: 2505.20096
   - Multi-agent system where agents specialize in different aspects of retrieval and generation
   - Relevance: Validates our architectural choice of specialized domain agents

9. **"Multi-Agent RAG System for Complex Research Tasks"**
   - arXiv: 2412.05838
   - Agents collaborate on complex queries requiring multiple retrieval steps
   - Relevance: Our compare_versions and cross-domain search implement similar patterns

10. **A-RAG: Agentic Retrieval-Augmented Generation**
    - arXiv: 2602.03442
    - Autonomous agents that decide when and how to retrieve
    - Relevance: Our Gemini tool-calling loop is a form of agentic retrieval

---

## Research Area 4: Graph-Enhanced RAG (GraphRAG)

### Key Papers

11. **Microsoft GraphRAG: From Local to Global**
    - arXiv: 2404.16130
    - Builds entity-relationship graphs from documents, uses community detection for global queries
    - Relevance: WYDOT has a knowledge graph but uses it for structured metadata, not community-based summarization

12. **"Graph RAG: Unlocking LLM Discovery on Narrative Private Data"**
    - arXiv: 2408.08921
    - Survey of graph-augmented retrieval approaches
    - Relevance: Our Document-Section-Chunk hierarchy is a form of graph-structured retrieval

13. **HybridRAG: Integrating Knowledge Graphs and Vector Retrieval**
    - arXiv: 2408.04948
    - Combines KG traversal with vector similarity for better retrieval
    - Relevance: WYDOT uses both graph structure (SUPERSEDES relationships, entity facts) and vector search

---

## Research Area 5: Domain-Specific RAG Systems

### Key Papers

14. **RAG4CM: RAG for Construction Management**
    - DOI: 10.1016/j.aei.2025.103158
    - RAG system for construction domain documents
    - Relevance: Most directly comparable — same domain (transportation/construction), similar document types

15. **"Graph-RAG for Engineering Compliance Checking"**
    - arXiv: 2412.08593
    - Uses graph structure for regulatory compliance document retrieval
    - Relevance: WYDOT Standard Specs are regulatory compliance documents

---

## Research Area 6: Hybrid Search Strategies

### Key Papers

16. **Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers**
    - arXiv: 2404.07220
    - Combines dense (vector) and sparse (keyword) retrieval with learned blending
    - Relevance: WYDOT uses vector + fulltext search with priority-based merging

17. **Multi-Meta-RAG: Improving RAG for Multi-Hop Queries Using Database Filtering**
    - arXiv: 2406.13213
    - Metadata-based filtering before retrieval to improve precision
    - Relevance: Our document-scoped search is a form of metadata pre-filtering

---

## Research Area 7: Self-Correcting and Adaptive RAG

### Key Papers

18. **Self-RAG: Learning to Retrieve, Generate, and Critique**
    - arXiv: 2310.11511
    - Model learns when to retrieve and self-evaluates retrieval quality
    - Relevance: Future enhancement for WYDOT (not currently implemented)

19. **CRAG: Corrective RAG**
    - arXiv: 2401.15884
    - Evaluates retrieval quality and triggers corrective retrieval if needed
    - Relevance: Could add a quality check layer to our pipeline

20. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**
    - arXiv: 2401.18059
    - Builds hierarchical summaries for multi-level retrieval
    - Relevance: Could help with annual report comparison queries

---

## Gap Analysis

| Approach | Addresses Dilution? | Addresses Routing? | Domain-Specific? | Multi-Agent? | Production-Scale? |
|----------|--------------------|--------------------|-------------------|--------------|-------------------|
| RAGRouter | Indirectly | Yes | No | No | No |
| Adaptive-RAG | No | Yes (complexity) | No | No | No |
| MA-RAG | No | No | No | Yes | No |
| Microsoft GraphRAG | No | No | No | No | Yes |
| HybridRAG | Partially | No | No | No | No |
| RAG4CM | No | No | Yes (construction) | No | No |
| Multi-Meta-RAG | Partially | Yes (metadata) | No | No | No |
| **WYDOT (ours)** | **Yes** | **Yes (LLM + agent)** | **Yes** | **Yes** | **Yes (1,128 docs)** |

**Key Finding:** No existing paper addresses the complete pipeline from diagnosing vector search dilution to solving it through combined LLM-based query routing and multi-agent domain-scoped retrieval in a production-scale, domain-specific document collection.
