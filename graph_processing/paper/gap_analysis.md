# Gap Analysis: WYDOT Paper Positioning

## Where Our Work Sits in the Literature

### The Intersection No One Covers

Our paper sits at the intersection of four research threads:

```
    Vector Search Dilution (Area 1)
              |
              v
    Query Routing (Area 2) -----> Multi-Agent RAG (Area 3)
              |                          |
              v                          v
    Document Routing (Area 7)     Domain-Specific RAG (Area 5)
              |                          |
              +----------+---------------+
                         |
                         v
              WYDOT: Full Pipeline
              (Diagnosis → Routing → Scoped Agents)
```

**No single paper addresses the complete journey** from diagnosing dilution in a scaled system to solving it through combined routing + domain-scoped agents.

---

## Paper-by-Paper Gap Comparison

### vs. RAGRouter (arXiv 2505.23052)
- **They do**: Route queries to optimal retrieval strategy
- **They don't**: Address domain-scoped search spaces or multi-agent execution
- **We add**: Domain routing (not strategy routing), agent-based execution, knowledge graph scoping

### vs. Adaptive-RAG (arXiv 2403.14403)
- **They do**: Route by query complexity (simple/moderate/complex)
- **They don't**: Route by domain or document category
- **We add**: Orthogonal routing dimension (domain + complexity could be combined)

### vs. MA-RAG (arXiv 2505.20096)
- **They do**: Multi-agent retrieval and generation
- **They don't**: Diagnose why multi-agent is needed (dilution), or use graph-based scoping
- **We add**: Motivation from dilution analysis, graph-metadata agent boundaries

### vs. Microsoft GraphRAG (arXiv 2404.16130)
- **They do**: Build entity graphs, use community detection for global queries
- **They don't**: Address scaling failures or use agents for retrieval
- **We add**: Practical graph structure (Document→Section→Chunk) for agent scoping

### vs. HybridRAG (arXiv 2408.04948)
- **They do**: Combine KG traversal with vector search
- **They don't**: Address dilution at scale or domain routing
- **We add**: Dilution diagnosis, domain-scoped hybrid search

### vs. RAG4CM (DOI 10.1016/j.aei.2025.103158)
- **They do**: RAG for construction management documents
- **They don't**: Address multi-document-type scaling or multi-agent architecture
- **We add**: Scaling to 1,128 docs across 9 categories, multi-agent solution

### vs. Multi-Meta-RAG (arXiv 2406.13213)
- **They do**: Metadata filtering before retrieval
- **They don't**: Use LLM routing or multi-agent architecture
- **We add**: LLM-based routing, agent specialization, tool calling

### vs. Blended RAG (arXiv 2404.07220)
- **They do**: Combine dense and sparse retrieval
- **They don't**: Address dilution from heterogeneous corpora
- **We add**: Domain scoping makes blended search more effective within agent boundaries

---

## Our Unique Contributions Matrix

| Contribution | Closest Related Work | What We Add |
|---|---|---|
| Dilution diagnosis at production scale | arXiv 2404.00657 (observations) | Systematic analysis with root cause (vocab overlap + chunk imbalance) |
| LLM-based domain routing | RAGRouter, Adaptive-RAG | Zero-shot, 100% coverage, domain-level (not strategy/complexity) |
| Agent scoping via graph metadata | Microsoft GraphRAG | Document-series → agent boundary mapping |
| Multi-agent for retrieval quality | MA-RAG | Motivated by dilution, uses KG for boundaries |
| Real-world government doc evaluation | RAG4CM | 20x more documents, 9 categories, multi-agent |
| Combined pipeline (diagnose → route → scope) | None | Full end-to-end novel |

---

## Potential Reviewer Questions & Answers

**Q: How is this different from just using metadata filters?**
A: Metadata filters require knowing which filter to apply. Our LLM router automatically determines the correct document domain from natural language queries. The multi-agent architecture adds tool-calling capabilities (compare versions, get specific sections) that simple filtering cannot provide.

**Q: Why not just train a classifier instead of using LLM routing?**
A: LLM routing is a zero-shot approach that works immediately without training data. For a government agency deploying a new system, this is critical. We discuss training a classifier as future work once query logs accumulate.

**Q: Is the dilution problem specific to your dataset?**
A: No. Any domain with overlapping terminology across document categories will exhibit this. Transportation is a strong example, but legal (statutes vs. case law vs. regulations), medical (clinical guidelines vs. research vs. formularies), and engineering (specs vs. manuals vs. reports) domains share the same characteristics.

**Q: What about retrieval latency overhead from routing + agent calls?**
A: Routing adds ~200-400ms. However, the reduced search space (89K → 1.6K-14K chunks) partially compensates. Net latency increase is ~500ms-1s, which is acceptable for a chatbot use case. We discuss optimization strategies (embedding cache, connection pooling) that can offset this.

---

## Suggested Venues

1. **ACL/EMNLP (NLP)**: Findings track — practical NLP systems
2. **SIGIR (IR)**: Resource/reproducibility track — retrieval at scale
3. **AAAI (AI)**: AI for social good track — government applications
4. **ASCE Journal of Computing in Civil Engineering**: Domain-specific venue
5. **arXiv preprint**: For immediate visibility and citation

---

## Strengthening the Paper

### What We Have
- Real production system (not toy benchmark)
- Clear problem diagnosis with case studies
- Working solution with code
- Qualitative evaluation showing improvement

### What We Need
- **Quantitative evaluation**: Precision@K, Recall@K, NDCG metrics on labeled test set
- **Ablation study**: Remove routing → measure degradation; remove agents → measure degradation
- **Latency benchmarks**: P50/P95 for each pipeline stage
- **User study**: Have WYDOT engineers rate answer quality (if possible before paper submission)
- **Reproducibility**: Open-source the agent framework (not the WYDOT data, but the architecture)
