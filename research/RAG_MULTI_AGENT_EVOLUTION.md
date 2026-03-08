# How RAG Will Evolve in Multi-Agentic Systems
## A Research Report with Cited Papers

---

## 1. Introduction: The Paradigm Shift

RAG is undergoing a fundamental transformation. Between 2023–2024, RAG was a **pipeline**: Retrieve → Generate. By 2025–2026, RAG is becoming a **runtime** — an orchestration layer within multi-agent architectures where retrieval is just one of many tools that autonomous agents can invoke, critique, and iterate upon.

> **The key insight**: RAG is no longer the "system." It is the **memory backbone** of agent systems.

This report surveys the current state and trajectory of this evolution, citing the most relevant papers.

---

## 2. The Evolution Timeline

```
2020 ──── 2023 ──── 2024 ──── 2025 ──── 2026 ──── Future
  │         │         │         │         │         │
  │    Naive RAG  Advanced   Agentic   Multi-     Knowledge
  │    (Lewis     RAG        RAG       Agent      Runtime
  │    et al.)    (Self-RAG, (Surveys, RAG        (RAG as OS)
  │               CRAG)      MA-RAG)   (RAGentA)
  │
  └── RAG = pipeline ──────► RAG = tool ──────► RAG = memory layer
```

---

## 3. Phase 1: From Pipeline to Self-Correction (2023–2024)

### 3.1 Self-RAG: The Model Learns to Critique Itself

> **Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H.** (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. [arXiv:2310.11511]

Self-RAG introduced **reflection tokens** — special markers (`[Retrieval]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) that the model generates inline to self-assess whether retrieval is needed, whether the retrieved content is relevant, and whether the generated text is supported. This was the first step toward "agentic" behavior: the model makes autonomous decisions about its own retrieval process.

**Key results**: Self-RAG outperformed ChatGPT and standard RAG by 10–17% on citation accuracy across knowledge-intensive benchmarks.

**Limitation**: Self-RAG is still a **single-model** system. It doesn't orchestrate multiple agents or tools.

### 3.2 CRAG: The Decision Gate

> **Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H.** (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884. [Under review, ICLR 2025]

CRAG introduced a **Decision Gate** that scores retrieved documents as Correct, Ambiguous, or Incorrect. If the gate triggers "Incorrect," the system falls back to a **web search** to find better sources. This is significant because it showed that RAG systems can have **contingency plans** — a hallmark of agentic behavior.

**Key result**: CRAG improved EM scores by 5–10% on PopQA and PubMedQA by catching and correcting bad retrievals.

---

## 4. Phase 2: The Rise of Agentic RAG (2025)

### 4.1 The Agentic RAG Survey

> **Singh, A., et al.** (2025). *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.* arXiv:2501.09136.

This is the foundational survey that formalized the term "Agentic RAG." It defines four **agentic design patterns**:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Reflection** | Agent evaluates its own output and decides if it needs to retry | Self-RAG's critique tokens |
| **Planning** | Agent decomposes complex queries into sub-tasks | "First find the cement spec, then find the related test method" |
| **Tool Use** | Agent decides which tool to invoke (vector search, graph query, web search, calculator) | Selecting Neo4j for entity lookup vs. Milvus for semantic search |
| **Multi-Agent Collaboration** | Multiple specialized agents communicate to solve a problem | Planner Agent → Retriever Agent → Verifier Agent |

**The survey's taxonomy of architectures:**

```
Agentic RAG
├── Single-Agent RAG (Self-RAG, CRAG)
├── Multi-Agent RAG (MA-RAG, RAGentA)
├── Hierarchical RAG (Router → Specialist agents)
├── Corrective RAG (Decision gate → fallback)
├── Adaptive RAG (Dynamic retrieval depth)
└── Graph-Based RAG (GraphRAG, LightRAG)
```

### 4.2 Reasoning RAG: System 1 vs System 2

> **Anonymous.** (2025). *Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic RAG for Industry Challenges.* arXiv (June 2025).

This survey borrows from Kahneman's dual-process theory:

| System | RAG Behavior | When Used |
|--------|-------------|-----------|
| **System 1** (Fast) | Single retrieval + immediate generation | Simple factual lookups |
| **System 2** (Slow) | Multi-step reasoning, planning, reflection, tool orchestration | Complex analysis, comparison, impact assessment |

**Key insight**: Most real-world enterprise queries require **System 2** — but most deployed RAG systems only implement **System 1**. The gap between research and production is enormous.

### 4.3 Deep Reasoning + RAG

> **Anonymous.** (2025). *Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs.* arXiv (July 2025).

This paper maps how reasoning **optimizes** each stage of RAG:
- **Pre-retrieval reasoning**: Query decomposition, HyDE (Hypothetical Document Embeddings)
- **During-retrieval reasoning**: Adaptive depth, re-ranking, iterative refinement
- **Post-retrieval reasoning**: Self-correction, claim verification, citation checking

**Key finding**: Agentic LLMs that interleave search and reasoning outperform both RAG-only and reasoning-only approaches on complex benchmarks.

---

## 5. Phase 3: Multi-Agent RAG Architectures (2025)

### 5.1 MA-RAG: Collaborative Chain-of-Thought

> **Nguyen, T., Chin, P., & Tai, Y.-W.** (2025). *MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning.* arXiv:2505.20689.

MA-RAG deploys **four specialized agents**:

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
│ Planner  │───►│ Step Definer │───►│ Extractor │───►│ QA Agent │
│          │    │              │    │           │    │          │
│ Breaks   │    │ Defines each │    │ Extracts  │    │ Generates│
│ query    │    │ retrieval    │    │ evidence  │    │ final    │
│ into     │    │ step         │    │ from docs │    │ answer   │
│ sub-tasks│    │              │    │           │    │          │
└──────────┘    └──────────────┘    └───────────┘    └──────────┘
```

**Key results**: On HotpotQA, MA-RAG significantly outperformed standalone LLMs and single-agent RAG baselines. **Training-free** — uses only prompting, no fine-tuning.

**Key limitation**: Agents communicate through a fixed pipeline. No dynamic negotiation or conflict resolution between agents.

### 5.2 RAGentA: Attributed QA with Iterative Filtering

> **Besrour, I., He, J., Schreieder, T., & Färber, M.** (2025). *RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering.* SIGIR 2025 (LiveRAG Challenge).

RAGentA uses a **4-agent pipeline** with iterative refinement:
- Agent 1: Initial answer generation
- Agent 2: Document-Question-Answer triplet filtering
- Agent 3: Final answer with inline citations
- Agent 4: Completeness checker + query reformulation

**Key results**: +1.09% correctness, +10.72% faithfulness over standard RAG. **+12.5% Recall@20** over the best single retrieval model.

**Contribution**: First system to systematically verify citations at the claim level in a multi-agent setup.

### 5.3 CIIR@LiveRAG: Self-Training for Multi-Agent Optimization

> **Anonymous.** (2025). *CIIR@LiveRAG 2025: Optimizing Multi-Agent RAG through Self-Training.* arXiv.

Uses a **self-training paradigm** where agents learn from their own successful interactions to improve inter-agent collaboration. This is a step toward **learning multi-agent RAG systems** rather than hand-designed pipelines.

---

## 6. Phase 4: RAG + Knowledge Graphs in Agent Systems (2025–2026)

### 6.1 GraphRAG: Community Summaries as Agent Memory

> **Edge, D., Trinh, H., Cheng, N., Bradley, J., et al.** (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130. Microsoft Research.

GraphRAG builds a **knowledge graph → community clusters → hierarchical summaries**. In a multi-agent context, these pre-computed summaries serve as **shared long-term memory** that any agent can query for global context.

### 6.2 DRIFT Search: Hybrid Global-Local for Agents

> **Microsoft Research.** (2024). *Introducing DRIFT Search: Combining Global and Local Search Methods.* Microsoft Research Blog (Oct 2024).

DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) lets an agent start with a **global overview** (community summaries) and then "drift" into **local specifics** (entity neighbors). This mirrors how human experts reason: "First, understand the big picture. Then, drill into details."

**Agent context**: An orchestrator agent can use DRIFT to decide: "I need global context first" → reads community summaries → "Now I need specific data" → drills into local graph.

### 6.3 LightRAG: Fast Graph RAG for Real-Time Agents

> **Guo, Z., et al.** (2024). *LightRAG: Simple and Fast Retrieval-Augmented Generation.* arXiv:2410.05779. University of Hong Kong.

LightRAG offers **dual-level retrieval** (entity-level + topic-level) with **incremental graph updates** — critical for agents that need to work with evolving knowledge bases. Its speed makes it viable for real-time agent workflows.

---

## 7. Phase 5: The Emerging Future (2026+)

### 7.1 RAG as a "Knowledge Runtime"

The trajectory suggests RAG is evolving from a **component** to an **operating system layer**:

```
2024: RAG = retrieve() + generate()

2025: Agentic RAG = plan() + retrieve() + reflect() + generate()

2026: Knowledge Runtime = {
    retrieve(),
    verify(),
    reason(),
    access_control(),    ← WHO can see what?
    audit_trail(),       ← WHY did the agent retrieve this?
    version_management(),← WHICH version of the document?
    cache_management()   ← WHAT has been pre-loaded?
}
```

### 7.2 MCP: The "USB-C" for Agent-RAG Integration

The **Model Context Protocol (MCP)** is emerging as the standard for connecting agents to data sources:
- RAG systems exposed as MCP **resources** (documents, embeddings)
- Graph queries exposed as MCP **tools**
- Agent frameworks (LangGraph, CrewAI) connect to RAG via MCP **clients**

This standardization means agents can **discover** available knowledge sources at runtime rather than having them hard-coded.

### 7.3 Multi-Agent Specialization: The "Team of Experts" Model

The future architecture isn't one big agent — it's a **team**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                       │
│  Receives query, classifies complexity, delegates to team   │
└──────────┬─────────────┬────────────────┬───────────────────┘
           │             │                │
    ┌──────▼──────┐ ┌────▼────────┐ ┌─────▼──────────┐
    │ RETRIEVER   │ │ GRAPH       │ │ VERIFIER       │
    │ AGENT       │ │ NAVIGATOR   │ │ AGENT          │
    │             │ │ AGENT       │ │                │
    │ Vector      │ │ Neo4j       │ │ Cross-checks   │
    │ search,     │ │ traversal,  │ │ claims against │
    │ keyword     │ │ entity      │ │ source docs,   │
    │ search,     │ │ linking,    │ │ detects        │
    │ BM25        │ │ community   │ │ hallucinations │
    │             │ │ summaries   │ │                │
    └─────────────┘ └─────────────┘ └────────────────┘
           │             │                │
    ┌──────▼──────┐ ┌────▼────────┐ ┌─────▼──────────┐
    │ TEMPORAL    │ │ TABLE       │ │ REPORT         │
    │ AGENT       │ │ READER      │ │ GENERATOR      │
    │             │ │ AGENT       │ │                │
    │ Version     │ │ Multimodal  │ │ Synthesizes    │
    │ comparison, │ │ table       │ │ findings into  │
    │ change      │ │ extraction, │ │ structured     │
    │ detection   │ │ numerical   │ │ output with    │
    │             │ │ validation  │ │ citations      │
    └─────────────┘ └─────────────┘ └────────────────┘
```

---

## 8. Key Research Papers Summary Table

| Paper | Authors | Venue/Year | Key Contribution | Cited As |
|-------|---------|-----------|------------------|----------|
| Self-RAG | Asai et al. | ICLR 2024 | Reflection tokens for self-assessment | [1] |
| CRAG | Yan et al. | arXiv 2024 | Decision gate for retrieval quality | [2] |
| RAPTOR | Sarthi et al. | ICLR 2024 | Tree-organized recursive summarization | [3] |
| GraphRAG | Edge et al. | arXiv 2024 (Microsoft) | Community-based global search | [4] |
| LightRAG | Guo et al. | arXiv 2024 (HKU) | Dual-level graph retrieval | [5] |
| Agentic RAG Survey | Singh et al. | arXiv Jan 2025 | Taxonomy of agentic RAG architectures | [6] |
| Reasoning RAG Survey | Anonymous | arXiv Jun 2025 | System 1 vs System 2 RAG reasoning | [7] |
| Towards Agentic RAG | Anonymous | arXiv Jul 2025 | Deep reasoning + retrieval synergy | [8] |
| MA-RAG | Nguyen, Chin, Tai | arXiv May 2025 | 4-agent collaborative chain-of-thought | [9] |
| RAGentA | Besrour et al. | SIGIR 2025 | Multi-agent attributed QA with citations | [10] |
| CIIR@LiveRAG | Anonymous | arXiv 2025 | Self-training for multi-agent RAG | [11] |
| DRIFT Search | Microsoft Research | Blog Oct 2024 | Global-to-local hybrid graph search | [12] |
| SCMRAG | Agrawal et al. | AAMAS 2025 | Self-corrective multi-hop with dynamic KG | [13] |
| HGMem | Zhou et al. | arXiv Dec 2025 | Hypergraph memory for multi-step RAG | [14] |
| VersionRAG | Huwiler et al. | arXiv Oct 2025 | Version-aware retrieval for evolving docs | [15] |
| HiRAG | Huang et al. | EMNLP Findings 2025 | Hierarchical knowledge in RAG indexing | [16] |

---

## 9. Open Problems & Research Opportunities

### 9.1 Unsolved Problems

| Problem | Current State | What's Missing |
|---------|-------------|---------------|
| **Agent conflict resolution** | Agents follow fixed pipelines [9, 10] | What happens when Retriever and Verifier disagree? |
| **Cost control** | Multi-agent = multi-LLM-calls = expensive | No cost-aware agent scheduling |
| **Provenance across agents** | RAGentA [10] tracks citations | No formal provenance DAG across the full agent chain |
| **Domain-specific agent roles** | All current work uses generic agents | No agent specialized for engineering, medicine, or law |
| **Agent memory across sessions** | HGMem [14] is single-session | Cross-session learning for multi-agent teams |
| **Benchmark for agentic RAG** | No standard benchmark exists | Need to measure agent coordination, not just answer quality |

### 9.2 Where YOUR Project Fits

Your WYDOT system is ahead of most research papers because you already have:

```
Current Research                    Your System
───────────────────                 ─────────────────────
Single retrieval tool      →       3 retrieval axes (structural, semantic, temporal)
Generic agents             →       Domain-specialized (engineering specs)
No memory                  →       Neo4j persistent graph
No provenance              →       [:SUPERSEDES] edges + page citations
Text-only                  →       Table extraction + PageIndex trees
```

**The breakthrough opportunity**: Build the first **domain-specialized multi-agent RAG system** where:
1. Each agent maps to a retrieval axis (Structural Agent, Semantic Agent, Temporal Agent)
2. The Orchestrator uses query classification to route to the right agent team
3. The Verifier checks answers against a formal constraint database
4. The full pipeline produces a **Provenance DAG** showing exactly how the answer was constructed

This would be the **SpecRAG** system from the Breakthrough Analysis — but framed as a **multi-agent architecture**, which makes it even more timely.

---

## 10. Conclusion

RAG is not dying — it is **dissolving** into the agent stack. By 2027, we won't talk about "RAG systems" as standalone products. We'll talk about **agents that happen to use retrieval** as one of their many capabilities. The papers that will define this era are the ones that show how retrieval can be **orchestrated, verified, and composed** across multiple specialized agents.

Your project is positioned at exactly this frontier.
