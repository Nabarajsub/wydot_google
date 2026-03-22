# Research Plan 07: CAG vs RAG vs Long-Context — The Efficiency Frontier for Static Engineering Corpora
## Target: ICML 2026 Main Track

## 1. Problem Statement

With Gemini 2.5 Flash supporting 1M+ tokens, a fundamental question arises: **Do we even need RAG for a static 890-page specification?** Why not just load the entire document into the context window?

The answer involves a nuanced tradeoff between **cost, latency, accuracy, and the "Lost in the Middle" problem.** But no one has rigorously measured this tradeoff on **real engineering documents.**

### The Research Gap

| Paper | Year | Finding | Missing |
|-------|------|---------|---------|
| **Liu et al. "Lost in the Middle"** | 2024 | LLMs struggle with facts in the middle of long contexts | No engineering document experiments |
| **RetroLM** (arXiv 2025) | 2025 | KV-level retrieval for long context | Tested on general QA, not domain-specific |
| **CAG paper** (ACM WebConf 2025) | 2025 | Preloaded KV cache beats RAG for static KBs | Tested on Wikipedia, not structured specs |
| **Our Proposal** | 2026 | **First head-to-head comparison** on real engineering specs | **Fills the gap** |

---

## 2. Experimental Design

### 2.1 The Three Contenders

| System | How It Works | Infrastructure |
|--------|-------------|---------------|
| **RAG** | Retrieve top-K chunks, feed to LLM | Neo4j + Vector Index |
| **CAG** | Preload entire spec into KV cache, query directly | Gemini Flash KV cache API |
| **Long Context** | Feed full 890-page PDF as raw text into prompt | Gemini 1.5 Pro (1M tokens) |

### 2.2 Efficiency Dimensions

```
                    Cost ($)
                     ▲
                     │      Long Context
                     │      ●
                     │
                     │          CAG (amortized)
                     │          ●
                     │
                     │  RAG
                     │  ●
                     └───────────────────► Accuracy (%)
```

### 2.3 Controlled Variables

| Variable | Held Constant |
|----------|--------------|
| LLM | Gemini 2.5 Flash for all three |
| Document | 2021 WYDOT Standard Specs (771 pages) |
| Queries | Same 100 queries from TransportBench (Plan 04) |
| Temperature | 0.0 (deterministic) |
| Evaluation | Same metrics (accuracy, faithfulness, citation) |

---

## 3. Key Hypotheses

1. **H1**: RAG will have the **lowest cost per query** but miss "global" context.
2. **H2**: Long Context will have the **highest accuracy for multi-hop questions** but cost 10-50x more.
3. **H3**: CAG will be the **Pareto-optimal** choice for static specs (near-Long-Context accuracy at near-RAG cost).
4. **H4**: All three will exhibit the **"Lost in the Middle"** effect for facts in Division 400-500 (the physical center of the document).
5. **H5**: **Hierarchical RAG** (PageIndex) will outperform all three for structural navigation queries.

---

## 4. Novel Contributions

### 4.1 "Efficiency Frontier" Framework
Define a formal framework for comparing retrieval architectures across 4 dimensions:

```
Efficiency Score = f(Accuracy, Cost, Latency, Faithfulness)

Pareto Optimal = system where no other system is better on ALL 4 dimensions
```

### 4.2 "Lost in the Middle" in Engineering Specs
First study of positional bias in structured documents. WYDOT specs have 8 Divisions:
- Division 100 (start) / Division 800 (end) = Expected high recall
- Division 400-500 (middle) = Expected low recall
- **Test**: Does this U-shape exist for engineering docs?

### 4.3 Hybrid CAG-RAG Architecture
If H3 is confirmed, propose a **"SpecCache"** system:
- Preload Division summaries into KV cache (~50K tokens)
- Use RAG for fine-grained chunk retrieval
- Combine: Global understanding (CAG) + Precise citations (RAG)

---

## 5. Evaluation Protocol

### 5.1 Query Types (100 queries)

| Type | # | What It Tests |
|------|---|---------------|
| Needle-in-haystack (specific fact) | 30 | Precision of retrieval |
| Multi-hop (connect 2+ sections) | 20 | Reasoning across distance |
| Global summary | 15 | "What are the main themes of Division 200?" |
| Table extraction | 15 | Numerical accuracy from tables |
| Positional bias test | 20 | 5 queries per Division pair (100/200, 400/500, 700/800) |

### 5.2 Metrics

| Metric | Formula/Method |
|--------|---------------|
| Answer Accuracy | LLM-as-judge (Claude) + human verification |
| Cost per Query | Token count × price per token |
| Latency | Time-to-first-token + total generation time |
| Positional Bias Score | Accuracy(Div100) - Accuracy(Div400) |
| Faithfulness | NLI entailment score |

### 5.3 Statistical Rigor
- 3 runs per query (for variance estimation)
- Paired t-test for significance
- Report confidence intervals

---

## 6. Implementation Roadmap

| Phase | Duration | Deliverable |
|-------|----------|------------|
| Setup all 3 systems | 1 week | RAG (existing), CAG (KV cache API), Long Context (full prompt) |
| Build TransportBench-100 | 2 weeks | 100 annotated queries with gold answers |
| Run experiments | 1 week | 3 runs × 100 queries × 3 systems = 900 inference calls |
| Analysis + Ablations | 1 week | Efficiency frontier plots, positional bias analysis |
| Paper writing | 2 weeks | ICML-format paper (8 pages + appendix) |

---

## 7. Why This Wins at ICML

1. **Timely debate**: "RAG vs Long Context" is the hottest topic in 2026.
2. **Clean experimental design**: Head-to-head comparison with controlled variables.
3. **Novel framework**: "Efficiency Frontier" formalism is reusable by the community.
4. **Real-world data**: Not synthetic — actual government engineering specs.
5. **Actionable result**: Clear recommendation for practitioners: "When to use what."

---

## 8. Publication Target

- **Primary**: ICML 2026 Main Track (Deadline: Jan 2026 → paper ready by Dec 2026 cycle)
- **Backup**: NeurIPS 2026 (May 2026)
- **Title**: *"The Efficiency Frontier: CAG vs. RAG vs. Long Context for Static Engineering Corpora"*
