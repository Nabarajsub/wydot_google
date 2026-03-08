# Literature Review & Research Gap Analysis
## For All 9 WYDOT Research Plans

> This document provides a critical review of current literature for each research plan, cites specific papers, and identifies precise research gaps that your project can address.

---

## References Index

For quick lookup, here are all cited papers with their identifiers used throughout this document:

| ID | Paper | Authors | Venue | Year |
|----|-------|---------|-------|------|
| [1] | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Asai, Wu, Wang, Sil, Hajishirzi | ICLR | 2024 |
| [2] | Corrective Retrieval Augmented Generation (CRAG) | Yan, Gu, Zhu, Ling | arXiv:2401.15884 | 2024 |
| [3] | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning | ICLR | 2024 |
| [4] | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | Edge, Trinh, Cheng, Bradley, et al. (Microsoft) | arXiv:2404.16130 | 2024 |
| [5] | LightRAG: Simple and Fast Retrieval-Augmented Generation | Guo et al. (U. Hong Kong) | arXiv:2410.05779 | 2024 |
| [6] | HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge | Huang et al. | EMNLP Findings | 2025 |
| [7] | VersionRAG: Version-Aware RAG for Evolving Documents | Huwiler et al. | arXiv | 2025 |
| [8] | T-GRAG: Temporal GraphRAG for Resolving Temporal Conflicts | — | arXiv | 2025 |
| [9] | TG-RAG: Temporal Graphs — Time-Sensitive Modeling and Retrieval | — | arXiv | 2025 |
| [10] | MA-RAG: Multi-Agent RAG via Collaborative Chain-of-Thought | Nguyen, Chin, Tai | arXiv:2505.20689 | 2025 |
| [11] | RAGBench: Explainable Benchmark for RAG Systems | — | arXiv:2407.11005 | 2024 |
| [12] | Lost in the Middle: How Language Models Use Long Contexts | Liu, Lin, Hewitt, Paranjape, et al. | TACL | 2024 |
| [13] | SCMRAG: Self-Corrective Multihop RAG for LLM Agents | Agrawal, Asrani, Youssef, Narayan | AAMAS | 2025 |
| [14] | SimRAG: Self-Improving RAG for Specialized Domains | Xu et al. | NAACL | 2025 |
| [15] | HGMem: Hypergraph-based Memory for Multi-step RAG | Zhou, Zhang, Yu, Meng, et al. | arXiv:2412.18794 | 2025 |
| [16] | M-RAG: Multimodal RAG for Visually Rich Knowledge in Architecture | — | ResearchGate | 2025 |
| [17] | UniDoc-Bench: Unified Benchmark for Document-Centric Multimodal RAG | — | arXiv | 2025 |
| [18] | LegalBench-RAG: Legal RAG Retrieval Benchmark | — | arXiv | 2024 |
| [19] | RetroLM: KV-Level Retrieval Augmentation for Long-Context Processing | — | arXiv:2502.11444 | 2025 |
| [20] | CAG: Enhancing Cache-Augmented Generation with Adaptive Contextual Compression | — | arXiv | 2025 |
| [21] | Agentic Retrieval-Augmented Generation: A Survey | — | arXiv | 2025 |
| [22] | RAGentA: Multi-Agent RAG for Attributed QA | — | ResearchGate | 2025 |

---

## Plan 01: Temporal/Versioned GraphRAG

### Current Literature

**VersionRAG** [7] is the most directly relevant paper. Published in October 2025, it models document evolution through a hierarchical graph capturing version sequences and change boundaries. On its VersionQA benchmark (100 questions, 34 documents), it achieved 90% accuracy vs. 58% for naive RAG and 64% for GraphRAG. However, its documents are **software changelogs and API docs**, not regulatory specifications.

**T-GRAG** [8] introduces temporal knowledge graphs with a Temporal Query Decomposition mechanism, but it is designed for **news articles** where timestamps change daily — not for engineering specs that change across decades.

**TG-RAG** [9] models a bi-level temporal graph with a hierarchical time abstraction. Its focus is on **Wikipedia-scale factual evolution**, not section-level structural diffs in legal/regulatory documents.

**Microsoft GraphRAG** [4] builds community summaries via Leiden clustering but has **no temporal awareness** — it treats all documents as a single static corpus.

### Critical Gap

> **No temporal RAG system has been applied to hierarchically structured government regulatory documents.** VersionRAG [7] comes closest but: (a) it uses software docs, not engineering specs; (b) it does not leverage existing document structure trees (your PageIndex); (c) it does not model section-level alignment across editions.

> Your unique advantage: you have **two editions of the same spec** with identical Division/Section numbering conventions, plus existing PageIndex trees for structural alignment.

---

## Plan 02: DualPath Hybrid RAG (PageIndex + GraphRAG)

### Current Literature

**RAPTOR** [3] (ICLR 2024) creates a tree of recursive summaries for hierarchical retrieval. It clusters text chunks and summarizes at each level, achieving 20% accuracy improvement on QuALITY. However, it **creates its own hierarchy from scratch** rather than leveraging the document's native structure (TOC, sections).

**HiRAG** [6] (EMNLP 2025 Findings) uses hierarchical knowledge to enhance RAG during indexing and retrieval. It showed gains in ROUGE and F1 over baselines. But it operates within **a single index type** — it does not fuse two fundamentally different retrieval paradigms (structure vs. semantics).

**LightRAG** [5] combines low-level (entity-specific) and high-level (thematic) retrieval through knowledge graph structures. It supports incremental updates and showed improvements in both granular and conceptual queries.  However, it still operates within **one unified graph index**, not across two independent retrieval systems.

**Microsoft GraphRAG** [4] offers Local Search (entity neighbors) and Global Search (community summaries), but both operate on the **same knowledge graph**. It does not combine graph search with a structural document tree.

### Critical Gap

> **No system fuses a structural document tree (like PageIndex) with an entity-relationship graph (like Neo4j GraphRAG).** RAPTOR [3] builds its own tree but ignores the document's native structure. HiRAG [6] uses hierarchical knowledge but within a single-index paradigm. LightRAG [5] has dual-level retrieval but both levels are graph-based.

> The risk: if one retrieval path (e.g., PageIndex) answers 90% of queries on its own, the fusion provides marginal benefit. This plan requires carefully designed queries where **both paths are needed** (cross-reference + structural location).

---

## Plan 03: Multimodal RAG for Engineering Specs

### Current Literature

**M-RAG** [16] is a 2025 framework for multimodal RAG in architecture. It handles floorplans, structural diagrams, and construction images through a shared semantic embedding space and dual-mode table decomposition. It reduced technical hallucinations by providing supporting diagrams. However, it focuses on **architectural design** (buildings), not on **regulatory specification tables** with gradation curves and material property values.

**UniDoc-Bench** [17] (arXiv 2025) is the closest benchmark, including "Construction" and "Government" as two of eight domains. It creates multimodal QA pairs from real PDF documents. But its construction domain covers **generic construction documents**, not the highly specific table-heavy format of state DOT specifications.

**Mistral OCR** (2025) and **NVIDIA NIM PDF Extraction Pipeline** provide state-of-the-art extraction of tables, figures, and equations from PDFs. These are **industrial tools**, not research contributions. Your contribution would need to focus on the **domain-specific evaluation** (how much information is lost in engineering tables), not on the extraction technique.

### Critical Gap

> **No study has quantified the information loss** caused by text-only RAG in engineering specification tables. UniDoc-Bench [17] touches construction but doesn't have WYDOT-style gradation tables. M-RAG [16] handles architectural diagrams but not regulatory spec tables with strict numerical values.

> The gap is specifically about **table-structure preservation for numerical engineering queries** (e.g., "What sieve sizes pass 100%?") — not about general multimodal extraction.

---

## Plan 04: TransportBench (Domain Benchmark)

### Current Literature

**RAGBench** [11] (arXiv 2024) is the largest RAG evaluation dataset (100K examples) with the TRACe framework measuring Utilization, Relevance, Adherence, and Completeness. However, it covers five **generic** industry domains (user manuals, tech support) — no transportation or construction.

**LegalBench-RAG** [18] (arXiv 2024) is a domain-specific benchmark for legal RAG, with thousands of human-annotated query-answer pairs over NDAs, M&A agreements, and contracts. It sets a precedent for **domain-specific RAG benchmarks** but is limited to the **legal sector**.

**CRAG (Comprehensive RAG Benchmark)** (Meta, 2024) tests RAG across diverse question types but uses **Wikipedia and web data**, not specialized government documents.

**UniDoc-Bench** [17] includes construction as a domain but is a **multimodal benchmark**, not a text-focused RAG benchmark for engineering QA.

### Critical Gap

> **There is no RAG benchmark for government transportation or construction engineering documents.** LegalBench-RAG [18] proves the value of domain-specific benchmarks (it was well-received by the NLP community). A "TransportBench" would be the first for this vertical.

> The risk: your corpus (2 PDFs) may be too small. Reviewers at EMNLP will expect 5–10+ documents. Expanding with additional WYDOT manuals and meeting minutes would strengthen the contribution.

---

## Plan 05: Agentic Compliance Monitor

### Current Literature

**MA-RAG** [10] (arXiv 2025) is a multi-agent framework with collaborative chain-of-thought reasoning, using Planner, Extractor, and QA agents. It outperforms standalone LLMs on multi-hop benchmarks. But it is designed for **ad-hoc question answering**, not continuous **proactive monitoring**.

**The Agentic RAG Survey** [21] (arXiv 2025) categorizes systems into single-agent, multi-agent, hierarchical, corrective, adaptive, and graph-based RAG. It identifies tool use, reflection, and planning as key principles. But it does not discuss **compliance monitoring** or **change-impact analysis** as an application.

**RAGentA** [22] (2025) is a multi-agent RAG for attributed QA with iterative filtering and in-line citations. It verifies answer completeness through dynamic refinement. But again, it is **reactive** (responds to queries) not **proactive** (detects changes autonomously).

### Critical Gap

> **No agentic RAG system operates in "proactive monitoring" mode** — detecting document changes and automatically assessing impact. All existing agentic systems [10, 21, 22] are **reactive**: they wait for a user query. A compliance monitor that autonomously triggers when a new spec edition is ingested would be novel.

> The risk: This is a **systems paper**, not a methods paper. It depends on Plan 01 (temporal diffs) being completed first. Publication venue should be **applied AI** (AAAI AI4Good, NeurIPS AI4Science), not methods-focused (NeurIPS main track).

---

## Plan 06: VerifyRAG (Formal Verification)

### Current Literature

**Self-RAG** [1] (ICLR 2024) teaches LLMs to generate "critique tokens" — `[Retrieval]`, `[IsRel]`, `[IsSup]`, `[IsUse]` — to self-assess retrieval quality. On knowledge-intensive benchmarks, it outperformed ChatGPT and standard RAG by 10–17% on citation accuracy.

**CRAG** [2] (arXiv 2024) evaluates retrieval quality with a "Decision Gate" that scores documents as Correct, Ambiguous, or Incorrect. If internal retrieval is poor, it triggers web search fallback. The approach is plug-and-play and compatible with Self-RAG.

**SCMRAG** [13] (AAMAS 2025) combines a self-corrective agent with a dynamic knowledge graph and external web search. It focuses on multi-hop factual questions, not engineering safety verification.

**SimRAG** [14] (NAACL 2025) uses self-training for domain adaptation but focuses on generating synthetic training data, not on formal verification of answers.

### Critical Gap

> Self-RAG [1] and CRAG [2] provide **probabilistic self-assessment** — the model "believes" its answer is correct. **No system provides formal verification** by checking the LLM's numerical claims against a structured constraint database.

> For engineering: "The minimum concrete cover is 2 inches" can be formally verified — not just self-assessed. The difference is between "the model thinks it's right" (Self-RAG) and "we can prove it matches the specification" (VerifyRAG).

> The risk: **Constraint extraction is an unsolved sub-problem.** LLMs extract structured constraints from natural language at ~80% accuracy. The remaining 20% (complex tables, conditional requirements) are the hard cases. You need Plan 03 (table extraction) to work first.

---

## Plan 07: Efficiency Frontier (CAG vs RAG vs Long Context)

### Current Literature

**"Lost in the Middle"** [12] (TACL 2024, Liu et al.) established the U-shaped performance curve: LLMs recall facts at the beginning and end of long contexts but miss facts in the middle. This has 700+ citations and is the most influential positional bias study.

**RetroLM** [19] (arXiv 2025) introduces KV-level retrieval augmentation for long-context processing, showing that RAG and long context are complementary, not competing.

**CAG papers** [20] (arXiv 2025) demonstrate that preloading documents into KV cache eliminates retrieval latency and simplifies architecture. A Hybrid CAG-RAG framework with Adaptive Contextual Compression was proposed for scalability.

**LongRAG** (2024) retrieves entire chapters (5,000+ words) instead of small chunks, giving the model more "breathing room" for reasoning while maintaining RAG's cost efficiency.

### Critical Gap

> **No study has compared RAG vs. CAG vs. Long Context on real engineering documents.** All existing comparisons [12, 19, 20] use Wikipedia, news, or generic QA datasets. Engineering specs have a unique structure (hierarchical, table-heavy, cross-referenced) that may produce different performance patterns than the "Lost in the Middle" [12] U-curve found in generic text.

> Key hypothesis: **positional bias may align with document structure** (Division numbers), not just absolute position. If Division 400 content is always "lost" because it's in the physical middle, this is a novel finding about document-structured positional bias.

> The risk: API cost. Running 300+ Long Context calls at 1M tokens is **$60–150**. And the "obvious result" — RAG is cheaper, Long Context is more accurate — needs a surprising finding to be publishable.

---

## Plan 08: RecurseRAG (Provenance Chains)

### Current Literature

**Self-RAG** [1] evaluates retrieval quality but does not **follow** cross-references. If the retrieved chunk says "see AASHTO M-147," Self-RAG would either ignore it or hallucinate the AASHTO content.

**SCMRAG** [13] (AAMAS 2025) uses a self-corrective agent with a dynamic knowledge graph to identify and retrieve missing information. This is the closest to recursive retrieval, but it **goes to the web** for missing info rather than following structured cross-references within a document corpus.

**MA-RAG** [10] breaks queries into sub-tasks with specialized agents, but the sub-tasks are defined by the **query**, not by cross-references found **during retrieval**. There is no "follow this reference" agent.

**Citation verification research** (ACL Anthology, 2025) has explored sentence-level fine-grained attribution for verifiability in long-form QA. But this focuses on **validating existing citations**, not on **following cross-references to retrieve additional context.**

### Critical Gap

> **No RAG system automatically detects and follows intra-document cross-references** (Section A → Section B → Table C) and records the traversal as a formal provenance chain. Self-RAG [1] self-critiques but doesn't follow references. SCMRAG [13] goes external. MA-RAG [10] plans sub-queries but doesn't detect implicit references.

> The novel contribution: a **Provenance DAG** where every claim in the final answer has a traceable chain of retrieval steps, each annotated with source, method, and confidence.

> The risk: **AASHTO copyright.** Most cross-references in WYDOT specs point to AASHTO standards, which are copyrighted and not in your corpus. You must scope to **intra-WYDOT cross-references** only.

---

## Plan 09: HyperSpec (Hypergraph Memory)

### Current Literature

**HGMem** [15] (arXiv 2025, Zhou et al.) is the direct precursor. It represents memory as a hypergraph where hyperedges correspond to memory units. It supports INSERT, UPDATE, and MERGE operations, enabling progressive formation of higher-order interactions. It outperformed existing multi-step RAG baselines and enabled open-weight models to match GPT-4o performance.

**mem0** (2024) is a practical cross-session memory system using key-value storage, but it has **no higher-order relationship modeling** — each memory is independent.

**Chainlit chat history** stores message logs but provides **no semantic structure** — it's a flat list of past messages, not a knowledge graph.

### Critical Gap

> HGMem [15] operates **within a single session** (multi-step reasoning). **No system extends hypergraph memory to cross-session, domain-specific QA.** The gap is: can higher-order memory relationships from Session 1 (concrete specs) enhance answers in Session 5 (bridge deck design)?

> The risk: this is the **most speculative** plan. Engineers typically ask isolated questions, not multi-session research sequences. The 10-session benchmark would be **synthetic**, which weakens the evaluation. HGMem [15] is also very recent (Dec 2025), so your "extension" might feel **incremental** rather than novel.

---

## Summary: Gap Strength by Plan

| Plan | Gap Clarity | Gap Uniqueness | Your Data Advantage | Overall Gap Strength |
|------|:-----------:|:--------------:|:-------------------:|:--------------------:|
| 01 Temporal | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟢 **Strongest** |
| 02 DualPath | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🟡 |
| 03 Multimodal | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🟡 |
| 04 Benchmark | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟢 **Strongest** |
| 05 Compliance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🟢 |
| 06 VerifyRAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 🟡 (hard execution) |
| 07 Efficiency | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🟡 |
| 08 RecurseRAG | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (copyright) | 🟡 |
| 09 HyperSpec | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 🔴 **Weakest** |

---

## Recommended Publication Strategy

Based on gap strength and feasibility:

1. **Immediate** (high confidence): **Plan 01 + Plan 04 combined** → *"VersionSpec: Temporal RAG with TransportBench Evaluation for Evolving Government Specifications"*
   - Combines the strongest gap (temporal) with the strongest dataset contribution (benchmark)
   - Target: **EMNLP 2026 Main + Resource Track**

2. **High-risk, high-reward**: **Plan 06 (VerifyRAG)** → if constraint extraction achieves >85% accuracy
   - Target: **NeurIPS 2026**

3. **Timely but expensive**: **Plan 07 (Efficiency Frontier)** → if budget allows $100+
   - Target: **ICML 2026**
