# 🎯 FINAL RESEARCH PLAN: SpecRAG — Three-Dimensional Retrieval-Augmented Generation for Domain-Critical Engineering Specifications

> **Target:** Top-tier CS venue (ACL / EMNLP / NeurIPS / SIGIR 2026)
> **Author(s):** [Your Name]
> **Date:** March 6, 2026
> **Status:** READY FOR EXECUTION

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Critical Gap Analysis: What the Literature Misses](#2-critical-gap-analysis)
3. [Analysis of Existing Plans (01–09): Why They Fall Short](#3-analysis-of-existing-plans)
4. [The Novel Contribution: SpecRAG](#4-the-novel-contribution-specrag)
5. [Paper Architecture & Writing Plan](#5-paper-architecture)
6. [Implementation Roadmap (6 Weeks)](#6-implementation-roadmap)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Risk Mitigation](#8-risk-mitigation)
9. [Publication Strategy](#9-publication-strategy)
10. [Appendix: Paper-by-Paper Gap Matrix](#10-appendix)

---

## 1. EXECUTIVE SUMMARY

### The One-Line Pitch
> **SpecRAG** is the first retrieval framework that unifies three orthogonal retrieval dimensions — **structural hierarchy** (document tree navigation), **semantic entity graphs** (knowledge graph traversal), and **temporal version evolution** (cross-edition change tracking) — into a single learned routing system for domain-critical engineering specifications.

### Why This Wins

After reading all 12 research papers in this folder and analyzing 9 existing research plans, the BREAKTHROUGH_ANALYSIS, CRITICAL_ANALYSIS, LITERATURE_REVIEW, and the full project codebase, I conclude:

1. **No published paper operates in 3 retrieval dimensions simultaneously.** GraphRAG (survey, 2025) catalogs hundreds of systems — all operate in 1D (semantic) or 2D (semantic + graph). VersionRAG (2025) adds temporal but drops structure. HM-RAG (2025) adds multi-source but all sources are semantic. RouteRAG (2025) learns to route between text and graph — but only 2 modalities. Nobody combines structural hierarchy + semantic graph + temporal evolution.

2. **No published paper has a comparable infrastructure moat.** Your WYDOT system already has: Neo4j knowledge graph with 9 entity types + SUPERSEDES edges, PageIndex structural trees, Gemini embeddings, FlashRank reranking, multi-agent orchestration, audio/image multimodal input, evaluation pipeline, and Cloud Run deployment. This is 6+ months of engineering that cannot be replicated in a review cycle.

3. **The domain (government construction specifications) is untouched.** All 12 papers operate on: Wikipedia (RouteRAG, TimeRAG), SEC filings (PageIndex paper), scientific collections (CollEX), generic benchmarks (HM-RAG on ScienceQA), or videos (VideoRAG, Multi-RAG). Zero papers address hierarchically-structured, version-evolving, safety-critical engineering specifications.

4. **Existing plans 01–09 are individually incremental.** The CRITICAL_ANALYSIS correctly diagnoses this. Each plan extends one dimension. SpecRAG subsumes Plans 01, 02, and 04 into a unified framework with stronger novelty claims.

---

## 2. CRITICAL GAP ANALYSIS: WHAT THE LITERATURE MISSES

### 2.1 Paper-by-Paper Gaps Relevant to Your Project

| Paper | What It Does | What It CANNOT Do | Gap You Fill |
|-------|-------------|-------------------|-------------|
| **GraphRAG Survey** (Han et al., 2025) | Catalogs 10 domains of GraphRAG | Does not address temporal or structural retrieval dimensions | SpecRAG adds structural + temporal dimensions |
| **VersionRAG** (Huwiler et al., 2025) | Version-aware graph for software docs | No structural hierarchy; only 34 docs; domain-limited | SpecRAG applies temporal tracking to hierarchical engineering specs with 1500+ docs |
| **TimeRAG** (Wang et al., 2025) | Temporal reasoning via query decomposition | Relies on web search; no document structure or versioning | SpecRAG handles temporal reasoning within a structured document corpus |
| **PageIndex** (Lumer et al., 2025) | Hierarchical tree navigation for SEC filings | No entity awareness; vector-based still wins 68% | SpecRAG fuses tree navigation with entity graph traversal — neither alone suffices |
| **HM-RAG** (Liu et al., 2025) | Multi-agent multimodal RAG | All 3 retrieval sources are semantic (vector/graph/web); no structural or temporal dimension | SpecRAG routes across fundamentally different retrieval paradigms, not just different semantic sources |
| **MMGraphRAG** (Wan & Yu, 2025) | Multimodal knowledge graph with scene graphs | Text+image only; no temporal versioning; no document structure | SpecRAG could extend to multimodal but starts with the harder 3D retrieval problem |
| **RouteRAG** (Guo et al., 2025) | RL-based routing between text and graph retrieval | Only 2 retrieval modes; Wikipedia-only; no domain specialization | SpecRAG has 3 retrieval modes + domain-specific query classifier + engineered constraint verification |
| **CollEX** (Schneider et al., 2025) | Agentic RAG for scientific collections | No formal evaluation; no temporal/structural awareness | SpecRAG provides rigorous evaluation + 3D retrieval |
| **VideoRAG** (Ren et al., 2025) | Graph-based video RAG | Video-only; no document structure; no version tracking | Different modality entirely — but validates graph-based retrieval for cross-document reasoning |
| **Multi-RAG** (Mao et al., 2025) | Multimodal video RAG via text conversion | Single primary modality; no graph structure | Different domain — but validates multi-source fusion |
| **Multimodal Route** (Ajirak et al., 2025) | Per-sample routing for clinical prediction | Not RAG; clinical domain only | Routing concept is transferable — SpecRAG adapts learned routing to retrieval dimensions |
| **RAG Review** (Oche et al., 2025) | Comprehensive RAG survey 2017–2025 | Identifies but does not solve: domain-specific RAG, temporal reasoning, structural retrieval | SpecRAG is the system the review calls for |

### 2.2 The Three Gaps Nobody Has Filled

**Gap 1: No Multi-Dimensional Retrieval Framework**
- Every existing system retrieves in 1 dimension (semantic similarity) or at most 2 (semantic + graph entity traversal)
- RouteRAG (closest competitor) routes between text retrieval and graph retrieval — but both are semantic
- Nobody routes between *fundamentally different retrieval paradigms*: tree traversal vs. graph traversal vs. vector search
- **SpecRAG fills this gap** with a 3D retrieval space: `R(q) = α·Structural(q) + β·Semantic(q) + γ·Temporal(q)` with learned weights

**Gap 2: No RAG System for Version-Evolving Hierarchical Documents**
- VersionRAG handles versions but ignores document hierarchy (Division → Section → Subsection)
- PageIndex handles hierarchy but ignores version evolution and entity relationships
- GraphRAG handles entities but ignores both hierarchy and versions
- **SpecRAG fills this gap** because WYDOT specs are *simultaneously* hierarchical (Division/Section structure), version-evolving (2010 → 2021 with SUPERSEDES links), and entity-rich (Materials, Standards, TestMethods)

**Gap 3: No Domain-Specific RAG Benchmark for Transportation/Construction Engineering**
- The RAG Review (2025) explicitly identifies the gap between academic benchmarks (NaturalQuestions, TriviaQA) and real-world domain needs
- FinanceBench exists for finance, LegalBench for law — nothing for transportation engineering
- **SpecQA benchmark fills this gap** with 100 queries requiring 1, 2, or 3 retrieval dimensions

---

## 3. ANALYSIS OF EXISTING PLANS (01–09): WHY THEY FALL SHORT INDIVIDUALLY

### Verdict: Each Plan Is a Feature, Not a Paper

| Plan | Core Idea | Standalone Paper Strength | Problem |
|------|----------|--------------------------|---------|
| **01: Temporal VersionSpec** | Section-level temporal KG | ⭐⭐⭐ Medium | Incremental over VersionRAG; different domain is not enough |
| **02: DualPath RAG** | Fuse PageIndex + GraphRAG | ⭐⭐ Weak | Risk of null result; "combining two things" is not novel without theory |
| **03: Multimodal Spec RAG** | Tables + figures extraction | ⭐⭐ Weak | Engineering contribution, not research; PyMuPDF table quality is ~70% |
| **04: TransportBench** | Domain benchmark | ⭐⭐⭐ Medium | Resource paper only; needs a system paper to accompany it |
| **05: ComplianceAgent** | Proactive change monitoring | ⭐⭐⭐ Medium | Depends on Plan 01; multi-agent orchestration is over-engineered |
| **06: VerifyRAG** | Formal verification of RAG | ⭐⭐⭐⭐ Strong | Excellent novelty but 9-week timeline and constraint extraction bottleneck |
| **07: Efficiency Frontier** | CAG vs RAG vs Long Context | ⭐⭐⭐ Medium | Most publishable but $100-200 cost; risk of "obvious result" |
| **08: RecurseRAG** | Cross-reference provenance | ⭐⭐⭐ Medium | AASHTO copyright kills external refs; 10-week timeline |
| **09: HyperSpec** | Hypergraph cross-session memory | ⭐⭐ Weak | 12-month project; speculative; lowest feasibility |

### The Winning Strategy: Subsume Plans 01, 02, and 04 into SpecRAG

SpecRAG is not a new plan — it is the *unification* that makes Plans 01, 02, and 04 into a single paper with triple the novelty:

- **Plan 01's temporal graph** becomes SpecRAG's temporal dimension (γ)
- **Plan 02's DualPath fusion** becomes SpecRAG's structural+semantic routing (α + β)
- **Plan 04's TransportBench** becomes SpecQA — the evaluation benchmark proving SpecRAG works

This is exactly what the BREAKTHROUGH_ANALYSIS recommended. I am operationalizing it.

---

## 4. THE NOVEL CONTRIBUTION: SpecRAG

### 4.1 Formal Definition

**SpecRAG** formalizes retrieval over domain-critical specifications as navigation through a 3-dimensional retrieval space:

```
R(q) = α(q) · Structural(q) + β(q) · Semantic(q) + γ(q) · Temporal(q)
```

Where:
- **Structural(q)**: PageIndex tree traversal — an LLM navigates the document's Division/Section hierarchy to locate relevant page ranges. Zero embeddings. Structure-based reasoning.
- **Semantic(q)**: Neo4j GraphRAG — entity-linked vector search with knowledge graph traversal across 9 entity types (Material, Standard, TestMethod, etc.) with FlashRank reranking.
- **Temporal(q)**: Version-aware graph traversal — follows SUPERSEDES edges between 2010 and 2021 specification editions, retrieves change diffs and temporal context.
- **α(q), β(q), γ(q)**: Learned query-dependent weights from a query-type classifier.

### 4.2 Query-Type Classifier

A lightweight classifier (fine-tuned on 100 annotated queries) determines the optimal dimensional subspace:

| Query Type | Example | Optimal Dimension(s) |
|-----------|---------|---------------------|
| **Structural** | "What does Division 500 cover?" | α dominant |
| **Entity-Semantic** | "What is the minimum compressive strength for Class A concrete?" | β dominant |
| **Temporal** | "How did aggregate specs change between 2010 and 2021?" | γ dominant |
| **Structural + Semantic** | "Which section specifies Type IL cement requirements and what are they?" | α + β |
| **Semantic + Temporal** | "Were any new materials added in the 2021 edition?" | β + γ |
| **Structural + Temporal** | "Was Division 800 reorganized in the 2021 update?" | α + γ |
| **Full 3D** | "How did the concrete mix design requirements in Section 501 change, and what new test standards were referenced?" | α + β + γ |

### 4.3 Architecture Diagram

```
                          ┌─────────────────────┐
                          │   User Query (q)     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Query-Type Classifier│
                          │  (α, β, γ weights)    │
                          └──┬───────┬────────┬──┘
                             │       │        │
                    ┌────────▼──┐ ┌──▼─────┐ ┌▼────────┐
                    │Structural │ │Semantic │ │Temporal  │
                    │  Retrieval│ │Retrieval│ │Retrieval │
                    │(PageIndex)│ │(GraphRAG│ │(Version  │
                    │           │ │+ Neo4j) │ │ Graph)   │
                    └────────┬──┘ └──┬─────┘ └┬────────┘
                             │       │        │
                          ┌──▼───────▼────────▼──┐
                          │   3D Context Fusion   │
                          │  (Merge + Dedup +     │
                          │   Provenance Tags)    │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Grounded Generation  │
                          │  (Gemini 2.5 Flash)   │
                          │  + Citation Injection  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Answer with Sources, │
                          │  Version Tags, and    │
                          │  Structural Path      │
                          └─────────────────────┘
```

### 4.4 What Makes This Novel (vs. All 12 Papers)

| Dimension | Closest Competitor | Why SpecRAG Wins |
|-----------|-------------------|-----------------|
| 3D Retrieval | RouteRAG (2D: text + graph) | SpecRAG adds structural hierarchy as a *fundamentally different* retrieval paradigm, not just another semantic source |
| Temporal Versioning | VersionRAG (versions for software docs) | SpecRAG applies versioning to *hierarchically structured* specs with SUPERSEDES edges already in Neo4j |
| Structural Hierarchy | PageIndex paper (hierarchy for SEC filings) | SpecRAG fuses hierarchy with entity graph and temporal tracking; PageIndex alone loses 32% of queries |
| Domain Specificity | None in transportation | First RAG system for government construction specs; 52 US state DOTs as potential adopters |
| Learned Routing | RouteRAG (RL-based) | SpecRAG uses query-type classification with domain-specific categories, not generic RL |
| Multi-Agent | HM-RAG (3 parallel agents) | SpecRAG's agents map to retrieval *dimensions*, not retrieval *sources* |
| Benchmark | None for transportation | SpecQA: 100 queries tagged by required retrieval dimensions |

### 4.5 Five Key Claims for the Paper

1. **Claim 1 (Framework):** We formalize multi-dimensional retrieval as a 3D space — the first framework to unify structural, semantic, and temporal retrieval with learned query-dependent routing.

2. **Claim 2 (System):** SpecRAG, instantiated on WYDOT specifications (1,500+ documents, 2 editions), demonstrates that 3D retrieval outperforms any single dimension or 2D combination.

3. **Claim 3 (Benchmark):** SpecQA, a 100-query benchmark with dimensional annotations, enables reproducible evaluation of multi-dimensional retrieval.

4. **Claim 4 (Empirical):** On SpecQA, SpecRAG achieves X% accuracy, outperforming GraphRAG-only (β-only) by Y%, PageIndex-only (α-only) by Z%, and a 2D fusion by W%.

5. **Claim 5 (Generalizability):** The 3D retrieval framework generalizes to any domain with hierarchically structured, version-evolving, entity-rich documents (legal codes, medical guidelines, building codes, military specifications).

---

## 5. PAPER ARCHITECTURE & WRITING PLAN

### 5.1 Target: ACL 2026 Main Conference or EMNLP 2026

- **ACL 2026:** Submission deadline typically January; if passed, use ARR (rolling)
- **EMNLP 2026:** Submission deadline typically June 2026
- **Backup:** SIGIR 2026 (resource track), NeurIPS 2026 (datasets & benchmarks)

### 5.2 Paper Structure (8 pages + references)

```
Title: SpecRAG: Three-Dimensional Retrieval-Augmented Generation
       for Domain-Critical Engineering Specifications

Abstract (250 words)
  - Problem: RAG operates in 1D (semantic); domain-critical specs need 3D
  - Solution: SpecRAG unifies structural, semantic, temporal retrieval
  - Results: X% improvement over strongest baseline on SpecQA

1. Introduction (1.5 pages)
   1.1 Motivation: Safety-critical engineering specs are simultaneously
       hierarchical, entity-rich, and version-evolving
   1.2 Limitations of existing approaches (1D retrieval)
   1.3 Our contribution: 3D retrieval framework + SpecQA benchmark
   1.4 Results preview

2. Related Work (1 page)
   2.1 RAG Evolution: Naive → Self-RAG → Agentic → Graph
   2.2 Structural Retrieval: RAPTOR, HiRAG, PageIndex
   2.3 Version-Aware Retrieval: VersionRAG, TimeRAG
   2.4 Graph-Based RAG: GraphRAG survey, RouteRAG, HM-RAG
   2.5 Domain-Specific RAG: Legal, Medical, Financial → Gap: Engineering

3. The SpecRAG Framework (2 pages)
   3.1 Problem Formulation: 3D Retrieval Space
   3.2 Dimension 1: Structural Retrieval (PageIndex)
   3.3 Dimension 2: Semantic Retrieval (GraphRAG + Neo4j)
   3.4 Dimension 3: Temporal Retrieval (Version Graph)
   3.5 Query-Type Classifier & Dimensional Routing
   3.6 3D Context Fusion & Grounded Generation

4. SpecQA Benchmark (0.75 pages)
   4.1 Corpus Description (WYDOT 2010, 2021, 1500+ docs)
   4.2 Query Taxonomy (7 types × 3 dimensions)
   4.3 Annotation Protocol (2 annotators, inter-rater agreement)
   4.4 Metrics: Accuracy, Citation Precision, Faithfulness,
       Dimensional Coverage

5. Experiments (1.5 pages)
   5.1 Baselines:
       - Naive RAG (vector-only)
       - GraphRAG-only (β dimension)
       - PageIndex-only (α dimension)
       - VersionRAG-adapted (γ dimension)
       - 2D combinations (α+β, β+γ, α+γ)
       - SpecRAG-full (α+β+γ)
   5.2 Main Results (Table 1: Accuracy by query type)
   5.3 Ablation Study (Table 2: Contribution of each dimension)
   5.4 Routing Analysis (Figure 2: Learned α,β,γ distributions)
   5.5 Efficiency Analysis (Table 3: Latency, cost, tokens)

6. Analysis & Discussion (0.75 pages)
   6.1 When Does 3D Help Most? (3D queries >> 1D queries)
   6.2 Error Analysis: Where SpecRAG Still Fails
   6.3 Generalizability to Other Domains

7. Conclusion (0.5 pages)

Appendix:
   A. SpecQA Query Examples
   B. Prompt Templates
   C. Neo4j Schema
   D. Reproducibility Checklist
```

### 5.3 Draft Abstract

> Retrieval-Augmented Generation (RAG) systems overwhelmingly operate in a single retrieval dimension — semantic similarity — which is insufficient for domain-critical documents that are simultaneously hierarchically structured, entity-rich, and version-evolving. We present **SpecRAG**, the first retrieval framework that unifies three orthogonal retrieval dimensions: (1) *structural retrieval* via hierarchical document tree navigation, (2) *semantic retrieval* via entity-linked knowledge graph traversal, and (3) *temporal retrieval* via cross-edition version tracking. A learned query-type classifier dynamically routes each query to its optimal dimensional subspace. We instantiate SpecRAG on government construction specifications from the Wyoming Department of Transportation (WYDOT), comprising 1,500+ documents across two specification editions, with a Neo4j knowledge graph containing 9 entity types and explicit version-evolution edges. We introduce **SpecQA**, a 100-query benchmark annotated with required retrieval dimensions, enabling principled evaluation of multi-dimensional retrieval. On SpecQA, SpecRAG achieves [X]% accuracy, outperforming the strongest single-dimension baseline by [Y]% and the best two-dimensional combination by [Z]%, with the largest gains on queries requiring all three dimensions. Our framework generalizes to any domain with hierarchically structured, version-evolving, entity-rich documents — including legal codes, medical guidelines, and military specifications.

---

## 6. IMPLEMENTATION ROADMAP (6 WEEKS)

### What Already Exists (Completed Infrastructure)

| Component | Status | Location |
|-----------|--------|----------|
| Neo4j Knowledge Graph (V4) | ✅ Production | Neo4j Aura |
| 9 entity types + SUPERSEDES edges | ✅ Working | `ingestneo4j_updated.py` |
| PageIndex structural trees | ✅ Working | `pageindex/` |
| Gemini embeddings (768d) | ✅ Working | `chatapp.py` |
| FlashRank reranking | ✅ Working | `chatapp.py` |
| Multi-agent orchestration | ✅ Working | `wydot_agents/` |
| Audio multimodal input | ✅ Working | `chatapp.py` |
| Online/Offline evaluation | ✅ Working | `evaluation/`, `utils/evaluation.py` |
| Cloud Run deployment | ✅ Working | `Dockerfile`, `cloudbuild.yaml` |

### What Needs to Be Built (6-Week Sprint)

#### Week 1: Temporal Dimension (γ)
- [ ] Build section-level alignment between 2010 and 2021 specs using existing PageIndex trees
- [ ] Generate LLM-powered diffs for ~140 aligned section pairs using Gemini 2.5 Flash
- [ ] Store diffs as `CHANGE` nodes in Neo4j with `MODIFIED_IN`, `ADDED_IN`, `REMOVED_IN` edges
- [ ] Build temporal query engine: given a temporal query, traverse version graph to find relevant changes
- **Estimated cost:** $5-10 (Gemini API)
- **Dependencies:** PageIndex trees for both 2010 and 2021 (already exist)

#### Week 2: Query-Type Classifier + 3D Routing
- [ ] Create 100 annotated queries (SpecQA v1) with dimensional tags
- [ ] Train/prompt-engineer a query classifier (Gemini 2.5 Flash few-shot):
  - Input: user query
  - Output: `{type: "structural"|"semantic"|"temporal"|"struct+sem"|"sem+temp"|"struct+temp"|"full3d", α: float, β: float, γ: float}`
- [ ] Build routing logic: based on classifier output, invoke appropriate retrieval dimension(s)
- [ ] Implement 3D context fusion: merge contexts from all invoked dimensions with provenance tags
- **Estimated cost:** $10-15 (API calls for classification + testing)

#### Week 3: SpecQA Benchmark Construction
- [ ] Expand query set to 100 queries across 7 types (see Section 4.2)
- [ ] Write gold-standard answers with exact source passages and page numbers
- [ ] Compute inter-annotator agreement (recruit 1 additional annotator — domain expert if possible)
- [ ] Create evaluation harness: automated scoring with LLM-as-judge + keyword overlap + citation precision
- **Estimated cost:** ~20 hours annotation time
- [ ] Distribution: 15 structural, 15 semantic, 10 temporal, 15 struct+sem, 15 sem+temp, 15 struct+temp, 15 full-3D

#### Week 4: Baselines + Experiments
- [ ] Run all 7 baselines on SpecQA:
  1. Naive RAG (vector-only, no graph)
  2. GraphRAG-only (β: Neo4j + entities)
  3. PageIndex-only (α: tree navigation)
  4. Temporal-only (γ: version graph)
  5. α + β (Structural + Semantic)
  6. β + γ (Semantic + Temporal)
  7. α + γ (Structural + Temporal)
- [ ] Run SpecRAG-full (α + β + γ) on SpecQA
- [ ] Record: accuracy, citation precision, faithfulness, latency, token count, cost
- [ ] 3 runs per query, report mean + std
- **Estimated cost:** $30-50 (API calls for all baselines + SpecRAG)

#### Week 5: Analysis + Paper Writing
- [ ] Generate all tables and figures:
  - Table 1: Main results by query type
  - Table 2: Ablation (contribution of each dimension)
  - Table 3: Efficiency comparison
  - Figure 1: Architecture diagram (clean LaTeX version)
  - Figure 2: Learned α,β,γ weight distributions per query type
  - Figure 3: Error analysis breakdown
- [ ] Write Sections 1-4 (Introduction, Related Work, Framework, Benchmark)
- [ ] Write Section 5 (Experiments) — results-driven

#### Week 6: Paper Polishing + Submission
- [ ] Write Sections 6-7 (Analysis, Conclusion)
- [ ] Write Abstract (final version with actual numbers)
- [ ] Internal review and revision
- [ ] Format for target venue (ACL/EMNLP style)
- [ ] Prepare supplementary materials (code, benchmark, prompts)
- [ ] Submit to ARR or target venue

---

## 7. EVALUATION STRATEGY

### 7.1 Primary Metrics

| Metric | Definition | How Measured |
|--------|-----------|--------------|
| **Answer Accuracy** | Does the answer correctly answer the question? | LLM-as-judge (Gemini 2.5 Pro) + 2 human annotators |
| **Citation Precision** | Are cited sources actually relevant? | Automated: check if `[SOURCE_X]` text appears in answer context |
| **Citation Recall** | Are all relevant sources cited? | Manual: compare against gold-standard source list |
| **Faithfulness** | Is the answer grounded in retrieved context? | LLM-as-judge: does answer contain claims not in context? |
| **Dimensional Coverage** | Did the system use the correct retrieval dimension(s)? | Compare classifier output to gold-standard dimensional tags |
| **Latency** | End-to-end response time | Wall clock (3 runs average) |
| **Token Efficiency** | Total tokens consumed (input + output) | API token counts |

### 7.2 Baselines

| Baseline | Description | Why Included |
|----------|------------|-------------|
| Naive RAG | Top-10 vector search → LLM | Standard baseline |
| GraphRAG (β-only) | Neo4j entity search + graph traversal → LLM | Tests semantic dimension alone |
| PageIndex (α-only) | Tree navigation → page extraction → LLM | Tests structural dimension alone |
| VersionRAG-adapted (γ-only) | Version graph traversal → LLM | Tests temporal dimension alone |
| α+β | PageIndex + GraphRAG fusion | Tests 2D without temporal |
| β+γ | GraphRAG + Version graph | Tests 2D without structure |
| α+γ | PageIndex + Version graph | Tests 2D without semantics |
| **SpecRAG (α+β+γ)** | Full 3D retrieval with learned routing | Our system |

### 7.3 Expected Results Pattern

Based on analysis of existing infrastructure and paper results:

- **1D baselines:** 55-70% accuracy (Naive RAG lowest; GraphRAG highest among 1D)
- **2D combinations:** 70-80% accuracy (α+β strongest because most queries are structural+semantic)
- **SpecRAG (3D):** 82-90% accuracy (largest gain on temporal and full-3D queries)
- **Key insight:** For queries requiring 1 dimension, SpecRAG matches the best 1D baseline. The gain comes from queries requiring 2-3 dimensions, where 1D systems catastrophically fail.

### 7.4 Statistical Rigor

- 3 runs per query (temperature = 0.0 for determinism, but 3 runs to verify)
- Paired bootstrap test (p < 0.05) for main comparisons
- Cohen's kappa for inter-annotator agreement on SpecQA
- Report confidence intervals on all metrics

---

## 8. RISK MITIGATION

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Section alignment between 2010/2021 fails for edge cases | Medium | Low | Focus on the ~120/140 sections that align cleanly; report coverage |
| Query classifier accuracy < 80% | Medium | High | Use few-shot prompting with Gemini 2.5 Flash; if < 80%, fall back to oracle classifier and note as limitation |
| 3D fusion adds marginal improvement over 2D | Low | Critical | If α+β already achieves 85%+, emphasize temporal queries where 3D is essential; worst case, paper becomes a negative result paper (still publishable at EMNLP Findings) |
| Latency too high for production | Low | Low | Report latency but note this is a research contribution; optimize later |
| PageIndex trees missing for some documents | Low | Low | Focus benchmark on documents with complete trees |

### 8.2 Research Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Reviewers say "just engineering, not research" | Medium | High | Emphasize the *formal 3D retrieval framework* and *learned routing*; ablation studies prove each dimension contributes |
| Reviewers say "domain too narrow" | Medium | Medium | Include generalizability discussion (legal codes, medical guidelines, building codes); framework is domain-agnostic |
| Similar work published during review | Low | High | The infrastructure moat (Neo4j + PageIndex + SUPERSEDES + 1500 docs) cannot be replicated quickly; if similar framework appears, differentiate on domain specialization |
| SpecQA benchmark too small (100 queries) | Medium | Medium | 100 queries with dimensional annotations is comparable to VersionQA (100 queries), FinanceBench (150), DocBench (229); emphasize annotation quality over quantity |

---

## 9. PUBLICATION STRATEGY

### 9.1 Primary Target: EMNLP 2026 (Main Conference)

- **Why:** EMNLP accepts systems papers with strong empirical results; the SpecQA benchmark is a resource contribution that EMNLP values; the domain-specific angle aligns with EMNLP's Industry Track if main track is too competitive.
- **Submission deadline:** ~June 2026 (6 weeks to implement + 4 weeks to write = 10 weeks → target mid-May start for June submission)
- **OR use ACL Rolling Review (ARR):** Submit anytime, reviews cycle monthly

### 9.2 Backup Targets

| Venue | Track | Why | Deadline |
|-------|-------|-----|----------|
| ACL 2026 (ARR) | Main / Industry | Top NLP venue; industry track values deployed systems | Rolling |
| SIGIR 2026 | Resource Track | SpecQA benchmark is a natural fit | ~February 2026 (may be passed) |
| NeurIPS 2026 | Datasets & Benchmarks | SpecQA as a standalone benchmark submission | ~May 2026 |
| AAAI 2027 | AI for Social Good | Government/transportation angle; practical impact | ~August 2026 |
| CIKM 2026 | Applied Track | Information retrieval + knowledge management | ~May 2026 |

### 9.3 Dual Submission Strategy

If the main SpecRAG paper is a full system paper, consider splitting:

1. **Paper A (System + Framework):** "SpecRAG: Three-Dimensional RAG for Engineering Specifications" → ACL/EMNLP main
2. **Paper B (Benchmark):** "SpecQA: A Multi-Dimensional Retrieval Benchmark for Transportation Engineering" → NeurIPS Datasets & Benchmarks or EMNLP Resource Track

This doubles publication output from the same work.

### 9.4 Follow-Up Papers (After SpecRAG)

Once SpecRAG is published, the remaining plans become strong follow-ups:

| Follow-Up | Built On | Target |
|-----------|----------|--------|
| **VerifyRAG** (Plan 06) | SpecRAG + constraint database | NeurIPS 2027 |
| **ComplianceAgent** (Plan 05) | SpecRAG + temporal dimension | AAAI 2027 |
| **Efficiency Frontier** (Plan 07) | SpecRAG vs. CAG vs. Long Context | ICML 2027 |
| **RecurseRAG** (Plan 08) | SpecRAG + provenance chains | ACL 2027 |

---

## 10. APPENDIX: PAPER-BY-PAPER GAP MATRIX

### How Each Paper Maps to SpecRAG's Dimensions

| Paper | Structural (α) | Semantic (β) | Temporal (γ) | Domain-Specific | Learned Routing | Benchmark |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| GraphRAG Survey | ❌ | ✅ | ❌ | Partial (10 domains cataloged) | ❌ | ❌ |
| VersionRAG | ❌ | ✅ | ✅ | Software docs only | ❌ | ✅ (100 queries) |
| TimeRAG | ❌ | ❌ | ✅ | News/factual only | ❌ | ❌ |
| PageIndex | ✅ | ❌ | ❌ | Finance (SEC) | ❌ | ✅ (150 queries) |
| HM-RAG | ❌ | ✅ | ❌ | General (ScienceQA) | ❌ | ❌ |
| MMGraphRAG | ❌ | ✅ | ❌ | Multi-domain | ❌ | ✅ (CMEL) |
| RouteRAG | ❌ | ✅ (2D) | ❌ | Wikipedia | ✅ (RL) | ❌ |
| CollEX | ❌ | ✅ | ❌ | Scientific collections | ❌ | ❌ |
| VideoRAG | ❌ | ✅ | ❌ | Videos | ❌ | ✅ (LongerVideos) |
| Multi-RAG | ❌ | ✅ | ❌ | Videos | ❌ | ❌ |
| Multimodal Route | ❌ | N/A | ❌ | Clinical | ✅ | ❌ |
| RAG Review | Survey | Survey | Survey | Survey | Survey | Survey |
| **SpecRAG (Ours)** | **✅** | **✅** | **✅** | **✅ (Transportation)** | **✅** | **✅ (SpecQA)** |

### Key Takeaway
SpecRAG is the **only** system that checks all six boxes. No existing paper even checks four.

---

## FINAL VERDICT

### What to Do RIGHT NOW (Priority Order)

1. **Week 1:** Build the temporal dimension (section alignment + diff generation). This is the only major missing piece — structural and semantic dimensions already exist in production.

2. **Week 2:** Build the query-type classifier and 3D routing logic. This is the intellectual core of the paper — the learned α,β,γ routing is what makes this a *framework* paper, not just an engineering paper.

3. **Week 3:** Construct SpecQA benchmark (100 queries with dimensional annotations). This is the evaluation backbone and a standalone publishable artifact.

4. **Weeks 4-6:** Run experiments, analyze results, write paper.

### Cost Estimate
- **API costs:** $50-80 total (temporal diffs $5-10, classifier training $10-15, experiments $30-50)
- **Human time:** ~100-120 hours over 6 weeks (benchmarking + writing + experiments)
- **Infrastructure:** $0 additional (everything runs on existing Neo4j Aura + Gemini API)

### Why This Will Get Accepted

1. **Clear novelty:** 3D retrieval framework — no prior work unifies structural + semantic + temporal
2. **Strong empirical setup:** 7 baselines, 100-query benchmark, ablation studies, statistical tests
3. **Real-world impact:** 52 state DOTs with similar specification documents; WYDOT is a deployed system
4. **Reproducibility:** All infrastructure exists; benchmark + code can be released
5. **Timeliness:** GraphRAG, agentic RAG, and domain-specific RAG are the hottest topics in NLP/IR right now (2025-2026)

---

*This plan was generated through critical analysis of 12 research papers, 9 existing research plans, 7 analysis documents, and the full WYDOT project codebase. It represents the highest-impact, most feasible path to a top-venue CS publication.*
