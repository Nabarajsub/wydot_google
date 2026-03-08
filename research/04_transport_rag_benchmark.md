# Research Plan 04: TransportBench — A Domain-Specific RAG Benchmark for Transportation Documents

## 1. Problem Statement

RAG evaluation benchmarks exist for general knowledge (NaturalQuestions), legal documents (LegalBench-RAG), and finance (FinanceBench). **No benchmark exists for transportation or construction engineering documents.** This is a critical gap because:

1. Transportation documents have **unique structures** (Division/Section numbering, material specification tables, test method cross-references) that generic benchmarks don't test.
2. Transportation queries require **domain-specific reasoning** (e.g., understanding that "Type B aggregate" refers to a specific gradation table, not just a keyword).
3. Government agencies worldwide maintain similar spec structures (AASHTO, state DOTs), making a benchmark broadly applicable.

### The Research Gap

| Benchmark | Domain | Year | Covers Transport? |
|-----------|--------|------|-------------------|
| NaturalQuestions | General | 2019 | ❌ |
| HotpotQA | Multi-hop | 2018 | ❌ |
| LegalBench-RAG | Legal | 2024 | ❌ |
| FinanceBench | Finance | 2023 | ❌ |
| UniDoc-Bench | Multi-domain | 2025 | Partially (construction category) |
| **TransportBench (Ours)** | **Transportation** | **2025** | **✅ First dedicated benchmark** |

---

## 2. Benchmark Design

### 2.1 Document Corpus

| Document | Pages | Type | Source |
|----------|-------|------|--------|
| WYDOT 2021 Standard Specifications | 771 | Regulatory Manual | Your project |
| WYDOT 2010 Standard Specifications | 890 | Regulatory Manual | Your project |
| Additional WYDOT PDFs (meeting minutes, manuals) | ~2000+ | Mixed | Your `data/` folder |

### 2.2 Query Categories

| Category | Description | # Queries | Example |
|----------|-------------|-----------|---------|
| **Factual Lookup** | Find a specific fact in a spec section | 30 | "What is the minimum cement content for Class A concrete?" |
| **Table Extraction** | Answer requires reading a specification table | 20 | "What sieve sizes are required for Type A aggregate base?" |
| **Cross-Reference** | Answer spans multiple sections/documents | 15 | "Which test methods are required for Section 301 aggregates?" |
| **Temporal/Version** | Answer requires comparing document versions | 15 | "What changed in Division 500 between 2010 and 2021?" |
| **Multi-Hop** | Answer requires connecting 2+ facts | 10 | "If I'm building a Class A bridge deck, what aggregate and cement specs apply?" |
| **Unanswerable** | Query has no answer in the corpus | 10 | "What are CDOT's specifications for hot mix asphalt?" |
| **Total** | | **100** | |

### 2.3 Annotation Framework

For each query, annotate:
- **Ground truth answer** (human-written)
- **Gold source passages** (exact text spans from the PDFs)
- **Gold page numbers** (for citation verification)
- **Difficulty level** (Easy / Medium / Hard)
- **Required capability** (Lookup / Table / Reasoning / Comparison)

---

## 3. Evaluation Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Answer Accuracy** | Is the answer factually correct? | LLM-as-judge (GPT-4 or Claude) |
| **Citation Precision** | Does the system cite the correct source? | Exact match on page number ± 2 pages |
| **Citation Recall** | Does it find all relevant sources? | Gold source overlap |
| **Faithfulness** | Is the answer supported by the context? | NLI-based check |
| **Table Extraction Accuracy** | Can it read table values correctly? | Exact numerical match |
| **Temporal Reasoning** | Can it identify changes across versions? | Human evaluation |

---

## 4. Implementation Roadmap

### Phase 1: Query Generation (Week 1-2)
- [ ] Manually create 50 queries across all categories
- [ ] Use LLM to generate 50 additional queries, then human-verify
- [ ] For each query, manually identify gold answer and source passages
- [ ] Assign difficulty levels and required capabilities

### Phase 2: Annotation & Validation (Week 3)
- [ ] Have 2 annotators independently verify gold answers
- [ ] Compute inter-annotator agreement (Cohen's kappa)
- [ ] Resolve disagreements
- [ ] Create final JSON dataset with schema:
  ```json
  {
    "query_id": "TQ-001",
    "query": "What is the minimum cement content for Class A concrete?",
    "category": "Factual Lookup",
    "difficulty": "Easy",
    "gold_answer": "The minimum cement content for Class A concrete is 564 lb/yd³.",
    "gold_sources": [
      {"doc": "2021_specs.pdf", "page": 445, "section": "513.03", "text": "..."}
    ],
    "requires": ["lookup"]
  }
  ```

### Phase 3: Baseline Evaluation (Week 4)
- [ ] Run all 100 queries through:
  - (a) Naive RAG (vector search only)
  - (b) Your GraphRAG (`chatapp_gemini.py`)
  - (c) PageIndex
  - (d) DualPath RAG (if implemented from Plan 02)
- [ ] Score each system on all metrics
- [ ] Generate leaderboard

### Phase 4: Release & Paper (Week 5)
- [ ] Package benchmark as downloadable dataset (JSON + PDFs)
- [ ] Write evaluation paper with baseline results
- [ ] Release on HuggingFace Datasets or GitHub

---

## 5. Expected Contributions

1. **TransportBench**: The first dedicated RAG evaluation benchmark for transportation engineering documents.
2. **Baseline results**: Performance of 4+ RAG architectures on domain-specific engineering queries.
3. **Category-level analysis**: Which RAG architectures excel at which query types.
4. **Reusable by other DOTs**: Any state DOT with similar specs can use TransportBench for their own evaluation.

---

## 6. Publication Target

- **Conference**: EMNLP 2026 Findings, ACL 2026 Resource Track
- **Title Proposal**: *"TransportBench: A Domain-Specific Benchmark for Evaluating RAG Systems on Government Transportation Specifications"*
- **Dataset Release**: HuggingFace Datasets Hub
