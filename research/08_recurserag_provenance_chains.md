# Research Plan 08: Recursive Agentic RAG with Provenance Chains for Engineering Compliance
## Target: ACL 2026 Main Track / NAACL 2027

## 1. Problem Statement

When an agent answers "The aggregate for base course must conform to AASHTO M-147 Table 1," a follow-up question arises: **"What does AASHTO M-147 Table 1 actually require?"** Current RAG systems either:

1. **Stop** — "Please refer to AASHTO M-147."
2. **Hallucinate** — Invent plausible-sounding requirements.

A **Recursive Agentic RAG** system would **automatically detect** the cross-reference, **launch a sub-retrieval** for AASHTO M-147, and **chain the provenance** so the user can trace every claim back to its source.

### The Research Gap

| System | Cross-Reference Handling | Provenance |
|--------|------------------------|------------|
| Standard RAG | Ignores cross-references | Source chunk ID only |
| Self-RAG | Evaluates relevance, but doesn't follow references | Critique tokens, no chain |
| LangGraph Agents | Can call tools recursively, but no formal provenance model | Implicit in agent state |
| **RecurseRAG (Ours)** | **Detects, follows, and chains cross-references** | **Formal provenance DAG** |

---

## 2. Core Innovation: Provenance DAG (Directed Acyclic Graph)

Every claim in the final answer has a traceable chain:

```
Final Answer Claim: "Base course aggregate must pass 100% through a 1" sieve"
    │
    ├── [LEVEL 0] User Query → "aggregate requirements for base course"
    │
    ├── [LEVEL 1] Retrieved: WYDOT Section 301.03
    │   └── Text: "Aggregate shall conform to AASHTO M-147, Table 1"
    │       └── Cross-reference detected: "AASHTO M-147 Table 1"
    │
    ├── [LEVEL 2] Sub-Retrieval: AASHTO M-147 (if available in corpus)
    │   └── Text: "Table 1 — Gradation Requirements: 1" sieve = 100%"
    │       └── Provenance: VERIFIED from AASHTO M-147
    │
    └── [LEVEL 2-ALT] If not in corpus:
        └── FLAGGED: "Referenced standard AASHTO M-147 not available.
             Claim extracted from WYDOT Section 301.03 only."
```

### 2.1 The Provenance DAG

```
                    ┌─────────────────┐
                    │  Final Answer   │
                    │  (Claim C₁)     │
                    └────────┬────────┘
                             │ supported_by
                    ┌────────┴────────┐
                    │                 │
            ┌───────▼──────┐  ┌──────▼────────┐
            │ WYDOT §301   │  │ AASHTO M-147  │
            │ (Source S₁)  │  │ (Source S₂)   │
            │ Level: 1     │  │ Level: 2      │
            │ Method: Vec  │  │ Method: Xref  │
            └──────────────┘  └───────────────┘
```

Each node stores: `{source, page, retrieval_method, level, confidence, cross_ref_trigger}`

---

## 3. System Architecture

### 3.1 Agent Pipeline

```python
class RecurseRAGAgent:
    def answer(self, query, max_depth=3):
        # Level 0: Initial retrieval
        chunks = self.retrieve(query)
        
        # Detect cross-references
        xrefs = self.detect_cross_references(chunks)
        
        # Level 1..N: Recursive sub-retrieval
        for xref in xrefs:
            if depth < max_depth:
                sub_chunks = self.retrieve(xref.target)
                chunks.extend(sub_chunks)
                # Record provenance edge
                self.provenance.add_edge(xref.source, sub_chunks, level=depth+1)
        
        # Generate with full provenance context
        answer = self.generate(query, chunks, provenance=self.provenance)
        return answer, self.provenance.to_dag()
```

### 3.2 Cross-Reference Detection

Train a lightweight classifier to detect:
- **Explicit references**: "per AASHTO M-147", "see Section 301", "as specified in Table 513-1"
- **Implicit references**: "the requirements of the preceding section", "standard test methods"
- **Circular references**: Detect and break cycles (Section A → Section B → Section A)

---

## 4. Key Research Questions

1. **How many levels of recursion are needed?** (Hypothesis: 2-3 levels suffice for 95% of engineering queries.)
2. **Does the provenance DAG improve user trust?** (User study with engineers.)
3. **What is the accuracy gain** from following cross-references vs. ignoring them?
4. **Can circular references be reliably detected and broken?**
5. **Does the cross-reference detector generalize** across DOT specifications?

---

## 5. Evaluation Protocol

### 5.1 Cross-Reference Detection Task
- **Dataset**: Manually annotate 500 sentences from WYDOT specs for cross-references.
- **Metrics**: Precision, Recall, F1 for cross-reference detection.
- **Baselines**: Regex-only, NER-based, LLM-based.

### 5.2 End-to-End QA Task
- **Dataset**: 50 queries that REQUIRE cross-reference following to answer correctly.
- **Example**: "What sieve sizes are required for aggregate base per WYDOT specifications?"
  - Correct answer requires: WYDOT §301 → AASHTO M-147 Table 1
- **Baselines**: Standard RAG, Self-RAG, CRAG, PageIndex.
- **Metrics**: Answer accuracy, provenance completeness, depth of retrieval.

### 5.3 User Study (for ACL)
- **Participants**: 10 civil engineering students/professionals.
- **Task**: Rate answers from (a) Standard RAG, (b) RecurseRAG with provenance.
- **Metrics**: Trust score, perceived accuracy, willingness to act on the answer.

---

## 6. Implementation Roadmap

| Phase | Duration | Deliverable |
|-------|----------|------------|
| Cross-reference annotator | 2 weeks | 500 annotated sentences |
| Cross-reference detector | 1 week | Classifier (LLM-based + rules) |
| Recursive retrieval engine | 2 weeks | Max-depth-3 recursive pipeline |
| Provenance DAG visualizer | 1 week | Interactive DAG display in Chainlit |
| Evaluation + user study | 2 weeks | Full results |
| Paper writing | 2 weeks | ACL-format paper |

---

## 7. Why This Wins at ACL

1. **Clear linguistic contribution**: Cross-reference detection is a discourse analysis task.
2. **Formal provenance model**: The DAG formalism is novel for RAG.
3. **User study**: ACL values human evaluation.
4. **Real engineering impact**: Engineers need to trace every claim.
5. **Reproducible**: All data from public WYDOT specs.

---

## 8. Publication Target

- **Primary**: ACL 2026 Main Track (Deadline: Feb 2026)
- **Backup**: EMNLP 2026, NAACL 2027
- **Title**: *"RecurseRAG: Recursive Cross-Reference Resolution with Provenance Chains for Technical Document QA"*
