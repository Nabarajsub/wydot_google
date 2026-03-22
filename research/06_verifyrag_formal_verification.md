# Research Plan 06: Self-Corrective RAG with Formal Verification for Safety-Critical Specifications
## Target: NeurIPS 2026 Main Track / ICLR 2027

## 1. Problem Statement

In safety-critical domains (bridges, highways, dams), a **wrong answer from a RAG system is worse than no answer.** If a contractor asks "What is the minimum concrete cover for reinforcement?" and the system returns an incorrect value from the wrong section or edition, the result could be a **structural failure.**

Existing Self-RAG (ICLR 2024) and CRAG (ICLR 2025) introduced "critique tokens" and "decision gates" to evaluate retrieval quality. But:

> **No one has formalized a verification pipeline for RAG in safety-critical engineering domains.**

### The Research Gap (Why This is NeurIPS-Worthy)

| Paper | What It Does | What It Misses |
|-------|-------------|---------------|
| **Self-RAG** (Asai et al., 2024) | Generates critique tokens for self-assessment | No domain-specific safety constraints; critique is learned, not verified |
| **CRAG** (Yan et al., 2025) | Gates on retrieval quality (Correct/Ambiguous/Incorrect) | Binary quality assessment; no formal proof of correctness |
| **SCMRAG** (AAMAS 2025) | Self-corrective multi-hop with dynamic KG | Focuses on web search fallback, not engineering verification |
| **Our Proposal** | **Formal verification chain** with domain constraint checking | **Fills the gap**: Provably correct answers for numerical engineering specs |

---

## 2. Core Innovation: "VerifyRAG"

A three-stage pipeline where the system doesn't just "retrieve" — it **proves** its answer is consistent with the specification.

### Architecture

```
User Query: "What is the minimum compressive strength for Class A concrete?"

Stage 1: RETRIEVAL (Standard)
├── Vector search → Candidate chunks
├── PageIndex tree → Candidate sections
└── Result: Section 513.02, Table 513-1

Stage 2: SELF-CRITIQUE (Self-RAG Style)
├── Relevance Token: Is this about Class A concrete? → ✅ [IsRel]
├── Support Token: Does source explicitly state a value? → ✅ [IsSup]
├── Recency Token: Is this from the latest edition? → ✅ [IsCur]
└── If ANY token = ❌ → Trigger corrective retrieval

Stage 3: FORMAL VERIFICATION (Novel)
├── Extract numerical claim: "4,000 psi"
├── Cross-check against specification constraint database:
│   Constraint: Class_A_Concrete.min_compressive_strength = 4000 psi (Section 513.02)
│   Verification: CLAIM == CONSTRAINT → ✅ VERIFIED
├── Check unit consistency: psi = psi → ✅
├── Check temporal validity: 2021 edition, current → ✅
└── Output: "4,000 psi [VERIFIED: Section 513.02, 2021 Standard Specifications]"
```

### 2.1 Constraint Database (The Key Contribution)

Extract machine-readable constraints from WYDOT specs:

```json
{
  "constraint_id": "C-513-001",
  "parameter": "min_compressive_strength",
  "material": "Class A Concrete",
  "value": 4000,
  "unit": "psi",
  "source_section": "513.02",
  "source_edition": "2021",
  "type": "minimum_threshold"
}
```

This enables **formal verification**: the LLM's answer can be checked against a structured rule, not just "vibes."

---

## 3. Key Research Questions

1. **Can LLMs reliably extract machine-readable constraints** from natural-language spec text? (Precision/Recall on constraint extraction.)
2. **Does formal verification reduce hallucination rate** compared to Self-RAG alone? (Measure factual accuracy on numerical engineering queries.)
3. **What is the cost-accuracy tradeoff** of adding a verification stage? (Tokens + latency vs. error rate.)
4. **Can the constraint database generalize** across state DOT specifications? (Test on Wyoming + Colorado + Texas specs.)

---

## 4. Evaluation Protocol (Rigorous for Top Conference)

### 4.1 Dataset
- **100 numerical engineering queries** with gold answers from WYDOT specs.
- **50 adversarial queries** designed to trigger hallucinations (wrong edition, wrong section, similar-but-different values).
- **Human expert annotations** from a licensed civil engineer.

### 4.2 Baselines
| System | Description |
|--------|-------------|
| Naive RAG | Vector search + Gemini generation |
| Self-RAG | With learned critique tokens |
| CRAG | With decision gate + web fallback |
| **VerifyRAG (Ours)** | Self-RAG + Formal constraint verification |

### 4.3 Metrics
| Metric | Definition |
|--------|-----------|
| **Factual Accuracy** | % of numerically correct answers |
| **Hallucination Rate** | % of answers containing fabricated values |
| **Verification Coverage** | % of answers that can be formally verified |
| **Safety Score** | 0 critical errors / 150 queries (target: **zero-tolerance**) |

### 4.4 Ablation Studies
- Self-RAG alone vs. Self-RAG + Verification
- Constraint extraction: Rule-based vs. LLM-extracted
- Effect of constraint database size on verification coverage

---

## 5. Implementation Roadmap

| Phase | Duration | Deliverable |
|-------|----------|------------|
| Constraint Extraction Pipeline | 2 weeks | Extract 200+ constraints from 2021 specs |
| Self-Critique Module | 1 week | Implement 4 critique token types |
| Formal Verifier | 2 weeks | Constraint matching + unit/temporal checks |
| Evaluation | 2 weeks | Full benchmark on 150 queries |
| Paper Writing | 2 weeks | NeurIPS-format paper |

---

## 6. Why This Wins at NeurIPS

1. **Novel formalism**: First paper to bring formal verification to RAG for engineering.
2. **Safety-critical framing**: NeurIPS has a growing track on AI Safety.
3. **Real-world data**: Not synthetic benchmarks — actual government specifications.
4. **Zero-tolerance evaluation**: The "Safety Score" metric is provocative and memorable.
5. **Generalizable**: The constraint database concept applies to ANY regulatory domain (FDA, SEC, building codes).

---

## 7. Publication Target

- **Primary**: NeurIPS 2026 Main Track (Deadline: May 2026)
- **Backup**: ICLR 2027 (Deadline: Oct 2026)
- **Title**: *"VerifyRAG: Formal Verification of Retrieval-Augmented Generation for Safety-Critical Engineering Specifications"*
