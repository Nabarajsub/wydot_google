# Critical Analysis: All 9 Research Plans
### Applicability, Implementation Feasibility, and Cost

> This document is a **brutally honest** assessment. Each plan is scored on a 1–5 scale for three dimensions, followed by specific risks and a final verdict.

---

## Scoring Legend

| Score | Meaning |
|-------|---------|
| ⭐ | Very Low / Major concerns |
| ⭐⭐ | Low / Significant challenges |
| ⭐⭐⭐ | Moderate / Doable with effort |
| ⭐⭐⭐⭐ | High / Straightforward |
| ⭐⭐⭐⭐⭐ | Very High / Almost trivial |

---

## Plan 01: Temporal/Versioned GraphRAG

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐⭐ | Perfect fit. You literally have 2010 and 2021 versions of the same spec. The `[:SUPERSEDES]` edge already exists in your Neo4j. |
| **Feasibility** | ⭐⭐⭐⭐ | Section alignment is the hard part. ~80% of sections have identical numbers across editions; the remaining 20% (merged/split/renamed) require fuzzy matching. PageIndex trees make this tractable. |
| **Cost** | ⭐⭐⭐⭐⭐ | ~$5–10 in Gemini API calls for 140 diff pairs. No new infrastructure needed. |

**Risks:**
- ❌ **Section alignment failures**: If Section 217 (2010) was split into 216A and 216B (2021), automatic alignment fails. Requires manual correction for ~10–20 edge cases.
- ❌ **Diff quality**: LLM-generated diffs may miss subtle numerical changes (e.g., "3000 psi" → "3500 psi" buried in a table). Need a secondary numerical diff check.
- ✅ **Mitigation**: Use PageIndex trees for structural alignment + regex for numerical extraction. Manageable.

**Verdict**: 🟢 **START HERE.** Lowest risk, highest immediate value, and builds the foundation for Plans 05, 06, and 07. Can produce results in 2–3 weeks.

---

## Plan 02: DualPath Hybrid RAG (PageIndex + GraphRAG)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐ | Both systems exist. The question is whether fusion genuinely improves answers or just adds complexity. |
| **Feasibility** | ⭐⭐⭐ | Building a working router is easy. Proving it helps is hard. The evaluation requires 50+ carefully designed queries with ground truth — significant human effort. |
| **Cost** | ⭐⭐⭐⭐ | ~$10–15 in API calls. The real cost is **human time** for query design and annotation (estimate: 20–30 hours). |

**Risks:**
- ❌ **The "null result" problem**: If PageIndex alone answers 90% of queries correctly, the fusion doesn't add much. You need to carefully select queries where BOTH paths are needed.
- ❌ **Latency penalty**: Running two retrieval paths doubles latency. For a real-time chat app, this matters.
- ❌ **Deduplication complexity**: PageIndex and GraphRAG may return overlapping content from the same pages. Naive concatenation wastes context window tokens.
- ✅ **Mitigation**: Focus evaluation on the ~20% of queries where single-path fails (cross-reference, multi-hop). This narrows the contribution but makes it defensible.

**Verdict**: 🟡 **Moderate priority.** A solid engineering paper but risks a "null result" that makes it unpublishable. Best done AFTER Plans 01 and 04 establish baselines. Probably a **workshop paper** unless the fusion shows >10% accuracy gain.

---

## Plan 03: Multimodal RAG for Engineering Specs

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐ | WYDOT specs are full of tables. Figures are less common but exist in structural diagrams. |
| **Feasibility** | ⭐⭐⭐ | Table extraction via PyMuPDF `find_tables()` works ~70% of the time. The remaining 30% are complex merged-cell tables that break. Figure extraction requires Vision API calls, which are expensive and noisy. |
| **Cost** | ⭐⭐⭐ | ~$20–40. Vision API calls for figure descriptions are expensive ($0.01–0.05 per image). For 200+ figures across both specs, this adds up. |

**Risks:**
- ❌ **Table extraction quality**: PyMuPDF's table detection fails on spanning headers and multi-page tables (common in WYDOT specs). You may need manual correction for 30–40% of tables.
- ❌ **Figure description noise**: Gemini Vision describing a cross-section diagram may produce a vague description ("A diagram showing layers of material") that doesn't help retrieval.
- ❌ **Evaluation difficulty**: How do you gold-annotate "correct table extraction"? You need pixel-level ground truth, which is extremely labor-intensive.
- ⚠️ **Competition**: NVIDIA's PDF extraction pipeline (NIM) and Mistral OCR (2025) are well-funded competitors. Your contribution needs to be about the **domain-specific evaluation**, not the extraction technique itself.

**Verdict**: 🟡 **Medium priority.** The table work is valuable as a FEATURE for your product but weak as a standalone PAPER. Best combined with Plan 04 (benchmark) — include table-dependent queries in TransportBench to create a unique evaluation.

---

## Plan 04: TransportBench (Domain Benchmark)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐⭐ | There is genuinely no RAG benchmark for transportation. This is a gap. |
| **Feasibility** | ⭐⭐⭐ | Creating 100 high-quality queries with gold answers requires deep domain expertise. You need a civil engineer to verify answers. Annotation is the bottleneck. |
| **Cost** | ⭐⭐⭐⭐ | ~$15–20 in API calls for running baselines. The real cost is **human annotation time**: 40–60 hours for 100 queries with gold answers and source passages. |

**Risks:**
- ❌ **Annotation bottleneck**: You need someone who can verify "Is 4000 psi correct for Class A concrete?" This requires domain expertise. If you're the sole annotator, inter-annotator agreement is impossible to report (reviewers will flag this).
- ❌ **Corpus size**: 2 PDFs (2010 + 2021) is small. Reviewers at EMNLP may say "This is just a test set for 2 documents, not a benchmark." You'd need 5+ documents for credibility.
- ❌ **Licensing**: Can WYDOT specs be redistributed as a benchmark? You need to verify they're public domain (most state DOT specs are, but check).
- ✅ **Mitigation**: Include 3–5 additional WYDOT documents (manuals, meeting minutes) to reach 5+ document corpus. Get one engineering colleague to serve as second annotator.

**Verdict**: 🟡 **High value but labor-intensive.** If you can get a second annotator and expand the corpus, this becomes a **strong resource paper**. Best submitted to **EMNLP or ACL Resource Track** where dataset papers are welcomed. Without a second annotator, it's a workshop paper.

---

## Plan 05: Agentic Compliance Monitor

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐⭐ | This is the most practically useful plan. Every DOT engineer struggles with spec updates. |
| **Feasibility** | ⭐⭐ | Multi-agent orchestration is complex. Building the Diff Agent alone is Plan 01. The Classification Agent needs labeled training data you don't have yet. The Stakeholder Mapper needs project data you may not have access to. |
| **Cost** | ⭐⭐⭐ | ~$30–50 for the full diff + classification pipeline. The multi-agent framework (LangGraph) is free but requires significant development time. |

**Risks:**
- ❌ **Depends on Plan 01**: You can't build a compliance monitor without first building the temporal diff graph. This is a **Phase 2** project.
- ❌ **Classification accuracy**: Classifying changes as Critical/Major/Minor requires domain expertise. Without a labeled training set (50+ manually classified diffs), the agent will make dangerous misclassifications (calling a safety change "Minor").
- ❌ **Stakeholder data**: The "who does this affect?" agent needs project-to-spec mapping data that may not exist in your current system.
- ❌ **Over-engineering risk**: 5 specialized agents may be overkill. A single well-prompted agent with tool access might perform identically.

**Verdict**: 🟠 **High value, but Phase 2.** Do Plan 01 first. Then the compliance monitor becomes achievable. As a paper, it's better suited for an **applied AI workshop** (AAAI AI4Good, NeurIPS AI4Science) than a main conference — reviewers at main tracks may view it as "engineering" rather than "research."

---

## Plan 06: VerifyRAG (Formal Verification)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐ | Safety-critical framing is compelling. Real need in engineering. |
| **Feasibility** | ⭐⭐ | The "constraint database" is the critical bottleneck. Extracting 200+ machine-readable constraints from natural-language spec text is essentially a **structured information extraction** task. LLMs are imperfect at this (~80% accuracy), and manual verification of 200 constraints takes significant effort. |
| **Cost** | ⭐⭐⭐ | ~$15–25 for extraction + verification pipeline. The real cost is the **human expert time** needed to verify extracted constraints (20–30 hours). |

**Risks:**
- ❌ **Constraint extraction is HARD**: "Aggregate shall conform to the gradation requirements shown in Table 301-1" — how do you extract a machine-readable constraint from a table reference? You need table parsing (Plan 03) to work first.
- ❌ **Coverage ceiling**: Many engineering requirements are qualitative ("Engineer's judgment shall prevail"), not quantitative. The constraint database can only verify ~40–60% of answers.
- ❌ **NeurIPS bar**: Reviewers may ask "Why not just use the spec as a lookup table?" The contribution needs to be clearly beyond simple pattern matching.
- ❌ **Formal methods community**: True "formal verification" has a specific meaning (theorem proving, model checking). Using the term loosely will attract criticism from formal methods reviewers.
- ✅ **Mitigation**: Frame as "automated consistency checking" rather than "formal verification." Focus on the 40% of answers that ARE verifiable as the contribution.

**Verdict**: 🟠 **Ambitious and novel, but high risk.** The constraint extraction bottleneck could stall the project. Best attempted AFTER Plans 01 and 03 provide the structural and table-parsing foundations. If successful, it's genuinely new and publishable — but execution risk is high.

---

## Plan 07: Efficiency Frontier (CAG vs RAG vs Long Context)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐⭐ | The "RAG vs Long Context" debate is the #1 topic in AI engineering right now. |
| **Feasibility** | ⭐⭐⭐⭐ | All three systems can be set up in ~1 week. The experiment is clean and well-defined. The main challenge is ensuring fair comparison (same model, same queries, controlled variables). |
| **Cost** | ⭐⭐ | **This is the most expensive plan.** Long Context mode (1M tokens per query × 100 queries × 3 runs) = ~300M tokens = **$60–150 in API costs.** CAG with KV caching is cheaper but still significant. |

**Risks:**
- ❌ **API cost**: This is the biggest risk. 300 Long Context inference calls at 1M tokens each is expensive. Budget carefully.
- ❌ **"Obvious result" risk**: If RAG wins on cost and Long Context wins on accuracy (which is the expected result), reviewers may say "We already knew this."
- ❌ **Model-specific results**: Results with Gemini may not transfer to GPT-4 or Claude. Reviewers will want multi-model experiments, which multiplies cost by 2–3x.
- ❌ **CAG availability**: KV cache preloading via API may have limited availability or undocumented behavior. Needs API access verification.
- ✅ **Mitigation**: The "Lost in the Middle" positional analysis on engineering specs IS novel. Focus the paper on that finding + the SpecCache hybrid proposal.

**Verdict**: 🟡 **Most publishable topic, but expensive.** If you can budget $100–200 for API costs, this is the most timely and citable paper. The novelty comes from the **domain** (engineering specs, not Wikipedia) and the **positional bias analysis** — not just the comparison itself.

---

## Plan 08: RecurseRAG (Provenance Chains)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐⭐⭐ | WYDOT specs are FULL of cross-references ("per AASHTO M-147", "see Table 301-1"). This is a genuine retrieval challenge. |
| **Feasibility** | ⭐⭐⭐ | Cross-reference detection is doable (~90% accuracy with regex + LLM). The hard part is that **you may not have AASHTO standards in your corpus.** AASHTO standards are copyrighted, so recursive retrieval into external standards may be impossible. |
| **Cost** | ⭐⭐⭐ | ~$20–30. Additional retrieval calls per query (2–3x normal). Annotation of 500 sentences for cross-reference detection training requires 10–15 hours. |

**Risks:**
- ❌ **AASHTO copyright**: The biggest risk. WYDOT specs reference AASHTO standards heavily, but AASHTO standards are **not freely available.** Your recursive agent will hit a dead end at Level 2 for most cross-references. This severely limits the "recursive" part.
- ❌ **Recursion depth**: In practice, most useful answers are at depth 1 (same document) or depth 2 (cross-document). Depth 3+ is rare. The "recursive" framing may be overstated.
- ❌ **Provenance DAG complexity**: Displaying a DAG to an engineer may be more confusing than helpful. The user study might show engineers prefer simple page citations.
- ✅ **Mitigation**: Focus on intra-document cross-references (Section A → Section B within WYDOT specs). These are common, resolvable, and don't hit copyright walls. Frame it as "intra-document cross-reference resolution" rather than "recursive retrieval."

**Verdict**: 🟡 **Good idea, but scope must shrink.** Drop the external AASHTO references (copyright issue). Focus on intra-spec cross-references (abundant and resolvable). The provenance DAG is still novel even within a single document. User study with engineers is the strongest contribution.

---

## Plan 09: HyperSpec (Hypergraph Memory)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Applicability** | ⭐⭐⭐ | Cross-session memory is a real UX need, but engineers rarely use chatbots over multi-day sessions for the same project spec. They tend to ask isolated questions. |
| **Feasibility** | ⭐⭐ | Hypergraph implementation is non-trivial. The HGMem paper (2025) required significant engineering. Building a domain-aware version adds another layer. The 10-session benchmark is artificial — real user sessions are messy and unpredictable. |
| **Cost** | ⭐⭐⭐ | ~$15–20 in API costs. The real cost is **development time**: 3–4 weeks to build the hypergraph data structure, memory operations, and Chainlit integration. |

**Risks:**
- ❌ **Overfit to simulation**: The 10-session benchmark is scripted. Real engineering sessions don't follow neat progressions. Reviewers will question ecological validity.
- ❌ **Sparse user interactions**: If a user only asks 3 questions per session, the hypergraph has too few nodes to form meaningful hyperedges. You need high-volume, related questions.
- ❌ **Evaluation difficulty**: How do you measure "cross-session coherence"? There's no established metric. You'd be proposing both the system AND the evaluation metric, which is risky.
- ❌ **Memory corruption**: Hyperedges formed from incorrect retrievals will propagate errors across sessions. One bad memory unit contaminates future sessions.
- ❌ **NeurIPS competition**: HGMem is already advanced. Your "domain-aware" extension may feel incremental.

**Verdict**: 🔴 **Lowest priority.** The most novel idea on paper, but the hardest to execute and evaluate convincingly. The gap between "cool concept" and "publishable result" is the widest here. Consider this a **12-month project**, not a 6-week sprint.

---

## Summary Comparison

| Plan | Applicability | Feasibility | Cost | Time to Results | Publication Risk | **Overall** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 Temporal | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $5–10 | 3 weeks | Low | 🟢 **Best start** |
| 02 DualPath | ⭐⭐⭐⭐ | ⭐⭐⭐ | $10–15 | 4 weeks | Medium (null result) | 🟡 |
| 03 Multimodal | ⭐⭐⭐⭐ | ⭐⭐⭐ | $20–40 | 5 weeks | Medium (competition) | 🟡 |
| 04 Benchmark | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $15–20 | 6 weeks | Medium (annotation) | 🟡 |
| 05 Compliance | ⭐⭐⭐⭐⭐ | ⭐⭐ | $30–50 | 8 weeks | Low (applied venue) | 🟠 |
| 06 VerifyRAG | ⭐⭐⭐⭐ | ⭐⭐ | $15–25 | 8 weeks | High (constraint extraction) | 🟠 |
| 07 Efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **$100–200** | 5 weeks | Medium (obvious result) | 🟡 |
| 08 RecurseRAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $20–30 | 6 weeks | Medium (scope) | 🟡 |
| 09 HyperSpec | ⭐⭐⭐ | ⭐⭐ | $15–20 | 10 weeks | **High** | 🔴 |

---

## Recommended Execution Order

```
Phase 1 (Weeks 1–3): Plan 01 — Temporal GraphRAG
    ├── Foundation for Plans 05, 06, 07
    └── Produces immediate results

Phase 2 (Weeks 4–8): Plan 04 — TransportBench
    ├── Creates the evaluation framework for ALL other plans
    ├── Combine with Plan 03 (table queries) for depth
    └── Submit as Resource Track paper

Phase 3 (Weeks 9–14): Choose ONE aggressive plan:
    ├── Option A: Plan 07 (Efficiency Frontier) — if budget allows ($100+)
    ├── Option B: Plan 08 (RecurseRAG) — if budget is tight
    └── Option C: Plan 06 (VerifyRAG) — if you have engineering expert support

Phase 4 (Long-term): Plan 05 (Compliance Monitor) — built on foundations
```

> **Bottom line**: Don't try to do all 9. Pick Plan 01 + Plan 04 as your foundation, then choose ONE aggressive plan (06, 07, or 08) for the top-conference submission.  Plan 09 is the most intellectually exciting but the least likely to produce results in a reasonable timeframe.
