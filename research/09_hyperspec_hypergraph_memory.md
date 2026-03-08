# Research Plan 09: HyperSpec — Hypergraph Memory for Multi-Session Engineering QA
## Target: NeurIPS 2026 / AAAI 2027

## 1. Problem Statement

Engineers interact with the WYDOT assistant over **multiple sessions** spanning days or weeks. In Session 1, they ask about "Section 513 concrete specs." In Session 5, they ask about "rebar cover requirements." **These are deeply related** — but current RAG systems have **no memory** across sessions. Each session starts from scratch.

**HGMem** (arXiv 2025) introduced hypergraph-based memory for multi-step RAG, but it only works **within a single session.** No one has applied hypergraph memory to **cross-session engineering QA** where facts accumulate and evolve over time.

### The Research Gap

| System | Memory Type | Scope | Engineering-Specific? |
|--------|-----------|-------|----------------------|
| **HGMem** (2025) | Hypergraph | Single session, multi-step | ❌ General NLP |
| **mem0** | Key-value store | Cross-session | ❌ General chatbot |
| **Chainlit History** | Message log | Cross-session | ❌ No semantic structure |
| **HyperSpec (Ours)** | **Domain-aware hypergraph** | **Cross-session + Cross-document** | **✅ Engineering specs** |

---

## 2. Core Innovation: Domain-Aware Hypergraph Memory

### 2.1 What is a Hypergraph?

In a normal graph, an edge connects 2 nodes. In a **hypergraph**, a **hyperedge connects N nodes simultaneously**, capturing higher-order relationships.

```
Normal Graph:        Hypergraph:
A ── B              A ─┐
B ── C              B ─┤── HE₁ (All used in bridge deck concrete)
                    C ─┘
                    D ─┐
                    E ─┤── HE₂ (All modified between 2010→2021)
                    F ─┘
```

### 2.2 HyperSpec Architecture

```
Session 1: User asks about concrete specs
    → Memory Unit M₁: {Section 513, Class A Concrete, 4000 psi, 2021}

Session 2: User asks about rebar requirements
    → Memory Unit M₂: {Section 509, #4 rebar, 2" cover, Grade 60}

Session 3: User asks about bridge deck
    → System recognizes: M₁ + M₂ are both relevant to bridge decks
    → Creates Hyperedge HE₁: {M₁, M₂} → "Bridge Deck Requirements"
    → Now can reason: "For a bridge deck, you need Class A concrete
       (§513) with #4 rebar at 2" cover (§509)"

Session 5: User asks about 2010 vs 2021 changes
    → System traverses temporal hyperedges to find all changed specs
       that were previously discussed
```

### 2.3 Memory Operations

| Operation | Description | When |
|-----------|-------------|------|
| **INSERT** | Add new memory unit from retrieval | Every query |
| **UPDATE** | Modify existing unit with new info | When same section is re-queried |
| **MERGE** | Create hyperedge linking related units | When 2+ units share entities |
| **DECAY** | Reduce weight of old, unaccessed units | Over time |
| **PROMOTE** | Increase weight of frequently accessed units | On re-access |

---

## 3. Key Research Questions

1. **Does cross-session memory improve answer quality** for follow-up questions? (Accuracy with/without memory.)
2. **Can hyperedges capture domain-specific relationships** (e.g., "all specs needed for a bridge deck") that flat memory cannot?
3. **What is the optimal memory size** before noise overwhelms signal?
4. **Does temporal decay help or hurt** for engineering knowledge?
5. **Can HyperSpec generalize** to other technical domains (medical protocols, legal contracts)?

---

## 4. Evaluation Protocol

### 4.1 Multi-Session Benchmark
Design a 10-session simulation:

| Session | Query | Expected Memory Effect |
|---------|-------|----------------------|
| 1 | "What concrete class for bridge decks?" | Creates M₁ |
| 2 | "What reinforcement for bridge decks?" | Creates M₂, Merges with M₁ |
| 3 | "Summarize bridge deck requirements" | Uses HE₁ (M₁ + M₂) |
| 4 | "What about retaining walls?" | Creates M₃ (different context) |
| 5 | "Compare bridge deck vs retaining wall concrete" | Connects M₁ and M₃ |
| ... | ... | Progressive complexity |
| 10 | "What are the key specs I've been researching?" | Global memory summary |

### 4.2 Baselines

| System | Memory |
|--------|--------|
| No Memory (stateless RAG) | None |
| Flat Memory (message history) | Last N messages |
| KV Memory (mem0-style) | Key-value pairs |
| **HyperSpec (Ours)** | Domain-aware hypergraph |

### 4.3 Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Session Coherence** | Do later sessions benefit from earlier ones? |
| **Cross-Reference Discovery** | Does the system find connections the user didn't ask for? |
| **Memory Precision** | Are recalled memories actually relevant? |
| **Answer Completeness** | Does memory-augmented answer cover more aspects? |

---

## 5. Implementation Roadmap

| Phase | Duration | Deliverable |
|-------|----------|------------|
| Hypergraph data structure | 1 week | Python implementation with INSERT/MERGE/DECAY |
| Integration with Chainlit | 1 week | Persistent memory across sessions |
| Domain-aware entity linker | 2 weeks | Recognize WYDOT-specific entities in memory |
| 10-session benchmark | 2 weeks | Scripted simulation with ground truth |
| Evaluation + ablations | 1 week | Full comparison |
| Paper writing | 2 weeks | NeurIPS-format paper |

---

## 6. Why This Wins at NeurIPS

1. **Novel architecture**: First domain-aware hypergraph memory for technical QA.
2. **Higher-order reasoning**: Hyperedges capture relationships that graphs cannot.
3. **Practical need**: Engineers actually do multi-session research on specs.
4. **Clean ablation**: Flat memory → Graph memory → Hypergraph memory.
5. **Generalizable**: Framework applies to medical, legal, financial domains.

---

## 7. Publication Target

- **Primary**: NeurIPS 2026 (Deadline: May 2026)
- **Backup**: AAAI 2027 (Deadline: Aug 2026)
- **Title**: *"HyperSpec: Domain-Aware Hypergraph Memory for Cross-Session Multi-Turn Technical QA"*
