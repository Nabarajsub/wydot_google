# Research Plan 05: Agentic Compliance Monitor — Proactive Specification Change Detection

## 1. Problem Statement

When WYDOT publishes a new edition of the Standard Specifications (e.g., 2021 replacing 2010), construction contractors, engineers, and project managers must **manually identify** which parts of their existing projects, contracts, and procedures are affected by the changes. This is a **time-consuming, error-prone** process that can lead to non-compliance, safety risks, and legal liability.

### The Research Gap

- **Agentic RAG (2025 survey)**: Describes autonomous agents for retrieval, but focuses on "information seeking" — not "proactive compliance monitoring."
- **MA-RAG (2025)**: Multi-agent QA framework, but designed for ad-hoc queries, not continuous monitoring.
- **Regulatory AI**: Most compliance AI focuses on financial regulations (SEC, FDA), not construction engineering standards.
- **No research** combines **agentic RAG** with **temporal document graphs** for **proactive change-impact analysis** in construction.

---

## 2. Proposed System: "ComplianceAgent"

### 2.1 Concept

An autonomous AI agent that:
1. **Detects** when a new specification version is ingested.
2. **Automatically diffs** every section against the previous version (using Plan 01's temporal graph).
3. **Classifies** each change by severity: Critical (safety), Major (procedural), Minor (editorial).
4. **Generates** a structured "Change Impact Report" for each Division.
5. **Proactively notifies** stakeholders about changes that affect their current projects.

### 2.2 Architecture

```
[New PDF Ingested]
        │
        ▼
[Agent: Diff Detector]
   "Which sections changed?"
        │
        ▼
[Agent: Impact Classifier]
   "Is this a safety change, a cost change, or editorial?"
        │
        ├── CRITICAL: "Minimum concrete strength changed from 3000 to 4000 psi"
        ├── MAJOR: "New test method (AASHTO T-335) required for aggregate"
        └── MINOR: "Typo fixed in Section 301.02"
        │
        ▼
[Agent: Stakeholder Mapper]
   "Who does this affect?"
        │
        ├── Current Projects using Division 300 → Project Manager A, B
        ├── Inspectors using Section 513 → Inspector Team C
        └── Materials Lab using AASHTO T-27 → Lab Manager D
        │
        ▼
[Agent: Report Generator]
   "Generate a Change Impact Report"
        │
        ▼
[Output: Structured Report + Notifications]
```

---

## 3. Agent Design

### 3.1 Agent Roles (Multi-Agent Architecture)

| Agent | Role | Tools | LLM |
|-------|------|-------|-----|
| **Diff Agent** | Extract structural diffs between versions | PageIndex trees, PDF text extraction | Gemini 2.5 Flash |
| **Classification Agent** | Classify change severity | Diff output, domain rules | Gemini 2.5 Flash |
| **Impact Agent** | Map changes to affected stakeholders/projects | Neo4j graph (entity relationships) | Gemini 2.5 Flash |
| **Report Agent** | Generate structured compliance report | All agent outputs | Gemini 2.5 Flash |
| **Orchestrator** | Coordinate the multi-agent workflow | LangGraph or custom | Gemini 2.5 Flash |

### 3.2 Classification Taxonomy

| Severity | Definition | Example | Action Required |
|----------|-----------|---------|-----------------|
| **🔴 Critical** | Safety-related change, new material requirement, load capacity change | "Minimum cover for reinforcement changed from 2" to 3"" | Immediate review, project re-evaluation |
| **🟡 Major** | New test method, changed procedure, new documentation requirement | "Added AASHTO T-335 as required test for aggregate" | Update QA procedures, retrain inspectors |
| **🟢 Minor** | Editorial, formatting, clarification without substance change | "Changed 'shall' to 'must' in Section 104.02" | Acknowledge, no action needed |

---

## 4. Key Research Questions

1. **Can LLMs reliably classify specification changes** into Critical/Major/Minor categories? (Measure accuracy vs. human expert labels.)
2. **How effective is multi-agent orchestration** compared to a single-agent approach for compliance analysis? (Measure completeness and accuracy of impact reports.)
3. **Can the system predict downstream impacts** (e.g., "Changing cement requirements affects Sections 413, 414, 501, and 513")? (Measure against expert-identified impact chains.)
4. **What is the optimal workflow** for human-in-the-loop validation? (Agent generates draft → Human approves/modifies.)

---

## 5. Implementation Roadmap

### Phase 1: Diff Infrastructure (Week 1)
- [ ] Build on Plan 01's temporal diff graph
- [ ] Create structured diff output format:
  ```json
  {
    "section": "301.03",
    "change_type": "modified",
    "old_text": "...",
    "new_text": "...",
    "semantic_diff": "Added gradation requirement for RAP material"
  }
  ```

### Phase 2: Classification Agent (Week 2)
- [ ] Design classification prompt with domain-specific rules
- [ ] Train on 50 manually labeled diffs (Critical/Major/Minor)
- [ ] Evaluate accuracy on held-out 25 diffs
- [ ] Iterate on prompt engineering

### Phase 3: Impact Analysis Agent (Week 3)
- [ ] Use Neo4j graph to trace entity relationships from changed sections
- [ ] Build impact chain: Changed Section → Affected Entities → Related Sections → Projects
- [ ] Generate structured impact report

### Phase 4: End-to-End Evaluation (Week 4)
- [ ] Run full pipeline on 2010 → 2021 specification transition
- [ ] Compare agent-generated report vs. human expert report
- [ ] Measure: Precision, Recall, F1 for change detection and impact analysis

---

## 6. Expected Contributions

1. **First proactive compliance monitoring system** using Agentic RAG for construction specifications.
2. **Change severity classification model** for engineering documents.
3. **Impact chain analysis** methodology using knowledge graphs.
4. **Practical tool** for WYDOT and other DOTs to automate specification update reviews.

---

## 7. Broader Impact

This research addresses a **real operational need** at WYDOT and every state DOT in the US:
- **52 state DOTs** each maintain their own specification manuals.
- **Every update cycle** requires manual review by hundreds of engineers and contractors.
- **A single missed critical change** can lead to construction defects, safety incidents, or legal disputes.
- This system could save **thousands of hours** of manual review per update cycle.

---

## 8. Publication Target

- **Conference**: AAAI 2026 AI for Government/Social Good, NeurIPS 2026 AI for Science
- **Title Proposal**: *"ComplianceAgent: Multi-Agent RAG for Proactive Change Detection in Government Construction Specifications"*
