"""
Phase 5: SpecQA Benchmark — 100 Queries with Dimensional Annotations
======================================================================
The evaluation backbone for SpecRAG. 100 queries across 8 types,
each annotated with:
- Required retrieval dimensions (alpha, beta, gamma)
- Gold-standard answer
- Source documents and page ranges
- Difficulty level

Usage:
    python -m phase5_benchmark.specqa_queries --generate    # Generate initial set
    python -m phase5_benchmark.specqa_queries --export      # Export to JSON
    python -m phase5_benchmark.specqa_queries --stats       # Show distribution
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BENCHMARK_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── SpecQA Benchmark Queries ─────────────────────────────
# 100 queries across 8 types with dimensional annotations

SPECQA_QUERIES = [
    # ══════════════════════════════════════════════════════
    # TYPE 1: STRUCTURAL (alpha-only) — 12 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "S01", "query": "What topics are covered under Division 200 in the 2021 Standard Specifications?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["earthwork", "excavation", "embankment", "subgrade"],
        "notes": "Requires navigating Division structure to find Division 200 scope"
    },
    {
        "id": "S02", "query": "How many sections are in Division 500 of the 2021 Standard Specifications?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["section", "501", "502", "concrete"],
        "notes": "Pure structural navigation"
    },
    {
        "id": "S03", "query": "What is the overall organization of the 2026 Construction Manual?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["chapter", "organization"],
        "notes": "TOC-level question"
    },
    {
        "id": "S04", "query": "Which division in the Standard Specifications covers structural steel?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["division", "steel", "structural"],
        "notes": "Requires scanning division titles"
    },
    {
        "id": "S05", "query": "List all the chapters in the 2024 Construction Manual.",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["2024 Construction Manual.pdf"],
        "gold_answer_keywords": ["chapter"],
        "notes": "Full TOC listing"
    },
    {
        "id": "S06", "query": "Where in the 2021 Standard Specifications would I find information about painting?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["painting", "division", "section"],
        "notes": "Requires navigating to find painting-related sections"
    },
    {
        "id": "S07", "query": "What subsections exist under Section 501 in the 2021 Standard Specifications?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["501.01", "501.02", "501.03", "concrete"],
        "notes": "Drill into section structure"
    },
    {
        "id": "S08", "query": "How is the Materials Testing Manual FY 2023 organized?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["test", "method", "section"],
        "notes": "Testing Manual structure"
    },
    {
        "id": "S09", "query": "Which section of the Standard Specifications deals with hot mix asphalt?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["HMA", "asphalt", "section", "401"],
        "notes": "Locate specific topic in hierarchy"
    },
    {
        "id": "S10", "query": "What major divisions exist in the 2010 Standard Specifications?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["2010 Standard Specifications.pdf"],
        "gold_answer_keywords": ["division", "100", "200", "300"],
        "notes": "Top-level structure of older edition"
    },
    {
        "id": "S11", "query": "Where would I find guardrail specifications in the 2021 Standard Specs?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["guardrail", "section", "division"],
        "notes": "Topic-based structure search"
    },
    {
        "id": "S12", "query": "How many pages does Division 800 span in the 2021 Standard Specifications?",
        "type": "structural", "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["division 800", "pages"],
        "notes": "Page range calculation from tree"
    },

    # ══════════════════════════════════════════════════════
    # TYPE 2: SEMANTIC (beta-only) — 12 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "E01", "query": "What is the minimum compressive strength required for Class A concrete?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["compressive", "strength", "psi", "class A"],
        "notes": "Entity: Material (Class A Concrete) → property"
    },
    {
        "id": "E02", "query": "What AASHTO standards are referenced for aggregate testing?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["AASHTO", "aggregate", "T-27", "M-147"],
        "notes": "Entity search: Standards referenced by Material"
    },
    {
        "id": "E03", "query": "What are the gradation requirements for Type B aggregate?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["gradation", "sieve", "passing", "type B"],
        "notes": "Entity: Material property with table data"
    },
    {
        "id": "E04", "query": "What types of portland cement are approved for use by WYDOT?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "easy",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["portland", "cement", "type I", "type II", "type IL"],
        "notes": "Entity listing"
    },
    {
        "id": "E05", "query": "What is the required slump range for structural concrete?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["slump", "inches", "range"],
        "notes": "Specific numerical requirement"
    },
    {
        "id": "E06", "query": "What equipment is required for nuclear density testing?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["nuclear", "density", "gauge", "equipment"],
        "notes": "Entity: Equipment for TestMethod"
    },
    {
        "id": "E07", "query": "What is the maximum water-cement ratio for bridge deck concrete?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["water", "cement", "ratio", "bridge", "deck"],
        "notes": "Specific value for specific application"
    },
    {
        "id": "E08", "query": "What forms are required for concrete mix design submittal?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["form", "mix", "design", "submittal"],
        "notes": "Entity: Form used for Process"
    },
    {
        "id": "E09", "query": "What is WYDOT's specification for fly ash in concrete?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["fly ash", "class", "percentage", "replacement"],
        "notes": "Material specification details"
    },
    {
        "id": "E10", "query": "What are the acceptance criteria for compacted embankment?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["compaction", "density", "percent", "embankment"],
        "notes": "Numerical acceptance criteria"
    },
    {
        "id": "E11", "query": "List all ASTM standards referenced in Division 500.",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["ASTM", "C150", "C33"],
        "notes": "Entity traversal: Division → Sections → Referenced Standards"
    },
    {
        "id": "E12", "query": "What is the minimum concrete cover requirement for reinforcing steel in bridge decks?",
        "type": "semantic", "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["cover", "reinforcing", "inches", "bridge"],
        "notes": "Specific numeric requirement"
    },

    # ══════════════════════════════════════════════════════
    # TYPE 3: TEMPORAL (gamma-only) — 10 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "T01", "query": "What major changes occurred in the Standard Specifications between 2010 and 2021?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["changed", "updated", "added", "removed"],
        "notes": "Broad temporal query across editions"
    },
    {
        "id": "T02", "query": "Were any new sections added to Division 500 between the 2010 and 2021 editions?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["added", "new", "section", "500"],
        "notes": "Specific temporal: structural addition detection"
    },
    {
        "id": "T03", "query": "How did the Construction Manual change from 2022 to 2023?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "medium",
        "source_docs": ["2022 Construction Manual.pdf", "2023 Construction Manual.pdf"],
        "gold_answer_keywords": ["changed", "updated", "revision"],
        "notes": "Construction Manual temporal diff"
    },
    {
        "id": "T04", "query": "What updates were made to the Materials Testing Manual between 2020 and 2023?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "hard",
        "source_docs": ["Materials Testing Manual  FY 2020.pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["updated", "test", "method", "changed"],
        "notes": "Testing Manual temporal chain"
    },
    {
        "id": "T05", "query": "According to the Summary of Changes, what were the key 2019 Construction Manual updates?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "easy",
        "source_docs": ["Summary of Changes 2019 Construction Manual.pdf"],
        "gold_answer_keywords": ["2019", "change", "update"],
        "notes": "Direct from change summary document"
    },
    {
        "id": "T06", "query": "Has the Construction Manual grown or shrunk in page count from 2018 to 2026?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "easy",
        "source_docs": ["2018 Construction Manual.pdf", "2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["pages", "2018", "2026"],
        "notes": "Quantitative temporal comparison"
    },
    {
        "id": "T07", "query": "What revisions were documented in the 2022 Construction Manual Revision Summary?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "easy",
        "source_docs": ["2022 Construction Manual Revision Summary.pdf"],
        "gold_answer_keywords": ["revision", "2022", "change"],
        "notes": "Direct from revision summary"
    },
    {
        "id": "T08", "query": "How did the 2025 Construction Manual differ from the 2024 edition?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "medium",
        "source_docs": ["2024 Construction Manual.pdf", "2025 Construction Manual.pdf"],
        "gold_answer_keywords": ["change", "2024", "2025"],
        "notes": "Recent temporal diff"
    },
    {
        "id": "T09", "query": "What is the version history of the WYDOT Standard Specifications?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "easy",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["2010", "2021", "edition", "version"],
        "notes": "SUPERSEDES chain traversal"
    },
    {
        "id": "T10", "query": "What are the Summary of Changes documents available and what years do they cover?",
        "type": "temporal", "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "difficulty": "easy",
        "source_docs": ["Summary of Changes 2018 Construction Manual.pdf", "Summary of Changes 2019 Construction Manual.pdf", "Summary of Changes 2020 Construction Manual.pdf", "Summary of Changes 2021 Construction Manual.pdf"],
        "gold_answer_keywords": ["2018", "2019", "2020", "2021", "summary"],
        "notes": "Meta-temporal: listing change documents"
    },

    # ══════════════════════════════════════════════════════
    # TYPE 4: STRUCTURAL + SEMANTIC (alpha+beta) — 14 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "SS01", "query": "In Section 501 of the 2021 specs, what materials are specified for concrete?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["section 501", "concrete", "cement", "aggregate", "water"],
        "notes": "Navigate to section (structural) then extract materials (semantic)"
    },
    {
        "id": "SS02", "query": "What test methods are referenced in Division 300 for base course materials?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["division 300", "base course", "AASHTO", "test"],
        "notes": "Navigate division, extract test method entities"
    },
    {
        "id": "SS03", "query": "Which chapter of the 2025 Construction Manual covers earthwork, and what are the key compaction requirements?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["2025 Construction Manual.pdf"],
        "gold_answer_keywords": ["chapter", "earthwork", "compaction", "density"],
        "notes": "Find chapter (structural) + extract requirements (semantic)"
    },
    {
        "id": "SS04", "query": "In the 2023 Testing Manual, what is the procedure for AASHTO T-27 sieve analysis?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["T-27", "sieve", "procedure", "gradation"],
        "notes": "Navigate to test method (structural) + details (semantic)"
    },
    {
        "id": "SS05", "query": "Under what section is hot mix asphalt covered, and what are the mix design requirements?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["HMA", "section", "mix design", "asphalt"],
        "notes": "Locate section + extract entity details"
    },
    {
        "id": "SS06", "query": "What materials and standards are specified in Division 600 for structural concrete?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["division 600", "structural", "concrete", "standard"],
        "notes": "Navigate to division + extract entities"
    },
    {
        "id": "SS07", "query": "In the 2026 Construction Manual, what inspection requirements exist for bridge construction?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["bridge", "inspection", "chapter", "requirement"],
        "notes": "Navigate to bridge chapter + extract inspection details"
    },
    {
        "id": "SS08", "query": "What does Section 401 say about the Superpave mix design method?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["401", "Superpave", "mix", "design"],
        "notes": "Navigate to section + extract method details"
    },
    {
        "id": "SS09", "query": "What sampling requirements are listed in the Testing Manual for concrete cylinders?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["sampling", "cylinder", "concrete", "frequency"],
        "notes": "Find section + extract requirements"
    },
    {
        "id": "SS10", "query": "In what section are geotextile requirements found, and what AASHTO specs apply?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["geotextile", "section", "AASHTO"],
        "notes": "Locate topic + extract referenced standards"
    },
    {
        "id": "SS11", "query": "What does the Standard Spec say about aggregate requirements, and what test does the Testing Manual prescribe for gradation?",
        "type": "cross_document", "alpha": 0.4, "beta": 0.6, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["aggregate", "gradation", "sieve", "AASHTO"],
        "notes": "Cross-document: Spec (requirements) + Testing Manual (test method)"
    },
    {
        "id": "SS12", "query": "Compare the concrete specifications in the Standard Specs with the testing procedures in the Materials Testing Manual.",
        "type": "cross_document", "alpha": 0.4, "beta": 0.6, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["concrete", "test", "specification", "procedure"],
        "notes": "Cross-document synthesis"
    },
    {
        "id": "SS13", "query": "How does the Construction Manual's earthwork chapter relate to the Standard Spec Division 200?",
        "type": "cross_document", "alpha": 0.4, "beta": 0.6, "gamma": 0.0,
        "difficulty": "hard",
        "source_docs": ["WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["earthwork", "division 200", "construction manual"],
        "notes": "Cross-document: Manual procedures vs Spec requirements"
    },
    {
        "id": "SS14", "query": "What construction inspection procedures does the manual describe for HMA paving?",
        "type": "structural_semantic", "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "difficulty": "medium",
        "source_docs": ["2025 Construction Manual.pdf"],
        "gold_answer_keywords": ["HMA", "inspection", "paving", "procedure"],
        "notes": "Navigate chapter + extract procedures"
    },

    # ══════════════════════════════════════════════════════
    # TYPE 5: SEMANTIC + TEMPORAL (beta+gamma) — 14 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "ST01", "query": "Did the compressive strength requirement for Class A concrete change between 2010 and 2021?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["compressive", "strength", "changed", "psi"],
        "notes": "Entity property comparison across editions"
    },
    {
        "id": "ST02", "query": "Were any new cement types added to the approved list between 2010 and 2021 specs?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["cement", "type IL", "added", "new"],
        "notes": "Entity addition detection across editions"
    },
    {
        "id": "ST03", "query": "How did aggregate gradation requirements evolve from the 2010 to 2021 Standard Specifications?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["gradation", "aggregate", "changed", "sieve"],
        "notes": "Material specification evolution"
    },
    {
        "id": "ST04", "query": "Were any new test methods introduced in the Testing Manual between 2020 and 2023?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual  FY 2020.pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["new", "test", "method", "added"],
        "notes": "TestMethod entity addition across editions"
    },
    {
        "id": "ST05", "query": "Did the AASHTO standard references change between the 2010 and 2021 specifications?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["AASHTO", "reference", "updated", "edition"],
        "notes": "Standard entity version tracking"
    },
    {
        "id": "ST06", "query": "How have embankment compaction requirements evolved across Construction Manual editions (2018-2026)?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2018 Construction Manual.pdf", "2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["embankment", "compaction", "changed", "updated"],
        "notes": "Long-range temporal evolution of a concept"
    },
    {
        "id": "ST07", "query": "Were any materials removed from the approved list between 2010 and 2021?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["removed", "material", "deprecated"],
        "notes": "Entity removal detection"
    },
    {
        "id": "ST08", "query": "How did concrete mix design requirements change in the Construction Manual from 2020 to 2024?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2020 Construction Manual.pdf", "2024 Construction Manual.pdf"],
        "gold_answer_keywords": ["concrete", "mix", "design", "change"],
        "notes": "Concept evolution in construction manuals"
    },
    {
        "id": "ST09", "query": "Have nuclear density testing procedures been updated across Testing Manual editions?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual  FY 2020.pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["nuclear", "density", "updated", "procedure"],
        "notes": "TestMethod evolution tracking"
    },
    {
        "id": "ST10", "query": "Were there any changes to fly ash specifications between the 2010 and 2021 specs?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["fly ash", "changed", "specification"],
        "notes": "Specific material change tracking"
    },
    {
        "id": "ST11", "query": "What equipment requirements have been updated in recent Testing Manual editions?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual  FY 2020.pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["equipment", "updated", "requirement"],
        "notes": "Equipment entity evolution"
    },
    {
        "id": "ST12", "query": "Has the water-cement ratio limit changed across specification editions?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["water", "cement", "ratio", "change"],
        "notes": "Specific value tracking across editions"
    },
    {
        "id": "ST13", "query": "What new forms or documentation requirements were added between 2019 and 2023 Construction Manuals?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2019 Construction Manual.pdf", "2023 Construction Manual.pdf"],
        "gold_answer_keywords": ["form", "documentation", "new", "added"],
        "notes": "Form entity addition tracking"
    },
    {
        "id": "ST14", "query": "Have Superpave mix design parameters been updated since the 2010 specifications?",
        "type": "semantic_temporal", "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["Superpave", "mix design", "updated", "parameter"],
        "notes": "Complex method evolution"
    },

    # ══════════════════════════════════════════════════════
    # TYPE 6: STRUCTURAL + TEMPORAL (alpha+gamma) — 14 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "AT01", "query": "Was Division 500 reorganized between the 2010 and 2021 Standard Specifications?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["division 500", "reorganized", "section", "structure"],
        "notes": "Structural comparison across editions"
    },
    {
        "id": "AT02", "query": "Were any new divisions added to the 2021 Standard Specifications that didn't exist in 2010?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["new", "division", "added"],
        "notes": "Top-level structural change detection"
    },
    {
        "id": "AT03", "query": "How has the chapter structure of the Construction Manual evolved from 2018 to 2026?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2018 Construction Manual.pdf", "2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["chapter", "structure", "changed", "reorganized"],
        "notes": "Long-range structural evolution"
    },
    {
        "id": "AT04", "query": "Were any sections removed from the Standard Specifications between 2010 and 2021?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["removed", "section", "deleted"],
        "notes": "Section removal detection"
    },
    {
        "id": "AT05", "query": "Has the Testing Manual's organization changed between FY 2020 and 2023?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual  FY 2020.pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["organization", "changed", "testing manual"],
        "notes": "Testing Manual structural comparison"
    },
    {
        "id": "AT06", "query": "Compare the table of contents of the 2020 and 2025 Construction Manuals.",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2020 Construction Manual.pdf", "2025 Construction Manual.pdf"],
        "gold_answer_keywords": ["chapter", "contents", "compare"],
        "notes": "Direct TOC comparison"
    },
    {
        "id": "AT07", "query": "Were any new subsections added to Section 501 between the 2010 and 2021 specs?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["subsection", "501", "new", "added"],
        "notes": "Fine-grained structural change"
    },
    {
        "id": "AT08", "query": "Did the numbering scheme change between the 2010 and 2021 Standard Specifications?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["numbering", "section", "changed"],
        "notes": "Numbering scheme evolution"
    },
    {
        "id": "AT09", "query": "How many more pages does the 2026 Construction Manual have compared to 2018?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "easy",
        "source_docs": ["2018 Construction Manual.pdf", "2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["pages", "2018", "2026"],
        "notes": "Quantitative size comparison"
    },
    {
        "id": "AT10", "query": "Were chapters reordered in the Construction Manual between 2022 and 2024?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2022 Construction Manual.pdf", "2024 Construction Manual.pdf"],
        "gold_answer_keywords": ["chapter", "reordered", "reorganized"],
        "notes": "Chapter order comparison"
    },
    {
        "id": "AT11", "query": "How did the scope of Division 800 change between 2010 and 2021?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["division 800", "scope", "changed"],
        "notes": "Division scope evolution"
    },
    {
        "id": "AT12", "query": "Was the Testing Manual restructured when it changed from FY 2021 to January 2023 effective date?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "medium",
        "source_docs": ["Materials Testing Manual FY  2021 (Updated).pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["restructured", "testing", "manual"],
        "notes": "Testing Manual restructuring"
    },
    {
        "id": "AT13", "query": "Were any sections renumbered between the 2010 and 2021 Standard Specifications?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["renumbered", "section"],
        "notes": "Section renumbering detection"
    },
    {
        "id": "AT14", "query": "What new sections appear in the 2021 specs that have no equivalent in 2010?",
        "type": "structural_temporal", "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["new", "section", "no equivalent"],
        "notes": "Completely new section detection"
    },

    # ══════════════════════════════════════════════════════
    # TYPE 7: FULL 3D (alpha+beta+gamma) — 12 queries
    # ══════════════════════════════════════════════════════
    {
        "id": "F01", "query": "How did the concrete mix design requirements in Section 501 change between 2010 and 2021, and what new test standards are now referenced?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["section 501", "concrete", "changed", "test", "standard"],
        "notes": "Navigate section (α) + extract entities (β) + compare editions (γ)"
    },
    {
        "id": "F02", "query": "What materials in Division 300 were updated between 2010 and 2021, and how do current testing procedures validate them?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["division 300", "material", "updated", "testing"],
        "notes": "Full 3D + cross-document"
    },
    {
        "id": "F03", "query": "In the hot mix asphalt section, what specification changes occurred between 2010 and 2021, and where in the Construction Manual are the updated field procedures?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "2025 Construction Manual.pdf"],
        "gold_answer_keywords": ["HMA", "asphalt", "changed", "construction manual", "field"],
        "notes": "3D + cross-document"
    },
    {
        "id": "F04", "query": "How have the structural steel inspection requirements evolved across the Standard Specs (2010→2021) and Construction Manuals (2018→2026)?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "2018 Construction Manual.pdf", "2026 Construction Manual.pdf"],
        "gold_answer_keywords": ["structural steel", "inspection", "evolved", "changed"],
        "notes": "Multi-document temporal + structural + semantic"
    },
    {
        "id": "F05", "query": "What section covers bridge deck concrete, what are the current requirements, and how have they changed since 2010?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["bridge deck", "section", "requirement", "changed"],
        "notes": "Navigate + entity details + temporal comparison"
    },
    {
        "id": "F06", "query": "Find the earthwork section, list current compaction standards, and identify what changed from 2010 to 2021.",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["earthwork", "compaction", "standard", "changed"],
        "notes": "Full 3D retrieval"
    },
    {
        "id": "F07", "query": "How has the guardrail specification section been modified between editions, and what are the current material requirements?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["guardrail", "modified", "material", "requirement"],
        "notes": "Full 3D for safety-critical infrastructure"
    },
    {
        "id": "F08", "query": "Locate the painting specifications, determine current paint system requirements, and identify changes from the 2010 edition.",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["painting", "paint", "system", "changed"],
        "notes": "Navigate + entity + temporal"
    },
    {
        "id": "F09", "query": "What drainage pipe materials are specified, how has the approved list evolved since 2010, and what tests does the current Testing Manual require?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "Materials Testing Manual Effective_Jan_2023.pdf"],
        "gold_answer_keywords": ["drainage", "pipe", "material", "test", "changed"],
        "notes": "3D + cross-document retrieval"
    },
    {
        "id": "F10", "query": "In what section is reinforcing steel covered, what are the 2021 specs, and how do they differ from 2010?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["reinforcing", "steel", "section", "2021", "2010", "differ"],
        "notes": "Full 3D retrieval"
    },
    {
        "id": "F11", "query": "Trace the evolution of concrete curing requirements across both the Standard Specs (2010→2021) and Construction Manuals (2020→2025).",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "2020 Construction Manual.pdf", "2025 Construction Manual.pdf"],
        "gold_answer_keywords": ["curing", "concrete", "evolution", "changed"],
        "notes": "Multi-document full 3D"
    },
    {
        "id": "F12", "query": "What section addresses erosion control, what current BMPs are specified, and what was different in the 2010 edition?",
        "type": "full_3d", "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "difficulty": "hard",
        "source_docs": ["2010 Standard Specifications.pdf", "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf"],
        "gold_answer_keywords": ["erosion", "control", "BMP", "section", "changed"],
        "notes": "Full 3D environmental topic"
    },
]

# Verify query count
assert len(SPECQA_QUERIES) == 100, f"Expected 100 queries, got {len(SPECQA_QUERIES)}"


def export_benchmark():
    """Export SpecQA queries to JSON."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARK_DIR / "specqa_queries.json"
    with open(output_path, "w") as f:
        json.dump(SPECQA_QUERIES, f, indent=2)
    logger.info(f"Exported {len(SPECQA_QUERIES)} queries to {output_path}")
    return output_path


def print_stats():
    """Print distribution statistics."""
    from collections import Counter

    type_counts = Counter(q["type"] for q in SPECQA_QUERIES)
    difficulty_counts = Counter(q["difficulty"] for q in SPECQA_QUERIES)
    dim_counts = Counter()
    for q in SPECQA_QUERIES:
        dims = []
        if q["alpha"] > 0.1: dims.append("α")
        if q["beta"] > 0.1: dims.append("β")
        if q["gamma"] > 0.1: dims.append("γ")
        dim_counts["+".join(dims)] += 1

    print(f"\n{'='*60}")
    print(f"SpecQA BENCHMARK STATISTICS")
    print(f"{'='*60}")
    print(f"Total queries: {len(SPECQA_QUERIES)}")

    print(f"\nBy Type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<25} {c:>3}")

    print(f"\nBy Difficulty:")
    for d, c in sorted(difficulty_counts.items()):
        print(f"  {d:<25} {c:>3}")

    print(f"\nBy Active Dimensions:")
    for d, c in sorted(dim_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:<25} {c:>3}")

    # Dimension requirement distribution
    dim_1 = sum(1 for q in SPECQA_QUERIES if sum(1 for w in [q['alpha'], q['beta'], q['gamma']] if w > 0.1) == 1)
    dim_2 = sum(1 for q in SPECQA_QUERIES if sum(1 for w in [q['alpha'], q['beta'], q['gamma']] if w > 0.1) == 2)
    dim_3 = sum(1 for q in SPECQA_QUERIES if sum(1 for w in [q['alpha'], q['beta'], q['gamma']] if w > 0.1) == 3)
    print(f"\nBy Dimension Count:")
    print(f"  1D queries: {dim_1}")
    print(f"  2D queries: {dim_2}")
    print(f"  3D queries: {dim_3}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpecQA Benchmark")
    parser.add_argument("--generate", action="store_true", help="Generate/export queries")
    parser.add_argument("--export", action="store_true", help="Export to JSON")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()

    if args.generate or args.export:
        export_benchmark()
    if args.stats or not any([args.generate, args.export]):
        print_stats()
