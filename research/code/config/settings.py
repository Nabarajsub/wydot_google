"""
SpecRAG Configuration
Central config for all phases of the SpecRAG pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent  # research/code/
RESEARCH_ROOT = PROJECT_ROOT.parent          # research/
DATA_DIR = RESEARCH_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PAGEINDEX_DIR = OUTPUT_DIR / "pageindex_trees"
TEMPORAL_DIR = OUTPUT_DIR / "temporal_diffs"
BENCHMARK_DIR = OUTPUT_DIR / "benchmark"
RESULTS_DIR = OUTPUT_DIR / "results"

# ─── API Keys ────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ─── Models ──────────────────────────────────────────────
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMS = 768

# ─── Corpus Definition ───────────────────────────────────
# All 19 documents organized by type and temporal order
CORPUS = {
    "standard_specs": {
        "type": "Standard Specification",
        "series": "WYDOT Standard Specifications",
        "documents": [
            {"file": "2010 Standard Specifications.pdf", "year": 2010, "pages": 890},
            {"file": "WYDOT_Standard_Specifications_–_2021_Edition_(Road_&_Bridge_Construction).pdf", "year": 2021, "pages": 771},
        ]
    },
    "construction_manuals": {
        "type": "Construction Manual",
        "series": "WYDOT Construction Manual",
        "documents": [
            {"file": "2018 Construction Manual.pdf", "year": 2018, "pages": 374},
            {"file": "2019 Construction Manual.pdf", "year": 2019, "pages": 451},
            {"file": "2020 Construction Manual.pdf", "year": 2020, "pages": 456},
            {"file": "2022 Construction Manual.pdf", "year": 2022, "pages": 447},
            {"file": "2023 Construction Manual.pdf", "year": 2023, "pages": 445},
            {"file": "2024 Construction Manual.pdf", "year": 2024, "pages": 447},
            {"file": "2025 Construction Manual.pdf", "year": 2025, "pages": 381},
            {"file": "2026 Construction Manual.pdf", "year": 2026, "pages": 368},
        ]
    },
    "testing_manuals": {
        "type": "Materials Testing Manual",
        "series": "WYDOT Materials Testing Manual",
        "documents": [
            {"file": "Materials Testing Manual  FY 2020.pdf", "year": 2020, "pages": 401},
            {"file": "Materials Testing Manual FY  2021 (Updated).pdf", "year": 2021, "pages": 417},
            {"file": "Materials Testing Manual Effective_Jan_2023.pdf", "year": 2023, "pages": 433},
        ]
    },
    "change_summaries": {
        "type": "Summary of Changes",
        "series": "WYDOT Change Summary",
        "documents": [
            {"file": "Summary of Changes 2018 Construction Manual.pdf", "year": 2018, "pages": 4},
            {"file": "Summary of Changes 2019 Construction Manual.pdf", "year": 2019, "pages": 2},
            {"file": "Summary of Changes 2020 Construction Manual.pdf", "year": 2020, "pages": 4},
            {"file": "Summary of Changes 2021 Construction Manual.pdf", "year": 2021, "pages": 2},
            {"file": "2022 Construction Manual Revision Summary.pdf", "year": 2022, "pages": 1},
        ]
    }
}

# ─── Temporal Chains (Edition Pairs for Diffing) ─────────
TEMPORAL_CHAINS = {
    "standard_specs": [(2010, 2021)],
    "construction_manuals": [
        (2018, 2019), (2019, 2020), (2020, 2022),
        (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)
    ],
    "testing_manuals": [(2020, 2021), (2021, 2023)],
}

# ─── Query Types for SpecQA ──────────────────────────────
QUERY_TYPES = {
    "structural":        {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "label": "α"},
    "semantic":          {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "label": "β"},
    "temporal":          {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "label": "γ"},
    "structural_semantic":  {"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "label": "α+β"},
    "semantic_temporal":    {"alpha": 0.0, "beta": 0.5, "gamma": 0.5, "label": "β+γ"},
    "structural_temporal":  {"alpha": 0.5, "beta": 0.0, "gamma": 0.5, "label": "α+γ"},
    "full_3d":              {"alpha": 0.33, "beta": 0.34, "gamma": 0.33, "label": "α+β+γ"},
    "cross_document":       {"alpha": 0.4, "beta": 0.6, "gamma": 0.0, "label": "α+β (cross-doc)"},
}

# ─── Neo4j Cypher Templates ──────────────────────────────
CYPHER_VECTOR_SEARCH = """
CALL db.index.vector.queryNodes('wydot_gemini_index', $k, $embedding)
YIELD node, score
RETURN node.text AS text, node.page AS page, node.source AS source,
       node.section AS section, score
ORDER BY score DESC
"""

CYPHER_ENTITY_SEARCH = """
MATCH (e {name: $entity_name})
OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
RETURN e.name AS entity, labels(e) AS types,
       collect(DISTINCT {text: c.text, page: c.page, source: c.source}) AS mentions
LIMIT 10
"""

CYPHER_TEMPORAL_SEARCH = """
MATCH (d1:Document)-[:SUPERSEDES]->(d2:Document)
WHERE d1.document_series = $series
RETURN d1.display_title AS newer, d1.year AS newer_year,
       d2.display_title AS older, d2.year AS older_year
ORDER BY d1.year DESC
"""

CYPHER_SECTION_CHUNKS = """
MATCH (d:Document {source: $source})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
WHERE s.name CONTAINS $section_name
RETURN c.text AS text, c.page AS page, c.seq AS seq, s.name AS section
ORDER BY c.seq
"""

# ─── Rate Limiting ────────────────────────────────────────
GEMINI_RPM = 15           # requests per minute for free tier
GEMINI_BATCH_SIZE = 50     # embeddings per batch
GEMINI_BATCH_DELAY = 0.5   # seconds between batches
MAX_RETRIES = 8
BASE_RETRY_DELAY = 2.0     # seconds
