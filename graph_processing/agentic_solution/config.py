"""
Centralized configuration for the WYDOT Multi-Agent system.
Loads from .env in parent directory.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from graph_processing/
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ── Neo4j ──
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1c9edfe6.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "1c9edfe6")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "")

# ── Gemini ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

# ── Search defaults ──
DEFAULT_VECTOR_K = 30          # candidates from vector index
DEFAULT_SCOPED_LIMIT = 15      # final chunks returned per agent search
DEFAULT_FULLTEXT_LIMIT = 10    # fulltext hits per search

# ── Agent document-series filters ──
# Maps agent name → list of Neo4j WHERE clauses for document_series
AGENT_SERIES_FILTERS = {
    "specs_agent": [
        "d.document_series CONTAINS 'Standard Spec'",
        "d.display_title CONTAINS 'Standard Spec'",
        "d.document_series = 'Standard Specifications'",
        "d.document_series CONTAINS 'Standard Plan'",
    ],
    "construction_agent": [
        "d.document_series = 'Construction Manual'",
        "d.display_title CONTAINS 'Construction Manual'",
    ],
    "materials_agent": [
        "d.document_series = 'Materials Testing Manual'",
        "d.display_title CONTAINS 'Materials Testing Manual'",
    ],
    "design_agent": [
        "d.document_series = 'Design Manual'",
        "d.document_series = 'Road Design Manual'",
        "d.display_title CONTAINS 'Design Manual'",
        "d.document_series CONTAINS 'Design Guide'",
        "d.document_series CONTAINS 'WYDOT Design'",
    ],
    "safety_agent": [
        "d.document_series = 'Report on Traffic Crashes'",
        "d.display_title CONTAINS 'Traffic Crash'",
        "d.document_series CONTAINS 'Highway Safety'",
        "d.document_series CONTAINS 'Strategic Highway Safety'",
        "d.document_series CONTAINS 'Vulnerable Road'",
    ],
    "bridge_agent": [
        "d.document_series CONTAINS 'Bridge'",
        "d.display_title CONTAINS 'Bridge'",
        "d.document_series CONTAINS 'Approach Slab'",
    ],
    "planning_agent": [
        "d.document_series CONTAINS 'State Transportation Improvement'",
        "d.display_title CONTAINS 'STIP'",
        "d.document_series CONTAINS 'Corridor'",
        "d.document_series CONTAINS 'Long Range'",
    ],
    "admin_agent": [
        "d.document_series CONTAINS 'Annual Report'",
        "d.display_title CONTAINS 'Annual Report'",
        "d.document_series CONTAINS 'Financial'",
        "d.document_series CONTAINS 'Operating Budget'",
    ],
}
