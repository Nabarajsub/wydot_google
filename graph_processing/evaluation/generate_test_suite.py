#!/usr/bin/env python3
"""
Generate a curated evaluation test suite for the WYDOT RAG paper.

Uses Gemini 2.5 Flash to generate domain-expert queries from sampled PDF chunks
in Neo4j, producing queries with ground-truth categories, relevant sections,
and reference answers.

Usage:
    cd graph_processing/
    python evaluation/generate_test_suite.py
"""
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_solution.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    GEMINI_API_KEY, AGENT_SERIES_FILTERS,
)
from neo4j import GraphDatabase
import google.generativeai as genai


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

QUERIES_PER_CATEGORY = 4     # ~4 per domain agent = ~36 single-domain
CROSS_DOMAIN_QUERIES = 8     # cross-domain queries
VERSION_COMPARISON_QUERIES = 5
SECTION_LOOKUP_QUERIES = 5
AMBIGUOUS_QUERIES = 7        # general/ambiguous
TOTAL_TARGET = 50

# Map agent names to paper categories
AGENT_TO_CATEGORY = {
    "specs_agent": "STANDARD_SPECS",
    "construction_agent": "CONSTRUCTION_MANUAL",
    "materials_agent": "MATERIALS_TESTING",
    "design_agent": "DESIGN_MANUAL",
    "safety_agent": "TRAFFIC_CRASHES",
    "bridge_agent": "BRIDGE_PROGRAM",
    "planning_agent": "STIP",
    "admin_agent": "ANNUAL_REPORT",
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def db_kwargs():
    return {"database": NEO4J_DATABASE} if NEO4J_DATABASE else {}


def sample_chunks_for_agent(driver, agent_name, n=5):
    """Sample N random chunks from an agent's document scope."""
    filters = AGENT_SERIES_FILTERS.get(agent_name, [])
    if not filters:
        return []

    where = " OR ".join(f"({f})" for f in filters)
    with driver.session(**db_kwargs()) as session:
        # Get total count first
        count_q = f"""
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
            WHERE {where} AND size(c.text) > 100
            RETURN count(c) AS cnt
        """
        total = session.run(count_q).single()["cnt"]
        if total == 0:
            return []

        # Random skip
        skips = random.sample(range(min(total, 500)), min(n, total, 500))
        chunks = []
        for skip in skips[:n]:
            q = f"""
                MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                WHERE {where} AND size(c.text) > 100
                RETURN c.text AS text, d.display_title AS title, d.year AS year,
                       s.name AS section, d.source AS source, d.document_series AS series
                SKIP $skip LIMIT 1
            """
            rec = session.run(q, skip=skip).single()
            if rec:
                chunks.append(dict(rec))

        return chunks


def generate_single_domain_queries(model, chunks, category, n=4):
    """Generate N single-domain queries from sampled chunks."""
    if not chunks:
        return []

    chunk_texts = []
    for i, c in enumerate(chunks[:5]):
        chunk_texts.append(
            f"--- Chunk {i+1} (from '{c['title']}', Section: {c['section']}, Year: {c['year']}) ---\n"
            f"{c['text'][:600]}\n"
        )

    prompt = f"""You are a WYDOT (Wyoming Department of Transportation) domain expert.
Given these document chunks from the {category} category, generate exactly {n} realistic
questions that a transportation engineer or WYDOT employee would ask.

CHUNKS:
{''.join(chunk_texts)}

REQUIREMENTS:
1. Questions should be specific and answerable from the chunk content.
2. Mix difficulty: some factual lookups, some requiring synthesis.
3. Do NOT mention "chunk" or "document" — ask naturally.
4. For each question, provide:
   - The query text
   - A short reference answer (1-3 sentences based on the chunks)
   - The relevant section name
   - The relevant document title

OUTPUT as JSON array:
[
  {{
    "query": "What is the maximum allowable ...",
    "reference_answer": "According to Section ..., the maximum is ...",
    "relevant_section": "Section 414.3",
    "relevant_title": "2021 Standard Specifications"
  }},
  ...
]

Return ONLY valid JSON, no markdown fences."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        queries = json.loads(text)
        # Add category metadata
        for q in queries:
            q["category"] = category
            q["query_type"] = "single_domain"
        return queries[:n]
    except Exception as e:
        print(f"  ❌ Error generating queries for {category}: {e}")
        return []


def generate_cross_domain_queries(model, n=8):
    """Generate cross-domain queries that span multiple categories."""
    prompt = f"""You are a WYDOT (Wyoming Department of Transportation) domain expert.
Generate exactly {n} cross-domain questions that would require searching MULTIPLE
document categories to answer properly.

WYDOT categories:
- STANDARD_SPECS: Construction specifications, tolerances, materials requirements
- CONSTRUCTION_MANUAL: Field inspection procedures, project administration
- MATERIALS_TESTING: Lab test procedures, sampling methods
- DESIGN_MANUAL: Road/bridge design standards
- TRAFFIC_CRASHES: Crash statistics, fatality data by county/year
- BRIDGE_PROGRAM: Bridge design, load ratings
- STIP: Transportation improvement program funding, planned projects
- ANNUAL_REPORT: Department accomplishments, leadership, budgets
- HIGHWAY_SAFETY: Safety programs, vulnerable road users

EXAMPLES of cross-domain queries:
- "How do the bridge design standards relate to the construction inspection procedures?" (DESIGN + CONSTRUCTION)
- "What safety improvements were funded in the 2023 STIP?" (SAFETY + STIP)

OUTPUT as JSON array:
[
  {{
    "query": "...",
    "reference_answer": "This requires information from both ... and ...",
    "categories_needed": ["STANDARD_SPECS", "CONSTRUCTION_MANUAL"],
    "query_type": "cross_domain"
  }},
  ...
]

Return ONLY valid JSON, no markdown fences."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        queries = json.loads(text)
        for q in queries:
            q["category"] = q.get("categories_needed", ["GENERAL"])[0]
            q["query_type"] = "cross_domain"
        return queries[:n]
    except Exception as e:
        print(f"  ❌ Error generating cross-domain queries: {e}")
        return []


def generate_version_comparison_queries(model, n=5):
    """Generate version comparison queries (2010 vs 2021 specs)."""
    prompt = f"""You are a WYDOT domain expert familiar with Standard Specifications.
Generate exactly {n} questions comparing the 2010 and 2021 Standard Specifications.

Topics to compare: aggregate gradation, concrete requirements, asphalt mix design,
surface grinding, guardrail specifications, culvert installation, contractor penalties,
insurance requirements, pile driving, welding procedures.

OUTPUT as JSON array:
[
  {{
    "query": "What changed in aggregate gradation requirements between 2010 and 2021?",
    "reference_answer": "The 2021 specs updated gradation bands for ...",
    "category": "STANDARD_SPECS",
    "query_type": "version_comparison"
  }},
  ...
]

Return ONLY valid JSON, no markdown fences."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        queries = json.loads(text)
        for q in queries:
            q["category"] = "STANDARD_SPECS"
            q["query_type"] = "version_comparison"
        return queries[:n]
    except Exception as e:
        print(f"  ❌ Error generating version comparison queries: {e}")
        return []


def generate_section_lookup_queries(model, n=5):
    """Generate section lookup queries."""
    sections = ["401", "414", "506", "703", "801", "203", "602", "701", "301", "106"]
    queries = []
    for sec in sections[:n]:
        queries.append({
            "query": f"What does Section {sec} of the Standard Specifications cover?",
            "reference_answer": f"Section {sec} covers...",
            "category": "STANDARD_SPECS",
            "query_type": "section_lookup",
            "relevant_section": f"Section {sec}",
        })
    return queries


def generate_ambiguous_queries(model, n=7):
    """Generate ambiguous/general queries that test the GENERAL fallback."""
    prompt = f"""Generate exactly {n} questions about WYDOT that are general or ambiguous —
they don't clearly fit into any specific document category.

Examples:
- "Who is the director of WYDOT?"
- "What does WYDOT stand for?"
- "How do I get a commercial driver's license in Wyoming?"
- "What are WYDOT's core values?"

OUTPUT as JSON array:
[
  {{
    "query": "...",
    "reference_answer": "...",
    "category": "GENERAL",
    "query_type": "ambiguous"
  }},
  ...
]

Return ONLY valid JSON, no markdown fences."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        queries = json.loads(text)
        for q in queries:
            q["category"] = "GENERAL"
            q["query_type"] = "ambiguous"
        return queries[:n]
    except Exception as e:
        print(f"  ❌ Error generating ambiguous queries: {e}")
        return []


def main():
    print("=" * 60)
    print("WYDOT Evaluation Test Suite Generator")
    print("=" * 60)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    driver = get_driver()

    all_queries = []

    # ── 1. Single-domain queries ──
    print("\n📋 Generating single-domain queries...")
    for agent_name, category in AGENT_TO_CATEGORY.items():
        print(f"  Sampling chunks for {category}...")
        chunks = sample_chunks_for_agent(driver, agent_name, n=5)
        print(f"    Got {len(chunks)} chunks")

        if chunks:
            print(f"  Generating {QUERIES_PER_CATEGORY} queries...")
            queries = generate_single_domain_queries(
                model, chunks, category, n=QUERIES_PER_CATEGORY
            )
            all_queries.extend(queries)
            print(f"    ✅ Generated {len(queries)} queries for {category}")
            time.sleep(1)  # Rate limiting

    # ── 2. Cross-domain queries ──
    print(f"\n📋 Generating {CROSS_DOMAIN_QUERIES} cross-domain queries...")
    cross = generate_cross_domain_queries(model, n=CROSS_DOMAIN_QUERIES)
    all_queries.extend(cross)
    print(f"  ✅ Generated {len(cross)} cross-domain queries")
    time.sleep(1)

    # ── 3. Version comparison queries ──
    print(f"\n📋 Generating {VERSION_COMPARISON_QUERIES} version comparison queries...")
    version = generate_version_comparison_queries(model, n=VERSION_COMPARISON_QUERIES)
    all_queries.extend(version)
    print(f"  ✅ Generated {len(version)} version comparison queries")
    time.sleep(1)

    # ── 4. Section lookup queries ──
    print(f"\n📋 Generating {SECTION_LOOKUP_QUERIES} section lookup queries...")
    section = generate_section_lookup_queries(model, n=SECTION_LOOKUP_QUERIES)
    all_queries.extend(section)
    print(f"  ✅ Generated {len(section)} section lookup queries")

    # ── 5. Ambiguous/general queries ──
    print(f"\n📋 Generating {AMBIGUOUS_QUERIES} ambiguous queries...")
    ambiguous = generate_ambiguous_queries(model, n=AMBIGUOUS_QUERIES)
    all_queries.extend(ambiguous)
    print(f"  ✅ Generated {len(ambiguous)} ambiguous queries")

    # ── Add IDs ──
    for i, q in enumerate(all_queries):
        q["id"] = f"q_{i+1:03d}"

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Total queries: {len(all_queries)}")
    by_type = {}
    for q in all_queries:
        t = q.get("query_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")

    by_cat = {}
    for q in all_queries:
        cat = q.get("category", "UNKNOWN")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nBy category:")
    for cat, c in sorted(by_cat.items()):
        print(f"  {cat}: {c}")

    # ── Save ──
    out_path = os.path.join(os.path.dirname(__file__), "test_suite.json")
    with open(out_path, "w") as f:
        json.dump(all_queries, f, indent=2)
    print(f"\n✅ Saved {len(all_queries)} queries to {out_path}")

    driver.close()


if __name__ == "__main__":
    main()
