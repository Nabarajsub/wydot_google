#!/usr/bin/env python3
"""
Pull corpus statistics from Neo4j for paper Table 1 and other data tables.
Outputs JSON that can be used by fill_paper.py.

Usage:
    cd graph_processing/
    python evaluation/neo4j_stats.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_solution.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    AGENT_SERIES_FILTERS,
)
from neo4j import GraphDatabase


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def db_kwargs():
    return {"database": NEO4J_DATABASE} if NEO4J_DATABASE else {}


def get_corpus_stats(driver):
    """Get document and chunk counts per document_series category."""
    with driver.session(**db_kwargs()) as session:
        result = session.run("""
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.document_series AS series,
                   count(DISTINCT d) AS doc_count,
                   count(DISTINCT c) AS chunk_count
            ORDER BY chunk_count DESC
        """)
        return [dict(r) for r in result]


def get_node_relationship_counts(driver):
    """Get total node and relationship counts."""
    with driver.session(**db_kwargs()) as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        return {"nodes": nodes, "relationships": rels}


def get_agent_scope_sizes(driver):
    """Get chunk count per agent scope (using the actual Cypher filters from config)."""
    results = {}
    with driver.session(**db_kwargs()) as session:
        for agent_name, filters in AGENT_SERIES_FILTERS.items():
            where = " OR ".join(f"({f})" for f in filters)
            cypher = f"""
                MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
                WHERE {where}
                RETURN count(DISTINCT c) AS chunk_count, count(DISTINCT d) AS doc_count
            """
            rec = session.run(cypher).single()
            results[agent_name] = {
                "chunks": rec["chunk_count"],
                "docs": rec["doc_count"],
            }

        # Total
        rec = session.run("""
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
            RETURN count(DISTINCT c) AS total_chunks, count(DISTINCT d) AS total_docs
        """).single()
        results["total"] = {
            "chunks": rec["total_chunks"],
            "docs": rec["total_docs"],
        }
    return results


def categorize_series(raw_stats):
    """Map raw document_series values into paper categories."""
    category_map = {
        "Standard Specs": ["Standard Specifications", "Standard Plans", "Standard Spec"],
        "Construction Manual": ["Construction Manual"],
        "Materials Testing": ["Materials Testing Manual"],
        "Design Manual": ["Design Manual", "Road Design Manual", "Design Guide", "WYDOT Design"],
        "Traffic & Crashes": ["Report on Traffic Crashes", "Traffic Crash", "Highway Safety",
                              "Strategic Highway Safety", "Vulnerable Road"],
        "STIP": ["State Transportation Improvement", "STIP", "Corridor", "Long Range"],
        "Annual Reports": ["Annual Report", "Financial", "Operating Budget"],
        "Bridge Program": ["Bridge", "Approach Slab"],
        "Highway Safety": ["Highway Safety", "Strategic Highway Safety", "Vulnerable Road"],
    }

    categories = {}
    assigned = set()

    for cat_name, keywords in category_map.items():
        categories[cat_name] = {"docs": 0, "chunks": 0}
        for stat in raw_stats:
            series = stat["series"] or "Unknown"
            if any(kw.lower() in series.lower() for kw in keywords):
                if stat["series"] not in assigned:
                    categories[cat_name]["docs"] += stat["doc_count"]
                    categories[cat_name]["chunks"] += stat["chunk_count"]
                    assigned.add(stat["series"])

    # "Other" = everything not assigned
    categories["Other"] = {"docs": 0, "chunks": 0}
    for stat in raw_stats:
        if stat["series"] not in assigned:
            categories["Other"]["docs"] += stat["doc_count"]
            categories["Other"]["chunks"] += stat["chunk_count"]

    return categories


def main():
    print("Connecting to Neo4j...")
    driver = get_driver()

    print("\n1. Corpus Statistics (raw series):")
    raw_stats = get_corpus_stats(driver)
    for s in raw_stats:
        print(f"   {s['series']:50s}  docs={s['doc_count']:>4d}  chunks={s['chunk_count']:>6d}")

    print("\n2. Categorized for paper Table 1:")
    categories = categorize_series(raw_stats)
    total_docs = sum(c["docs"] for c in categories.values())
    total_chunks = sum(c["chunks"] for c in categories.values())

    table_data = {}
    for cat, data in categories.items():
        pct = (data["chunks"] / total_chunks * 100) if total_chunks else 0
        chk_doc = (data["chunks"] / data["docs"]) if data["docs"] else 0
        table_data[cat] = {
            "docs": data["docs"],
            "chunks": data["chunks"],
            "pct": round(pct, 1),
            "chk_per_doc": round(chk_doc, 0),
        }
        print(f"   {cat:25s}  docs={data['docs']:>4d}  chunks={data['chunks']:>6d}  "
              f"pct={pct:>5.1f}%  chk/doc={chk_doc:>6.0f}")

    print(f"\n   {'TOTAL':25s}  docs={total_docs:>4d}  chunks={total_chunks:>6d}")

    print("\n3. Node and Relationship counts:")
    counts = get_node_relationship_counts(driver)
    print(f"   Nodes: {counts['nodes']:,}")
    print(f"   Relationships: {counts['relationships']:,}")

    print("\n4. Agent Scope Sizes (for Table 3 — search space reduction):")
    scopes = get_agent_scope_sizes(driver)
    total_c = scopes["total"]["chunks"]
    scope_data = {}
    for agent, data in scopes.items():
        if agent == "total":
            continue
        reduction = (1 - data["chunks"] / total_c) * 100 if total_c else 0
        scope_data[agent] = {
            "chunks": data["chunks"],
            "docs": data["docs"],
            "reduction_pct": round(reduction, 1),
        }
        print(f"   {agent:25s}  chunks={data['chunks']:>6d}  "
              f"reduction={reduction:>5.1f}%")

    # Save all data
    output = {
        "table1_categories": table_data,
        "node_counts": counts,
        "agent_scopes": scope_data,
        "total_docs": total_docs,
        "total_chunks": total_chunks,
    }

    out_path = os.path.join(os.path.dirname(__file__), "neo4j_stats.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {out_path}")

    driver.close()


if __name__ == "__main__":
    main()
