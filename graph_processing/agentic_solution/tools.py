"""
Neo4j search tools used by all agents.
Each tool searches a scoped subset of the knowledge graph.
"""
import re
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    GEMINI_API_KEY, GEMINI_EMBED_MODEL,
    DEFAULT_VECTOR_K, DEFAULT_SCOPED_LIMIT, DEFAULT_FULLTEXT_LIMIT,
    AGENT_SERIES_FILTERS,
)


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def _db_kwargs():
    """Return database kwarg for session — Aura Free uses no database name."""
    return {"database": NEO4J_DATABASE} if NEO4J_DATABASE else {}


def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{GEMINI_EMBED_MODEL}",
        google_api_key=GEMINI_API_KEY,
    )


def _build_where_clause(agent_name: str) -> str:
    """Build a Neo4j WHERE clause from the agent's series filters."""
    filters = AGENT_SERIES_FILTERS.get(agent_name, [])
    if not filters:
        return "TRUE"
    return " OR ".join(f"({f})" for f in filters)


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query for fulltext search."""
    stop = {
        'what', 'which', 'where', 'when', 'how', 'who', 'does', 'did', 'the',
        'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
        'for', 'and', 'but', 'not', 'this', 'that', 'with', 'from', 'about',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'than', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'only', 'same', 'also', 'tell', 'show',
        'describe', 'explain', 'compare', 'list', 'give', 'find', 'need',
        'requirements', 'required', 'requirement', 'according', 'specific',
        'used', 'using', 'does', 'must', 'shall', 'should', 'would',
    }
    words = re.sub(r'[^\w\s]', ' ', query).split()
    keywords = [w for w in words if len(w) > 2 and w.lower() not in stop]
    return keywords[:8]


# ═══════════════════════════════════════════════════════════════
#  SCOPED SEARCH — the core tool used by every agent
# ═══════════════════════════════════════════════════════════════

def scoped_vector_search(
    query: str,
    agent_name: str,
    year: Optional[int] = None,
    section: Optional[str] = None,
    limit: int = DEFAULT_SCOPED_LIMIT,
) -> List[Dict]:
    """
    Vector similarity search scoped to a specific agent's document domain.
    Returns list of dicts with: text, source, title, year, section, page, score.
    """
    where_clause = _build_where_clause(agent_name)
    results = []
    seen_ids = set()

    try:
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)
        driver = _get_driver()

        with driver.session(**_db_kwargs()) as session:
            # Build year/section filters
            extra_filters = []
            params = {
                "k": DEFAULT_VECTOR_K * 3,
                "emb": query_embedding,
                "limit": limit,
            }
            if year:
                extra_filters.append("d.year = $year")
                params["year"] = year
            if section:
                extra_filters.append(f"s.name CONTAINS $section_filter")
                params["section_filter"] = section

            year_section_clause = (" AND " + " AND ".join(extra_filters)) if extra_filters else ""

            cypher = f"""
            CALL db.index.vector.queryNodes('wydot_gemini_index', $k, $emb)
            YIELD node, score
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
            WHERE ({where_clause}){year_section_clause}
            RETURN node.id AS id, node.text AS text, d.source AS source,
                   d.display_title AS title, d.year AS year,
                   s.name AS section, node.page AS page, score
            ORDER BY score DESC LIMIT $limit
            """
            for rec in session.run(cypher, **params):
                cid = rec["id"]
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append({
                        "id": cid,
                        "text": rec["text"],
                        "source": rec["source"],
                        "title": rec["title"],
                        "year": rec["year"],
                        "section": rec["section"],
                        "page": rec["page"],
                        "score": rec["score"],
                        "search_type": "scoped_vector",
                    })

        driver.close()
    except Exception as e:
        print(f"    [scoped_vector_search] Error: {e}")

    return results


def scoped_fulltext_search(
    query: str,
    agent_name: str,
    year: Optional[int] = None,
    limit: int = DEFAULT_FULLTEXT_LIMIT,
) -> List[Dict]:
    """
    Fulltext keyword search scoped to a specific agent's document domain.
    """
    where_clause = _build_where_clause(agent_name)
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    ft_query = " ".join(keywords)
    results = []
    seen_ids = set()

    try:
        driver = _get_driver()
        with driver.session(**_db_kwargs()) as session:
            year_filter = " AND d.year = $year" if year else ""
            params = {"ftq": ft_query, "limit": limit}
            if year:
                params["year"] = year

            cypher = f"""
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $ftq)
            YIELD node, score
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
            WHERE ({where_clause}){year_filter}
            RETURN node.id AS id, node.text AS text, d.source AS source,
                   d.display_title AS title, d.year AS year,
                   s.name AS section, node.page AS page, score
            ORDER BY score DESC LIMIT $limit
            """
            for rec in session.run(cypher, **params):
                cid = rec["id"]
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append({
                        "id": cid,
                        "text": rec["text"],
                        "source": rec["source"],
                        "title": rec["title"],
                        "year": rec["year"],
                        "section": rec["section"],
                        "page": rec["page"],
                        "score": rec["score"],
                        "search_type": "scoped_fulltext",
                    })

        driver.close()
    except Exception as e:
        print(f"    [scoped_fulltext_search] Error: {e}")

    return results


def get_section_content(
    section_number: str,
    agent_name: str,
    year: Optional[int] = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Retrieve all chunks from a specific section (e.g., '414', '506.4.4').
    """
    where_clause = _build_where_clause(agent_name)
    results = []

    try:
        driver = _get_driver()
        with driver.session(**_db_kwargs()) as session:
            year_filter = " AND d.year = $year" if year else ""
            params = {"sec": section_number, "limit": limit}
            if year:
                params["year"] = year

            cypher = f"""
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
            WHERE ({where_clause}){year_filter}
              AND s.name CONTAINS $sec
            RETURN c.id AS id, c.text AS text, d.source AS source,
                   d.display_title AS title, d.year AS year,
                   s.name AS section, c.page AS page
            ORDER BY c.page, c.id
            LIMIT $limit
            """
            for rec in session.run(cypher, **params):
                results.append({
                    "id": rec["id"],
                    "text": rec["text"],
                    "source": rec["source"],
                    "title": rec["title"],
                    "year": rec["year"],
                    "section": rec["section"],
                    "page": rec["page"],
                    "search_type": "section_lookup",
                })

        driver.close()
    except Exception as e:
        print(f"    [get_section_content] Error: {e}")

    return results


def compare_versions(
    topic: str,
    agent_name: str,
    year_old: int = 2010,
    year_new: int = 2021,
    limit: int = 10,
) -> Dict:
    """
    Search for a topic in two different versions and return both for comparison.
    """
    old_results = scoped_vector_search(topic, agent_name, year=year_old, limit=limit)
    old_fulltext = scoped_fulltext_search(topic, agent_name, year=year_old, limit=5)
    new_results = scoped_vector_search(topic, agent_name, year=year_new, limit=limit)
    new_fulltext = scoped_fulltext_search(topic, agent_name, year=year_new, limit=5)

    # Merge and deduplicate
    def _merge(vec, ft):
        seen = {r["id"] for r in vec}
        merged = list(vec)
        for r in ft:
            if r["id"] not in seen:
                seen.add(r["id"])
                merged.append(r)
        return merged

    return {
        "old_version": {
            "year": year_old,
            "chunks": _merge(old_results, old_fulltext),
        },
        "new_version": {
            "year": year_new,
            "chunks": _merge(new_results, new_fulltext),
        },
    }


def global_search(
    query: str,
    year: Optional[int] = None,
    limit: int = DEFAULT_SCOPED_LIMIT,
) -> List[Dict]:
    """
    Global vector + fulltext search across ALL documents.
    Used by the general agent as a fallback.
    """
    results = []
    seen_ids = set()

    try:
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(query)
        driver = _get_driver()

        with driver.session(**_db_kwargs()) as session:
            year_filter = " AND d.year = $year" if year else ""
            params = {
                "k": DEFAULT_VECTOR_K * 2,
                "emb": query_embedding,
                "limit": limit,
            }
            if year:
                params["year"] = year

            cypher = f"""
            CALL db.index.vector.queryNodes('wydot_gemini_index', $k, $emb)
            YIELD node, score
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(node)
            WHERE TRUE{year_filter}
            RETURN node.id AS id, node.text AS text, d.source AS source,
                   d.display_title AS title, d.year AS year,
                   s.name AS section, node.page AS page, score
            ORDER BY score DESC LIMIT $limit
            """
            for rec in session.run(cypher, **params):
                cid = rec["id"]
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append({
                        "id": cid,
                        "text": rec["text"],
                        "source": rec["source"],
                        "title": rec["title"],
                        "year": rec["year"],
                        "section": rec["section"],
                        "page": rec["page"],
                        "score": rec["score"],
                        "search_type": "global_vector",
                    })

        driver.close()
    except Exception as e:
        print(f"    [global_search] Error: {e}")

    return results
