"""
Phase 4: SpecRAG 3D Router — The Core System
==============================================
Unified 3-dimensional retrieval pipeline:

R(q) = alpha * Structural(q) + beta * Semantic(q) + gamma * Temporal(q)

Orchestrates all three retrieval dimensions, fuses context,
and generates grounded answers with citations.

Usage:
    python -m phase4_router.specrag_router --query "How did concrete specs change?"
    python -m phase4_router.specrag_router --interactive
"""
import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMS
)
from utils.gemini_client import gemini_generate, gemini_embed
from phase4_router.query_classifier import classify_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Activation threshold — dimension is used if weight > this
ACTIVATION_THRESHOLD = 0.1


# ─── Semantic Retriever (Beta Dimension) ──────────────────

def semantic_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic retrieval via Neo4j vector search + entity graph traversal.
    This is the SEMANTIC dimension (beta) of SpecRAG.
    """
    from neo4j import GraphDatabase

    # Generate query embedding
    query_embedding = gemini_embed([query], task_type="retrieval_query")[0]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []

    with driver.session() as session:
        # Vector search
        vector_results = session.run("""
            CALL db.index.vector.queryNodes('wydot_gemini_index', $k, $embedding)
            YIELD node, score
            RETURN node.text AS text, node.page AS page, node.source AS source,
                   node.section AS section, node.id AS chunk_id, score
            ORDER BY score DESC
        """, k=top_k, embedding=query_embedding)

        for record in vector_results:
            results.append({
                "text": record["text"],
                "page": record["page"],
                "file": record["source"],
                "section": record["section"],
                "chunk_id": record["chunk_id"],
                "score": record["score"],
                "dimension": "semantic",
                "retrieval_method": "vector_search",
            })

        # Entity-based expansion — for top chunks, get connected entities
        if results:
            top_chunk_ids = [r["chunk_id"] for r in results[:3]]
            for cid in top_chunk_ids:
                entity_results = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e)
                    OPTIONAL MATCH (e)-[r]->(e2)
                    WHERE type(r) <> 'MENTIONS'
                    RETURN e.name AS entity, labels(e) AS types,
                           type(r) AS rel_type, e2.name AS related_entity
                    LIMIT 20
                """, chunk_id=cid)

                entities = []
                for rec in entity_results:
                    entities.append({
                        "entity": rec["entity"],
                        "types": rec["types"],
                        "relation": rec["rel_type"],
                        "related": rec["related_entity"],
                    })

                if entities:
                    entity_text = "Related entities: " + "; ".join(
                        f"{e['entity']} ({'/'.join(e['types'])})"
                        + (f" --{e['relation']}--> {e['related']}" if e['relation'] else "")
                        for e in entities
                    )
                    results.append({
                        "text": entity_text,
                        "file": results[0]["file"],
                        "page": 0,
                        "section": "Entity Graph",
                        "dimension": "semantic",
                        "retrieval_method": "entity_graph_expansion",
                    })

        # Neighbor expansion — get adjacent chunks for top results
        for r in results[:3]:
            if r.get("chunk_id"):
                neighbor_results = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:NEXT_CHUNK]->(next:Chunk)
                    RETURN next.text AS text, next.page AS page, next.source AS source,
                           next.section AS section
                    LIMIT 2
                """, chunk_id=r["chunk_id"])

                for rec in neighbor_results:
                    results.append({
                        "text": rec["text"],
                        "page": rec["page"],
                        "file": rec["source"],
                        "section": rec["section"],
                        "dimension": "semantic",
                        "retrieval_method": "neighbor_expansion",
                    })

    driver.close()
    logger.info(f"Semantic retrieval: {len(results)} results")
    return results


# ─── 3D Context Fusion ───────────────────────────────────

def fuse_contexts(structural_ctx: list[dict],
                  semantic_ctx: list[dict],
                  temporal_ctx: list[dict],
                  alpha: float, beta: float, gamma: float,
                  max_context_length: int = 15000) -> str:
    """
    Fuse contexts from all three dimensions into a single prompt context.

    Strategy:
    1. Allocate context budget proportional to dimensional weights
    2. Deduplicate overlapping content
    3. Add provenance tags for citation
    4. Order: structural first (sets scene), semantic second (details), temporal last (changes)
    """
    # Budget allocation
    total_budget = max_context_length
    structural_budget = int(total_budget * alpha) if alpha > ACTIVATION_THRESHOLD else 0
    semantic_budget = int(total_budget * beta) if beta > ACTIVATION_THRESHOLD else 0
    temporal_budget = int(total_budget * gamma) if gamma > ACTIVATION_THRESHOLD else 0

    # Normalize if total exceeds budget
    allocated = structural_budget + semantic_budget + temporal_budget
    if allocated > total_budget and allocated > 0:
        factor = total_budget / allocated
        structural_budget = int(structural_budget * factor)
        semantic_budget = int(semantic_budget * factor)
        temporal_budget = int(temporal_budget * factor)

    parts = []
    source_counter = 1
    seen_texts = set()

    def add_context(contexts, budget, dimension_label):
        nonlocal source_counter
        added = 0
        for ctx in contexts:
            text = ctx.get("text", "")
            # Dedup by first 100 chars
            text_key = text[:100].lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            if added + len(text) > budget:
                remaining = budget - added
                if remaining > 200:
                    text = text[:remaining] + "..."
                else:
                    break

            file_info = ctx.get("file", "unknown")
            page_info = ctx.get("page", "")
            section_info = ctx.get("section", "")
            method = ctx.get("retrieval_method", "")

            source_tag = f"[SOURCE_{source_counter}]"
            provenance = f"[{dimension_label}|{method}|{file_info}|p{page_info}|{section_info}]"

            parts.append(f"{source_tag} {provenance}\n{text}")
            source_counter += 1
            added += len(text)

    # Fuse in order: structural → semantic → temporal
    if structural_budget > 0:
        parts.append("=== STRUCTURAL CONTEXT (Document Hierarchy) ===")
        add_context(structural_ctx, structural_budget, "STRUCTURAL")

    if semantic_budget > 0:
        parts.append("\n=== SEMANTIC CONTEXT (Knowledge Graph) ===")
        add_context(semantic_ctx, semantic_budget, "SEMANTIC")

    if temporal_budget > 0:
        parts.append("\n=== TEMPORAL CONTEXT (Version Changes) ===")
        add_context(temporal_ctx, temporal_budget, "TEMPORAL")

    return "\n\n".join(parts)


# ─── Grounded Generation ─────────────────────────────────

def generate_answer(query: str, fused_context: str,
                    classification: dict) -> dict:
    """
    Generate a grounded answer with citations using Gemini.
    """
    dim_info = classification.get("dimensions_active", [])
    dim_str = ", ".join(dim_info) if dim_info else "general"

    prompt = f"""You are SpecRAG, an expert assistant for WYDOT (Wyoming Department of Transportation) engineering specifications.

Answer the following query using ONLY the context provided below. Be precise and cite your sources using [SOURCE_N] tags.

RETRIEVAL DIMENSIONS USED: {dim_str}
QUERY TYPE: {classification.get('type', 'unknown')}

CONTEXT:
{fused_context}

QUERY: {query}

Instructions:
1. Answer the query thoroughly using ONLY information from the context
2. Cite specific sources with [SOURCE_N] tags for every claim
3. If the context contains temporal/change information, highlight what changed and when
4. If comparing across document types (Specs vs Manuals vs Testing), note the source type
5. If the context is insufficient, say so explicitly
6. Be precise with numbers, specifications, and requirements — accuracy is critical
7. Structure your answer clearly with sections if the query is complex

ANSWER:
"""

    start_time = time.time()
    answer = gemini_generate(prompt, temperature=0.0)
    generation_time = time.time() - start_time

    return {
        "answer": answer,
        "query": query,
        "classification": classification,
        "generation_time_s": round(generation_time, 2),
    }


# ─── Main SpecRAG Pipeline ───────────────────────────────

def specrag_query(query: str, verbose: bool = True) -> dict:
    """
    Full SpecRAG 3D retrieval pipeline.

    Steps:
    1. Classify query → get (alpha, beta, gamma) weights
    2. Activate relevant dimensions based on weights
    3. Retrieve from each active dimension
    4. Fuse contexts with budget allocation
    5. Generate grounded answer

    Returns complete result dict with answer, sources, and metrics.
    """
    total_start = time.time()

    # Step 1: Classify
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"SpecRAG Query: {query}")
        logger.info(f"{'='*60}")

    classify_start = time.time()
    classification = classify_query(query)
    classify_time = time.time() - classify_start

    alpha = classification.get("alpha", 0)
    beta = classification.get("beta", 0)
    gamma = classification.get("gamma", 0)

    if verbose:
        logger.info(f"Classification: {classification['type']} "
                    f"(α={alpha}, β={beta}, γ={gamma})")
        logger.info(f"Active dimensions: {classification['dimensions_active']}")

    # Step 2-3: Retrieve from active dimensions
    structural_ctx = []
    semantic_ctx = []
    temporal_ctx = []

    retrieval_start = time.time()

    if alpha > ACTIVATION_THRESHOLD:
        if verbose:
            logger.info("Retrieving: STRUCTURAL dimension...")
        try:
            from phase1_pageindex.structural_retriever import structural_retrieve, load_all_trees
            trees = load_all_trees()
            structural_ctx = structural_retrieve(query, trees)
        except Exception as e:
            logger.warning(f"Structural retrieval failed: {e}")

    if beta > ACTIVATION_THRESHOLD:
        if verbose:
            logger.info("Retrieving: SEMANTIC dimension...")
        try:
            semantic_ctx = semantic_retrieve(query)
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")

    if gamma > ACTIVATION_THRESHOLD:
        if verbose:
            logger.info("Retrieving: TEMPORAL dimension...")
        try:
            from phase3_temporal.diff_engine import temporal_retrieve
            temporal_ctx = temporal_retrieve(query)
        except Exception as e:
            logger.warning(f"Temporal retrieval failed: {e}")

    retrieval_time = time.time() - retrieval_start

    if verbose:
        logger.info(f"Retrieved: {len(structural_ctx)} structural, "
                    f"{len(semantic_ctx)} semantic, {len(temporal_ctx)} temporal")

    # Step 4: Fuse contexts
    fused_context = fuse_contexts(
        structural_ctx, semantic_ctx, temporal_ctx,
        alpha, beta, gamma
    )

    # Step 5: Generate answer
    if verbose:
        logger.info("Generating grounded answer...")
    result = generate_answer(query, fused_context, classification)

    total_time = time.time() - total_start

    # Compile full result
    result["metrics"] = {
        "total_time_s": round(total_time, 2),
        "classify_time_s": round(classify_time, 2),
        "retrieval_time_s": round(retrieval_time, 2),
        "generation_time_s": result["generation_time_s"],
        "structural_results": len(structural_ctx),
        "semantic_results": len(semantic_ctx),
        "temporal_results": len(temporal_ctx),
        "context_length_chars": len(fused_context),
        "dimensions_used": classification["dimensions_active"],
        "dimension_count": classification["dimension_count"],
    }

    result["provenance"] = {
        "structural_sources": [
            {"file": c.get("file"), "pages": f"{c.get('page_start')}-{c.get('page_end')}",
             "section": c.get("section_path")}
            for c in structural_ctx
        ],
        "semantic_sources": [
            {"file": c.get("file"), "page": c.get("page"),
             "section": c.get("section"), "score": c.get("score")}
            for c in semantic_ctx if c.get("retrieval_method") == "vector_search"
        ],
        "temporal_sources": [
            {"change_type": c.get("change_type"), "severity": c.get("severity"),
             "old_year": c.get("old_year"), "new_year": c.get("new_year")}
            for c in temporal_ctx if c.get("change_type")
        ],
    }

    if verbose:
        logger.info(f"\nTotal time: {total_time:.2f}s")
        logger.info(f"Answer length: {len(result['answer'])} chars")

    return result


# ─── Baseline Runners (for comparison) ────────────────────

def run_baseline_alpha_only(query: str) -> dict:
    """Structural only (alpha=1, beta=0, gamma=0)."""
    return _run_single_dimension(query, alpha=1.0, beta=0.0, gamma=0.0, name="alpha_only")


def run_baseline_beta_only(query: str) -> dict:
    """Semantic only (alpha=0, beta=1, gamma=0)."""
    return _run_single_dimension(query, alpha=0.0, beta=1.0, gamma=0.0, name="beta_only")


def run_baseline_gamma_only(query: str) -> dict:
    """Temporal only (alpha=0, beta=0, gamma=1)."""
    return _run_single_dimension(query, alpha=0.0, beta=0.0, gamma=1.0, name="gamma_only")


def run_baseline_alpha_beta(query: str) -> dict:
    """Structural + Semantic (alpha=0.5, beta=0.5, gamma=0)."""
    return _run_single_dimension(query, alpha=0.5, beta=0.5, gamma=0.0, name="alpha_beta")


def run_baseline_beta_gamma(query: str) -> dict:
    """Semantic + Temporal (alpha=0, beta=0.5, gamma=0.5)."""
    return _run_single_dimension(query, alpha=0.0, beta=0.5, gamma=0.5, name="beta_gamma")


def run_baseline_alpha_gamma(query: str) -> dict:
    """Structural + Temporal (alpha=0.5, beta=0, gamma=0.5)."""
    return _run_single_dimension(query, alpha=0.5, beta=0.0, gamma=0.5, name="alpha_gamma")


def run_naive_rag(query: str) -> dict:
    """Naive vector-only RAG — no graph, no structure, no temporal."""
    semantic_ctx = semantic_retrieve(query, top_k=10)
    # Filter to vector_search only (no graph expansion)
    vector_only = [c for c in semantic_ctx if c.get("retrieval_method") == "vector_search"]

    fused = fuse_contexts([], vector_only, [], 0.0, 1.0, 0.0)
    classification = {"type": "naive_rag", "alpha": 0, "beta": 1, "gamma": 0,
                     "dimensions_active": ["semantic"], "dimension_count": 1}
    result = generate_answer(query, fused, classification)
    result["baseline"] = "naive_rag"
    return result


def _run_single_dimension(query: str, alpha: float, beta: float, gamma: float,
                          name: str) -> dict:
    """Run retrieval with fixed dimensional weights (no classifier)."""
    structural_ctx = []
    semantic_ctx = []
    temporal_ctx = []

    if alpha > ACTIVATION_THRESHOLD:
        try:
            from phase1_pageindex.structural_retriever import structural_retrieve, load_all_trees
            trees = load_all_trees()
            structural_ctx = structural_retrieve(query, trees)
        except Exception as e:
            logger.warning(f"Structural retrieval failed: {e}")

    if beta > ACTIVATION_THRESHOLD:
        try:
            semantic_ctx = semantic_retrieve(query)
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")

    if gamma > ACTIVATION_THRESHOLD:
        try:
            from phase3_temporal.diff_engine import temporal_retrieve
            temporal_ctx = temporal_retrieve(query)
        except Exception as e:
            logger.warning(f"Temporal retrieval failed: {e}")

    fused = fuse_contexts(structural_ctx, semantic_ctx, temporal_ctx, alpha, beta, gamma)
    classification = {"type": name, "alpha": alpha, "beta": beta, "gamma": gamma,
                     "dimensions_active": [d for d, w in [("structural", alpha), ("semantic", beta), ("temporal", gamma)] if w > ACTIVATION_THRESHOLD],
                     "dimension_count": sum(1 for w in [alpha, beta, gamma] if w > ACTIVATION_THRESHOLD)}
    result = generate_answer(query, fused, classification)
    result["baseline"] = name
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpecRAG 3D Router")
    parser.add_argument("--query", type=str, help="Run single query")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.query:
        result = specrag_query(args.query)
        print(f"\n{'='*60}")
        print(f"ANSWER:")
        print(f"{'='*60}")
        print(result["answer"])
        print(f"\nMetrics: {json.dumps(result['metrics'], indent=2)}")
    elif args.interactive:
        print("SpecRAG Interactive Mode")
        print("Type a query and press Enter. Type 'quit' to exit.\n")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                result = specrag_query(query)
                print(f"\n{result['answer']}")
    else:
        # Demo
        demo_queries = [
            "What does Division 500 cover?",
            "What is the minimum compressive strength for Class A concrete?",
            "How did aggregate specifications change between 2010 and 2021?",
        ]
        for q in demo_queries:
            result = specrag_query(q)
            print(f"\nQ: {q}")
            print(f"A: {result['answer'][:200]}...")
            print(f"Dims: {result['metrics']['dimensions_used']}")
