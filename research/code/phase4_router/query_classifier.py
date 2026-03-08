"""
Phase 4: Query-Type Classifier & 3D Router
============================================
The intellectual core of SpecRAG.

Classifies queries into 8 types and assigns dimensional weights (alpha, beta, gamma)
that determine which retrieval dimensions to activate.

R(q) = alpha(q) * Structural(q) + beta(q) * Semantic(q) + gamma(q) * Temporal(q)

Usage:
    python -m phase4_router.query_classifier              # Interactive mode
    python -m phase4_router.query_classifier --query "..."
    python -m phase4_router.query_classifier --benchmark   # Classify all SpecQA queries
"""
import json
import sys
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import QUERY_TYPES
from utils.gemini_client import gemini_generate_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Few-Shot Examples for Classification ─────────────────

FEW_SHOT_EXAMPLES = [
    {
        "query": "What does Division 500 cover?",
        "type": "structural",
        "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "reasoning": "Asks about document structure/organization — needs tree navigation"
    },
    {
        "query": "What is the minimum compressive strength for Class A concrete?",
        "type": "semantic",
        "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "reasoning": "Asks about a specific entity property — needs entity graph search"
    },
    {
        "query": "How did aggregate specifications change between 2010 and 2021?",
        "type": "temporal",
        "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "reasoning": "Asks about changes between editions — needs temporal diff retrieval"
    },
    {
        "query": "Which section specifies Type IL cement requirements and what are they?",
        "type": "structural_semantic",
        "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
        "reasoning": "Needs to find the section (structural) then extract entity details (semantic)"
    },
    {
        "query": "Were any new test methods added in the 2023 Testing Manual compared to 2020?",
        "type": "semantic_temporal",
        "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
        "reasoning": "Asks about entity-level changes (test methods) across versions"
    },
    {
        "query": "Was Chapter 5 of the Construction Manual reorganized between 2022 and 2025?",
        "type": "structural_temporal",
        "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
        "reasoning": "Asks about structural reorganization across editions"
    },
    {
        "query": "How did concrete mix design requirements in Section 501 change, and what new test standards are now referenced?",
        "type": "full_3d",
        "alpha": 0.33, "beta": 0.34, "gamma": 0.33,
        "reasoning": "Needs section navigation (structural), entity extraction (semantic), and change tracking (temporal)"
    },
    {
        "query": "What does the Standard Spec say about aggregate, and what test does the Testing Manual prescribe?",
        "type": "cross_document",
        "alpha": 0.4, "beta": 0.6, "gamma": 0.0,
        "reasoning": "Needs cross-document retrieval — find in Standard Spec (structural) and match entity in Testing Manual (semantic)"
    },
]


def classify_query(query: str) -> dict:
    """
    Classify a query into one of 8 types and assign dimensional weights.

    Returns:
    {
        "query": str,
        "type": str,
        "alpha": float,  # structural weight
        "beta": float,   # semantic weight
        "gamma": float,  # temporal weight
        "reasoning": str,
        "dimensions_active": ["structural", "semantic", "temporal"],
    }
    """
    examples_text = "\n".join([
        f"Query: \"{ex['query']}\"\n"
        f"Type: {ex['type']}\n"
        f"Alpha (structural): {ex['alpha']}, Beta (semantic): {ex['beta']}, Gamma (temporal): {ex['gamma']}\n"
        f"Reasoning: {ex['reasoning']}\n"
        for ex in FEW_SHOT_EXAMPLES
    ])

    type_descriptions = """
QUERY TYPES:
1. structural (alpha=1.0) — Questions about document organization, sections, divisions, table of contents
2. semantic (beta=1.0) — Questions about specific entities, materials, standards, values, requirements
3. temporal (gamma=1.0) — Questions about changes between document editions/years
4. structural_semantic (alpha+beta) — Need to find a section AND extract entity details from it
5. semantic_temporal (beta+gamma) — Questions about how entities/requirements changed over time
6. structural_temporal (alpha+gamma) — Questions about structural reorganization across editions
7. full_3d (alpha+beta+gamma) — Complex queries needing section navigation + entity extraction + temporal changes
8. cross_document (alpha+beta) — Questions requiring info from multiple document types (specs, manuals, testing)

DOCUMENT CONTEXT:
- Standard Specifications (2010, 2021): Organized by Division/Section/Subsection numbers
- Construction Manuals (2018-2026): Organized by Chapter, 8 annual editions
- Materials Testing Manuals (2020, 2021, 2023): Organized by test method
- Summary of Changes documents: Document edition differences
"""

    prompt = f"""You are a query classifier for SpecRAG, a 3-dimensional retrieval system for WYDOT engineering specifications.

Given a user query, classify it into one of 8 types and assign dimensional weights (alpha, beta, gamma) that sum to 1.0.

{type_descriptions}

EXAMPLES:
{examples_text}

QUERY TO CLASSIFY: "{query}"

Instructions:
1. Determine which retrieval dimensions are needed to fully answer this query
2. Assign weights proportional to each dimension's importance
3. Weights must sum to approximately 1.0
4. Provide brief reasoning

Return a JSON object with:
- "type": one of the 8 types listed above
- "alpha": float (structural weight, 0.0-1.0)
- "beta": float (semantic weight, 0.0-1.0)
- "gamma": float (temporal weight, 0.0-1.0)
- "reasoning": brief explanation of classification

Return ONLY the JSON object.
"""

    result = gemini_generate_json(prompt, temperature=0.0)

    # Normalize weights to sum to 1.0
    total = result.get("alpha", 0) + result.get("beta", 0) + result.get("gamma", 0)
    if total > 0:
        result["alpha"] = round(result.get("alpha", 0) / total, 3)
        result["beta"] = round(result.get("beta", 0) / total, 3)
        result["gamma"] = round(result.get("gamma", 0) / total, 3)

    # Determine active dimensions (threshold > 0.1)
    active = []
    if result.get("alpha", 0) > 0.1:
        active.append("structural")
    if result.get("beta", 0) > 0.1:
        active.append("semantic")
    if result.get("gamma", 0) > 0.1:
        active.append("temporal")

    result["query"] = query
    result["dimensions_active"] = active
    result["dimension_count"] = len(active)

    return result


def classify_batch(queries: list[str]) -> list[dict]:
    """Classify a batch of queries. Returns list of classification results."""
    results = []
    for i, q in enumerate(queries):
        logger.info(f"Classifying {i+1}/{len(queries)}: {q[:60]}...")
        result = classify_query(q)
        results.append(result)
        logger.info(f"  → {result['type']} (α={result['alpha']}, β={result['beta']}, γ={result['gamma']})")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpecRAG Query Classifier")
    parser.add_argument("--query", type=str, help="Classify a single query")
    parser.add_argument("--benchmark", action="store_true", help="Classify all SpecQA queries")
    args = parser.parse_args()

    if args.query:
        result = classify_query(args.query)
        print(json.dumps(result, indent=2))
    elif args.benchmark:
        # Load SpecQA queries
        from config.settings import BENCHMARK_DIR
        specqa_path = BENCHMARK_DIR / "specqa_queries.json"
        if specqa_path.exists():
            with open(specqa_path) as f:
                queries = [q["query"] for q in json.load(f)]
            results = classify_batch(queries)
            output_path = BENCHMARK_DIR / "specqa_classifications.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} classifications to {output_path}")
        else:
            print("SpecQA queries not found. Run phase5 first.")
    else:
        # Interactive mode
        print("SpecRAG Query Classifier — Interactive Mode")
        print("Type a query and press Enter. Type 'quit' to exit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if query:
                result = classify_query(query)
                print(f"\n  Type: {result['type']}")
                print(f"  Alpha (structural): {result['alpha']}")
                print(f"  Beta (semantic):     {result['beta']}")
                print(f"  Gamma (temporal):    {result['gamma']}")
                print(f"  Active dimensions:   {result['dimensions_active']}")
                print(f"  Reasoning: {result.get('reasoning', 'N/A')}\n")
