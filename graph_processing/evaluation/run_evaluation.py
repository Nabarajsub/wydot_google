#!/usr/bin/env python3
"""
Run evaluation of 4 RAG pipeline variants on the test suite.

Baselines:
  1. Monolithic RAG — global vector search, no routing
  2. Regex + Scoped — regex-based routing + scoped search
  3. LLM + Scoped — LLM routing + single-agent scoped search
  4. MASDR-RAG — full multi-agent with Gemini tool calling

For each query × system, records:
  - Retrieved chunk IDs and their document_series
  - Whether chunks match the target category (for precision/recall)
  - Answer text from the LLM
  - Latency breakdown
  - Routing decision (for LLM/multi-agent systems)

Usage:
    cd graph_processing/
    python evaluation/run_evaluation.py
"""
import json
import os
import sys
import time
import re
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic_solution.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    GEMINI_API_KEY, AGENT_SERIES_FILTERS,
)
from agentic_solution.tools import (
    scoped_vector_search, scoped_fulltext_search, global_search,
)
from agentic_solution.agents import get_agent, AGENT_REGISTRY
from agentic_solution.orchestrator import run_orchestrator, TOOL_AGENT_LABELS
from neo4j import GraphDatabase
import google.generativeai as genai


# ═══════════════════════════════════════════════════════════════
#  CATEGORY → AGENT MAPPING
# ═══════════════════════════════════════════════════════════════

CATEGORY_TO_AGENT = {
    "STANDARD_SPECS": "specs_agent",
    "CONSTRUCTION_MANUAL": "construction_agent",
    "MATERIALS_TESTING": "materials_agent",
    "DESIGN_MANUAL": "design_agent",
    "TRAFFIC_CRASHES": "safety_agent",
    "BRIDGE_PROGRAM": "bridge_agent",
    "STIP": "planning_agent",
    "ANNUAL_REPORT": "admin_agent",
    "HIGHWAY_SAFETY": "safety_agent",
    "GENERAL": "general_agent",
}

# Regex patterns for rule-based routing
REGEX_ROUTES = {
    r"(?i)section\s+\d": "STANDARD_SPECS",
    r"(?i)standard\s+spec": "STANDARD_SPECS",
    r"(?i)construction\s+manual": "CONSTRUCTION_MANUAL",
    r"(?i)materials?\s+test": "MATERIALS_TESTING",
    r"(?i)design\s+manual": "DESIGN_MANUAL",
    r"(?i)(crash|fatal|accident|traffic\s+death)": "TRAFFIC_CRASHES",
    r"(?i)bridge\s+(design|load|rating|plan)": "BRIDGE_PROGRAM",
    r"(?i)(stip|improvement\s+program)": "STIP",
    r"(?i)annual\s+report": "ANNUAL_REPORT",
}


def regex_route(query: str) -> Optional[str]:
    """Rule-based routing using regex patterns."""
    for pattern, category in REGEX_ROUTES.items():
        if re.search(pattern, query):
            return category
    return None  # unroutable


def llm_route(query: str, model) -> str:
    """LLM-based routing using Gemini."""
    prompt = f"""Classify this WYDOT query into ONE category:
- STANDARD_SPECS: Construction specifications, material specs, tolerances, thresholds
- CONSTRUCTION_MANUAL: Field inspection, project administration
- MATERIALS_TESTING: Lab test procedures, sampling methods
- DESIGN_MANUAL: Road/bridge design standards
- TRAFFIC_CRASHES: Crash statistics, fatalities, accident data
- BRIDGE_PROGRAM: Bridge design, load ratings, structural details
- STIP: Project funding, transportation improvement programming
- ANNUAL_REPORT: Department reports, leadership, budgets
- HIGHWAY_SAFETY: Safety programs, vulnerable road users
- GENERAL: Cross-domain, unclear, or doesn't fit above

Query: {query}

Respond with ONLY the category name, nothing else."""

    try:
        response = model.generate_content(prompt)
        cat = response.text.strip().upper().replace(" ", "_")
        # Validate
        valid = {"STANDARD_SPECS", "CONSTRUCTION_MANUAL", "MATERIALS_TESTING",
                 "DESIGN_MANUAL", "TRAFFIC_CRASHES", "BRIDGE_PROGRAM", "STIP",
                 "ANNUAL_REPORT", "HIGHWAY_SAFETY", "GENERAL"}
        return cat if cat in valid else "GENERAL"
    except Exception as e:
        print(f"    LLM route error: {e}")
        return "GENERAL"


def check_chunk_relevance(chunk: Dict, target_category: str) -> bool:
    """Check if a retrieved chunk belongs to the target category's document domain."""
    agent_name = CATEGORY_TO_AGENT.get(target_category, "general_agent")
    if agent_name == "general_agent":
        return True  # General queries — any chunk could be relevant

    filters = AGENT_SERIES_FILTERS.get(agent_name, [])
    if not filters:
        return True

    # Check document_series against filters
    series = (chunk.get("series") or chunk.get("document_series") or "").lower()
    title = (chunk.get("title") or "").lower()

    for f in filters:
        # Parse the filter: "d.document_series CONTAINS 'X'"
        match = re.search(r"CONTAINS\s+'([^']+)'", f)
        if match and match.group(1).lower() in series:
            return True
        match = re.search(r"CONTAINS\s+'([^']+)'", f)
        if match and match.group(1).lower() in title:
            return True
        match = re.search(r"=\s+'([^']+)'", f)
        if match and match.group(1).lower() == series:
            return True

    return False


def get_chunk_series(driver, chunk_ids: List[str]) -> Dict[str, str]:
    """Look up document_series for a list of chunk IDs."""
    if not chunk_ids:
        return {}

    db_kw = {"database": NEO4J_DATABASE} if NEO4J_DATABASE else {}
    with driver.session(**db_kw) as session:
        result = session.run("""
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.id IN $ids
            RETURN c.id AS id, d.document_series AS series, d.display_title AS title
        """, ids=chunk_ids)
        return {r["id"]: {"series": r["series"], "title": r["title"]} for r in result}


# ═══════════════════════════════════════════════════════════════
#  BASELINE 1: Monolithic RAG (global search, no routing)
# ═══════════════════════════════════════════════════════════════

def run_monolithic(query: str, k: int = 15) -> Tuple[List[Dict], float]:
    """Run global vector search without any routing or scoping."""
    t0 = time.time()
    chunks = global_search(query, limit=k)
    latency = time.time() - t0
    return chunks, latency


# ═══════════════════════════════════════════════════════════════
#  BASELINE 2: Regex + Scoped
# ═══════════════════════════════════════════════════════════════

def run_regex_scoped(query: str, k: int = 15) -> Tuple[List[Dict], float, Optional[str]]:
    """Regex routing + scoped search. Falls back to global if unroutable."""
    t0 = time.time()
    category = regex_route(query)

    if category:
        agent_name = CATEGORY_TO_AGENT.get(category, "general_agent")
        agent = get_agent(agent_name)
        chunks = agent.search(query, limit=k)
    else:
        chunks = global_search(query, limit=k)
        category = None

    latency = time.time() - t0
    return chunks, latency, category


# ═══════════════════════════════════════════════════════════════
#  BASELINE 3: LLM + Scoped (single agent)
# ═══════════════════════════════════════════════════════════════

def run_llm_scoped(query: str, model, k: int = 15) -> Tuple[List[Dict], float, str]:
    """LLM routing + scoped search via single agent."""
    t0 = time.time()
    category = llm_route(query, model)
    agent_name = CATEGORY_TO_AGENT.get(category, "general_agent")
    agent = get_agent(agent_name)
    chunks = agent.search(query, limit=k)
    latency = time.time() - t0
    return chunks, latency, category


# ═══════════════════════════════════════════════════════════════
#  BASELINE 4: MASDR-RAG (full multi-agent)
# ═══════════════════════════════════════════════════════════════

def run_masdr_rag(query: str) -> Tuple[str, List[Dict], List[Dict], float]:
    """Full multi-agent orchestration with Gemini tool calling."""
    t0 = time.time()
    answer, sources, tool_events = run_orchestrator(query)
    latency = time.time() - t0
    return answer, sources, tool_events, latency


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def precision_at_k(relevant_flags: List[bool], k: int) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    topk = relevant_flags[:k]
    return sum(topk) / len(topk) if topk else 0.0


def recall_at_k(relevant_flags: List[bool], k: int, total_relevant: int) -> float:
    """Recall@k: fraction of total relevant found in top-k."""
    if total_relevant == 0:
        return 0.0
    topk = relevant_flags[:k]
    return sum(topk) / total_relevant


def ndcg_at_k(relevant_flags: List[bool], k: int) -> float:
    """NDCG@k with binary relevance."""
    import math
    topk = relevant_flags[:k]
    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(topk))
    # Ideal: all relevant first
    ideal = sorted(topk, reverse=True)
    idcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_dilution_factor(
    query: str, target_category: str, driver, k: int = 15
) -> float:
    """
    Compute δ = 1 - P_global/P_scoped for a single query.
    """
    # Global search
    global_chunks = global_search(query, limit=k)
    global_series = get_chunk_series(driver, [c["id"] for c in global_chunks])
    global_relevant = sum(
        1 for c in global_chunks
        if check_chunk_relevance({**c, **global_series.get(c["id"], {})}, target_category)
    )
    p_global = global_relevant / k if k > 0 else 0

    # Scoped search
    agent_name = CATEGORY_TO_AGENT.get(target_category, "general_agent")
    if agent_name == "general_agent":
        return 0.0  # No dilution for general queries

    agent = get_agent(agent_name)
    scoped_chunks = agent.search(query, limit=k)
    scoped_series = get_chunk_series(driver, [c["id"] for c in scoped_chunks])
    scoped_relevant = sum(
        1 for c in scoped_chunks
        if check_chunk_relevance({**c, **scoped_series.get(c["id"], {})}, target_category)
    )
    p_scoped = scoped_relevant / k if k > 0 else 0

    if p_scoped == 0:
        return 0.0

    delta = 1 - (p_global / p_scoped)
    return max(0.0, delta)


# ═══════════════════════════════════════════════════════════════
#  LLM-based Answer Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_answer_with_llm(
    query: str, answer: str, reference: str, chunks_text: str, model
) -> Dict:
    """Use Gemini to evaluate answer correctness, faithfulness, relevance."""
    prompt = f"""Evaluate this RAG system answer on 3 dimensions (0.0 to 1.0 scale):

QUERY: {query}

REFERENCE ANSWER: {reference}

SYSTEM ANSWER: {answer[:2000]}

RETRIEVED CONTEXT (first 1500 chars): {chunks_text[:1500]}

Score each dimension:
1. CORRECTNESS: Does the answer match the reference answer's key facts? (0.0-1.0)
2. FAITHFULNESS: Is the answer grounded in the retrieved context (not hallucinated)? (0.0-1.0)
3. RELEVANCE: Does the answer address the question? (0.0-1.0)

Respond as JSON ONLY:
{{"correctness": 0.X, "faithfulness": 0.X, "relevance": 0.X, "correct_binary": true/false}}
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"    Eval error: {e}")
        return {"correctness": 0.0, "faithfulness": 0.0, "relevance": 0.0, "correct_binary": False}


# ═══════════════════════════════════════════════════════════════
#  GENERATE ANSWER (for non-multi-agent baselines)
# ═══════════════════════════════════════════════════════════════

def generate_answer(query: str, chunks: List[Dict], model) -> str:
    """Generate an answer from retrieved chunks using Gemini."""
    context = ""
    for i, c in enumerate(chunks[:10], 1):
        title = c.get("title", "Unknown")
        section = c.get("section", "N/A")
        year = c.get("year", "N/A")
        text = c.get("text", "")[:500]
        context += f"[Source {i}: {title}, {section}, Year: {year}]\n{text}\n\n"

    prompt = f"""Answer this question using ONLY the provided sources. Cite sources as [Source X].

QUESTION: {query}

SOURCES:
{context}

If the sources don't contain enough information, say so. Be concise but thorough."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# ═══════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("WYDOT RAG Evaluation Harness")
    print("=" * 60)

    # Load test suite
    suite_path = os.path.join(os.path.dirname(__file__), "test_suite.json")
    if not os.path.exists(suite_path):
        print(f"❌ Test suite not found at {suite_path}")
        print("   Run: python evaluation/generate_test_suite.py first")
        sys.exit(1)

    with open(suite_path) as f:
        test_suite = json.load(f)

    print(f"Loaded {len(test_suite)} test queries")

    genai.configure(api_key=GEMINI_API_KEY)
    eval_model = genai.GenerativeModel("gemini-2.5-flash")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    results = []
    K = 10  # evaluation at k=10

    for i, test_q in enumerate(test_suite):
        qid = test_q["id"]
        query = test_q["query"]
        target_cat = test_q.get("category", "GENERAL")
        reference = test_q.get("reference_answer", "")
        query_type = test_q.get("query_type", "unknown")

        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(test_suite)}] {qid}: {query[:70]}...")
        print(f"  Target: {target_cat} ({query_type})")

        result = {
            "id": qid,
            "query": query,
            "target_category": target_cat,
            "query_type": query_type,
            "reference_answer": reference,
            "systems": {},
        }

        # ── System 1: Monolithic ──
        print("  Running Monolithic...")
        try:
            mono_chunks, mono_lat = run_monolithic(query, k=K)
            mono_series = get_chunk_series(driver, [c["id"] for c in mono_chunks])
            mono_relevant = [
                check_chunk_relevance({**c, **mono_series.get(c["id"], {})}, target_cat)
                for c in mono_chunks
            ]
            mono_answer = generate_answer(query, mono_chunks, eval_model)
            time.sleep(0.5)
            mono_eval = evaluate_answer_with_llm(
                query, mono_answer, reference,
                " ".join(c.get("text", "")[:200] for c in mono_chunks[:5]),
                eval_model
            )

            result["systems"]["monolithic"] = {
                "latency": round(mono_lat, 3),
                "chunks_retrieved": len(mono_chunks),
                "p_at_5": round(precision_at_k(mono_relevant, 5), 3),
                "p_at_10": round(precision_at_k(mono_relevant, K), 3),
                "r_at_5": round(recall_at_k(mono_relevant, 5, sum(mono_relevant)), 3),
                "r_at_10": round(recall_at_k(mono_relevant, K, sum(mono_relevant)), 3),
                "ndcg_at_5": round(ndcg_at_k(mono_relevant, 5), 3),
                "ndcg_at_10": round(ndcg_at_k(mono_relevant, K), 3),
                "routed_category": None,
                **mono_eval,
            }
            print(f"    P@10={precision_at_k(mono_relevant, K):.2f} lat={mono_lat:.1f}s")
        except Exception as e:
            print(f"    ❌ Monolithic error: {e}")
            result["systems"]["monolithic"] = {"error": str(e)}

        time.sleep(1)

        # ── System 2: Regex + Scoped ──
        print("  Running Regex+Scoped...")
        try:
            regex_chunks, regex_lat, regex_cat = run_regex_scoped(query, k=K)
            regex_series = get_chunk_series(driver, [c["id"] for c in regex_chunks])
            regex_relevant = [
                check_chunk_relevance({**c, **regex_series.get(c["id"], {})}, target_cat)
                for c in regex_chunks
            ]
            regex_answer = generate_answer(query, regex_chunks, eval_model)
            time.sleep(0.5)
            regex_eval = evaluate_answer_with_llm(
                query, regex_answer, reference,
                " ".join(c.get("text", "")[:200] for c in regex_chunks[:5]),
                eval_model
            )

            result["systems"]["regex_scoped"] = {
                "latency": round(regex_lat, 3),
                "chunks_retrieved": len(regex_chunks),
                "p_at_5": round(precision_at_k(regex_relevant, 5), 3),
                "p_at_10": round(precision_at_k(regex_relevant, K), 3),
                "r_at_5": round(recall_at_k(regex_relevant, 5, sum(regex_relevant)), 3),
                "r_at_10": round(recall_at_k(regex_relevant, K, sum(regex_relevant)), 3),
                "ndcg_at_5": round(ndcg_at_k(regex_relevant, 5), 3),
                "ndcg_at_10": round(ndcg_at_k(regex_relevant, K), 3),
                "routed_category": regex_cat,
                "routing_correct": regex_cat == target_cat if regex_cat else False,
                **regex_eval,
            }
            print(f"    P@10={precision_at_k(regex_relevant, K):.2f} routed={regex_cat} lat={regex_lat:.1f}s")
        except Exception as e:
            print(f"    ❌ Regex error: {e}")
            result["systems"]["regex_scoped"] = {"error": str(e)}

        time.sleep(1)

        # ── System 3: LLM + Scoped ──
        print("  Running LLM+Scoped...")
        try:
            llm_chunks, llm_lat, llm_cat = run_llm_scoped(query, eval_model, k=K)
            llm_series = get_chunk_series(driver, [c["id"] for c in llm_chunks])
            llm_relevant = [
                check_chunk_relevance({**c, **llm_series.get(c["id"], {})}, target_cat)
                for c in llm_chunks
            ]
            llm_answer = generate_answer(query, llm_chunks, eval_model)
            time.sleep(0.5)
            llm_eval = evaluate_answer_with_llm(
                query, llm_answer, reference,
                " ".join(c.get("text", "")[:200] for c in llm_chunks[:5]),
                eval_model
            )

            result["systems"]["llm_scoped"] = {
                "latency": round(llm_lat, 3),
                "chunks_retrieved": len(llm_chunks),
                "p_at_5": round(precision_at_k(llm_relevant, 5), 3),
                "p_at_10": round(precision_at_k(llm_relevant, K), 3),
                "r_at_5": round(recall_at_k(llm_relevant, 5, sum(llm_relevant)), 3),
                "r_at_10": round(recall_at_k(llm_relevant, K, sum(llm_relevant)), 3),
                "ndcg_at_5": round(ndcg_at_k(llm_relevant, 5), 3),
                "ndcg_at_10": round(ndcg_at_k(llm_relevant, K), 3),
                "routed_category": llm_cat,
                "routing_correct": llm_cat == target_cat,
                **llm_eval,
            }
            print(f"    P@10={precision_at_k(llm_relevant, K):.2f} routed={llm_cat} lat={llm_lat:.1f}s")
        except Exception as e:
            print(f"    ❌ LLM+Scoped error: {e}")
            result["systems"]["llm_scoped"] = {"error": str(e)}

        time.sleep(1)

        # ── System 4: MASDR-RAG ──
        print("  Running MASDR-RAG...")
        try:
            masdr_answer, masdr_sources, masdr_events, masdr_lat = run_masdr_rag(query)
            masdr_series = get_chunk_series(driver, [c["id"] for c in masdr_sources])
            masdr_relevant = [
                check_chunk_relevance({**c, **masdr_series.get(c["id"], {})}, target_cat)
                for c in masdr_sources
            ]

            time.sleep(0.5)
            masdr_eval = evaluate_answer_with_llm(
                query, masdr_answer, reference,
                " ".join(c.get("text", "")[:200] for c in masdr_sources[:5]),
                eval_model
            )

            agents_used = list(set(e["agent_label"] for e in masdr_events))

            result["systems"]["masdr_rag"] = {
                "latency": round(masdr_lat, 3),
                "chunks_retrieved": len(masdr_sources),
                "p_at_5": round(precision_at_k(masdr_relevant, 5), 3),
                "p_at_10": round(precision_at_k(masdr_relevant, min(K, len(masdr_relevant))), 3),
                "r_at_5": round(recall_at_k(masdr_relevant, 5, sum(masdr_relevant)), 3),
                "r_at_10": round(recall_at_k(masdr_relevant, K, sum(masdr_relevant)), 3),
                "ndcg_at_5": round(ndcg_at_k(masdr_relevant, 5), 3),
                "ndcg_at_10": round(ndcg_at_k(masdr_relevant, K), 3),
                "agents_used": agents_used,
                "tool_calls": len(masdr_events),
                "answer_length": len(masdr_answer),
                **masdr_eval,
            }
            print(f"    P@10={precision_at_k(masdr_relevant, K):.2f} agents={agents_used} lat={masdr_lat:.1f}s")
        except Exception as e:
            print(f"    ❌ MASDR-RAG error: {e}")
            import traceback; traceback.print_exc()
            result["systems"]["masdr_rag"] = {"error": str(e)}

        results.append(result)
        time.sleep(1)

        # Save intermediate results every 5 queries
        if (i + 1) % 5 == 0:
            _save_results(results)

    _save_results(results)

    # ── Compute dilution factors ──
    print("\n\n📊 Computing dilution factors per category...")
    dilution_results = {}
    single_domain = [q for q in test_suite if q.get("query_type") == "single_domain"]

    for q in single_domain:
        cat = q["category"]
        if cat == "GENERAL":
            continue
        try:
            delta = compute_dilution_factor(q["query"], cat, driver, k=K)
            if cat not in dilution_results:
                dilution_results[cat] = []
            dilution_results[cat].append(delta)
            print(f"  {cat}: δ={delta:.3f} for '{q['query'][:50]}'")
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ Dilution error for {cat}: {e}")

    # Average δ per category
    dilution_avg = {}
    for cat, deltas in dilution_results.items():
        dilution_avg[cat] = {
            "mean": round(sum(deltas) / len(deltas), 3),
            "values": [round(d, 3) for d in deltas],
            "n": len(deltas),
        }
        print(f"  {cat}: avg δ = {dilution_avg[cat]['mean']:.3f} (n={len(deltas)})")

    # Save dilution results
    dilution_path = os.path.join(os.path.dirname(__file__), "dilution_results.json")
    with open(dilution_path, "w") as f:
        json.dump(dilution_avg, f, indent=2)
    print(f"✅ Saved dilution results to {dilution_path}")

    _save_results(results)
    _print_summary(results)

    driver.close()


def _save_results(results):
    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Saved {len(results)} results to {out_path}")


def _print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    systems = ["monolithic", "regex_scoped", "llm_scoped", "masdr_rag"]
    metrics = ["p_at_5", "p_at_10", "r_at_5", "r_at_10", "ndcg_at_5", "ndcg_at_10",
               "correctness", "faithfulness", "relevance", "latency"]

    for system in systems:
        print(f"\n📊 {system.upper()}:")
        for metric in metrics:
            values = []
            for r in results:
                sys_data = r.get("systems", {}).get(system, {})
                if metric in sys_data and not isinstance(sys_data[metric], str):
                    values.append(sys_data[metric])

            if values:
                avg = sum(values) / len(values)
                values_sorted = sorted(values)
                p50 = values_sorted[len(values_sorted) // 2]
                p95_idx = min(int(len(values_sorted) * 0.95), len(values_sorted) - 1)
                p95 = values_sorted[p95_idx]
                print(f"  {metric:15s}: avg={avg:.3f}  P50={p50:.3f}  P95={p95:.3f}  n={len(values)}")

        # Routing accuracy for routed systems
        if system in ("regex_scoped", "llm_scoped"):
            correct = sum(
                1 for r in results
                if r.get("systems", {}).get(system, {}).get("routing_correct", False)
            )
            routed = sum(
                1 for r in results
                if r.get("systems", {}).get(system, {}).get("routed_category") is not None
            )
            total = len(results)
            coverage = routed / total if total else 0
            accuracy = correct / routed if routed else 0
            print(f"  {'routing_cov':15s}: {coverage:.1%} ({routed}/{total})")
            print(f"  {'routing_acc':15s}: {accuracy:.1%} ({correct}/{routed})")

        # Binary correctness
        correct_binary = sum(
            1 for r in results
            if r.get("systems", {}).get(system, {}).get("correct_binary", False)
        )
        total = sum(
            1 for r in results
            if system in r.get("systems", {}) and "error" not in r["systems"][system]
        )
        if total:
            print(f"  {'correct_pct':15s}: {correct_binary/total:.1%} ({correct_binary}/{total})")


if __name__ == "__main__":
    main()
