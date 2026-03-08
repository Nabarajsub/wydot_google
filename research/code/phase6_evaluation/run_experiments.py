"""
Phase 6: Experiment Runner & Evaluation Harness
=================================================
Runs all 8 baselines + SpecRAG on SpecQA benchmark queries.
Computes metrics, generates tables, and saves results.

Baselines:
1. naive_rag         — vector-only, no graph, no structure, no temporal
2. alpha_only        — structural retrieval only (PageIndex)
3. beta_only         — semantic retrieval only (GraphRAG + Neo4j)
4. gamma_only        — temporal retrieval only (version graph)
5. alpha_beta        — structural + semantic (2D)
6. beta_gamma        — semantic + temporal (2D)
7. alpha_gamma       — structural + temporal (2D)
8. specrag_full      — full 3D with learned routing (our system)

Usage:
    python -m phase6_evaluation.run_experiments                    # Run all
    python -m phase6_evaluation.run_experiments --baseline naive_rag
    python -m phase6_evaluation.run_experiments --query-id S01
    python -m phase6_evaluation.run_experiments --analyze          # Analyze saved results
"""
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RESULTS_DIR, BENCHMARK_DIR
from phase5_benchmark.specqa_queries import SPECQA_QUERIES
from utils.gemini_client import gemini_generate_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Baseline Definitions ─────────────────────────────────

BASELINES = {
    "naive_rag": {
        "name": "Naive RAG (Vector-Only)",
        "runner": "run_naive_rag",
        "dimensions": ["semantic"],
        "description": "Standard vector search, no graph, no structure, no temporal"
    },
    "alpha_only": {
        "name": "Structural Only (PageIndex)",
        "runner": "run_baseline_alpha_only",
        "dimensions": ["structural"],
        "description": "PageIndex tree navigation only"
    },
    "beta_only": {
        "name": "Semantic Only (GraphRAG)",
        "runner": "run_baseline_beta_only",
        "dimensions": ["semantic"],
        "description": "Neo4j vector + entity graph traversal"
    },
    "gamma_only": {
        "name": "Temporal Only (Version Graph)",
        "runner": "run_baseline_gamma_only",
        "dimensions": ["temporal"],
        "description": "Version graph traversal only"
    },
    "alpha_beta": {
        "name": "Structural + Semantic (2D)",
        "runner": "run_baseline_alpha_beta",
        "dimensions": ["structural", "semantic"],
        "description": "PageIndex + GraphRAG fusion"
    },
    "beta_gamma": {
        "name": "Semantic + Temporal (2D)",
        "runner": "run_baseline_beta_gamma",
        "dimensions": ["semantic", "temporal"],
        "description": "GraphRAG + Version graph"
    },
    "alpha_gamma": {
        "name": "Structural + Temporal (2D)",
        "runner": "run_baseline_alpha_gamma",
        "dimensions": ["structural", "temporal"],
        "description": "PageIndex + Version graph"
    },
    "specrag_full": {
        "name": "SpecRAG (Full 3D)",
        "runner": "specrag_query",
        "dimensions": ["structural", "semantic", "temporal"],
        "description": "Full 3D retrieval with learned routing"
    },
}


# ─── Answer Evaluation ────────────────────────────────────

def evaluate_answer(query_data: dict, answer: str, system_name: str) -> dict:
    """
    Evaluate a system answer against the SpecQA gold standard.

    Metrics:
    1. Keyword Overlap: fraction of gold_answer_keywords found in answer
    2. LLM-Judge Accuracy: Gemini rates answer quality (1-5)
    3. LLM-Judge Faithfulness: Is answer grounded in context? (1-5)
    4. Citation Present: Does answer include [SOURCE_N] citations?
    5. Dimension Match: Did the system use the correct dimensions?
    """
    query = query_data["query"]
    gold_keywords = query_data.get("gold_answer_keywords", [])

    # Metric 1: Keyword Overlap
    answer_lower = answer.lower()
    matched_keywords = [kw for kw in gold_keywords if kw.lower() in answer_lower]
    keyword_overlap = len(matched_keywords) / len(gold_keywords) if gold_keywords else 0.0

    # Metric 2 & 3: LLM-as-Judge
    judge_prompt = f"""You are evaluating a RAG system's answer quality. Rate on two dimensions.

QUERY: {query}
EXPECTED KEYWORDS: {', '.join(gold_keywords)}
SYSTEM ANSWER: {answer[:2000]}

Rate the following (1-5 scale):
1. ACCURACY: Does the answer correctly address the query? (1=wrong, 3=partial, 5=fully correct)
2. FAITHFULNESS: Is the answer grounded in cited sources, or does it appear to hallucinate? (1=hallucinated, 3=mixed, 5=fully grounded)
3. COMPLETENESS: Does the answer cover all aspects of the query? (1=misses most, 3=partial, 5=comprehensive)

Return JSON: {{"accuracy": int, "faithfulness": int, "completeness": int, "explanation": str}}
"""
    try:
        judge_result = gemini_generate_json(judge_prompt, temperature=0.0)
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        judge_result = {"accuracy": 0, "faithfulness": 0, "completeness": 0, "explanation": str(e)}

    # Metric 4: Citation presence
    import re
    citations = re.findall(r'\[SOURCE_\d+\]', answer)
    has_citations = len(citations) > 0

    return {
        "query_id": query_data["id"],
        "system": system_name,
        "keyword_overlap": round(keyword_overlap, 3),
        "matched_keywords": matched_keywords,
        "accuracy": judge_result.get("accuracy", 0),
        "faithfulness": judge_result.get("faithfulness", 0),
        "completeness": judge_result.get("completeness", 0),
        "has_citations": has_citations,
        "citation_count": len(citations),
        "judge_explanation": judge_result.get("explanation", ""),
        "answer_length": len(answer),
    }


# ─── Experiment Runner ────────────────────────────────────

def run_single_experiment(query_data: dict, system_name: str, system_func) -> dict:
    """Run a single query through a single system and evaluate."""
    query = query_data["query"]
    query_id = query_data["id"]

    logger.info(f"  [{query_id}] Running {system_name}...")

    start_time = time.time()
    try:
        result = system_func(query)
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        error = None
    except Exception as e:
        answer = f"ERROR: {str(e)}"
        metrics = {}
        error = str(e)

    elapsed = time.time() - start_time

    # Evaluate
    evaluation = evaluate_answer(query_data, answer, system_name)

    return {
        "query_id": query_id,
        "query": query,
        "query_type": query_data["type"],
        "difficulty": query_data["difficulty"],
        "system": system_name,
        "answer": answer,
        "evaluation": evaluation,
        "latency_s": round(elapsed, 2),
        "system_metrics": metrics,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }


def run_experiments(target_baseline: Optional[str] = None,
                    target_query_id: Optional[str] = None,
                    num_runs: int = 1,
                    save_incremental: bool = True):
    """
    Run the full experiment suite.

    Args:
        target_baseline: Run only this baseline (None = all)
        target_query_id: Run only this query ID (None = all)
        num_runs: Number of runs per query (for variance estimation)
        save_incremental: Save results after each system completes
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Import system functions
    from phase4_router.specrag_router import (
        specrag_query, run_naive_rag,
        run_baseline_alpha_only, run_baseline_beta_only, run_baseline_gamma_only,
        run_baseline_alpha_beta, run_baseline_beta_gamma, run_baseline_alpha_gamma,
    )

    runner_map = {
        "run_naive_rag": run_naive_rag,
        "run_baseline_alpha_only": run_baseline_alpha_only,
        "run_baseline_beta_only": run_baseline_beta_only,
        "run_baseline_gamma_only": run_baseline_gamma_only,
        "run_baseline_alpha_beta": run_baseline_alpha_beta,
        "run_baseline_beta_gamma": run_baseline_beta_gamma,
        "run_baseline_alpha_gamma": run_baseline_alpha_gamma,
        "specrag_query": specrag_query,
    }

    # Filter queries
    queries = SPECQA_QUERIES
    if target_query_id:
        queries = [q for q in queries if q["id"] == target_query_id]

    # Filter baselines
    baselines_to_run = BASELINES
    if target_baseline:
        baselines_to_run = {k: v for k, v in BASELINES.items() if k == target_baseline}

    all_results = []
    total_runs = len(queries) * len(baselines_to_run) * num_runs

    print(f"\n{'='*70}")
    print(f"SpecRAG EXPERIMENT SUITE")
    print(f"{'='*70}")
    print(f"Queries: {len(queries)} | Systems: {len(baselines_to_run)} | "
          f"Runs per query: {num_runs} | Total: {total_runs}")
    print(f"{'='*70}\n")

    run_count = 0
    for system_name, system_info in baselines_to_run.items():
        print(f"\n--- Running: {system_info['name']} ---")

        runner_func = runner_map.get(system_info["runner"])
        if not runner_func:
            logger.error(f"Runner not found: {system_info['runner']}")
            continue

        system_results = []
        for run_idx in range(num_runs):
            for query_data in queries:
                run_count += 1
                logger.info(f"[{run_count}/{total_runs}] {system_name} | {query_data['id']} | run {run_idx+1}")

                result = run_single_experiment(query_data, system_name, runner_func)
                result["run_index"] = run_idx
                system_results.append(result)
                all_results.append(result)

        # Save incremental results per system
        if save_incremental:
            system_path = RESULTS_DIR / f"results_{system_name}.json"
            with open(system_path, "w") as f:
                json.dump(system_results, f, indent=2)
            logger.info(f"Saved {len(system_results)} results to {system_path.name}")

    # Save all results
    all_path = RESULTS_DIR / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved all {len(all_results)} results to {all_path.name}")

    # Generate summary
    generate_summary(all_results)

    return all_results


# ─── Analysis & Summary ──────────────────────────────────

def generate_summary(results: list[dict] = None):
    """Generate summary tables from experiment results."""
    if results is None:
        # Load latest results
        result_files = sorted(RESULTS_DIR.glob("all_results_*.json"), reverse=True)
        if not result_files:
            print("No results found!")
            return
        with open(result_files[0]) as f:
            results = json.load(f)

    # ─── Table 1: Overall Accuracy by System ──────────
    print(f"\n{'='*90}")
    print("TABLE 1: OVERALL RESULTS BY SYSTEM")
    print(f"{'='*90}")
    print(f"{'System':<30} {'Accuracy':>9} {'Faithful':>9} {'Complete':>9} "
          f"{'Keywords':>9} {'Latency':>8} {'N':>4}")
    print("-" * 90)

    systems = {}
    for r in results:
        sys_name = r["system"]
        if sys_name not in systems:
            systems[sys_name] = []
        systems[sys_name].append(r)

    for sys_name in BASELINES:
        if sys_name not in systems:
            continue
        sys_results = systems[sys_name]
        n = len(sys_results)
        avg_acc = sum(r["evaluation"]["accuracy"] for r in sys_results) / n
        avg_faith = sum(r["evaluation"]["faithfulness"] for r in sys_results) / n
        avg_comp = sum(r["evaluation"]["completeness"] for r in sys_results) / n
        avg_kw = sum(r["evaluation"]["keyword_overlap"] for r in sys_results) / n
        avg_lat = sum(r["latency_s"] for r in sys_results) / n

        label = BASELINES[sys_name]["name"]
        marker = " **" if sys_name == "specrag_full" else ""
        print(f"{label:<30} {avg_acc:>8.2f} {avg_faith:>9.2f} {avg_comp:>9.2f} "
              f"{avg_kw:>9.3f} {avg_lat:>7.1f}s {n:>4}{marker}")

    # ─── Table 2: Accuracy by Query Type ──────────────
    print(f"\n{'='*110}")
    print("TABLE 2: ACCURACY BY QUERY TYPE (SpecRAG vs Best Baseline)")
    print(f"{'='*110}")

    query_types = set(r["query_type"] for r in results)
    for qtype in sorted(query_types):
        type_results = [r for r in results if r["query_type"] == qtype]

        print(f"\n  Query Type: {qtype}")
        print(f"  {'System':<30} {'Accuracy':>9} {'Keywords':>9} {'N':>4}")
        print(f"  {'-'*55}")

        for sys_name in BASELINES:
            sys_type_results = [r for r in type_results if r["system"] == sys_name]
            if not sys_type_results:
                continue
            n = len(sys_type_results)
            avg_acc = sum(r["evaluation"]["accuracy"] for r in sys_type_results) / n
            avg_kw = sum(r["evaluation"]["keyword_overlap"] for r in sys_type_results) / n
            label = BASELINES[sys_name]["name"]
            marker = " **" if sys_name == "specrag_full" else ""
            print(f"  {label:<30} {avg_acc:>8.2f} {avg_kw:>9.3f} {n:>4}{marker}")

    # ─── Table 3: Accuracy by Dimension Count ─────────
    print(f"\n{'='*90}")
    print("TABLE 3: ACCURACY BY DIMENSION COUNT")
    print(f"{'='*90}")

    for dim_count in [1, 2, 3]:
        dim_queries = [q for q in SPECQA_QUERIES
                      if sum(1 for w in [q['alpha'], q['beta'], q['gamma']] if w > 0.1) == dim_count]
        dim_ids = {q["id"] for q in dim_queries}
        dim_results = [r for r in results if r["query_id"] in dim_ids]

        if not dim_results:
            continue

        print(f"\n  {dim_count}D Queries ({len(dim_queries)} queries):")
        print(f"  {'System':<30} {'Accuracy':>9} {'N':>4}")
        print(f"  {'-'*45}")

        for sys_name in BASELINES:
            sys_dim_results = [r for r in dim_results if r["system"] == sys_name]
            if not sys_dim_results:
                continue
            n = len(sys_dim_results)
            avg_acc = sum(r["evaluation"]["accuracy"] for r in sys_dim_results) / n
            label = BASELINES[sys_name]["name"]
            marker = " **" if sys_name == "specrag_full" else ""
            print(f"  {label:<30} {avg_acc:>8.2f} {n:>4}{marker}")

    print(f"\n{'='*90}")
    print("** = Our system (SpecRAG)")
    print(f"{'='*90}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpecRAG Experiments")
    parser.add_argument("--baseline", type=str, help="Run only this baseline")
    parser.add_argument("--query-id", type=str, help="Run only this query")
    parser.add_argument("--runs", type=int, default=1, help="Runs per query (for variance)")
    parser.add_argument("--analyze", action="store_true", help="Analyze saved results only")
    args = parser.parse_args()

    if args.analyze:
        generate_summary()
    else:
        run_experiments(
            target_baseline=args.baseline,
            target_query_id=args.query_id,
            num_runs=args.runs,
        )
