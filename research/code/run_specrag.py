"""
SpecRAG Master Runner
======================
Orchestrates the full SpecRAG pipeline from data to paper-ready results.

Usage:
    python run_specrag.py --phase 1      # Build PageIndex trees
    python run_specrag.py --phase 2      # Ingest corpus to Neo4j
    python run_specrag.py --phase 3      # Generate temporal diffs
    python run_specrag.py --phase 4      # Test query classifier
    python run_specrag.py --phase 5      # Export SpecQA benchmark
    python run_specrag.py --phase 6      # Run experiments
    python run_specrag.py --all          # Run everything sequentially
    python run_specrag.py --query "..."  # Interactive SpecRAG query
"""
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("specrag_run.log"),
    ]
)
logger = logging.getLogger(__name__)


def run_phase1():
    """Phase 1: Build PageIndex trees for all 19 documents."""
    print("\n" + "=" * 70)
    print("PHASE 1: Building PageIndex Trees")
    print("=" * 70)
    from phase1_pageindex.build_trees import build_all_trees
    results = build_all_trees()
    return results


def run_phase2(dry_run=False, skip_entities=False):
    """Phase 2: Ingest full corpus into Neo4j."""
    print("\n" + "=" * 70)
    print("PHASE 2: Neo4j Corpus Ingestion")
    print("=" * 70)
    from phase2_ingestion.ingest_corpus import ingest_corpus
    ingest_corpus(dry_run=dry_run, skip_entities=skip_entities)


def run_phase3(chain=None):
    """Phase 3: Generate temporal diffs between editions."""
    print("\n" + "=" * 70)
    print("PHASE 3: Temporal Diff Generation")
    print("=" * 70)
    from phase3_temporal.diff_engine import run_diff_pipeline
    results = run_diff_pipeline(chain_name=chain)
    return results


def run_phase4(query=None):
    """Phase 4: Test query classifier."""
    print("\n" + "=" * 70)
    print("PHASE 4: Query-Type Classifier Test")
    print("=" * 70)
    from phase4_router.query_classifier import classify_query
    import json

    test_queries = [
        "What does Division 500 cover?",
        "What is the minimum compressive strength for Class A concrete?",
        "How did aggregate specifications change between 2010 and 2021?",
        "In Section 501, what materials are specified and how have they changed since 2010?",
    ]

    if query:
        test_queries = [query]

    for q in test_queries:
        result = classify_query(q)
        print(f"\nQuery: {q}")
        print(f"  Type: {result['type']}")
        print(f"  α={result['alpha']:.2f}  β={result['beta']:.2f}  γ={result['gamma']:.2f}")
        print(f"  Dims: {result['dimensions_active']}")


def run_phase5():
    """Phase 5: Export SpecQA benchmark."""
    print("\n" + "=" * 70)
    print("PHASE 5: SpecQA Benchmark Export")
    print("=" * 70)
    from phase5_benchmark.specqa_queries import export_benchmark, print_stats
    export_benchmark()
    print_stats()


def run_phase6(baseline=None, query_id=None):
    """Phase 6: Run experiments."""
    print("\n" + "=" * 70)
    print("PHASE 6: Running Experiments")
    print("=" * 70)
    from phase6_evaluation.run_experiments import run_experiments
    run_experiments(target_baseline=baseline, target_query_id=query_id)


def run_query(query):
    """Run a single SpecRAG query interactively."""
    from phase4_router.specrag_router import specrag_query
    import json

    result = specrag_query(query, verbose=True)

    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"\n{'='*60}")
    print("METRICS:")
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nPROVENANCE:")
    print(json.dumps(result["provenance"], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="SpecRAG Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_specrag.py --phase 1                # Build PageIndex trees (no API cost)
  python run_specrag.py --phase 2 --dry-run      # Preview ingestion
  python run_specrag.py --phase 2 --skip-entities # Fast ingestion (no entity extraction)
  python run_specrag.py --phase 3 --chain standard_specs  # Diff only standard specs
  python run_specrag.py --phase 5                # Export benchmark
  python run_specrag.py --phase 6 --baseline specrag_full  # Run only SpecRAG
  python run_specrag.py --query "What is Class A concrete?"  # Interactive query
  python run_specrag.py --all                    # Full pipeline
        """
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="Run specific phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--query", type=str, help="Run a single SpecRAG query")
    parser.add_argument("--dry-run", action="store_true", help="Phase 2: preview only")
    parser.add_argument("--skip-entities", action="store_true",
                       help="Phase 2: skip entity extraction")
    parser.add_argument("--chain", type=str,
                       help="Phase 3: specific temporal chain")
    parser.add_argument("--baseline", type=str, help="Phase 6: specific baseline")
    parser.add_argument("--query-id", type=str, help="Phase 6: specific query ID")

    args = parser.parse_args()

    if args.query:
        run_query(args.query)
    elif args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2(dry_run=args.dry_run, skip_entities=args.skip_entities)
    elif args.phase == 3:
        run_phase3(chain=args.chain)
    elif args.phase == 4:
        run_phase4(query=args.query)
    elif args.phase == 5:
        run_phase5()
    elif args.phase == 6:
        run_phase6(baseline=args.baseline, query_id=args.query_id)
    elif args.all:
        run_phase1()
        run_phase2(skip_entities=args.skip_entities)
        run_phase3()
        run_phase5()
        run_phase6()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
