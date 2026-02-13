#!/usr/bin/env python3
"""
Run RAG evaluation locally using a curated dataset (JSONL).
- Loads from evaluation/datasets/ (e.g. wydot_golden_sample.jsonl).
- Uses rag_evaluator.RAGEvaluator (Neo4j + Vertex AI or fallback heuristics).
- Writes report to evaluation/reports/local_report_<timestamp>.json (no BigQuery required).

Usage:
  cd evaluation && python run_local.py
  python run_local.py --dataset datasets/wydot_golden_sample.jsonl --output reports/local_report.json

Requires: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env.
For full LLM-based metrics: GCP_PROJECT_ID and Vertex AI enabled.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Run from evaluation/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(Path(__file__).resolve().parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

from rag_evaluator import (
    RAGEvaluator,
    load_examples_from_jsonl,
    EvalExample,
)


def main():
    parser = argparse.ArgumentParser(description="Run WYDOT RAG evaluation locally")
    parser.add_argument("--dataset", default="datasets/wydot_golden_sample.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output", default="", help="Output report path (default: reports/local_report_<timestamp>.json)")
    parser.add_argument("--output-dir", default="reports", help="Output directory for report")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent / dataset_path
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Create evaluation/datasets/wydot_golden_sample.jsonl with one JSON object per line: question, reference_answer, id")
        sys.exit(1)

    examples = load_examples_from_jsonl(str(dataset_path))
    if not examples:
        print("No examples loaded from dataset")
        sys.exit(1)
    print(f"Loaded {len(examples)} examples from {dataset_path}")

    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(examples)
    report = evaluator.generate_report(results)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = out_dir / out_path.name
    else:
        out_path = out_dir / f"local_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path}")

    print("\n" + "=" * 60)
    print("LOCAL EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
