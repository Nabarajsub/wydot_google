#!/usr/bin/env python3
"""
Cloud Run Job for scheduled RAG evaluation.
Runs as a batch job triggered by Cloud Scheduler.
"""

import os
import json
import logging
from datetime import datetime

from rag_evaluator import (
    RAGEvaluator,
    load_golden_dataset_from_validator,
    load_examples_from_jsonl,
    generate_synthetic_examples,
)

from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GCS_BUCKET = os.getenv("GCS_EVAL_BUCKET", "wydot-evaluations")
EVAL_SOURCE = os.getenv("EVAL_SOURCE", "synthetic")  # synthetic, validator, jsonl
JSONL_PATH = os.getenv("JSONL_PATH", "geminituning_val.jsonl")
DB_PATH = os.getenv("DB_PATH", "feedback.db")
NUM_SYNTHETIC = int(os.getenv("NUM_SYNTHETIC", "10"))


def upload_report_to_gcs(report: dict, bucket_name: str) -> str:
    """Upload evaluation report to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    blob_name = f"reports/eval_report_{timestamp}.json"
    
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(report, indent=2),
        content_type="application/json"
    )
    
    logger.info(f"Report uploaded to gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"


def main():
    """Run scheduled evaluation job."""
    logger.info("Starting scheduled RAG evaluation...")
    
    # Load examples based on source
    if EVAL_SOURCE == "validator":
        examples = load_golden_dataset_from_validator(DB_PATH)
    elif EVAL_SOURCE == "jsonl":
        examples = load_examples_from_jsonl(JSONL_PATH)
    else:
        examples = generate_synthetic_examples(NUM_SYNTHETIC)
    
    if not examples:
        logger.warning("No examples to evaluate - exiting")
        return
    
    logger.info(f"Evaluating {len(examples)} examples...")
    
    # Run evaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(examples)
    
    # Generate report
    report = evaluator.generate_report(results)
    report["job_metadata"] = {
        "source": EVAL_SOURCE,
        "num_examples": len(examples),
        "run_timestamp": datetime.utcnow().isoformat(),
    }
    
    # Upload to GCS
    try:
        gcs_path = upload_report_to_gcs(report, GCS_BUCKET)
        logger.info(f"Report saved to: {gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload report: {e}")
    
    # Log summary
    logger.info("="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total evaluated: {report['total_evaluated']}")
    logger.info(f"Groundedness (mean): {report['metrics']['groundedness']['mean']:.2f}")
    logger.info(f"Answer Relevance (mean): {report['metrics']['answer_relevance']['mean']:.2f}")
    logger.info(f"Citation Accuracy (mean): {report['metrics']['citation_accuracy']['mean']:.2f}")
    logger.info(f"Latency p50: {report['metrics']['latency_ms']['p50']:.0f}ms")


if __name__ == "__main__":
    main()
