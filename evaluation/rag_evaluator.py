#!/usr/bin/env python3
"""
WYDOT RAG Evaluation System
Uses Vertex AI GenAI Evaluation Service to assess RAG quality.

Features:
- Groundedness, relevance, coherence metrics
- Custom WYDOT-specific evaluation metrics
- BigQuery storage for evaluation history
- Export from validator.py feedback for golden dataset
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

from dotenv import load_dotenv

# Google Cloud
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate

# Neo4j for testing against your RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BQ_DATASET = os.getenv("BQ_EVAL_DATASET", "wydot_eval")
BQ_TABLE = os.getenv("BQ_EVAL_TABLE", "rag_evaluations")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_INDEX_NAME = os.getenv("NEO4J_INDEX_NAME", "wydot_vector_index")

# Initialize Vertex AI
if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class EvalExample:
    """Single evaluation example."""
    id: str
    question: str
    reference_answer: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvalResult:
    """Evaluation result for a single example."""
    id: str
    question: str
    retrieved_context: str
    generated_answer: str
    reference_answer: Optional[str]
    
    # Core metrics
    groundedness: float
    answer_relevance: float
    context_relevance: float
    coherence: float
    
    # Custom metrics
    citation_accuracy: float
    section_reference_accuracy: float
    
    # Metadata
    evaluation_timestamp: str
    model_used: str
    retrieval_k: int
    latency_ms: float


# =========================================================
# RAG SYSTEM WRAPPER
# =========================================================

class WYDOTRagSystem:
    """Wrapper around your existing RAG system for evaluation."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.retriever = None
        self.model = GenerativeModel("gemini-2.0-flash")
        self._init_retriever()
    
    def _init_retriever(self):
        """Initialize Neo4j retriever."""
        try:
            vector_store = Neo4jVector.from_existing_index(
                self.embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name=NEO4J_INDEX_NAME,
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
            )
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            logger.info("Neo4j retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            self.retriever = None
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query."""
        if not self.retriever:
            return []
        try:
            docs = self.retriever.invoke(query)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer using context."""
        prompt = f"""You are a WYDOT expert assistant. Answer based on the context below.

CONTEXT:
{context}

QUESTION: {query}

Instructions:
- Provide accurate answers based on the context
- Cite specific sections using [SOURCE_X] format
- If the answer isn't in the context, say so clearly
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {e}"
    
    def query(self, question: str) -> tuple[str, str]:
        """Full RAG query - retrieve + generate."""
        chunks = self.retrieve(question)
        context = "\n\n".join(chunks) if chunks else "No relevant documents found."
        answer = self.generate(question, context)
        return context, answer


# =========================================================
# EVALUATION METRICS
# =========================================================

def create_groundedness_metric() -> PointwiseMetric:
    """Create groundedness metric - is the answer supported by context?"""
    return PointwiseMetric(
        metric="groundedness",
        metric_prompt_template=PointwiseMetricPromptTemplate(
            criteria={
                "groundedness": (
                    "Is the response fully grounded in the provided context? "
                    "The response should only contain information that can be "
                    "directly verified from the context. Any claims not supported "
                    "by the context should be penalized."
                ),
            },
            rating_rubric={
                "1": "Response contains significant claims not supported by context",
                "2": "Response contains some unsupported claims",
                "3": "Response is mostly grounded but has minor unsupported details",
                "4": "Response is well grounded with only trivial unsupported elements",
                "5": "Response is completely grounded in the provided context",
            },
        ),
    )


def create_answer_relevance_metric() -> PointwiseMetric:
    """Create answer relevance metric - does answer address the question?"""
    return PointwiseMetric(
        metric="answer_relevance",
        metric_prompt_template=PointwiseMetricPromptTemplate(
            criteria={
                "answer_relevance": (
                    "Does the response directly and completely address the user's "
                    "question? Consider whether the response provides the specific "
                    "information requested, is on-topic, and doesn't include "
                    "unnecessary information."
                ),
            },
            rating_rubric={
                "1": "Response is completely off-topic or doesn't address the question",
                "2": "Response partially addresses the question but misses key aspects",
                "3": "Response addresses the question but could be more complete",
                "4": "Response addresses the question well with minor omissions",
                "5": "Response perfectly addresses all aspects of the question",
            },
        ),
    )


def create_citation_accuracy_metric() -> PointwiseMetric:
    """Custom metric for WYDOT - are citations used correctly?"""
    return PointwiseMetric(
        metric="citation_accuracy",
        metric_prompt_template=PointwiseMetricPromptTemplate(
            criteria={
                "citation_accuracy": (
                    "Does the response correctly cite sources using [SOURCE_X] format? "
                    "Citations should be placed after relevant claims and reference "
                    "actual content from the context. Missing citations for factual "
                    "claims or incorrect source references should be penalized."
                ),
            },
            rating_rubric={
                "1": "No citations or completely incorrect citations",
                "2": "Few citations, many factual claims uncited",
                "3": "Some citations but inconsistent usage",
                "4": "Good citation coverage with minor gaps",
                "5": "Excellent citation accuracy and coverage",
            },
        ),
    )


def create_section_reference_metric() -> PointwiseMetric:
    """Custom metric for WYDOT - are section references accurate?"""
    return PointwiseMetric(
        metric="section_reference_accuracy",
        metric_prompt_template=PointwiseMetricPromptTemplate(
            criteria={
                "section_reference_accuracy": (
                    "When the response mentions WYDOT specification sections "
                    "(e.g., 'Section 501', 'Division 200'), are these references "
                    "accurate and verifiable in the provided context? Incorrect "
                    "or fabricated section numbers should be heavily penalized."
                ),
            },
            rating_rubric={
                "1": "Section references are fabricated or incorrect",
                "2": "Multiple section reference errors",
                "3": "Some section references are accurate, some questionable",
                "4": "Section references are mostly accurate",
                "5": "All section references are accurate and verifiable",
            },
        ),
    )


# =========================================================
# EVALUATION RUNNER
# =========================================================

class RAGEvaluator:
    """Run evaluations on the RAG system."""
    
    def __init__(self):
        self.rag_system = WYDOTRagSystem()
        self.bq_client = bigquery.Client(project=PROJECT_ID) if PROJECT_ID else None
        self._ensure_bq_table()
    
    def _ensure_bq_table(self):
        """Create BigQuery dataset and table if they don't exist."""
        if not self.bq_client:
            logger.warning("BigQuery client not initialized - results won't be stored")
            return
        
        dataset_ref = f"{PROJECT_ID}.{BQ_DATASET}"
        table_ref = f"{dataset_ref}.{BQ_TABLE}"
        
        # Create dataset
        try:
            self.bq_client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = LOCATION
            self.bq_client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Created BigQuery dataset: {dataset_ref}")
        
        # Create table with schema
        schema = [
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("question", "STRING"),
            bigquery.SchemaField("retrieved_context", "STRING"),
            bigquery.SchemaField("generated_answer", "STRING"),
            bigquery.SchemaField("reference_answer", "STRING"),
            bigquery.SchemaField("groundedness", "FLOAT"),
            bigquery.SchemaField("answer_relevance", "FLOAT"),
            bigquery.SchemaField("context_relevance", "FLOAT"),
            bigquery.SchemaField("coherence", "FLOAT"),
            bigquery.SchemaField("citation_accuracy", "FLOAT"),
            bigquery.SchemaField("section_reference_accuracy", "FLOAT"),
            bigquery.SchemaField("evaluation_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("model_used", "STRING"),
            bigquery.SchemaField("retrieval_k", "INTEGER"),
            bigquery.SchemaField("latency_ms", "FLOAT"),
        ]
        
        try:
            self.bq_client.get_table(table_ref)
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            self.bq_client.create_table(table, exists_ok=True)
            logger.info(f"Created BigQuery table: {table_ref}")
    
    def evaluate_single(self, example: EvalExample) -> EvalResult:
        """Evaluate a single example."""
        import time
        
        start_time = time.time()
        
        # Run RAG query
        context, answer = self.rag_system.query(example.question)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare eval data
        eval_data = {
            "prompt": example.question,
            "response": answer,
            "context": context,
        }
        if example.reference_answer:
            eval_data["reference"] = example.reference_answer
        
        # Run Vertex AI evaluation
        try:
            eval_task = EvalTask(
                dataset=[eval_data],
                metrics=[
                    create_groundedness_metric(),
                    create_answer_relevance_metric(),
                    create_citation_accuracy_metric(),
                    create_section_reference_metric(),
                    "coherence",
                    "fluency",
                ],
            )
            
            result = eval_task.evaluate()
            metrics = result.summary_metrics
            
            groundedness = metrics.get("groundedness/mean", 0.0)
            answer_relevance = metrics.get("answer_relevance/mean", 0.0)
            citation_accuracy = metrics.get("citation_accuracy/mean", 0.0)
            section_accuracy = metrics.get("section_reference_accuracy/mean", 0.0)
            coherence = metrics.get("coherence/mean", 0.0)
            
        except Exception as e:
            logger.error(f"Vertex AI evaluation failed: {e}")
            # Fallback to simple heuristic metrics
            groundedness = 0.0
            answer_relevance = 0.0
            citation_accuracy = self._simple_citation_check(answer)
            section_accuracy = self._simple_section_check(answer, context)
            coherence = 0.0
        
        return EvalResult(
            id=example.id,
            question=example.question,
            retrieved_context=context[:5000],  # Truncate for storage
            generated_answer=answer,
            reference_answer=example.reference_answer,
            groundedness=groundedness,
            answer_relevance=answer_relevance,
            context_relevance=0.0,  # Computed separately if needed
            coherence=coherence,
            citation_accuracy=citation_accuracy,
            section_reference_accuracy=section_accuracy,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            model_used="gemini-2.0-flash",
            retrieval_k=8,
            latency_ms=latency_ms,
        )
    
    def _simple_citation_check(self, answer: str) -> float:
        """Simple heuristic: count [SOURCE_X] citations."""
        import re
        citations = re.findall(r'\[SOURCE_\d+\]', answer)
        # Basic scoring: at least 1 citation = 0.5, 3+ = 1.0
        if len(citations) >= 3:
            return 1.0
        elif len(citations) >= 1:
            return 0.5
        return 0.0
    
    def _simple_section_check(self, answer: str, context: str) -> float:
        """Simple heuristic: verify section references exist in context."""
        import re
        sections_in_answer = set(re.findall(r'Section\s+\d+', answer, re.IGNORECASE))
        sections_in_context = set(re.findall(r'Section\s+\d+', context, re.IGNORECASE))
        
        if not sections_in_answer:
            return 1.0  # No sections mentioned, no errors
        
        valid = sections_in_answer.intersection(sections_in_context)
        return len(valid) / len(sections_in_answer) if sections_in_answer else 1.0
    
    def evaluate_batch(self, examples: List[EvalExample]) -> List[EvalResult]:
        """Evaluate a batch of examples."""
        results = []
        for i, example in enumerate(examples):
            logger.info(f"Evaluating {i+1}/{len(examples)}: {example.question[:50]}...")
            result = self.evaluate_single(example)
            results.append(result)
            
            # Store to BigQuery
            if self.bq_client:
                self._store_result(result)
        
        return results
    
    def _store_result(self, result: EvalResult):
        """Store evaluation result to BigQuery."""
        table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        
        row = {
            "id": result.id,
            "question": result.question,
            "retrieved_context": result.retrieved_context,
            "generated_answer": result.generated_answer,
            "reference_answer": result.reference_answer,
            "groundedness": result.groundedness,
            "answer_relevance": result.answer_relevance,
            "context_relevance": result.context_relevance,
            "coherence": result.coherence,
            "citation_accuracy": result.citation_accuracy,
            "section_reference_accuracy": result.section_reference_accuracy,
            "evaluation_timestamp": result.evaluation_timestamp,
            "model_used": result.model_used,
            "retrieval_k": result.retrieval_k,
            "latency_ms": result.latency_ms,
        }
        
        try:
            errors = self.bq_client.insert_rows_json(table_ref, [row])
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
    
    def generate_report(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Generate summary report from evaluation results."""
        if not results:
            return {"error": "No results to report"}
        
        def avg(values):
            return sum(values) / len(values) if values else 0.0
        
        return {
            "total_evaluated": len(results),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "groundedness": {
                    "mean": avg([r.groundedness for r in results]),
                    "min": min([r.groundedness for r in results]),
                    "max": max([r.groundedness for r in results]),
                },
                "answer_relevance": {
                    "mean": avg([r.answer_relevance for r in results]),
                    "min": min([r.answer_relevance for r in results]),
                    "max": max([r.answer_relevance for r in results]),
                },
                "citation_accuracy": {
                    "mean": avg([r.citation_accuracy for r in results]),
                    "min": min([r.citation_accuracy for r in results]),
                    "max": max([r.citation_accuracy for r in results]),
                },
                "section_reference_accuracy": {
                    "mean": avg([r.section_reference_accuracy for r in results]),
                    "min": min([r.section_reference_accuracy for r in results]),
                    "max": max([r.section_reference_accuracy for r in results]),
                },
                "coherence": {
                    "mean": avg([r.coherence for r in results]),
                },
                "latency_ms": {
                    "mean": avg([r.latency_ms for r in results]),
                    "p50": sorted([r.latency_ms for r in results])[len(results)//2],
                    "p95": sorted([r.latency_ms for r in results])[int(len(results)*0.95)] if len(results) >= 20 else max([r.latency_ms for r in results]),
                },
            },
        }


# =========================================================
# DATASET UTILITIES
# =========================================================

def load_golden_dataset_from_validator(db_path: str = "feedback.db") -> List[EvalExample]:
    """Load validated Q&A pairs from validator.py's SQLite database."""
    import sqlite3
    
    if not os.path.exists(db_path):
        logger.warning(f"Validator database not found: {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            item_id,
            original_question,
            suggested_answer,
            original_context
        FROM feedback 
        WHERE correctness = 'Correct'
    """)
    
    examples = []
    for row in cursor.fetchall():
        item_id, question, answer, context = row
        examples.append(EvalExample(
            id=item_id,
            question=question,
            reference_answer=answer,
            context=context,
        ))
    
    conn.close()
    logger.info(f"Loaded {len(examples)} validated examples from feedback database")
    return examples


def load_examples_from_jsonl(path: str) -> List[EvalExample]:
    """Load evaluation examples from JSONL file."""
    examples = []
    
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Handle Gemini tuning format
            if "contents" in data:
                user_text = ""
                model_text = ""
                for msg in data.get("contents", []):
                    for part in msg.get("parts", []):
                        text = part.get("text", "")
                        if msg.get("role") == "user":
                            user_text += text + "\n"
                        elif msg.get("role") == "model":
                            model_text += text + "\n"
                
                # Extract question from user text
                import re
                parts = re.split(r'(?i)\bQUESTION:\s*', user_text, maxsplit=1)
                question = parts[1].strip() if len(parts) > 1 else user_text.strip()
                
                example_id = hashlib.md5(question.encode()).hexdigest()[:12]
                examples.append(EvalExample(
                    id=example_id,
                    question=question,
                    reference_answer=model_text.strip(),
                ))
            else:
                # Standard format
                question = data.get("question", data.get("query", ""))
                example_id = data.get("id", hashlib.md5(question.encode()).hexdigest()[:12])
                examples.append(EvalExample(
                    id=example_id,
                    question=question,
                    reference_answer=data.get("answer", data.get("reference_answer")),
                    context=data.get("context"),
                ))
    
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def generate_synthetic_examples(num_examples: int = 20) -> List[EvalExample]:
    """Generate synthetic test queries using Gemini."""
    if not PROJECT_ID:
        logger.warning("GCP_PROJECT_ID not set, cannot generate synthetic examples")
        return []
    
    model = GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""Generate {num_examples} diverse test questions that someone might ask about WYDOT (Wyoming Department of Transportation) specifications.

Include a mix of:
1. Simple lookups: "What is the concrete strength requirement in Section 501?"
2. Comparison questions: "What are the differences between Type A and Type B aggregate?"
3. Multi-part questions: "What are all the requirements for bridge deck construction?"
4. Edge cases: Misspellings, ambiguous phrasing

Output as JSONL format, one JSON object per line:
{{"question": "...", "difficulty": "simple|medium|complex"}}

Generate exactly {num_examples} questions:"""
    
    try:
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        
        examples = []
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    question = data.get("question", "")
                    if question:
                        example_id = hashlib.md5(question.encode()).hexdigest()[:12]
                        examples.append(EvalExample(
                            id=f"synthetic_{example_id}",
                            question=question,
                            metadata={"difficulty": data.get("difficulty", "unknown")},
                        ))
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Generated {len(examples)} synthetic examples")
        return examples
    
    except Exception as e:
        logger.error(f"Failed to generate synthetic examples: {e}")
        return []


# =========================================================
# CLI
# =========================================================

def main():
    """Run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WYDOT RAG Evaluation System")
    parser.add_argument("--source", choices=["validator", "jsonl", "synthetic"], 
                        default="synthetic", help="Data source for evaluation")
    parser.add_argument("--jsonl-path", help="Path to JSONL file (if source=jsonl)")
    parser.add_argument("--db-path", default="feedback.db", help="Validator database path")
    parser.add_argument("--num-synthetic", type=int, default=10, help="Number of synthetic examples")
    parser.add_argument("--output", help="Output JSON file for report")
    
    args = parser.parse_args()
    
    # Load examples
    if args.source == "validator":
        examples = load_golden_dataset_from_validator(args.db_path)
    elif args.source == "jsonl":
        if not args.jsonl_path:
            print("Error: --jsonl-path required when source=jsonl")
            return
        examples = load_examples_from_jsonl(args.jsonl_path)
    else:
        examples = generate_synthetic_examples(args.num_synthetic)
    
    if not examples:
        print("No examples to evaluate")
        return
    
    print(f"Evaluating {len(examples)} examples...")
    
    # Run evaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(examples)
    
    # Generate report
    report = evaluator.generate_report(results)
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(json.dumps(report, indent=2))
    
    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
