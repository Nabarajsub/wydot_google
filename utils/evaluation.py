"""
Evaluation module for WYDOT RAG Chatbot.
Handles:
1. Offline Evaluation: Scoring 5 hardcoded QnA pairs against the live pipeline.
2. Online Evaluation: Computing per-question scores (relevancy, etc.) and aggregating.
"""
import time
import uuid
import threading
import sqlite3
import os
import json
import logging
from typing import List, Dict, Optional, Any

# You may need to import your embedding function or LLM here for scoring
# For simplicity, we will use heuristic/keyword based scoring for now to avoid
# heavy dependencies or extra API costs, but structure it for model-based later.

logger = logging.getLogger("chainlit")

# -----------------------------------------------------------------------------
# Database Setup
# -----------------------------------------------------------------------------
EVAL_DB_PATH = os.getenv("EVAL_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "evaluation.sqlite3"))

_lock = threading.Lock()
_conn = None

def _get_conn():
    global _conn
    if _conn is not None:
        return _conn
    import sqlite3
    db_dir = os.path.dirname(EVAL_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(EVAL_DB_PATH, check_same_thread=False, timeout=5)
    _conn.execute("PRAGMA journal_mode=WAL;")
    
    # 1. Online Validation Table (per question)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS online_evals (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            session_id TEXT,
            question TEXT,
            answer TEXT,
            context TEXT,
            num_sources INTEGER,
            latency_ms REAL,
            -- Scores
            answer_relevancy REAL,   -- 0-1
            context_utilization REAL,-- 0-1
            completeness REAL,       -- 0-1
            has_error INTEGER DEFAULT 0
        )
    """)
    
    # 2. Aggregates Table (every 10 questions)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS online_aggregates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start REAL,
            ts_end REAL,
            count INTEGER,
            avg_relevancy REAL,
            avg_utilization REAL,
            avg_completeness REAL,
            error_rate REAL
        )
    """)
    
    # 3. Offline Results Table
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS offline_runs (
            id TEXT PRIMARY KEY,
            ts REAL,
            total_score REAL,
            details_json TEXT -- Stored JSON of individual QnA results
        )
    """)
    
    _conn.commit()
    return _conn

# -----------------------------------------------------------------------------
# 1. Verification/Offline Evaluation Logic
# -----------------------------------------------------------------------------

GOLDEN_DATASET = [
    {
        "question": "What is the required concrete strength for bridge decks?",
        "expected_answer_keywords": ["4000", "psi", "deck", "concrete"],
        "expected_context_keywords": ["Section", "4000", "psi", "Class", "A"]
    },
    {
        "question": "What is the maximum aggregate size for asphalt mix?",
        "expected_answer_keywords": ["aggregate", "size", "inch", "maximum"],
        "expected_context_keywords": ["gradation", "aggregate", "sieve"]
    },
    {
        "question": "How often should traffic control devices be inspected?",
        "expected_answer_keywords": ["daily", "inspection", "traffic", "control"],
        "expected_context_keywords": ["traffic", "control", "maintenance", "check"]
    },
    {
        "question": "What are the requirements for silt fence installation?",
        "expected_answer_keywords": ["silt", "fence", "post", "embedment"],
        "expected_context_keywords": ["erosion", "control", "silt", "fence"]
    },
    {
        "question": "What is the procedure for bidding on a construction project?",
        "expected_answer_keywords": ["bid", "proposal", "submit", "electronic"],
        "expected_context_keywords": ["bidder", "proposal", "contract", "letting"]
    }
]

def calculate_overlap_score(text: str, keywords: List[str]) -> float:
    """Simple keyword overlap score (0.0 to 1.0)."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for k in keywords if k.lower() in text_lower)
    return matches / len(keywords)

async def run_offline_evaluation(search_func, generate_func) -> Dict:
    """
    Run evaluation against the live pipeline functions.
    search_func: (query) -> context, sources
    generate_func: (query, context, history) -> answer
    """
    results = []
    total_score = 0
    
    for item in GOLDEN_DATASET:
        q = item["question"]
        t0 = time.time()
        
        # 1. Retrieval
        try:
            context, sources = search_func(q, "All Documents") # Default to All Docs
            context_recall = calculate_overlap_score(context, item["expected_context_keywords"])
        except Exception as e:
            context = ""
            context_recall = 0.0
            
        # 2. Generation
        try:
            # Empty history for independent eval
            answer = generate_func(q, context, [])
            answer_correctness = calculate_overlap_score(answer, item["expected_answer_keywords"])
        except Exception as e:
            answer = ""
            answer_correctness = 0.0
            
        latency = (time.time() - t0) * 1000
        
        # Composite score
        item_score = (context_recall * 0.4) + (answer_correctness * 0.6)
        
        results.append({
            "question": q,
            "answer_preview": answer[:100] + "..." if answer else "",
            "context_recall": round(context_recall, 2),
            "answer_correctness": round(answer_correctness, 2),
            "latency_ms": round(latency, 1),
            "total_score": round(item_score, 2)
        })
        total_score += item_score

    avg_score = total_score / len(GOLDEN_DATASET) if GOLDEN_DATASET else 0
    
    # Save to DB
    run_id = str(uuid.uuid4())
    try:
        with _lock:
            c = _get_conn()
            c.execute(
                "INSERT INTO offline_runs (id, ts, total_score, details_json) VALUES (?, ?, ?, ?)",
                (run_id, time.time(), round(avg_score, 2), json.dumps(results))
            )
            c.commit()
    except Exception as e:
        logger.error(f"Error saving offline eval: {e}")

    return {
        "run_id": run_id,
        "average_score": round(avg_score, 2),
        "results": results
    }

def get_latest_offline_result() -> Optional[Dict]:
    """Get the most recent offline run."""
    try:
        with _lock:
            c = _get_conn()
            cur = c.execute("SELECT ts, total_score, details_json FROM offline_runs ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                return {
                    "ts": row[0],
                    "score": row[1],
                    "results": json.loads(row[2])
                }
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# 2. Online Evaluation Logic
# -----------------------------------------------------------------------------

def util_compute_relevancy(question: str, answer: str) -> float:
    """Heuristic: Check if answer length > 50 chars and shares words with question."""
    if not answer or len(answer) < 10:
        return 0.1
    
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    a_words = set(w.lower() for w in answer.split())
    
    if not q_words: return 0.5
    common = q_words.intersection(a_words)
    # If we answer the keywords, likely relevant. Plus length bonus.
    return min(0.5 + (len(common) / len(q_words)) * 0.5, 1.0)

def record_online_eval(
    session_id: str,
    question: str,
    answer: str,
    context: str,
    num_sources: int,
    latency_ms: float,
    has_error: bool
):
    """Compute scores and save to DB. Check if aggregation needed."""
    
    # Compute scores
    answer_relevancy = util_compute_relevancy(question, answer)
    context_utilization = 1.0 if num_sources > 0 else 0.0 # binary helper
    if num_sources > 0 and "[SOURCE" in answer:
        context_utilization = 1.0
    
    completeness = min(len(answer) / 200.0, 1.0) # Cap at 200 chars for "complete"
    
    if has_error:
        answer_relevancy = 0
        context_utilization = 0
        completeness = 0

    def _write():
        try:
            with _lock:
                c = _get_conn()
                # Insert row
                c.execute("""
                    INSERT INTO online_evals (
                        id, ts, session_id, question, answer, context, 
                        num_sources, latency_ms, answer_relevancy, 
                        context_utilization, completeness, has_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()), time.time(), session_id, question, answer, 
                    context[:500], num_sources, latency_ms, 
                    answer_relevancy, context_utilization, completeness, 
                    1 if has_error else 0
                ))
                c.commit()
                
                # Aggregate on every request for real-time dashboard updates
                _aggregate_last_10(c)
                    
        except Exception as e:
            logger.error(f"Error recording online eval: {e}")

    threading.Thread(target=_write, daemon=True).start()

def _aggregate_last_10(conn):
    """Aggregate stats for the last 10 requests."""
    # Create a new cursor for this operation
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*), 
            AVG(answer_relevancy), 
            AVG(context_utilization), 
            AVG(completeness),
            SUM(has_error),
            MIN(ts),
            MAX(ts)
        FROM (SELECT * FROM online_evals ORDER BY ts DESC LIMIT 10)
    """)
    row = cursor.fetchone()
    if row:
        count, avg_rel, avg_util, avg_comp, errs, t_start, t_end = row
        err_rate = (errs / count) if count > 0 else 0
        cursor.execute("""
            INSERT INTO online_aggregates (
                ts_start, ts_end, count, avg_relevancy, 
                avg_utilization, avg_completeness, error_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (t_start, t_end, count, avg_rel, avg_util, avg_comp, err_rate))
        conn.commit()

def get_online_stats() -> Dict:
    """Get current aggregate stats (from latest aggregate window)."""
    try:
        with _lock:
            c = _get_conn()
            cur = c.execute("""
                SELECT avg_relevancy, avg_utilization, avg_completeness, error_rate 
                FROM online_aggregates ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    "relevancy": round(row[0], 2),
                    "utilization": round(row[1], 2),
                    "completeness": round(row[2], 2),
                    "error_rate": round(row[3], 2)
                }
            
            # Fallback to raw average of all time if no aggregates yet
            cur = c.execute("""
                SELECT AVG(answer_relevancy), AVG(context_utilization), AVG(completeness), AVG(has_error)
                FROM online_evals
            """)
            row = cur.fetchone()
            if row and row[0] is not None:
                 return {
                    "relevancy": round(row[0], 2),
                    "utilization": round(row[1], 2),
                    "completeness": round(row[2], 2),
                    "error_rate": round(row[3], 2)
                }
    except Exception:
        pass
    return {"relevancy": 0, "utilization": 0, "completeness": 0, "error_rate": 0}

def get_online_trends() -> List[Dict]:
    """Get trend of scores over time (last 20 aggregates)."""
    try:
        with _lock:
            c = _get_conn()
            cur = c.execute("""
                SELECT ts_end, avg_relevancy, avg_utilization, error_rate 
                FROM online_aggregates ORDER BY id DESC LIMIT 20
            """)
            rows = cur.fetchall()
            # Return in chronological order
            return [
                {
                    "ts": r[0], 
                    "relevancy": round(r[1], 2), 
                    "utilization": round(r[2], 2), 
                    "error_rate": round(r[3], 2)
                } 
                for r in reversed(rows)
            ]
    except Exception:
        return []
