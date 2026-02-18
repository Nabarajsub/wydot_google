"""
Online RAG telemetry: record latency and metrics per request (local SQLite; Cloud: BigQuery).
"""
import os
import re
import time
import threading
import uuid
from typing import Optional

# Local SQLite path (overridable by env)
TELEMETRY_DB_PATH = os.getenv("TELEMETRY_DB_PATH", "")

_lock = threading.Lock()
_conn = None

def _get_placeholder():
    return "%s" if os.getenv("DATABASE_URL") and os.getenv("DATABASE_URL", "").startswith("postgres") else "?"


def _get_conn():
    global _conn
    if _conn is not None:
        return _conn
        
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres"):
        import psycopg2
        from urllib.parse import urlparse, unquote, parse_qs
        try:
            # Parse URL into explicit params (avoids %23 encoding issues with psycopg2)
            clean = db_url.replace("postgresql+psycopg2://", "postgresql://")
            parsed = urlparse(clean)
            pg_params = {
                "user": unquote(parsed.username) if parsed.username else "postgres",
                "password": unquote(parsed.password) if parsed.password else "",
                "dbname": parsed.path.lstrip("/") or "postgres",
            }
            qs = parse_qs(parsed.query)
            if "host" in qs:
                pg_params["host"] = qs["host"][0]
            elif parsed.hostname:
                pg_params["host"] = parsed.hostname
                if parsed.port:
                    pg_params["port"] = str(parsed.port)
            print(f"📊 [telemetry] Connecting to PG: dbname={pg_params['dbname']}, host={pg_params.get('host', '(default)')}", flush=True)
            _conn = psycopg2.connect(**pg_params)
            # Create table with Postgres syntax
            cur = _conn.cursor()
            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS request_metrics (
                        id TEXT PRIMARY KEY,
                        ts DOUBLE PRECISION NOT NULL,
                        user_id INTEGER,
                        thread_id TEXT,
                        session_id TEXT,
                        model_used TEXT,
                        index_used TEXT,
                        retrieval_latency_ms DOUBLE PRECISION,
                        generation_latency_ms DOUBLE PRECISION,
                        total_latency_ms DOUBLE PRECISION,
                        num_sources INTEGER,
                        citation_count INTEGER,
                        has_error INTEGER DEFAULT 0
                    )
                """)
            finally:
                cur.close()
            _conn.commit()
            return _conn
        except Exception as e:
            print(f"⚠️ Telemetry fallback to SQLite (Postgres failed): {e}")

    # Fallback to SQLite
    path = TELEMETRY_DB_PATH or os.path.join(os.path.dirname(__file__), "..", "telemetry.sqlite3")
    import sqlite3
    db_dir = os.path.dirname(path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS request_metrics (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            user_id INTEGER,
            thread_id TEXT,
            session_id TEXT,
            model_used TEXT,
            index_used TEXT,
            retrieval_latency_ms REAL,
            generation_latency_ms REAL,
            total_latency_ms REAL,
            num_sources INTEGER,
            citation_count INTEGER,
            has_error INTEGER DEFAULT 0
        )
    """)
    _conn.commit()
    return _conn


def count_citations(text: str) -> int:
    """Count [Source N] or [N] style citations in response."""
    if not text:
        return 0
    # [Source 1], [Source 2] or [1], [2]
    return len(re.findall(r"\[Source\s*\d+\]|\[\d+\]", text))


def record_request(
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    num_sources: int = 0,
    citation_count: Optional[int] = None,
    response_text: Optional[str] = None,
    model_used: str = "",
    index_used: str = "",
    user_id: Optional[int] = None,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    has_error: bool = False,
) -> None:
    """Record one RAG request. Runs in background so it does not block the response."""
    if citation_count is None and response_text is not None:
        citation_count = count_citations(response_text)

    def _write():
        try:
            with _lock:
                c = _get_conn()
                cur = c.cursor()
                try:
                    cur.execute(
                        """INSERT INTO request_metrics (
                            id, ts, user_id, thread_id, session_id, model_used, index_used,
                            retrieval_latency_ms, generation_latency_ms, total_latency_ms,
                            num_sources, citation_count, has_error
                        ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""".format(ph=_get_placeholder()),
                        (
                            str(uuid.uuid4()),
                            time.time(),
                            user_id,
                            thread_id,
                            session_id,
                            model_used,
                            index_used,
                            retrieval_latency_ms,
                            generation_latency_ms,
                            total_latency_ms,
                            num_sources,
                            citation_count or 0,
                            1 if has_error else 0,
                        ),
                    )
                finally:
                    cur.close()
                c.commit()
        except Exception:
            try:
                c = _get_conn()
                if hasattr(c, "rollback"):
                    c.rollback()
            except:
                pass

    t = threading.Thread(target=_write, daemon=True)
    t.start()


def get_recent_metrics(limit: int = 100):
    """Return recent rows for local dashboards/debugging."""
    try:
        with _lock:
            c = _get_conn()
            cur = c.cursor()
            try:
                cur.execute(
                    f"SELECT * FROM request_metrics ORDER BY ts DESC LIMIT {_get_placeholder()}",
                    (limit,),
                )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in rows]
            finally:
                cur.close()
    except Exception:
        return []

def get_latency_stats() -> dict:
    """Return AVG, P50, P95, P99 latency stats."""
    try:
        with _lock:
            c = _get_conn()
            # SQLite specific approximation for P50/95/99 using ORDER BY + LIMIT/OFFSET
            # For simplicity, we'll fetch all non-error latencies and compute in Python
            # unless the dataset is huge (metrics are small rows, should be fine for <100k rows)
            cur = c.cursor()
            try:
                cur.execute(
                    "SELECT total_latency_ms, retrieval_latency_ms, generation_latency_ms FROM request_metrics WHERE has_error=0 ORDER BY total_latency_ms ASC"
                )
                rows = cur.fetchall()
            finally:
                cur.close()
            
            if not rows:
                return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0,
                        "avg_retrieval": 0, "avg_generation": 0}

            total_count = len(rows)
            # Compute averages
            totals = [r[0] for r in rows]
            avg = sum(totals) / total_count
            avg_retrieval = sum(r[1] for r in rows) / total_count
            avg_generation = sum(r[2] for r in rows) / total_count
            
            # Compute percentiles
            def pct(p):
                idx = int(total_count * p)
                return totals[min(idx, total_count - 1)]

            return {
                "count": total_count,
                "avg": round(avg, 1),
                "p50": round(pct(0.50), 1),
                "p95": round(pct(0.95), 1),
                "p99": round(pct(0.99), 1),
                "avg_retrieval": round(avg_retrieval, 1),
                "avg_generation": round(avg_generation, 1)
            }
    except Exception:
        return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "avg_retrieval": 0, "avg_generation": 0}

def get_timeseries(interval_hours: int = 1) -> list:
    """Return metrics aggregated by time bucket."""
    # SQLite doesn't have great date truncation, we'll group by TS / 3600
    try:
        with _lock:
            c = _get_conn()
            # Group by hour
            group_seconds = interval_hours * 3600
            sql = f"""
                SELECT 
                    (CAST(ts AS INTEGER) / {group_seconds}) * {group_seconds} as bucket_ts,
                    COUNT(*) as count,
                    AVG(total_latency_ms) as avg_latency,
                    SUM(has_error) as error_count
                FROM request_metrics
                GROUP BY bucket_ts
                ORDER BY bucket_ts ASC
                LIMIT 100
            """
            cur = c.cursor()
            try:
                cur.execute(sql)
                rows = cur.fetchall()
                return [
                    {
                        "ts": r[0],
                        "count": r[1],
                        "avg_latency": round(r[2] or 0, 1),
                        "error_count": r[3]
                    }
                    for r in rows
                ]
            finally:
                cur.close()
    except Exception:
        return []

def get_model_comparison() -> list:
    """Return stats grouped by model."""
    try:
        with _lock:
            c = _get_conn()
            sql = """
                SELECT 
                    model_used,
                    COUNT(*) as count,
                    AVG(total_latency_ms) as avg_latency,
                    AVG(generation_latency_ms) as avg_gen_latency,
                    SUM(has_error) as errors
                FROM request_metrics
                GROUP BY model_used
            """
            cur = c.cursor()
            try:
                cur.execute(sql)
                rows = cur.fetchall()
                return [
                    {
                        "model": r[0] or "Unknown",
                        "count": r[1],
                        "avg_latency": round(r[2] or 0, 1),
                        "avg_gen_latency": round(r[3] or 0, 1),
                        "error_rate": round((r[4] / r[1] * 100) if r[1] > 0 else 0, 1)
                    }
                    for r in rows
                ]
            finally:
                cur.close()
    except Exception:
        return []
