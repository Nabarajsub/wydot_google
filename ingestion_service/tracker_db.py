"""
Ingestion tracker: SQLite for local (and Cloud SQL later).
Replaces JSON file so the same code works locally and on Cloud Run with a persistent volume or Cloud SQL.
"""
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

_db_path = os.getenv("INGESTION_TRACKER_DB", "")
_conn = None
_lock = threading.Lock()


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
            _conn = psycopg2.connect(**pg_params)
            with _conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_metadata (
                        filename TEXT PRIMARY KEY,
                        media_type TEXT NOT NULL,
                        chunks INTEGER NOT NULL,
                        ingested_at TEXT NOT NULL,
                        metadata_json TEXT
                    )
                """)
            _conn.commit()
            return _conn
        except Exception as e:
            print(f"⚠️ Ingestion tracker fallback to SQLite: {e}")

    path = _db_path or os.path.join(os.path.dirname(__file__), "ingestion_tracker.sqlite3")
    import sqlite3
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_metadata (
            filename TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            chunks INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            metadata_json TEXT
        )
    """)
    _conn.commit()
    return _conn


def load_tracker() -> Dict[str, Any]:
    """Return {"files": [ {...}, ... ]} compatible with existing frontend."""
    with _lock:
        c = _get_conn()
        cur = c.cursor() if hasattr(c, "cursor") else c
        try:
            cur.execute(
                "SELECT filename, media_type, chunks, ingested_at, metadata_json FROM ingestion_metadata ORDER BY ingested_at DESC"
            )
            rows = cur.fetchall()
        finally:
            if hasattr(cur, "close"): cur.close()
            
    files = []
    for r in rows:
        fn, mtype, chunks, date, meta_json = r
        meta = {}
        if meta_json:
            try:
                import json
                meta = json.loads(meta_json)
            except Exception:
                pass
        files.append({
            "filename": fn,
            "type": mtype,
            "chunks": chunks,
            "date": date,
            "metadata": meta,
        })
    return {"files": files}


def add_to_tracker(filename: str, media_type: str, chunks: int, metadata: Optional[Dict] = None) -> None:
    import json
    meta_json = json.dumps(metadata or {}, default=str)
    ph = _get_placeholder()
    # SQL query difference: SQLite uses REPLACE INTO or INSERT OR REPLACE
    # Postgres uses INSERT INTO ... ON CONFLICT
    is_postgres = ph == "%s"
    
    with _lock:
        c = _get_conn()
        cur = c.cursor() if hasattr(c, "cursor") else c
        try:
            if is_postgres:
                cur.execute(
                    """INSERT INTO ingestion_metadata (filename, media_type, chunks, ingested_at, metadata_json)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (filename) DO UPDATE SET
                       media_type=EXCLUDED.media_type, chunks=EXCLUDED.chunks, 
                       ingested_at=EXCLUDED.ingested_at, metadata_json=EXCLUDED.metadata_json""",
                    (filename, media_type, chunks, datetime.utcnow().isoformat(), meta_json),
                )
            else:
                cur.execute(
                    """INSERT OR REPLACE INTO ingestion_metadata (filename, media_type, chunks, ingested_at, metadata_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (filename, media_type, chunks, datetime.utcnow().isoformat(), meta_json),
                )
            c.commit()
        finally:
            if hasattr(cur, "close"): cur.close()


def remove_from_tracker(filename: str) -> None:
    ph = _get_placeholder()
    with _lock:
        c = _get_conn()
        cur = c.cursor() if hasattr(c, "cursor") else c
        try:
            cur.execute(f"DELETE FROM ingestion_metadata WHERE filename = {ph}", (filename,))
            c.commit()
        finally:
            if hasattr(cur, "close"): cur.close()
