"""
Ingestion tracker: SQLite for local (and Cloud SQL later).
Replaces JSON file so the same code works locally and on Cloud Run with a persistent volume or Cloud SQL.
"""
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

_db_path = os.getenv("INGESTION_TRACKER_DB", "")
_conn = None
_lock = threading.Lock()


def _get_conn():
    global _conn
    if _conn is not None:
        return _conn
    path = _db_path or os.path.join(os.path.dirname(__file__), "ingestion_tracker.sqlite3")
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
        cur = c.execute(
            "SELECT filename, media_type, chunks, ingested_at, metadata_json FROM ingestion_metadata ORDER BY ingested_at DESC"
        )
        rows = cur.fetchall()
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
    with _lock:
        c = _get_conn()
        c.execute(
            """INSERT OR REPLACE INTO ingestion_metadata (filename, media_type, chunks, ingested_at, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (filename, media_type, chunks, datetime.utcnow().isoformat(), meta_json),
        )
        c.commit()


def remove_from_tracker(filename: str) -> None:
    with _lock:
        c = _get_conn()
        c.execute("DELETE FROM ingestion_metadata WHERE filename = ?", (filename,))
        c.commit()
